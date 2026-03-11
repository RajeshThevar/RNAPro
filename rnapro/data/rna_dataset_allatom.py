# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
RNA Dataset for Structure Prediction

This module provides the RNADataset class for loading and processing RNA structure data
from the Stanford RNA 3D Folding Kaggle competition. It handles:

- Sequence and coordinate loading from CSV files
- MSA feature generation from FASTA files
- Template structure features
- Chemical mapping profile prediction
- Random cropping with aligned MSA/template/chemical profile cropping
- Temporal filtering for train/validation splits

Data Pipeline:
    CSV Files → Filtering & Validation → Cropping → Feature Extraction → 
    MSA/Template Features → Coordinate Building → Model-Ready Dictionary

Key Components:
    - RNADataset: Main dataset class with dual output modes (training/inference)
    - Constants: ATOM_NAMES, nucleotide mappings, default values
    - Helper methods: cropping, MSA reading, template processing, feature assembly
"""

import os
import json
import traceback
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter("ignore", BiopythonWarning)
warnings.simplefilter("ignore", PDBConstructionWarning)

from tqdm import tqdm
from biotite.structure import AtomArray
from biotite.structure.atoms import AtomArray
from torch.utils.data import Dataset

from rnapro.data.msa_featurizer import MSA_MAX_SIZE, SEQ_LIMITS
from rnapro.data.json_to_feature import SampleDictToFeatures
from rnapro.data.tokenizer import TokenArray
from rnapro.data.utils import (
    data_type_transform,
    make_dummy_feature,
)
from rnapro.utils.logger import get_logger
from rnapro.utils.torch_utils import dict_to_tensor

logger = get_logger(__name__)

# ==================== RNA Atom and Nucleotide Constants ====================
# Standard RNA atoms (27 total) in canonical order
ATOM_NAMES = [
    "P",
    "OP1", 
    "OP2",
    "O5'",
    "O3'",
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "O4'",
    "C5'",
    "N1",
    "C2",
    "O2",
    "N3",
    "C4",
    "N4",
    "C5",
    "C6",
    "O4",
    "N9",
    "N7",
    "C8",
    "N6",
    "N2",
    "O6"
]

VALID_RNA_NUCLEOTIDES = {'A', 'C', 'G', 'U'}
INVALID_SEQUENCE_CHARS = {'-', 'X'}

# ==================== Default Values and Limits ====================
DEFAULT_CROP_SIZE = 256
INVALID_COORDINATE_VALUE = -1e8
TOKEN_COUNT_DEFAULT = 1
ZERO_COORDINATE = np.array([0.0, 0.0, 0.0])
ATOM_NAME_TO_INDEX = {atom_name: idx for idx, atom_name in enumerate(ATOM_NAMES)}

# Template limits
MAX_TEMPLATES = 4  # Maximum number of templates read per target (up to 40 available)
MAX_TEMPLATE_FEATURES = 4  # Maximum number of templates after featurization

# ==================== Chemical Mapping Constants ====================
# Odd indices (1,3,5,7,9) for 2A3 reactivity in RibonanzaNet2 output
CHEMICAL_MAPPING_DMS_INDICES = slice(1, None, 2)



class RNADataset(Dataset):
    """
    PyTorch Dataset for RNA structure prediction from the Stanford RNA 3D Folding Kaggle competition.
    
    Processes RNA sequences, all-atom coordinates, MSAs, templates, and optional chemical mapping
    profiles for training and inference of RNA structure prediction models.
    
    Data Format:
    -----------
    Expects CSV files with the following structure:
    
    1. sequences.csv (required):
       - target_id, sequence, temporal_cutoff, description, all_sequences
    
    2. coords.csv (required):
       - ID (target_id_residue_num), resname, resid, {atom_name}_{x/y/z}_1
       - Supports 27 standard RNA atoms
    
    3. templates.csv (optional):
       - ID, resname, resid, {x/y/z}_{template_idx} for C1' atoms
       - Up to 40 templates per target, uses first MAX_TEMPLATES (4)
    
    4. MSA FASTA files (optional):
       - {target_id}.MSA.fasta in rna_msa_dir
       - Standard RNA sequences (A,C,G,U) with gaps (-)
    
    Key Features:
    ------------
    - Random sequence cropping with aligned MSA/template/chemical profile cropping
    - Temporal filtering for train/validation splits
    - Chemical mapping profile prediction via RibonanzaNet2 or SGNM
    - MSA feature generation with proper gap/deletion encoding
    - Template structure features (C1' coordinates)
    - Dual output modes: dict (training) or tuple (inference with structure saving)
    
    Usage:
    ------
    Training:
        dataset = RNADataset(..., inference_mode=False)  # Returns dict
    
    Inference/Evaluation:
        dataset = RNADataset(..., inference_mode=True)   # Returns (dict, atom_array, entity_poly_type)
    
    References:
    ----------
    https://www.kaggle.com/competitions/stanford-rna-3d-folding/data
    """
    def __init__(
        self,
        data_dir: str,
        use_msa: bool = True,
        crop_size=420,
        is_eval=False,
        use_template='None',
        msa_dir='',
        use_cluster='na',
        num_templates=4,
        temporal_cutoff=None,
        mode='train',
    ) -> None:
        """
        Initialize the RNA Dataset with validation and configuration.
        
        Args:
            sequences_csv_fpath: Path to sequences CSV (required).
            coords_csv_fpath: Path to all-atom coordinates CSV (required).
            max_n_token: Max residues per sequence (-1 for unlimited).
            cropping_configs: Dict with 'crop_size' key (default: 256).
            template_featurizer: Template featurizer instance (currently unused).
            name: Dataset name for logging.
            temporal_cutoff: ISO date (YYYY-MM-DD) to exclude newer structures.
                Example: "2022-05-27" excludes CASP15 from training.
            enable_chemical_mapping: Generate chemical profiles via RibonanzaNet2.
            ribonanzanet2_config_fpath: RibonanzaNet2 config (required if enable_chemical_mapping=True).
            ribonanzanet2_checkpoint_fpath: RibonanzaNet2 checkpoint (required if enable_chemical_mapping=True).
            enable_templates: Load and use template structures.
            templates_csv_fpath: Templates CSV with C1' coordinates (up to 40 per target).
            inference_mode: False=dict output (training), True=tuple output (structure saving).
            enable_msa: Generate MSA features from FASTA files.
            rna_msa_dir: Directory with {target_id}.MSA.fasta files.
            **kwargs: Unused additional arguments.
        """
        super(RNADataset, self).__init__()
        self.mode = mode
        self.use_template = use_template
        sequences_csv_fpath = Path(os.path.join(data_dir, 'train_sequences.v2.1.csv'))
        coords_csv_fpath = Path(os.path.join(data_dir, 'train_allatom.v2.1.csv'))
        templates_csv_fpath = Path(os.path.join(data_dir, 'train_templates.v2.1.csv'))
        target_resolution_fpath = Path(os.path.join(data_dir, 'target_resolution.json'))
        self.template_features = {}
        self.template_features_names = []
        if self.use_template == 'ca_precomputed':
            template_pt_fpath = os.path.join(data_dir, 'template_features.pt')
            self.template_features = torch.load(template_pt_fpath, weights_only=False)
            self.template_features_names = list(self.template_features.keys())
        self.inference_mode = mode == 'test'
        # Validate inputs
        # Check file existence
        if not Path(sequences_csv_fpath).exists():
            raise FileNotFoundError(f"Sequences CSV file not found: {sequences_csv_fpath}")
        if not Path(coords_csv_fpath).exists():
            raise FileNotFoundError(f"Coordinates CSV file not found: {coords_csv_fpath}")
            
        # if max_n_token <= 0 and max_n_token != -1:
            # raise ValueError(f"max_n_token must be positive or -1 (no limit), got: {max_n_token}")
            
        # Validate templates file if provided
        if self.use_template == 'ca' and templates_csv_fpath and not Path(templates_csv_fpath).exists():
            raise FileNotFoundError(f"Templates file not found: {templates_csv_fpath}")
        
        rna_msa_dir = Path(msa_dir)
        # Validate MSA directory if provided
        if rna_msa_dir and not Path(rna_msa_dir).exists():
            raise FileNotFoundError(f"RNA MSA directory not found: {rna_msa_dir}")
        
        # Store configuration parameters
        self.sequences_csv_fpath = Path(sequences_csv_fpath)
        self.coords_csv_fpath = Path(coords_csv_fpath)
        self.crop_size = crop_size
        self.temporal_cutoff = pd.to_datetime(temporal_cutoff) if temporal_cutoff else None
        self.templates_csv_fpath = Path(templates_csv_fpath) if templates_csv_fpath else None
        self.template_featurizer = None
        self.rna_msa_dir = Path(rna_msa_dir) if rna_msa_dir else None
        self.rna_msa_seq_limit = min(MSA_MAX_SIZE, SEQ_LIMITS.get("nucleotide", 10000)) + 1
        self.target_resolution = self._load_target_resolution_map(target_resolution_fpath)

        # Read data - always use all-atom format
        # First read sequences with temporal filtering to get the list of valid target_ids
        self.data_list = self.read_sequences(sequences_csv_fpath=self.sequences_csv_fpath)
        
        # Then read coords only for the sequences that passed filtering
        valid_target_ids = {item["name"] for item in self.data_list}
        self.coords_dict = self.read_coords(
            coords_csv_fpath=self.coords_csv_fpath, 
            valid_target_ids=valid_target_ids
        )
        
        # Validate that sequences match between sequence CSV and coords CSV
        self._validate_sequence_consistency()

        # Read template data if provided
        if self.templates_csv_fpath and self.use_template == 'ca':
            self.templates_dict = self.read_templates(
                templates_csv_fpath=self.templates_csv_fpath,
                valid_target_ids=valid_target_ids
            )
        else:
            self.templates_dict = {}


        logger.info(
            f"Loaded {len(self.data_list)} samples from {self.sequences_csv_fpath}"
        )
        if self.rna_msa_dir:
            # Check how many samples have corresponding MSA files
            msa_files = {f.name[:-len(".MSA.fasta")] for f in self.rna_msa_dir.glob("*.MSA.fasta")}
            n_with_msa = sum(1 for item in self.data_list if item["name"] in msa_files)
            logger.info(f"RNA MSA directory: {self.rna_msa_dir}, {n_with_msa}/{len(self.data_list)} samples have MSA files")

        self.ribonanza_net_tokenizer= {
            'A': 0,
            'C': 1,
            'G': 2,
            'U': 3,
            'PAD': 4,
            'X': 5,
        }

    @staticmethod
    def _load_target_resolution_map(
        target_resolution_fpath: Path,
    ) -> dict[str, float]:
        if not target_resolution_fpath.exists():
            return {}
        try:
            with target_resolution_fpath.open() as f:
                raw_map = json.load(f)
        except Exception as e:
            logger.warning(
                "Failed to load target resolution map %s: %s",
                target_resolution_fpath,
                e,
            )
            return {}

        resolution_map = {}
        for target_id, resolution in raw_map.items():
            try:
                resolution_map[str(target_id)] = float(resolution)
            except (TypeError, ValueError):
                continue

        logger.info(
            "Loaded target resolution map: %d entries from %s",
            len(resolution_map),
            target_resolution_fpath,
        )
        return resolution_map

    
    def read_sequences(
        self, sequences_csv_fpath: Union[str, Path]
    ) -> list[dict]:
        """
        Read and filter RNA sequences from CSV.
        
        Args:
            sequences_csv_fpath: Path to sequences CSV.
                
        Returns:
            List of dicts with "name" and "sequences" (containing rnaSequence data).
            
        Filters:
            - Invalid characters ('-', 'X')
            - Exceeds max_n_token
            - Published after temporal_cutoff
        """
        try:
            df = pd.read_csv(sequences_csv_fpath)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read sequences CSV file {sequences_csv_fpath}: {e}") from e
        
        try:
            df["temporal_cutoff"] = pd.to_datetime(df["temporal_cutoff"])
        except Exception as e:
            logger.warning(f"Failed to parse temporal_cutoff dates: {e}. Continuing without temporal filtering.")
            df["temporal_cutoff"] = pd.NaT

        data_list = []
        print('#'*20, 'before filtering', len(df))
        num_cutoff = 0
        num_limit = 0
        num_invalid = 0
        for _, row in df.iterrows():
            target_id = row["target_id"]
            sequence = row["sequence"]

            # Filter sequences with invalid nucleotide characters
            if any(char in sequence for char in INVALID_SEQUENCE_CHARS):
                logger.debug(f"Skipping sequence {target_id} with invalid characters")
                continue

            # # Skip sequences exceeding maximum token limit
            # if self.max_n_token > 0 and len(sequence) > self.max_n_token:
            #     logger.debug(f"Skipping sequence {target_id} (length {len(sequence)} > {self.max_n_token})")
            #     continue
            
            if self.temporal_cutoff is not None:
                # Skip if cutoff is after self.temporal_cutoff
                if self.mode == 'test':
                    if row["temporal_cutoff"] < self.temporal_cutoff:
                        num_cutoff += 1
                        continue
                else:
                    if row["temporal_cutoff"] >= self.temporal_cutoff:
                        num_cutoff += 1
                        continue

            data_list.append(
                {
                    "sequences": [
                        {
                            "rnaSequence": {
                                "sequence": sequence,
                                "count": TOKEN_COUNT_DEFAULT,
                                "msa": {
                                    "precomputed_msa_dir": self.rna_msa_dir,
                                    "pairing_db": "",
                                },
                            },
                        }
                    ],
                    "name": target_id,
                }
            )
        print('#'*20, 'after filtering', len(data_list))
        print('#'*20, f"num_cutoff: {num_cutoff}, num_limit: {num_limit}, num_invalid: {num_invalid}")
        return data_list

    def read_coords(
        self, coords_csv_fpath: Union[str, Path], valid_target_ids: set[str]
    ) -> dict[str, dict]:
        """
        Read and process all-atom coordinates from CSV with vectorized operations.
        
        Args:
            coords_csv_fpath: Path to coordinates CSV.
            valid_target_ids: Target IDs to include (for filtering).
            
        Returns:
            Dict mapping target_id to {"seq": str, "xyz": list[dict]}.
            Missing coordinates filled with INVALID_COORDINATE_VALUE (-1e8).
        """
        
        # Prepare coordinate column names for vectorized extraction
        coord_cols = []
        for atom_name in ATOM_NAMES:
            coord_cols.extend([f"{atom_name}_x_1", f"{atom_name}_y_1", f"{atom_name}_z_1"])
        usecols = ["ID", "resname"] + coord_cols

        coords_dict = {}
        try:
            chunk_iter = pd.read_csv(
                coords_csv_fpath,
                usecols=usecols,
                chunksize=50000,
            )
        except Exception as e:
            raise FileNotFoundError(f"Failed to read coordinates CSV file {coords_csv_fpath}: {e}") from e

        saw_any_rows = False
        for chunk in tqdm(chunk_iter, desc="Processing coordinate chunks"):
            chunk.fillna(INVALID_COORDINATE_VALUE, inplace=True)
            chunk["target_id"] = chunk["ID"].str.rsplit("_", n=1).str[0]
            chunk = chunk[chunk["target_id"].isin(valid_target_ids)]
            if chunk.empty:
                continue
            saw_any_rows = True
            for target_id, group in chunk.groupby("target_id", sort=False):
                seq = "".join(group["resname"].values)
                coords_matrix = group[coord_cols].to_numpy(dtype=np.float32, copy=False)
                coords_reshaped = coords_matrix.reshape(len(group), len(ATOM_NAMES), 3)

                if target_id in coords_dict:
                    coords_dict[target_id]["seq_parts"].append(seq)
                    coords_dict[target_id]["xyz_parts"].append(coords_reshaped.copy())
                else:
                    coords_dict[target_id] = {
                        "seq_parts": [seq],
                        "xyz_parts": [coords_reshaped.copy()],
                    }

        if not saw_any_rows:
            logger.warning("No labels found for any of the valid target IDs")
            return {}

        compact_coords_dict = {}
        for target_id, parts in tqdm(coords_dict.items(), desc="Compacting coordinates"):
            compact_coords_dict[target_id] = {
                "seq": "".join(parts["seq_parts"]),
                "xyz": np.concatenate(parts["xyz_parts"], axis=0),
            }

        return compact_coords_dict

    def read_templates(
        self, templates_csv_fpath: Union[str, Path], valid_target_ids: set[str]
    ) -> dict[str, list[dict]]:
        """
        Read and process RNA template C1' coordinates from CSV.
        
        Args:
            templates_csv_fpath: Path to templates CSV with columns:
                ID, resname, resid, {x/y/z}_{template_idx} (1-40).
            valid_target_ids: Target IDs to include.
            
        Returns:
            Dict mapping target_id to list of template dicts with:
                {"coords": np.ndarray (n_res, 3), "coords_mask": np.ndarray (n_res,)}
        """
        try:
            df = pd.read_csv(templates_csv_fpath)
        except Exception as e:
            raise FileNotFoundError(f"Failed to read templates CSV file {templates_csv_fpath}: {e}") from e
        
        df.fillna(INVALID_COORDINATE_VALUE, inplace=True)
        
        # Pre-compute target_ids to avoid string operations in loop
        df['target_id'] = df['ID'].str.rsplit('_', n=1).str[0]
        
        # Filter to only valid target IDs early to save processing time
        df = df[df['target_id'].isin(valid_target_ids)]
        
        if len(df) == 0:
            logger.warning("No template data found for any of the valid target IDs")
            return {}
        
        templates_dict = defaultdict(list)
        
        # Process each template index (1 to MAX_TEMPLATES)
        for template_idx in tqdm(range(1, MAX_TEMPLATES + 1), desc="Processing templates"):
            # Check if this template index has C1' coordinate data
            coord_cols = [
                f"x_{template_idx}",
                f"y_{template_idx}", 
                f"z_{template_idx}"
            ]
            
            # Check if template columns exist
            existing_coord_cols = [col for col in coord_cols if col in df.columns]
            
            if len(existing_coord_cols) != 3:
                # Need all 3 coordinate columns for this template
                continue
            
            # Filter rows where this template has valid data (not all NaN/invalid)
            template_df = df.dropna(subset=existing_coord_cols, how='all')
            if len(template_df) == 0:
                continue
                
            # Group by target_id and process efficiently
            for target_id, group in template_df.groupby('target_id'):
                # Extract C1' coordinates only
                coords_matrix = group[existing_coord_cols].values.astype(np.float32)  # Shape: (n_residues, 3)
                n_residues = len(group)
                
                # Create coordinate mask (1 for valid coords, 0 for missing)
                coords_mask = (coords_matrix[:, 0] > INVALID_COORDINATE_VALUE).astype(np.int32)  # Shape: (n_residues,)
                
                # Create template data structure
                template_data = {
                    "coords": coords_matrix,  # Shape: (n_residues, 3)
                    "coords_mask": coords_mask,  # Shape: (n_residues,)
                }
                
                templates_dict[target_id].append(template_data)
        
        # Convert defaultdict to regular dict
        templates_dict = dict(templates_dict)
        
        logger.info(f"Loaded templates for {len(templates_dict)} targets, "
                   f"total templates: {sum(len(templates) for templates in templates_dict.values())}")
        
        return templates_dict

    # ==================== Validation Methods ====================
    
    def _validate_sequence_consistency(self) -> None:
        """
        Remove samples with missing coordinates or sequence mismatches.
        
        Modifies self.data_list to exclude invalid samples and logs warnings.
        """
        mismatched = []
        missing_labels = []
        
        for item in self.data_list:
            target_id = item["name"]
            sequence_from_csv = item["sequences"][0]["rnaSequence"]["sequence"]
            
            if target_id not in self.coords_dict:
                missing_labels.append(target_id)
                continue
                
            sequence_from_labels = self.coords_dict[target_id]["seq"]
            
            if sequence_from_csv != sequence_from_labels:
                mismatched.append((target_id, len(sequence_from_csv), len(sequence_from_labels)))
        
        if missing_labels:
            logger.warning(f"Found {len(missing_labels)} sequences without corresponding coords: {missing_labels[:5]}...")
            # Remove sequences without coords from data_list
            self.data_list = [item for item in self.data_list if item["name"] not in missing_labels]
            
        if mismatched:
            logger.warning(f"Found {len(mismatched)} sequences with mismatched lengths between CSV and labels")
            for target_id, csv_len, label_len in mismatched[:5]:
                logger.warning(f"  {target_id}: CSV={csv_len}, Labels={label_len}")
            # Remove mismatched sequences from data_list
            mismatched_ids = {item[0] for item in mismatched}
            self.data_list = [item for item in self.data_list if item["name"] not in mismatched_ids]

    # ==================== Cropping and Processing Helper Methods ====================
    
    def _apply_sequence_cropping(
        self,
        single_sample_dict: Mapping[str, Any],
        xyz_dict_list: np.ndarray,
        seq: str
    ) -> tuple[dict, list[dict], Optional[np.ndarray], list[dict], int]:
        """
        Apply random cropping to sequences, coordinates, templates, and chemical profiles.
        
        Returns:
            (cropped_dict, cropped_coords, cropped_chemical_profile, cropped_templates, crop_start)
        """
        crop_start = 0  # Default for no cropping
        cropped_chemical_profile = None
        cropped_single_sample_dict = deepcopy(single_sample_dict)
        sample_name = single_sample_dict["name"]
        
        # Get template data for this sample
        template_data = self.templates_dict.get(sample_name, [])
        print('#'*20, 'before crop', len(seq))
        if len(seq) > self.crop_size and self.crop_size > 0:
            # Determine crop boundaries
            crop_start = np.random.randint(0, len(seq) - self.crop_size + 1)
            crop_end = crop_start + self.crop_size
            
            # Crop the sequence in the sample dict
            cropped_seq = seq[crop_start:crop_end]
            cropped_single_sample_dict["sequences"][0]["rnaSequence"]["sequence"] = cropped_seq
            
            # Crop xyz coordinates
            cropped_xyz_dict_list = xyz_dict_list[crop_start:crop_end]
            
            # Crop templates
            cropped_templates = self._crop_templates(template_data, crop_start, crop_end)
            
            # Get and crop chemical mapping profile if available
            # cropped_chemical_profile = self._crop_chemical_profile(
                # sample_name, seq, crop_start, crop_end
            # )
        else:
            cropped_xyz_dict_list = xyz_dict_list
            cropped_templates = template_data
            cropped_seq = single_sample_dict["sequences"][0]["rnaSequence"]["sequence"]
            
            # Handle chemical profile for non-cropped case
            # cropped_chemical_profile = self._get_chemical_profile(sample_name)
        print('#'*20, 'after crop', len(cropped_seq))

        return cropped_single_sample_dict, cropped_xyz_dict_list, cropped_templates, crop_start, cropped_seq
    
    
    def _get_chemical_profile(self, sample_name: str) -> Optional[np.ndarray]:
        """Get chemical mapping profile without cropping."""
        if (self.enable_chemical_mapping and 
            sample_name in self.coords_dict and 
            "chemical_mapping_profile" in self.coords_dict[sample_name]):
            return self.coords_dict[sample_name]["chemical_mapping_profile"]
        return None

    def _read_msa_file(self, target_id: str) -> Optional[list[str]]:
        """
        Read MSA sequences from {target_id}.MSA.fasta file.
        
        Args:
            target_id: Target identifier for filename construction.
            
        Returns:
            List of RNA sequences (with gaps '-') or None if unavailable.
            
        Format:
            Standard FASTA with RNA nucleotides (A,C,G,U,N) and gaps (-).
        """
            
        msa_file_path = self.rna_msa_dir / f"{target_id}.MSA.fasta"
        if not msa_file_path.exists():
            logger.debug(f"MSA file not found for target {target_id}: {msa_file_path}")
            return None
        
        try:
            sequences = []
            with open(msa_file_path, 'r') as f:
                current_seq = ""
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_seq:
                            sequences.append(current_seq)
                            current_seq = ""
                    else:
                        current_seq += line
                
                # Add the last sequence
                if current_seq:
                    sequences.append(current_seq)
            
            return sequences if sequences else None
            
        except (IOError, OSError) as e:
            logger.warning(f"Error reading MSA file for target {target_id}: {e}")
            return None

    def _crop_templates(
        self,
        templates: list[dict],
        crop_start: int,
        crop_end: int
    ) -> list[dict]:
        """Crop template coordinates to match sequence cropping."""
        cropped_templates = []
        
        for template in templates:
            # Crop template coordinates and masks
            coords = template["coords"][crop_start:crop_end]  # Shape: (cropped_len, 3)
            coords_mask = template["coords_mask"][crop_start:crop_end]  # Shape: (cropped_len,)
            
            cropped_template = {
                "coords": coords,
                "coords_mask": coords_mask,
            }
            
            cropped_templates.append(cropped_template)
        
        return cropped_templates

    # ==================== Coordinate Building Methods ====================
    
    def _build_coordinates_and_masks(
        self,
        token_array: TokenArray,
        atom_array: AtomArray,
        cropped_xyz_dict_list: list[dict]
    ) -> tuple[list[np.ndarray], list[int]]:
        """
        Extract atomic coordinates and validity masks from coordinate dictionaries.
        
        Returns:
            (coordinate_list, coordinate_mask_list) with 1=valid, 0=missing.
        """
        coordinate_list = []
        coordinate_mask_list = []

        for token_idx, token in enumerate(token_array):
            for atom_idx in token.atom_indices:
                atom = atom_array[atom_idx]
                atom_array_idx = ATOM_NAME_TO_INDEX.get(atom.atom_name)
                if atom_array_idx is None:
                    coordinate_list.append(ZERO_COORDINATE.copy())
                    coordinate_mask_list.append(0)
                    continue
                coord = cropped_xyz_dict_list[token_idx, atom_array_idx]
                coordinate_list.append(coord)
                if coord[0] <= INVALID_COORDINATE_VALUE:
                    coordinate_mask_list.append(0)
                else:
                    coordinate_mask_list.append(1)

        return coordinate_list, coordinate_mask_list

    # ==================== MSA Feature Methods ====================
    
    def _create_msa_features(
        self,
        single_sample_dict: Mapping[str, Any],
        atom_array: AtomArray,
        token_array: TokenArray,
        crop_start: int = 0,
        crop_end: Optional[int] = None
    ) -> dict:
        """
        Generate MSA features from FASTA files with cropping alignment.
        
        Nucleotide Encoding:
            Local indices: A=0, G=1, C=2, U=3, N=4, gap=5
            Unified indices: A=21, G=22, C=23, U=24, N=25, gap=31
            
        Args:
            single_sample_dict: Sample dict with cropped target sequence.
            atom_array: AtomArray (unused).
            token_array: TokenArray (unused).
            crop_start: MSA crop start position.
            crop_end: MSA crop end position (None if no cropping).
            
        Returns:
            Dict with keys: msa, has_deletion, deletion_value, deletion_mean, 
            profile, rna_unpair_num_alignments. Empty dict if no MSA available.
            
        Note:
            MSA sequences cropped to match target sequence length.
            Profile is 32D covering proteins (0-20), RNA (21-25), DNA (26-30), gap (31).
        """
        sample_name = single_sample_dict["name"]
        
        # Read MSA sequences for this target
        msa_sequences = self._read_msa_file(sample_name)
        if not msa_sequences:
            # Return empty MSA features if no data available
            logger.debug(f"No MSA data available for {sample_name}")
            return {}
        if len(msa_sequences) > self.rna_msa_seq_limit:
            msa_sequences = msa_sequences[:self.rna_msa_seq_limit]

        try:
            from rnapro.data.constants import RNA_NT_TO_ID
            
            # The sequence in single_sample_dict is already cropped, so use it as-is
            sequence = single_sample_dict["sequences"][0]["rnaSequence"]["sequence"]
            seq_len = len(sequence)
                
            num_msa_seq = len(msa_sequences)
            
            # Use official RNA mapping that includes gap character at index 5
            rna_mapping_with_gap = RNA_NT_TO_ID
            
            # Create MSA feature arrays for cropped sequence length using unified residue indices
            # MSA embedder expects integer indices in unified residue space (converts to one-hot internally)
            from rnapro.data.constants import RNA_STD_RESIDUES, STD_RESIDUES_WITH_GAP
            
            # Mapping from local RNA_NT_TO_ID indices to unified system indices
            rna_local_to_unified = {
                0: RNA_STD_RESIDUES["A"],  # A: 0 -> 21
                1: RNA_STD_RESIDUES["G"],  # G: 1 -> 22  
                2: RNA_STD_RESIDUES["C"],  # C: 2 -> 23
                3: RNA_STD_RESIDUES["U"],  # U: 3 -> 24
                4: RNA_STD_RESIDUES["N"],  # N: 4 -> 25
                5: STD_RESIDUES_WITH_GAP["-"],  # Gap: 5 -> 31
            }
            
            # Create MSA array with integer indices in unified 32D space
            # MSA embedder expects int64 indices, not one-hot encoding
            msa_array = np.full((num_msa_seq, seq_len), 25, dtype=np.int64)  # Default to 'N' (index 25)
            
            for seq_idx, msa_seq in enumerate(msa_sequences):
                # Apply cropping to MSA sequences to match target sequence cropping
                if crop_end is not None:
                    # Crop MSA sequence to match target sequence cropping
                    msa_seq_cropped = msa_seq[crop_start:crop_end] if len(msa_seq) > crop_start else msa_seq
                else:
                    msa_seq_cropped = msa_seq
                    
                for pos_idx, nucleotide in enumerate(msa_seq_cropped):
                    if pos_idx < seq_len:  # Ensure we don't exceed sequence length
                        # Map nucleotide to local index first, then to unified index
                        local_idx = rna_mapping_with_gap.get(nucleotide.upper(), 4)
                        unified_idx = rna_local_to_unified[local_idx]
                        # Store unified index directly
                        msa_array[seq_idx, pos_idx] = unified_idx
            
            # Create deletion features based on gap characters (index 31 in unified space)
            gap_idx = STD_RESIDUES_WITH_GAP["-"]  # Index 31
            has_deletion = (msa_array == gap_idx).astype(np.bool_)  # True where gaps are present
            
            # Apply arctan transformation to deletion counts for consistency with standard pipeline
            # Since we have binary gap presence (0 or 1), this maps: 0 -> 0, 1 -> ~0.187
            deletion_counts = has_deletion.astype(np.float32)  # Treat gap presence as count=1
            deletion_value = (2 / np.pi) * np.arctan(deletion_counts / 3)  # Standard arctan transform
            
            # Calculate deletion_mean (mean deletion probability across MSA sequences)
            deletion_mean = np.mean(deletion_value, axis=0).astype(np.float32)  # Shape: (seq_len,)
            
            # Create unified 32-dimensional profile by averaging MSA (including gaps)
            # Profile covers all residue types: proteins (0-20) + RNA (21-25) + DNA (26-30) + gap (31)
            # This matches the standard _make_msa_profile implementation used in InferenceMSAFeaturizer
            all_res_types = np.arange(32)
            res_type_hits = msa_array[..., None] == all_res_types[None, ...]
            res_type_counts = res_type_hits.sum(axis=0)
            profile = (res_type_counts / num_msa_seq).astype(np.float32)
            
            # Create MSA feature dictionary
            msa_features = {
                "msa": msa_array,
                "has_deletion": has_deletion, 
                "deletion_value": deletion_value,
                "deletion_mean": deletion_mean,
                "profile": profile,
                "rna_unpair_num_alignments": np.array([num_msa_seq], dtype=np.int32),
            }
            
            logger.debug(f"Created MSA features for {sample_name} with {num_msa_seq} sequences")
            return msa_features
            
        except Exception as e:
            logger.warning(f"Error generating MSA features for {sample_name}: {e}")
            return {}

    # ==================== Template Feature Methods ====================
    def _create_template_features_ca_precomputed(
        self,
        sample_name: str,
        sequence: str,
        crop_start: int,
        crop_end: int
    ) -> dict:
        """
        Convert template coordinates to model-compatible features.
        
        Args:
            sample_name: Name of the sample.
            
        Returns:
            Dict with template_coords, template_coords_mask, n_templates.
            Uses first MAX_TEMPLATE_FEATURES (4) templates, shuffled during training.
        """
        n_slots = 5  # Number of template slots in template_features.pt
        if sample_name in self.template_features_names:  # template exists
            if np.random.rand() < 0.5:
                random_idx = np.random.choice(n_slots, size=4, replace=True)
                template_ca = torch.from_numpy(self.template_features[sample_name]['xyz'][:, random_idx]).permute(1, 0, 2).float()
                template_ca = template_ca[:, crop_start:crop_end]
            else:
                template_ca = torch.zeros(4, len(sequence), 3)
        else:
            template_ca = torch.zeros(4, len(sequence), 3)

        template_features = {
            "template_coords": template_ca,  # Shape: (n_templates, n_residues, 3)
            "template_coords_mask": torch.ones(4, len(sequence)),  # Shape: (n_templates, n_residues)
            "n_templates": torch.tensor([4]),
        }
        
        return template_features
    def _create_template_features(
        self,
        templates: list[dict],
        token_array: TokenArray,
        atom_array: AtomArray
    ) -> dict:
        """
        Convert template coordinates to model-compatible features.
        
        Args:
            templates: List of template dicts with coords and coords_mask.
            token_array: Token array for dimension matching.
            atom_array: Atom array (unused).
            
        Returns:
            Dict with template_coords, template_coords_mask, n_templates.
            Uses first MAX_TEMPLATE_FEATURES (4) templates, shuffled during training.
        """
        if not templates:
            return {}
        
        # If template_featurizer is available, use it
        if self.template_featurizer is not None:
            # Convert templates to format expected by template_featurizer
            # This would need to be implemented based on the specific featurizer interface
            return self.template_featurizer.create_features(templates, token_array, atom_array)
        
        # Otherwise, initialize template feature arrays for C1' coordinates only
        n_residues = len(token_array)
        template_coords = np.zeros((MAX_TEMPLATE_FEATURES, n_residues, 3), dtype=np.float32)  # C1' coordinates
        template_coords_mask = np.zeros((MAX_TEMPLATE_FEATURES, n_residues), dtype=np.int32)  # C1' validity mask
        
        # Fill template data
        # Shuffle templates during training, use first MAX_TEMPLATE_FEATURES during evaluation
        if not self.inference_mode and len(templates) > MAX_TEMPLATE_FEATURES:
            # Training mode: randomly select templates via shuffled indices
            indices = np.random.permutation(len(templates))[:MAX_TEMPLATE_FEATURES]
            templates_to_use = [templates[i] for i in indices]
        else:
            # Evaluation mode (or fewer templates than needed): use first MAX_TEMPLATE_FEATURES templates
            templates_to_use = templates[:MAX_TEMPLATE_FEATURES]
        
        n_templates = 0  # Number of valid templates 
        for template_idx, template in enumerate(templates_to_use):
            coords = template["coords"]  # Shape: (template_residues, 3)
            coords_mask = template["coords_mask"]  # Shape: (template_residues,)
            
            # Ensure template data matches expected dimensions
            seq_len = min(coords.shape[0], n_residues)
            
            template_coords[template_idx, :seq_len] = coords[:seq_len]  # Copy C1' coordinates
            template_coords_mask[template_idx, :seq_len] = coords_mask[:seq_len]  # Copy validity mask

            # Count only templates with at least one valid coordinate
            if np.any(coords_mask[:seq_len]):
                n_templates += 1
        
        template_features = {
            "template_coords": torch.from_numpy(template_coords),  # Shape: (n_templates, n_residues, 3)
            "template_coords_mask": torch.from_numpy(template_coords_mask),  # Shape: (n_templates, n_residues)
            "n_templates": torch.tensor([n_templates]),
        }
        
        return template_features


    # ==================== Template Feature Methods ====================
    
    def _create_masked_template_features(
        self,
        cropped_xyz_dict_list: np.ndarray,
        n_templates=1
    ) -> dict:
        """
        Convert template coordinates to model-compatible features.
        
        Args:
            templates: List of template dicts with coords and coords_mask.
            token_array: Token array for dimension matching.
            atom_array: Atom array (unused).
            
        Returns:
            Dict with template_coords, template_coords_mask, n_templates.
            Uses first MAX_TEMPLATE_FEATURES (4) templates, shuffled during training.
        """

        ca_coordinates = torch.from_numpy(
            cropped_xyz_dict_list[:, ATOM_NAME_TO_INDEX["C1'"]]
        ).float()

        gt_xyz = ca_coordinates.clone()
        nan_mask = (gt_xyz <= -1e8).sum(1) > 0
        new_xyzs = []
        template_coords_mask = []
        for idx in range(n_templates):
            if idx == 0:
                gt_xyz_ = gt_xyz.clone()
            else:
                gt_xyz_ = gt_xyz.clone() +torch.randn_like(gt_xyz) * 0.1
            seq_len = gt_xyz_.shape[0]

            # Randomly choose block length (up to 50% of sequence)
            max_block_len = int(seq_len * 0.5)
            block_len = np.random.randint(1, max_block_len + 1)

            # Randomly choose start index
            start_idx = np.random.randint(0, seq_len - block_len + 1)
            end_idx = start_idx + block_len

            # Create mask for contiguous block
            template_mask = torch.zeros(seq_len, dtype=torch.bool, device=gt_xyz.device)
            template_mask[start_idx:end_idx] = True
            template_mask[nan_mask] = True

            # create mask for -10-8 coordinates
            

            masked_gt_xyz = gt_xyz_.clone()
            masked_gt_xyz[template_mask] = torch.tensor(np.nan, dtype=torch.float32, device=gt_xyz_.device)
            new_xyzs.append(masked_gt_xyz)
            template_coords_mask.append(~template_mask)
        template_coords = torch.stack(new_xyzs, dim=0)
        template_coords_mask = torch.stack(template_coords_mask, dim=0)



        template_features = {
            "template_coords": template_coords,  # Shape: (n_templates, n_residues, 3)
            "template_coords_mask": template_coords_mask,  # Shape: (n_templates, n_residues)
            "n_templates": torch.tensor([n_templates]),
        }
        
        return template_features

    # ==================== Main Processing Pipeline ====================
    
    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict, AtomArray, dict]:
        """
        Transform raw RNA sample into model-ready features.
        
        Pipeline: cropping → feature extraction → MSA/template generation → 
                 coordinate building → assembly
        
        Args:
            single_sample_dict: Sample dict with name and sequences.

        Returns:
            (data_dict, atom_array, entity_poly_type) where data_dict contains:
                input_feature_dict, coordinate, coordinate_mask, N_* counters,
                sample_name, chemical_mapping_profile (if enabled).
        """
        # Extract basic sequence and coordinate information
        try:
            sample_name = single_sample_dict["name"]
            xyz_dict_list = self.coords_dict[sample_name]["xyz"]
            seq = single_sample_dict["sequences"][0]["rnaSequence"]["sequence"]
        except KeyError as e:
            raise KeyError(f"Missing required data for sample processing: {e}") from e
        except IndexError as e:
            raise ValueError(f"Invalid sample data structure: {e}") from e

        # Apply sequence cropping if needed
        (cropped_single_sample_dict, 
         cropped_xyz_dict_list, 
         cropped_templates,
         crop_start,
         cropped_seq) = self._apply_sequence_cropping(single_sample_dict, xyz_dict_list, seq)

        # Generate molecular features from cropped data
        sample2feat = SampleDictToFeatures(cropped_single_sample_dict)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        if sample_name in self.target_resolution:
            features_dict["resolution"] = torch.tensor(
                [self.target_resolution[sample_name]], dtype=torch.float32
            )

        # Add specialized mask features for different model components
        features_dict["distogram_rep_atom_mask"] = torch.tensor(
            atom_array.distogram_rep_atom_mask, dtype=torch.long
        )
        # features_dict["sgnm_frame_atom_mask"] = torch.tensor(
        #     atom_array.sgnm_frame_atom_mask, dtype=torch.long
        # )
        entity_poly_type = sample2feat.entity_poly_type

        # Build coordinate arrays and validity masks
        coordinate_list, coordinate_mask_list = self._build_coordinates_and_masks(
            token_array, atom_array, cropped_xyz_dict_list
        )

        # Generate MSA features with cropping alignment
        # Calculate crop_end if cropping was applied (indicated by sequence length difference)
        original_seq_len = len(single_sample_dict["sequences"][0]["rnaSequence"]["sequence"])
        cropped_seq_len = len(cropped_single_sample_dict["sequences"][0]["rnaSequence"]["sequence"])
        crop_end = crop_start + cropped_seq_len if cropped_seq_len < original_seq_len else None
        msa_features = self._create_msa_features(
            cropped_single_sample_dict, atom_array, token_array, crop_start, crop_end
        )

        # Generate template features
        if self.use_template == 'ca':
            template_features = self._create_template_features(cropped_templates, token_array, atom_array)
        elif self.use_template == 'masked_templates':
            template_features = self._create_masked_template_features(cropped_xyz_dict_list)
        elif self.use_template == 'ca_precomputed':
            template_features = self._create_template_features_ca_precomputed(sample_name, cropped_seq, crop_start, crop_end)

        # Add dummy features for unimplemented components
        dummy_feats = []
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
            
        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)
        
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform features to appropriate data types
        feat = data_type_transform(feat_or_label_dict=features_dict)

        # tokenized sequence
        sequence=[self.ribonanza_net_tokenizer[nt] for nt in cropped_seq]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)
        feat['tokenized_seq'] = sequence
        # Assemble final data dictionary with all required components
        data = self._assemble_final_data(
            feat, coordinate_list, coordinate_mask_list, 
            entity_poly_type, token_array
        )

        return data, atom_array, entity_poly_type

    # ==================== Data Assembly Helper Methods ====================
    
    def _compute_molecular_statistics(self, feat: dict) -> dict:
        """Compute atom and token counts for each molecular type (ligand, protein, dna, rna)."""
        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )
        return stats

    def _format_statistics_key(self, key: str) -> str:
        """Convert 'molecule_type/unit' to 'N_abbreviated_type_unit' format."""
        type_, unit = key.split("/")
        if type_ == "protein":
            type_ = "prot"
        elif type_ == "ligand":
            type_ = "lig"
        # dna and rna remain unchanged
        return f"N_{type_}_{unit}"

    def _assemble_final_data(
        self,
        feat: dict,
        coordinate_list: list[np.ndarray],
        coordinate_mask_list: list[int],
        entity_poly_type: dict,
        token_array: TokenArray,
    ) -> dict:
        """
        Assemble complete data dictionary with features, coordinates, and statistics.
        
        Returns:
            Dict with input_feature_dict, coordinates, N_* counters, and chemical profiles.
        """
        data = {"input_feature_dict": feat}

        # Calculate basic dimensions
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]
        N_asym = len(torch.unique(feat["asym_id"]))

        # Add basic dimension counters
        data.update({
            "N_asym": torch.tensor([N_asym]),
            "N_token": torch.tensor([N_token]),
            "N_atom": torch.tensor([N_atom]),
            "N_msa": torch.tensor([N_msa]),
        })

        # Add molecular type statistics
        stats = self._compute_molecular_statistics(feat)
        stat_keys = [
            "protein/atom", "ligand/atom", "dna/atom", "rna/atom",
            "protein/token", "ligand/token", "dna/token", "rna/token",
        ]
        data.update({
            self._format_statistics_key(k): torch.tensor([stats[k]]) 
            for k in stat_keys
        })

        # Add entity type information
        data["entity_poly_type"] = entity_poly_type

        # Add coordinate data
        data["coordinate"] = torch.from_numpy(np.array(coordinate_list)).float()
        data["coordinate_mask"] = torch.from_numpy(np.array(coordinate_mask_list)).long()


        return data

    # ==================== Dataset Interface Methods ====================
    
    def __len__(self) -> int:
        return len(self.data_list)

    def augment_with_test_samples(self, test_dataset: 'RNADataset') -> None:
        """
        Add test samples to training set with masked structures.
        
        Enables learning sequence/MSA → chemical mapping relationships without
        leaking test structural information.
        
        Args:
            test_dataset: Test dataset to extract samples from.
            
        Side Effects:
            Adds samples to self.data_list and self.coords_dict.
            Coordinates masked with INVALID_COORDINATE_VALUE.
            Templates excluded for augmented samples.
        """
        n_test_samples = len(test_dataset.data_list)
        if n_test_samples == 0:
            logger.info("No test samples to add to training set")
            return
        
        # Store original training size
        original_train_size = len(self.data_list)
        
        # Add all test samples to training data_list
        for test_idx in range(n_test_samples):
            test_sample = test_dataset.data_list[test_idx].copy()
            
            # Mark as augmented sample for tracking
            test_sample["is_augmented"] = True
            test_sample["original_dataset"] = "test"
            
            self.data_list.append(test_sample)
        
        # Add test coords with masked coordinates (keep chemical mapping profile)
        for test_idx in range(n_test_samples):
            test_sample = test_dataset.data_list[test_idx]
            sample_name = test_sample["name"]
            
            if sample_name in test_dataset.coords_dict:
                test_coords = test_dataset.coords_dict[sample_name]
                
                # Create masked coordinate entry
                masked_coords = {
                    "seq": test_coords["seq"],
                    "xyz": self._create_masked_xyz(test_coords["xyz"]),
                }
                
                # Copy chemical mapping profile if available
                if self.enable_chemical_mapping and "chemical_mapping_profile" in test_coords:
                    masked_coords["chemical_mapping_profile"] = test_coords["chemical_mapping_profile"]
                
                self.coords_dict[sample_name] = masked_coords
        
        # Don't add test templates - templates_dict entries are simply not created
        # for augmented samples, which will result in empty template features
        
        logger.info(
            f"Augmented training set: {original_train_size} original samples + "
            f"{n_test_samples} test samples = "
            f"{len(self.data_list)} total samples"
        )
    
    def _create_masked_xyz(self, xyz_dict_list: np.ndarray) -> np.ndarray:
        """Set all coordinates to INVALID_COORDINATE_VALUE for test set augmentation."""
        return np.full_like(xyz_dict_list, INVALID_COORDINATE_VALUE, dtype=np.float32)

    def __getitem__(self, idx: int) -> Union[dict, tuple[dict, AtomArray, dict]]:
        """
        Retrieve and process RNA sample for model input.

        Args:
            idx: Sample index (0-based).

        Returns:
            If inference_mode=False (training):
                dict with input_feature_dict, coordinate, coordinate_mask, N_* counters,
                sample_name, sample_index, chemical_mapping_profile (if enabled).
                
            If inference_mode=True (inference/evaluation):
                (data_dict, atom_array, entity_poly_type) for structure saving.
        """
        try:
            single_sample_dict = self.data_list[idx].copy()
        except IndexError:
            raise IndexError(f"Sample index {idx} out of range (dataset size: {len(self.data_list)})")
        
        try:
            data, atom_array, entity_poly_type = self.process_one(single_sample_dict)
            data["sample_name"] = single_sample_dict["name"]
            data["sample_index"] = idx
        except Exception as e:
            logger.error(f"Failed to process sample {idx} ({single_sample_dict.get('name', 'unknown')}): {e}")
            raise
        
        if self.inference_mode:
            # Return tuple for structure saving (evaluation with structure saving)
            return data, atom_array, entity_poly_type
        else:
            # Return just data dict for normal training (backward compatibility)
            return data
