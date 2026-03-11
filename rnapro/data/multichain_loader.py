"""Data loader factory for multi-chain competition fine-tuning.

Creates DataLoader instances backed by MultiChainDataset, which uses
pre-computed bioassembly_dict .pkl.gz files (from preprocess_bioassembly.py)
through the BaseSingleDataset pipeline.

Mirrors the interface of rna_dataset_allatom_loader.get_dataloaders() so
that AF3Trainer.init_data() can swap between loaders based on config.
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

import pandas as pd
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, DistributedSampler

from rnapro.data.dataloader import DistributedDataLoader
from rnapro.data.multichain_dataset import MultiChainDataset
from rnapro.utils.distributed import DIST_WRAPPER
from rnapro.utils.logger import get_logger

logger = get_logger(__name__)


def _deterministic_holdout_pdbs(
    pdb_ids: list[str],
    holdout_fraction: float = 0.05,
) -> tuple[list[str], list[str]]:
    """Create a deterministic PDB-level holdout split.

    Used as a leakage-safe fallback when temporal_cutoff does not yield a
    non-empty validation set.
    """
    unique_ids = sorted(set(pdb_ids))
    if len(unique_ids) <= 1:
        return unique_ids, []

    n_holdout = max(1, int(round(len(unique_ids) * holdout_fraction)))
    ranked = sorted(
        unique_ids,
        key=lambda pdb_id: hashlib.md5(pdb_id.encode("utf-8")).hexdigest(),
    )
    val_pdbs = ranked[:n_holdout]
    train_pdbs = [pdb_id for pdb_id in ranked if pdb_id not in set(val_pdbs)]
    return train_pdbs, val_pdbs


def _build_pdb_split(
    indices_csv: str,
    temporal_cutoff: str,
) -> tuple[list[str], list[str]]:
    """Build leakage-safe train/val PDB splits from indices.csv."""
    indices_df = pd.read_csv(indices_csv)
    if "pdb_id" not in indices_df.columns:
        raise ValueError(f"indices.csv missing pdb_id column: {indices_csv}")

    pdb_ids = sorted(indices_df["pdb_id"].astype(str).unique())
    if len(pdb_ids) <= 1:
        return pdb_ids, []

    if "release_date" not in indices_df.columns:
        logger.warning(
            "indices.csv has no release_date column; using deterministic "
            "PDB-level holdout for validation."
        )
        return _deterministic_holdout_pdbs(pdb_ids)

    cutoff_ts = pd.to_datetime(temporal_cutoff, errors="coerce")
    if pd.isna(cutoff_ts):
        raise ValueError(f"Invalid temporal_cutoff: {temporal_cutoff}")

    dated = indices_df[["pdb_id", "release_date"]].copy()
    dated["release_date"] = pd.to_datetime(dated["release_date"], errors="coerce")
    pdb_dates = (
        dated.groupby("pdb_id", as_index=False)["release_date"]
        .min()
        .sort_values(["release_date", "pdb_id"], na_position="last")
    )

    train_pdbs = pdb_dates.loc[
        pdb_dates["release_date"].notna()
        & (pdb_dates["release_date"] <= cutoff_ts),
        "pdb_id",
    ].astype(str).tolist()
    val_pdbs = pdb_dates.loc[
        pdb_dates["release_date"].notna()
        & (pdb_dates["release_date"] > cutoff_ts),
        "pdb_id",
    ].astype(str).tolist()

    if not val_pdbs:
        logger.warning(
            "Temporal cutoff %s produced an empty validation split; holding out "
            "the newest PDBs instead.",
            temporal_cutoff,
        )
        dated_only = pdb_dates[pdb_dates["release_date"].notna()]
        if len(dated_only) > 1:
            n_holdout = max(1, int(round(len(dated_only) * 0.05)))
            val_pdbs = dated_only.tail(n_holdout)["pdb_id"].astype(str).tolist()
            train_pdbs = [
                pdb_id
                for pdb_id in pdb_ids
                if pdb_id not in set(val_pdbs)
            ]
        else:
            train_pdbs, val_pdbs = _deterministic_holdout_pdbs(pdb_ids)

    if not train_pdbs:
        logger.warning(
            "Temporal cutoff %s produced an empty training split; falling back "
            "to deterministic PDB-level holdout.",
            temporal_cutoff,
        )
        train_pdbs, val_pdbs = _deterministic_holdout_pdbs(pdb_ids)

    if not val_pdbs:
        logger.warning(
            "Validation split is still empty after fallbacks; reusing the "
            "training set for validation as a last resort."
        )
        val_pdbs = list(train_pdbs)

    return sorted(set(train_pdbs)), sorted(set(val_pdbs))


def get_multichain_dataloaders(
    configs: ConfigDict,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders for multi-chain fine-tuning.

    Args:
        configs: Configuration object. Expected keys:
            - configs.data.bioassembly_dir: Path to bioassembly .pkl.gz directory.
            - configs.data.indices_csv: Path to indices.csv.
            - configs.train_crop_size: Crop size for training.
            - configs.data.max_n_token: Max tokens per sample (filter).
            - configs.data.temporal_cutoff: Date string for train/val split.
        seed: Random seed for sampling.

    Returns:
        Tuple of (train_dataloader, validation_dataloader).
    """
    crop_size = configs.train_crop_size
    logger.info(f"Multi-chain loader: crop_size={crop_size}")

    bioassembly_dir = configs.data.bioassembly_dir
    indices_csv = configs.data.indices_csv
    max_n_token = configs.data.get("max_n_token", 5120)
    temporal_cutoff = configs.data.get("temporal_cutoff", "2025-05-09")
    train_pdbs, val_pdbs = _build_pdb_split(indices_csv, temporal_cutoff)
    overlap = set(train_pdbs) & set(val_pdbs)
    if overlap:
        raise ValueError(
            f"Train/val PDB overlap detected in multichain split: {sorted(overlap)[:5]}"
        )
    logger.info(
        "Multi-chain temporal split: cutoff=%s train_pdb=%d val_pdb=%d",
        temporal_cutoff,
        len(train_pdbs),
        len(val_pdbs),
    )

    # Common dataset kwargs
    dataset_kwargs = {
        "bioassembly_dict_dir": bioassembly_dir,
        "indices_fpath": indices_csv,
        "cropping_configs": {
            "crop_size": crop_size,
            "method_weights": [0.2, 0.4, 0.4],
            "contiguous_crop_complete_lig": True,
            "spatial_crop_complete_lig": True,
        },
        "max_n_token": max_n_token,
        "random_sample_if_failed": True,
        "shuffle_mols": True,
        "shuffle_sym_ids": True,
        "ref_pos_augment": True,
    }

    # MSA featurizer (optional)
    msa_featurizer = _build_msa_featurizer(configs)
    if msa_featurizer is not None:
        dataset_kwargs["msa_featurizer"] = msa_featurizer

    # Training dataset
    train_dataset = MultiChainDataset(
        name="multichain_train",
        pdb_list=train_pdbs,
        **dataset_kwargs,
    )

    # Validation dataset - held-out PDBs, no augmentation
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["ref_pos_augment"] = False
    val_kwargs["shuffle_mols"] = False
    val_kwargs["shuffle_sym_ids"] = False
    val_kwargs["random_sample_if_failed"] = False
    val_kwargs["name"] = "multichain_val"
    val_kwargs["pdb_list"] = val_pdbs

    val_dataset = MultiChainDataset(**val_kwargs)

    logger.info(
        f"Multi-chain datasets: train={len(train_dataset)}, "
        f"val={len(val_dataset)}"
    )

    # Create samplers and data loaders
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=True,
        seed=seed,
    )

    if DIST_WRAPPER.world_size > 1:
        train_dl = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sampler=train_sampler,
            collate_fn=lambda batch: batch,
        )
    else:
        train_dl = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sampler=train_sampler,
            collate_fn=lambda batch: batch,
        )

    val_sampler = DistributedSampler(
        dataset=val_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    val_dl = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        sampler=val_sampler,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )

    return train_dl, val_dl


def _build_msa_featurizer(configs: ConfigDict) -> Optional[object]:
    """Build MSA featurizer from configs if MSA is enabled.

    Args:
        configs: Configuration object.

    Returns:
        MSAFeaturizer instance or None.
    """
    if not configs.get("use_msa", False):
        return None

    msa_dir = configs.get("msa_dir", "")
    if not msa_dir:
        return None
    if os.path.isdir(msa_dir):
        try:
            dir_entries = os.listdir(msa_dir)
        except OSError:
            dir_entries = []
        if any(name.endswith(".MSA.fasta") for name in dir_entries):
            logger.warning(
                "Multi-chain MSA expects AF3-style RNA MSA directories, but %s "
                "contains RNA-only FASTA files; disabling multichain MSA for this run.",
                msa_dir,
            )
            return None

    try:
        from rnapro.data.msa_featurizer import MSAFeaturizer, RNAMSAFeaturizer

        rna_msa_args = {
            "msa_dir": msa_dir,
            "indexing_method": "sequence",
            "merge_method": "dense_max",
            "max_size": 16384,
            "dataset_name": "multichain_train",
        }

        return MSAFeaturizer(
            prot_msa_args=None,
            rna_msa_args=rna_msa_args,
            enable_rna_msa=True,
        )
    except Exception as e:
        logger.warning(f"Failed to build MSA featurizer: {e}")
        return None
