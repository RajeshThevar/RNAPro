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

import warnings
from collections import defaultdict
from typing import Any, Iterator, Mapping, Optional, Sequence
import random

from Bio import BiopythonWarning
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter("ignore", BiopythonWarning)
warnings.simplefilter("ignore", PDBConstructionWarning)

from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, DistributedSampler

from rnapro.data.dataloader import (
    DistributedDataLoader,
    KeySumBalancedSampler,
)
from rnapro.data.rna_dataset_allatom import RNADataset
from rnapro.utils.distributed import DIST_WRAPPER
from rnapro.utils.logger import get_logger
from rnapro.utils.torch_utils import collate_fn_first

logger = get_logger(__name__)


def get_dataloaders(
    configs: ConfigDict, 
    seed=42, 
    build_val: bool = True,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """
    Generate data loaders for training and testing based on the given configurations and seed.

    This function creates RNA datasets using configuration values instead of hardcoded paths.
    Supports MSA feature generation, template structures, chemical mapping, and sequence cropping.
    
    The configuration should include:
    - configs.data.rna_train: Training dataset configuration
    - configs.data.rna_test: Testing/validation dataset configuration  
    - configs.data.chemical_mapping.enable: Whether to enable chemical mapping
    - configs.data.template.enable: Whether to enable template structures
    - configs.ribonanzanet2: RibonanzaNet2 model paths
    - configs.data.augment_train_with_test: Whether to augment training with test samples

    Note:
        MSA features are generated internally by RNADataset using FASTA files from rna_msa_dir.

    Args:
        configs (ConfigDict): An object containing the data configuration information.
        world_size (int): The number of processes in the distributed environment.
        seed (int): The random seed used for data sampling.
        error_dir (str, optional): The directory to store error information. Defaults to None.

    Returns:
        tuple: A tuple containing the training data loader and a dictionary of testing data loaders.

    """
    temporal_cutoff = "2025-05-09"
    crop_size = configs.train_crop_size
    print(f"cropping size is {crop_size}")
    train_dataset = RNADataset(
        data_dir=configs.data.train_sets[0],
        use_msa=configs.use_msa,
        crop_size = crop_size,
        use_template=configs.use_template,
        msa_dir=configs.msa_dir,
        use_cluster=configs.use_cluster,
        num_templates=configs.num_templates,
        temporal_cutoff=temporal_cutoff,
        mode='train',
    )

    
    test_dataset = None
    if build_val:
        test_dataset = RNADataset(
            data_dir=configs.data.train_sets[0],
            use_msa=configs.use_msa,
            crop_size = crop_size,
            use_template=configs.use_template,
            msa_dir=configs.msa_dir,
            use_cluster=configs.use_cluster,
            num_templates=configs.num_templates,
            temporal_cutoff='2025-02-01',
            mode='test',
        )

    if DIST_WRAPPER.world_size > 1:
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            shuffle=True,
            seed=seed,
        )
        train_dl = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sampler=train_sampler,
            collate_fn=lambda batch: batch,
        )

    else:
        train_sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            shuffle=True,
            seed=seed,
        )
        train_dl = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sampler=train_sampler,
            collate_fn=lambda batch: batch,
        )

    if test_dataset is not None:
        test_sampler = DistributedSampler(
            dataset=test_dataset,
            num_replicas=DIST_WRAPPER.world_size,
            rank=DIST_WRAPPER.rank,
            shuffle=False,
        )
        test_dls = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            sampler=test_sampler,
            collate_fn=lambda batch: batch,
            num_workers=0,
        )
        logger.info(
            f"train data size: {len(train_dataset)}, test size: {len(test_dataset)}"
        )
    else:
        test_dls = None
        logger.info(f"train data size: {len(train_dataset)}, validation skipped")
    
    return train_dl, test_dls
