"""Data loader factory for multi-chain competition fine-tuning.

Creates DataLoader instances backed by MultiChainDataset, which uses
pre-computed bioassembly_dict .pkl.gz files (from preprocess_bioassembly.py)
through the BaseSingleDataset pipeline.

Mirrors the interface of rna_dataset_allatom_loader.get_dataloaders() so
that AF3Trainer.init_data() can swap between loaders based on config.
"""

from __future__ import annotations

from typing import Optional

from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, DistributedSampler

from rnapro.data.dataloader import DistributedDataLoader
from rnapro.data.multichain_dataset import MultiChainDataset
from rnapro.utils.distributed import DIST_WRAPPER
from rnapro.utils.logger import get_logger

logger = get_logger(__name__)


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
        **dataset_kwargs,
    )

    # Validation dataset - same data, no augmentation
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["ref_pos_augment"] = False
    val_kwargs["shuffle_mols"] = False
    val_kwargs["shuffle_sym_ids"] = False
    val_kwargs["random_sample_if_failed"] = False
    val_kwargs["name"] = "multichain_val"

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
            prot_msa_args={
                "msa_dir": "",
                "indexing_method": "sequence",
                "merge_method": "dense_max",
                "max_size": 16384,
                "dataset_name": "multichain_train",
            },
            rna_msa_args=rna_msa_args,
            enable_rna_msa=True,
        )
    except Exception as e:
        logger.warning(f"Failed to build MSA featurizer: {e}")
        return None
