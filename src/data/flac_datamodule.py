import random
from typing import List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.flac_dataset import FlacPairDataset


class FlacDataModule(pl.LightningDataModule):
    def __init__(
        self,
        n_channels: int,
        batch_size: int,
        meta_frame_length: int,
        fs: int,
        n_workers: int,
        train_clean_dir: str,
        train_noisy_dir: str,
        val_clean_dir: Optional[str] = None,
        val_noisy_dir: Optional[str] = None,
        test_clean_dir: Optional[str] = None,
        test_noisy_dir: Optional[str] = None,
        val_split: float = 0.0,
        train_fraction: Optional[float] = None,
        val_fraction: Optional[float] = None,
        target_dir: int = 0,
        seed: int = 0,
        dry_target: bool = True,
        extensions: Optional[List[str]] = None,
        **_unused,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.meta_frame_length = meta_frame_length
        self.fs = fs
        self.n_workers = n_workers
        self.train_clean_dir = train_clean_dir
        self.train_noisy_dir = train_noisy_dir
        self.val_clean_dir = val_clean_dir
        self.val_noisy_dir = val_noisy_dir
        self.test_clean_dir = test_clean_dir
        self.test_noisy_dir = test_noisy_dir
        self.val_split = val_split
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.target_dir = target_dir
        self.seed = seed
        self.dry_target = dry_target
        self.extensions = extensions

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is None:
            base_dataset = FlacPairDataset(
                clean_dir=self.train_clean_dir,
                noisy_dir=self.train_noisy_dir,
                n_channels=self.n_channels,
                meta_frame_length=self.meta_frame_length,
                fs=self.fs,
                dry_target=self.dry_target,
                disable_random=False,
                target_dir=self.target_dir,
                extensions=self.extensions,
            )
            names = list(base_dataset.file_list)
            if self.train_fraction is not None:
                if not (0 < self.train_fraction <= 1):
                    raise ValueError(
                        f"train_fraction must be in (0, 1], got {self.train_fraction}"
                    )
                rnd = random.Random(self.seed)
                rnd.shuffle(names)
                keep = max(1, int(len(names) * self.train_fraction))
                names = names[:keep]

            if self.val_clean_dir and self.val_noisy_dir:
                self.train_dataset = FlacPairDataset(
                    clean_dir=self.train_clean_dir,
                    noisy_dir=self.train_noisy_dir,
                    n_channels=self.n_channels,
                    meta_frame_length=self.meta_frame_length,
                    fs=self.fs,
                    dry_target=self.dry_target,
                    disable_random=False,
                    file_list=names,
                    target_dir=self.target_dir,
                    extensions=self.extensions,
                )
                val_dataset = FlacPairDataset(
                    clean_dir=self.val_clean_dir,
                    noisy_dir=self.val_noisy_dir,
                    n_channels=self.n_channels,
                    meta_frame_length=self.meta_frame_length,
                    fs=self.fs,
                    dry_target=self.dry_target,
                    disable_random=True,
                    target_dir=self.target_dir,
                    extensions=self.extensions,
                )
                if self.val_fraction is not None:
                    if not (0 < self.val_fraction <= 1):
                        raise ValueError(
                            f"val_fraction must be in (0, 1], got {self.val_fraction}"
                        )
                    val_names = list(val_dataset.file_list)
                    rnd = random.Random(self.seed)
                    rnd.shuffle(val_names)
                    keep = max(1, int(len(val_names) * self.val_fraction))
                    val_names = val_names[:keep]
                    val_dataset = FlacPairDataset(
                        clean_dir=self.val_clean_dir,
                        noisy_dir=self.val_noisy_dir,
                        n_channels=self.n_channels,
                        meta_frame_length=self.meta_frame_length,
                        fs=self.fs,
                        dry_target=self.dry_target,
                        disable_random=True,
                        file_list=val_names,
                        target_dir=self.target_dir,
                        extensions=self.extensions,
                    )
                self.val_dataset = val_dataset
            else:
                if self.val_split and self.val_split > 0:
                    rnd = random.Random(self.seed)
                    rnd.shuffle(names)
                    split_idx = int(len(names) * (1 - self.val_split))
                    train_names = names[:split_idx]
                    val_names = names[split_idx:]
                    self.train_dataset = FlacPairDataset(
                        clean_dir=self.train_clean_dir,
                        noisy_dir=self.train_noisy_dir,
                        n_channels=self.n_channels,
                        meta_frame_length=self.meta_frame_length,
                        fs=self.fs,
                        dry_target=self.dry_target,
                        disable_random=False,
                        file_list=train_names,
                        target_dir=self.target_dir,
                        extensions=self.extensions,
                    )
                    self.val_dataset = FlacPairDataset(
                        clean_dir=self.train_clean_dir,
                        noisy_dir=self.train_noisy_dir,
                        n_channels=self.n_channels,
                        meta_frame_length=self.meta_frame_length,
                        fs=self.fs,
                        dry_target=self.dry_target,
                        disable_random=True,
                        file_list=val_names,
                        target_dir=self.target_dir,
                        extensions=self.extensions,
                    )
                else:
                    self.train_dataset = base_dataset
                    self.val_dataset = None

        if self.test_dataset is None and self.test_clean_dir and self.test_noisy_dir:
            self.test_dataset = FlacPairDataset(
                clean_dir=self.test_clean_dir,
                noisy_dir=self.test_noisy_dir,
                n_channels=self.n_channels,
                meta_frame_length=self.meta_frame_length,
                fs=self.fs,
                dry_target=self.dry_target,
                disable_random=True,
                target_dir=self.target_dir,
                extensions=self.extensions,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
            pin_memory=True,
        )
