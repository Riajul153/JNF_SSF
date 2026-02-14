import os
import random
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


class FlacPairDataset(Dataset):
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: str,
        n_channels: int,
        meta_frame_length: int,
        fs: int,
        dry_target: bool = True,
        disable_random: bool = False,
        target_dir: int = 0,
        file_list: Optional[List[str]] = None,
        extensions: Optional[List[str]] = None,
    ):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.n_channels = n_channels
        self.meta_frame_length = meta_frame_length
        self.fs = fs
        self.use_dry_target = dry_target
        self.disable_random = disable_random
        self.target_dir = target_dir
        self.extensions = extensions or [".flac", ".wav"]

        if file_list is None:
            file_list = self._discover_pairs()
        self.file_list = file_list

    def _discover_pairs(self) -> List[str]:
        names = []
        for name in os.listdir(self.clean_dir):
            lower = name.lower()
            if any(lower.endswith(ext) for ext in self.extensions):
                noisy_path = os.path.join(self.noisy_dir, name)
                if os.path.isfile(noisy_path):
                    names.append(name)
        names.sort()
        if not names:
            raise FileNotFoundError(
                f"No paired files found in {self.clean_dir} and {self.noisy_dir}"
            )
        return names

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        name = self.file_list[idx]
        clean_path = os.path.join(self.clean_dir, name)
        noisy_path = os.path.join(self.noisy_dir, name)

        clean, clean_sr = sf.read(clean_path, always_2d=True, dtype="float32")
        noisy, noisy_sr = sf.read(noisy_path, always_2d=True, dtype="float32")
        if clean_sr != self.fs or noisy_sr != self.fs:
            raise ValueError(
                f"Sample rate mismatch for {name}: clean {clean_sr}, noisy {noisy_sr}, expected {self.fs}"
            )

        clean = clean[:, 0]
        noisy = noisy.T

        if noisy.shape[0] < self.n_channels:
            pad = self.n_channels - noisy.shape[0]
            noisy = np.concatenate([noisy, np.repeat(noisy[-1:, :], pad, axis=0)], axis=0)
        elif noisy.shape[0] > self.n_channels:
            noisy = noisy[: self.n_channels, :]

        min_len = min(clean.shape[-1], noisy.shape[-1])
        clean = clean[:min_len]
        noisy = noisy[:, :min_len]

        if self.meta_frame_length < 0:
            start = 0
            frame_len = min_len
        else:
            frame_len = self.meta_frame_length
            if min_len <= frame_len:
                start = 0
            elif self.disable_random:
                start = 0
            else:
                start = random.randint(0, min_len - frame_len)

        if frame_len > min_len:
            pad_len = frame_len - min_len
            clean = np.pad(clean, (0, pad_len), mode="constant")
            noisy = np.pad(noisy, ((0, 0), (0, pad_len)), mode="constant")
        else:
            clean = clean[start : start + frame_len]
            noisy = noisy[:, start : start + frame_len]

        clean = np.repeat(clean[None, :], self.n_channels, axis=0)

        noise = noisy - clean

        return {
            "noisy_td": noisy.astype(np.float32),
            "clean_td": clean.astype(np.float32) if self.use_dry_target else clean.astype(np.float32),
            "reverb_clean_td": clean.astype(np.float32),
            "noise_td": noise.astype(np.float32),
            "start_idx": start,
            "sample_idx": idx,
            "name": name,
            "target_dir": self.target_dir,
        }
