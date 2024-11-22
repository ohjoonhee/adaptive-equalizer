from pathlib import Path
from typing import Union, Optional

import os
import os.path as osp

import librosa
import numpy as np

import torch
import torchaudio
from torch.utils.data import Subset, DataLoader, Dataset


import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from transforms.base import BaseTransforms
from utils.make_eqs import make_random_eq


class GTZANSpec(Dataset):
    def __init__(
        self,
        root: str,
        subset: Optional[str] = None,
        audio_cache_dir: str = "segments_3sec",
        spec_cache_dir: str = "spec_3sec",
        return_complex: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.subset = subset
        self.audio_cache_dir = audio_cache_dir
        self.spec_cache_dir = spec_cache_dir
        self.return_complex = return_complex

        self.spec_paths = self.load_data(subset)
        self.spec_paths = sorted(self.spec_paths)
        self.eq_paths = self.load_eqs(subset)

    def load_data(self, subset: Optional[str] = None) -> None:
        root = osp.join(self.root, self.spec_cache_dir)
        genres = os.listdir(root)
        spec_paths = []
        if subset == "training":
            for genre in genres:
                genre_path = osp.join(root, genre)
                files = sorted(os.listdir(genre_path))[:-20]
                for file in files:
                    fs = os.listdir(osp.join(genre_path, file))
                    spec_paths.extend([osp.join(genre_path, file, f) for f in fs])

        elif subset == "validation":
            for genre in genres:
                genre_path = osp.join(root, genre)
                files = sorted(os.listdir(genre_path))[-20:-10]
                for file in files:
                    fs = os.listdir(osp.join(genre_path, file))
                    spec_paths.extend([osp.join(genre_path, file, f) for f in fs])

        elif subset == "test":
            for genre in genres:
                genre_path = osp.join(root, genre)
                files = sorted(os.listdir(genre_path))[-10:]
                for file in files:
                    fs = os.listdir(osp.join(genre_path, file))
                    spec_paths.extend([osp.join(genre_path, file, f) for f in fs])

        for path in spec_paths:
            path_dirs = path.split(osp.sep)
            path_dirs[-4] = self.audio_cache_dir
            path_dirs[-1] = path_dirs[-1][:-4] + ".wav"
            audio_path = osp.join(*path_dirs)
            assert os.path.exists(audio_path), f"File not found: {audio_path}"

        print("Loaded", len(spec_paths), "spectrograms and Audio Segments")

        return spec_paths

    def load_eqs(self, subset: Optional[str] = None) -> None:
        if subset == "training":
            return None
        elif subset == "validation":
            eq_path = osp.join(self.root, "random_walk_eqs")
            eq_files = os.listdir(eq_path)
            eqs = [osp.join(eq_path, f) for f in eq_files]
            return sorted(eqs)
        elif subset == "test":
            eq_path = osp.join(self.root, "random_walk_eqs")
            eq_files = os.listdir(eq_path)
            eqs = [osp.join(eq_path, f) for f in eq_files]
            return sorted(eqs)

    def __getitem__(self, index):
        path = self.spec_paths[index]
        spec = np.load(path)
        if not self.return_complex:
            spec = np.abs(spec).astype(np.float32)
        if self.subset == "training":
            eq = make_random_eq(spec.shape[0]).astype(np.float32)
            spec = spec * eq[:, None]
            inv_eq = 1 / eq
        elif self.subset == "validation":
            eq_path = self.eq_paths[index]
            eq = np.load(eq_path).astype(np.float32)
            spec = spec * eq[:, None]
            inv_eq = 1 / eq
        elif self.subset == "test":
            eq_path = self.eq_paths[index]
            eq = np.load(eq_path).astype(np.float32)
            spec = spec * eq[:, None]
            inv_eq = 1 / eq
        return spec, inv_eq

    def __len__(self):
        return len(self.spec_paths)


class GTZANDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        # transforms: BaseTransforms,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.root = root
        # self.transforms = transforms
        # self.train_transform = transforms.train_transform()
        # self.val_transform = transforms.val_transform()
        # self.test_transform = transforms.test_transform()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train_dataset = GTZANSpec(
                self.root,
                subset="training",
                # transform=self.train_transform,
            )
            self.val_dataset = GTZANSpec(
                self.root,
                subset="validation",
                # transform=self.val_transform,
            )
        else:
            self.test_dataset = GTZANSpec(
                self.root,
                subset="testing",
                # transform=self.test_transform,
            )

    # def _collate_fn(self, batch):
    #     tensors = [b[0].t() for b in batch if b]
    #     tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    #     tensors = tensors.transpose(1, -1)

    #     srs = [b[1] for b in batch if b]
    #     srs = torch.tensor(srs)

    #     return tensors, srs

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=self._collate_fn,
        )
