from pathlib import Path
from typing import Union, Optional

from collections import namedtuple

import os
import os.path as osp

import librosa
import numpy as np

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from transforms.base import BaseTransforms
from utils.make_eqs import make_random_eq

path_item = namedtuple("path_item", ["audio_path", "eq_path"])


class GTZANSpec(Dataset):
    def __init__(
        self,
        root: str,
        subset: Optional[str] = None,
        audio_cache_dir: str = "segments_3sec",
        val_eq_dir: str = "random_walk_eqs_db",
        eq_shape: int = 1025,  # FIXME: Hardcoded and type is limited to int
    ) -> None:
        super().__init__()
        self.root = root
        self.subset = subset
        self.audio_cache_dir = audio_cache_dir
        self.eq_shape = eq_shape
        self.val_eq_dir = val_eq_dir

        self.paths = self.load_data(subset)

    def load_data(self, subset: Optional[str] = None) -> list[path_item]:
        root = osp.join(self.root, self.audio_cache_dir)
        genres = os.listdir(root)
        paths = []
        missing_files = 0
        if subset == "training":
            for genre in genres:
                genre_path = osp.join(root, genre)
                files = sorted(os.listdir(genre_path))[:-20]
                for file in files:
                    fs = os.listdir(osp.join(genre_path, file))
                    for f in fs:
                        audio_path = osp.join(genre_path, file, f)
                        paths.append(path_item(audio_path, None))

        elif subset == "validation":
            eq_path_root = osp.join(self.root, self.val_eq_dir)
            eq_paths = sorted(os.listdir(eq_path_root))
            eq_paths_gen = (osp.join(eq_path_root, f) for f in eq_paths)
            for genre in genres:
                genre_path = osp.join(root, genre)
                files = sorted(os.listdir(genre_path))[-20:-10]
                for file in files:
                    fs = sorted(os.listdir(osp.join(genre_path, file)))
                    for f in fs:
                        audio_path = osp.join(genre_path, file, f)
                        paths.append(path_item(audio_path, next(eq_paths_gen)))

        elif subset == "test":
            pass  # TODO: Implement test set
        else:
            raise ValueError("Invalid subset")

        print(
            f"Loaded {len(paths)} spectrograms and audio segments with {missing_files} missing files."
        )

        return paths

    def __getitem__(self, index):
        path = self.paths[index]

        # load audio
        clean_audio, sr = librosa.load(path.audio_path, sr=None)

        # load eq and apply to spec
        if self.subset == "training":
            eq = make_random_eq(self.eq_shape).astype(np.float32)
        elif self.subset == "validation":
            eq = np.load(path.eq_path).astype(np.float32)
        elif self.subset == "test":
            # TODO: Implement test set
            pass

        return {
            "clean_audio": clean_audio,
            "label": eq,
        }

    def __len__(self):
        return len(self.paths)


class GTZANDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        # transforms: BaseTransforms,
        num_workers: int,
        audio_cache_dir: str = "segments_3sec",
        val_eq_dir: str = "random_walk_eqs",
        eq_shape: int = 1025,  # FIXME: Hardcoded and type is limited to int
    ) -> None:
        super().__init__()
        self.root = root
        # self.transforms = transforms
        # self.train_transform = transforms.train_transform()
        # self.val_transform = transforms.val_transform()
        # self.test_transform = transforms.test_transform()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.audio_cache_dir = audio_cache_dir
        self.val_eq_dir = val_eq_dir
        self.eq_shape = eq_shape

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train_dataset = GTZANSpec(
                self.root,
                subset="training",
                # transform=self.train_transform,
                audio_cache_dir=self.audio_cache_dir,
                val_eq_dir=self.val_eq_dir,
                eq_shape=self.eq_shape,
            )
            self.val_dataset = GTZANSpec(
                self.root,
                subset="validation",
                # transform=self.val_transform,
                audio_cache_dir=self.audio_cache_dir,
                val_eq_dir=self.val_eq_dir,
                eq_shape=self.eq_shape,
            )
        else:
            self.test_dataset = GTZANSpec(
                self.root,
                subset="testing",
                # transform=self.test_transform,
                audio_cache_dir=self.audio_cache_dir,
                val_eq_dir=self.val_eq_dir,
                eq_shape=self.eq_shape,
            )

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
