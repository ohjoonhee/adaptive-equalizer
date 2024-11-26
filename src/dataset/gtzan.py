from pathlib import Path
from typing import Union, Optional

from collections import namedtuple

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

path_item = namedtuple("path_item", ["audio_path", "spec_path", "eq_path"])


class GTZANSpec(Dataset):
    def __init__(
        self,
        root: str,
        subset: Optional[str] = None,
        audio_cache_dir: str = "segments_3sec",
        spec_cache_dir: str = "spec_3sec",
        val_eq_dir: str = "random_walk_eqs",
        amp_to_db: bool = False,
        return_complex: bool = False,
    ) -> None:
        super().__init__()
        self.root = root
        self.subset = subset
        self.audio_cache_dir = audio_cache_dir
        self.spec_cache_dir = spec_cache_dir
        self.val_eq_dir = val_eq_dir
        self.amp_to_db = amp_to_db
        self.return_complex = return_complex

        self.paths = self.load_data(subset)

    def load_data(self, subset: Optional[str] = None) -> list[path_item]:
        root = osp.join(self.root, self.spec_cache_dir)
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
                        spec_path = osp.join(genre_path, file, f)
                        spec_path_dirs = spec_path.split(osp.sep)
                        spec_path_dirs[-4] = self.audio_cache_dir
                        spec_path_dirs[-1] = spec_path_dirs[-1][:-4] + ".wav"
                        audio_path = osp.join(*spec_path_dirs)
                        if not os.path.exists(audio_path):
                            print(f"File not found: {audio_path} for {spec_path}")
                            missing_files += 1
                            continue
                        paths.append(path_item(audio_path, spec_path, None))

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
                        spec_path = osp.join(genre_path, file, f)
                        spec_path_dirs = spec_path.split(osp.sep)
                        spec_path_dirs[-4] = self.audio_cache_dir
                        spec_path_dirs[-1] = spec_path_dirs[-1][:-4] + ".wav"
                        audio_path = osp.join(*spec_path_dirs)
                        if not os.path.exists(audio_path):
                            print("File not found:", audio_path)
                            continue
                        paths.append(
                            path_item(audio_path, spec_path, next(eq_paths_gen))
                        )

        elif subset == "test":
            # TODO: Implement test set
            pass
        else:
            raise ValueError("Invalid subset")

        print(
            f"Loaded {len(paths)} spectrograms and audio segments with {missing_files} missing files."
        )

        return paths

    def load_eqs(self, subset: Optional[str] = None) -> None:
        if subset == "training":
            return None
        elif subset == "validation":
            eq_path = osp.join(self.root, self.val_eq_dir)
            eq_files = os.listdir(eq_path)
            eqs = [osp.join(eq_path, f) for f in eq_files]
            return sorted(eqs)
        elif subset == "test":
            eq_path = osp.join(self.root, self.val_eq_dir)
            eq_files = os.listdir(eq_path)
            eqs = [osp.join(eq_path, f) for f in eq_files]
            return sorted(eqs)

    def apply_eq(self, spec: np.ndarray, eq: np.ndarray) -> np.ndarray:
        eq = np.power(10, eq / 20)
        return spec * eq[:, None]

    def __getitem__(self, index):
        path = self.paths[index]

        # load spectrogram
        clean_spec = np.load(path.spec_path)

        # load audio
        clean_audio, sr = torchaudio.load(path.audio_path)

        # load eq and apply to spec
        if self.subset == "training":
            eq = make_random_eq(clean_spec.shape[0]).astype(np.float32)
        elif self.subset == "validation":
            eq = np.load(path.eq_path).astype(np.float32)
        elif self.subset == "test":
            # TODO: Implement test set
            pass
        # noisy_spec = clean_spec * eq[:, None]
        noisy_spec = self.apply_eq(clean_spec, eq)
        noisy_audio = librosa.istft(noisy_spec)
        # inv_eq = 1 / eq
        inv_eq = -eq / 20

        if not self.return_complex:
            noisy_spec = np.abs(noisy_spec).astype(np.float32)

        return {
            "clean_audio": clean_audio,
            "clean_spec": clean_spec,
            "noisy_audio": noisy_audio,
            "noisy_spec": noisy_spec,
            "label": inv_eq,
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
        val_eq_dir: str = "random_walk_eqs",
    ) -> None:
        super().__init__()
        self.root = root
        # self.transforms = transforms
        # self.train_transform = transforms.train_transform()
        # self.val_transform = transforms.val_transform()
        # self.test_transform = transforms.test_transform()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_eq_dir = val_eq_dir

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train_dataset = GTZANSpec(
                self.root,
                subset="training",
                # transform=self.train_transform,
                val_eq_dir=self.val_eq_dir,
            )
            self.val_dataset = GTZANSpec(
                self.root,
                subset="validation",
                # transform=self.val_transform,
                val_eq_dir=self.val_eq_dir,
            )
        else:
            self.test_dataset = GTZANSpec(
                self.root,
                subset="testing",
                # transform=self.test_transform,
                val_eq_dir=self.val_eq_dir,
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
