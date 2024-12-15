from pathlib import Path
from typing import Union, Optional

from collections import namedtuple

import os
import os.path as osp
import glob

import librosa
import numpy as np
import pandas as pd
import ast
from functools import partial

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from transforms.base import BaseTransforms
from utils.make_eqs import make_random_eq

# path_item = namedtuple("path_item", ["audio_path", "eq_path"])


class FMA(Dataset):
    def __init__(
        self,
        root: str,
        subset: Optional[str],
        split: Optional[str],
        metadata_path: str = "fma_metadata",
        val_eq_dir: str = "random_walk_eqs_db",
        eq_shape: int = 1025,  # FIXME: Hardcoded and type is limited to int
        sr: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.subset = subset
        self.split = split
        self.eq_shape = eq_shape
        self.val_eq_dir = val_eq_dir
        self.sr = sr

        assert subset in ["small", "medium", "large"]
        assert split in ["training", "validation", "test"]

        tracks = self._load(osp.join(metadata_path, "tracks.csv"))
        small = tracks[tracks["set", "subset"] <= self.subset]["set"]
        self.df = small[small["split"] == split]
        self.df["audio_path"] = self.df.index.map(
            partial(self.get_audio_path, osp.join(self.root), ext=".wav")
        )

        if self.split == "training":
            self.df["eq_path"] = None
        elif self.split == "validation":
            eq_path_root = osp.join(self.root, self.val_eq_dir)
            eq_paths = sorted(os.listdir(eq_path_root))
            self.df["eq_path"] = [
                osp.join(eq_path_root, f) for f in eq_paths[: len(self.df)]
            ]
        elif self.split == "test":
            eq_path_root = osp.join(self.root, self.val_eq_dir)
            eq_paths = sorted(os.listdir(eq_path_root))
            self.df["eq_path"] = [
                osp.join(eq_path_root, f) for f in eq_paths[: len(self.df)]
            ]

        self._scan_tracks()

    def _scan_tracks(self):
        drop_rows = []
        for index, row in self.df.iterrows():
            if not osp.exists(row["audio_path"]):
                drop_rows.append(index)
        self.df.drop(drop_rows, inplace=True)
        print(f"Dropped {len(drop_rows)} rows")

    def _load(self, filepath):
        filename = os.path.basename(filepath)
        if "features" in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
        if "echonest" in filename:
            return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
        if "genres" in filename:
            return pd.read_csv(filepath, index_col=0)
        if "tracks" in filename:
            tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

            COLUMNS = [
                ("track", "tags"),
                ("album", "tags"),
                ("artist", "tags"),
                ("track", "genres"),
                ("track", "genres_all"),
            ]
            for column in COLUMNS:
                tracks[column] = tracks[column].map(ast.literal_eval)

            COLUMNS = [
                ("track", "date_created"),
                ("track", "date_recorded"),
                ("album", "date_created"),
                ("album", "date_released"),
                ("artist", "date_created"),
                ("artist", "active_year_begin"),
                ("artist", "active_year_end"),
            ]
            for column in COLUMNS:
                tracks[column] = pd.to_datetime(tracks[column])

            SUBSETS = ("small", "medium", "large")
            try:
                tracks["set", "subset"] = tracks["set", "subset"].astype(
                    "category", categories=SUBSETS, ordered=True
                )
            except (ValueError, TypeError):
                # the categories and ordered arguments were removed in pandas 0.25
                tracks["set", "subset"] = tracks["set", "subset"].astype(
                    pd.CategoricalDtype(categories=SUBSETS, ordered=True)
                )

            COLUMNS = [
                ("track", "genre_top"),
                ("track", "license"),
                ("album", "type"),
                ("album", "information"),
                ("artist", "bio"),
            ]
            for column in COLUMNS:
                tracks[column] = tracks[column].astype("category")

            return tracks

    def get_audio_path(self, audio_dir, track_id, ext=".mp3"):
        """
        Return the path to the mp3 given the directory where the audio is stored
        and the track ID.

        Examples
        --------
        >>> import utils
        >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
        >>> utils.get_audio_path(AUDIO_DIR, 2)
        '../data/fma_small/000/000002.mp3'

        """
        tid_str = "{:06d}".format(track_id)
        return os.path.join(audio_dir, tid_str[:3], tid_str + ext)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        # load audio
        # clean_audio, sr = librosa.load(item.audio_path, sr=self.sr, mono=True)
        clean_audio, sr = torchaudio.load(item.audio_path)
        clean_audio = clean_audio.mean(dim=0).squeeze().numpy()

        # load eq and apply to spec
        if self.split == "training":
            min_db = np.random.randn() * 15 - 20
            max_db = np.random.randn() * 7.5 + 10
            ma_window = np.random.randint(10, 100)
            eq = make_random_eq(
                self.eq_shape, ma_window=ma_window, min_db=min_db, max_db=max_db
            ).astype(np.float32)
            # eq = make_random_eq(self.eq_shape).astype(np.float32)
        elif self.split == "validation":
            eq = np.load(item.eq_path).astype(np.float32)
        elif self.split == "test":
            # TODO: Implement test set
            eq = np.load(item.eq_path).astype(np.float32)

        return {
            "clean_audio": clean_audio,
            "label": eq,
        }

    def __len__(self):
        return len(self.df)


class FMADataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        # transforms: BaseTransforms,
        num_workers: int,
        # audio_cache_dir: str = "segments_3sec",
        metadata_path: str = "fma_metadata",
        subset: str = "small",
        val_eq_dir: str = "random_walk_eqs_db",
        eq_shape: int = 1025,  # FIXME: Hardcoded and type is limited to int
        sr: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = root
        # self.transforms = transforms
        # self.train_transform = transforms.train_transform()
        # self.val_transform = transforms.val_transform()
        # self.test_transform = transforms.test_transform()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.audio_cache_dir = audio_cache_dir
        self.metadata_path = metadata_path
        self.subset = subset
        self.val_eq_dir = val_eq_dir
        self.eq_shape = eq_shape
        self.sr = sr

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train_dataset = FMA(
                self.root,
                split="training",
                # transform=self.val_transform,
                # audio_cache_dir=self.audio_cache_dir,
                metadata_path=self.metadata_path,
                subset=self.subset,
                val_eq_dir=self.val_eq_dir,
                eq_shape=self.eq_shape,
                sr=self.sr,
            )
            self.val_dataset = FMA(
                self.root,
                split="validation",
                # transform=self.val_transform,
                # audio_cache_dir=self.audio_cache_dir,
                metadata_path=self.metadata_path,
                subset=self.subset,
                val_eq_dir=self.val_eq_dir,
                eq_shape=self.eq_shape,
                sr=self.sr,
            )
        else:
            self.test_dataset = FMA(
                self.root,
                split="test",
                # transform=self.val_transform,
                # audio_cache_dir=self.audio_cache_dir,
                metadata_path=self.metadata_path,
                subset=self.subset,
                val_eq_dir=self.val_eq_dir,
                eq_shape=self.eq_shape,
                sr=self.sr,
            )

    def _collate_fn(self, batch):
        clean_audios = [item["clean_audio"] for item in batch]
        labels = torch.stack([torch.tensor(item["label"]) for item in batch])

        min_len = min([len(audio) for audio in clean_audios])
        clean_audios = torch.stack(
            [torch.tensor(audio[:min_len]) for audio in clean_audios]
        )

        batch = {
            "clean_audio": clean_audios,
            "label": labels,
        }

        return batch

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # collate_fn=self._collate_fn,
        )
