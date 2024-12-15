from typing import Optional, Callable

import os.path as osp
import glob
import torch

import torchaudio
import librosa

from torch.utils.data import DataLoader, Dataset


class AudioDatasetFromFolder(Dataset):
    def __init__(
        self,
        root: str,
        ext: str = "wav",
        sr: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Assume
        """
        self.root = root
        self.ext = ext
        self.sr = sr
        self.transform = transform
        self.paths = self._load_paths()

    def _load_paths(self):
        return sorted(glob.glob(osp.join(self.root, f"*.{self.ext}")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        audio_path = self.paths[idx]
        # audio, sr = torchaudio.load(audio_path)
        # audio = audio.mean(0).numpy()  # convert to mono
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        if self.transform is not None:
            audio = self.transform(audio)
        return {
            "noisy_audio": audio,
            "sr": sr,
        }
