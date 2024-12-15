import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import pandas as pd
from glob import glob

import logging
import pdb
from tqdm import tqdm
import itertools
import os

opj = os.path.join

from sklearn.model_selection import train_test_split

import torchaudio


def load(path, sample_rate):
    """
    Load audio sample at `path` and resample to `sample_rate`
    """
    # sample, _ = torchaudio.sox_effects.apply_effects_file(path, [['rate', str(sample_rate)]])
    sample, sr = torchaudio.load(path)
    sample = torchaudio.transforms.Resample(sr, sample_rate)(sample)
    return sample


class InfiniteDataset(IterableDataset):
    def __init__(self):
        """
        Base class for datasets that repeats forever
        """
        self.index = 0
        self.data = []

    def __iter__(self):
        assert len(self.data) > 0, "Must initialize data before iterating"
        while True:
            yield self.data[self.index]
            self.index = (self.index + 1) % len(self.data)


class NoiseDataset(InfiniteDataset):
    def __init__(self, noise_source, sample_length=47500, rate=44100, split="training"):
        """
        noise_source: either list of directories with .wav files or path to .npy file
        sample_length: length to cut noise samples to
        rate: sampling rate
        """
        assert split in ["training", "test", "validation"]
        self.sample_length = sample_length
        self.rate = rate
        self.index = 0

        if isinstance(noise_source, list):
            self.noise_files = noise_source
            logging.info("Loading individual noise sample files")
            self.data = []
            for f in tqdm(self.noise_files):
                noise_sample = load(f, self.rate)
                noise_sample = noise_sample.mean(0, keepdim=True)
                noise_sample = torch.stack(
                    torch.split(noise_sample, sample_length, dim=1)[:-1]
                )
                self.data.append(noise_sample)
            self.data = torch.cat(self.data)
        else:
            logging.info(f"Batch loading from noise sample file for split {split}")
            noises = np.load(noise_source)[split]
            self.data = torch.from_numpy(noises[..., :sample_length])

        logging.info(f"Read {self.data.shape[0]} noise samples")

    def save(self, path):
        """
        Save data for batch loading later
        path: path to save data at
        """
        logging.info(f"Saving noise samples to {path}")
        np.save(path, self.data.numpy())


if __name__ == "__main__":

    def split_from_data(data, valid_fraction, test_fraction):
        data = data[torch.where(~torch.all(data == 0, dim=2))].unsqueeze(1)
        num_samples = data.shape[0]
        data = data[torch.randperm(num_samples)]
        splits = [
            int((1 - valid_fraction - test_fraction) * num_samples),
            int(test_fraction * num_samples),
        ]
        splits.append(num_samples - splits[0] - splits[1])
        train, valid, test = torch.split(data, splits)
        train, valid, test = train.numpy(), valid.numpy(), test.numpy()
        return train, valid, test

    def process_noise(
        data_dir_noise,
        sample_length,
        sample_rate,
        valid_fraction,
        test_fraction,
        save_root_dir,
    ):
        ## If there is no separate directory named ace-ambient and ace-bubble, make it from the full ACE dataset
        ## data_dir_noise = '...Single'. i.e., ACE dataset path
        noise_names = ["Ambient", "Babble"]
        total_wav_path_list = glob(
            os.path.join(data_dir_noise, "**", "*.wav"), recursive=True
        )
        noise_wav_list = [
            wav
            for wav in total_wav_path_list
            if any([name in os.path.basename(wav) for name in noise_names])
        ]
        dataset = NoiseDataset(
            noise_wav_list, rate=sample_rate, sample_length=sample_length
        )
        out_file = f"noise_samples_{sample_length}_length_{sample_rate}_hz.npz"

        train, valid, test = split_from_data(
            dataset.data, valid_fraction, test_fraction
        )
        np.savez(
            os.path.join(save_root_dir, out_file),
            training=train,
            validation=valid,
            test=test,
        )

    sr = 22050
    process_noise("data/ACE_Corpus_RIRN_Single/Single", sr * 31, sr, 0.1, 0.1, "data/")
