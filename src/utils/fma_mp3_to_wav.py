import os
import os.path as osp
import glob
import tqdm

import numpy as np
import torch
import soundfile as sf
import librosa


if __name__ == "__main__":
    root = "/mnt/ssd/datasets/fma_small"
    wav_root = "/mnt/ssd/datasets/fma_small_wav_22k_mono"
    audio_paths = glob.glob(osp.join(root, "**/*.mp3"))
    for path in tqdm.tqdm(audio_paths):
        wav, sr = librosa.load(path, sr=22050, mono=True)
        # audio = AudioSegment.from_mp3(path)
        path_segments = path.split(os.sep)
        wav_path = osp.join(
            wav_root, path_segments[-2], path_segments[-1].replace("mp3", "wav")
        )
        os.makedirs(osp.dirname(wav_path), exist_ok=True)
        # audio.export(wav_path, format="wav")
        sf.write(wav_path, wav, sr)
