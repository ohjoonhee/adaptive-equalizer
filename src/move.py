import os
import os.path as osp

import librosa
import tqdm
import glob
import soundfile as sf


if __name__ == "__main__":
    skip_files = [
        "data/fma_small/011/011298.mp3",
        "data/fma_small/021/021657.mp3",
        "data/fma_small/029/029245.mp3",
        "data/fma_small/054/054568.mp3",
        "data/fma_small/054/054576.mp3",
        "data/fma_small/098/098565.mp3",
        "data/fma_small/098/098567.mp3",
        "data/fma_small/098/098569.mp3",
        "data/fma_small/099/099134.mp3",
        "data/fma_small/108/108925.mp3",
        "data/fma_small/133/133297.mp3",
    ]
    src_folder = "data/fma_small"
    target_folder = "data/fma_small_wav_44k"
    files = glob.glob(osp.join(src_folder, "**/*.mp3"))
    print(files[:10])
    for idx, file in enumerate(tqdm.tqdm(files)):
        if file in skip_files:
            continue
        wav, sr = librosa.load(file, sr=44100, mono=True)
        wav_path = file.replace(src_folder, target_folder).replace(".mp3", ".wav")
        os.makedirs(osp.dirname(wav_path), exist_ok=True)
        sf.write(wav_path, wav, sr)
