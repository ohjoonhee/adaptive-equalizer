import os
import os.path as osp
import glob
import tqdm
import librosa
import numpy as np

if __name__ == "__main__":
    segment_root = "data/segments_3sec"
    spec_root = "data/spec_3sec"

    # Get all the audio files
    audio_files = glob.glob(osp.join(segment_root, "**/*.wav"), recursive=True)

    for file in tqdm.tqdm(audio_files):
        dirs = file.split(osp.sep)
        basename = dirs[-1][:-4]
        orig_name = dirs[-2]
        genre = dirs[-3]
        os.makedirs(osp.join(spec_root, genre, orig_name), exist_ok=True)

        # Load the audio file
        wav, sr = librosa.load(file, sr=None)
        spec = librosa.stft(wav)
        # spec = np.abs(spec)
        # spec = librosa.amplitude_to_db(spec, ref=np.max)

        # Save the spectrogram
        np.save(osp.join(spec_root, genre, orig_name, f"{basename}.npy"), spec)
