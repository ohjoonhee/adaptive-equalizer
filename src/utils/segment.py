import os
import os.path as osp
import glob
from pydub import AudioSegment
import tqdm

if __name__ == "__main__":
    data_root = "data/genres_original"
    segment_root = "data/segments_3sec"

    # Get all the audio files
    audio_files = glob.glob(osp.join(data_root, "**/*.wav"), recursive=True)

    for file in tqdm.tqdm(audio_files):
        dirs = file.split(osp.sep)
        basename = dirs[-1][:-4]
        genre = dirs[-2]
        os.makedirs(osp.join(segment_root, genre, basename), exist_ok=True)

        # Load the audio file
        audio = AudioSegment.from_file(file)

        # Split the audio file into 3-second segments

        # Save the segments
        for i in range(0, len(audio) // 3000):
            segment = audio[i * 3000 : (i + 1) * 3000]
            segment.export(
                osp.join(segment_root, genre, basename, f"{basename}_{i}.wav"),
                format="wav",
            )
