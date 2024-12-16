import sys

sys.path.append("./src")

import torch

import os.path as osp
import glob
import tqdm

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import IPython.display as ipd
import soundfile as sf

from frechet_audio_distance import FrechetAudioDistance

from net.eff import EfficientNet

if __name__ == "__main__":
    # ckpt_path = "logs/dev_fma-s_wav-22050_logfreq/v1/fit3/checkpoints/best-epoch=80-val_loss=0.0254.ckpt"
    # ckpt = torch.load(ckpt_path, map_location="cpu")
    # net = EfficientNet()
    # weights = {k[4:]: v for k, v in ckpt["state_dict"].items()}
    # net.load_state_dict(weights)
    # net = net.to("cuda")

    RW_PATH = "/mnt/ssd/datasets/youtube_live_music/wav_files_22k"
    OUTPUT_PATH = "logs/real_world"
    SAMPLE_RATE = 22050

    # for wav in tqdm.tqdm(glob.glob(osp.join(RW_PATH, "*.wav"))):
    #     audio, sr = librosa.load(wav, sr=SAMPLE_RATE)
    #     spec = librosa.stft(audio)
    #     magspec = np.abs(spec)
    #     magspec = torch.from_numpy(magspec).float()
    #     with torch.no_grad():
    #         eq = net(magspec[None, None, ...].to("cuda"))
    #     eq = eq[0].detach().cpu().numpy()
    #     eq = np.power(10, eq)

    #     recon_spec = spec * eq[:, None]
    #     recon_wav = librosa.istft(recon_spec)
    #     recon_wav = librosa.util.normalize(recon_wav)
    #     sf.write(osp.join(OUTPUT_PATH, osp.basename(wav)), recon_wav, SAMPLE_RATE)

    # frechet = FrechetAudioDistance(
    #     ckpt_dir="../checkpoints/vggish",
    #     model_name="vggish",
    #     # submodel_name="630k-audioset", # for CLAP only
    #     sample_rate=16000,
    #     use_pca=False,  # for VGGish only
    #     use_activation=False,  # for VGGish only
    #     verbose=False,
    #     audio_load_worker=8,
    #     # enable_fusion=False, # for CLAP only
    # )

    frechet = FrechetAudioDistance(
        ckpt_dir="../checkpoints/clap",
        model_name="clap",
        submodel_name="music_audioset",  # for CLAP only
        sample_rate=48000,
        # use_pca=False, # for VGGish only
        # use_activation=False, # for VGGish only
        verbose=False,
        audio_load_worker=8,
        enable_fusion=False,  # for CLAP only
    )

    score = frechet.score(
        background_dir="/mnt/ssd/datasets/youtube_live_music/background_fma_22k",
        # eval_dir="logs/dev_predict_addnoise/ep299/predict/noisy",
        eval_dir="logs/dev_predict_addnoise/ep299/predict/recon",
    )
    print(score)
