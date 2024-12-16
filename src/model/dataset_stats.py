from typing import Optional
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from torchmetrics.audio import (
    SignalDistortionRatio,
    ScaleInvariantSignalDistortionRatio,
)
from torchmetrics import MeanAbsoluteError

try:
    import wandb
finally:
    pass


@dataclass
class STFTConfig:
    """Dataclass containing STFT params."""

    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    center: bool = True
    window: str = "hann_window"


class DatasetStatsModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        stft_params: STFTConfig,
        sr: Optional[int] = None,
        compression: Optional[float] = None,
        vis_per_batch: int = 0,
        vis_batches: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = net
        self.criterion = criterion
        self.sr = sr
        self.compression = compression if compression is not None else 1

        # for validation metrics
        self.l1_spec = MeanAbsoluteError()
        self.sdr = SignalDistortionRatio()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()

        # For visualization
        self.vis_per_batch = vis_per_batch
        self.vis_batches = vis_batches if vis_batches is not None else float("inf")

        # STFT params
        win = self.get_window(stft_params.window, stft_params.win_length)
        self.register_buffer("window", win, persistent=False)
        self.window: torch.Tensor
        self.stft_params = asdict(stft_params)
        self.stft_params.pop("window")

    def get_window(self, window, win_length):
        if window is None or window == "hann_window":
            window_tensor = torch.hann_window(window_length=win_length, periodic=False)
        if window == "hamming_window":
            window_tensor = torch.hamming_window(
                window_length=win_length, periodic=False
            )
        return window_tensor

    def forward(self, x):
        return self.net(x)

    def apply_eq(self, spec, eq):
        eq = torch.pow(10, eq / 20)
        return spec * eq.unsqueeze(-1)

    def apply_inv_eq(self, spec, preds):
        # eq = eq * 20
        # eq = torch.pow(10, -eq / 20)
        eq = torch.pow(10, preds)
        return spec * eq.unsqueeze(-1)

    def on_after_batch_transfer(
        self, batch: dict, dataloader_idx: int
    ) -> dict:
        clean_audio = batch["clean_audio"]

        # seg_len = 66150
        seg_len = self.sr * 3
        segs = []
        for i in range(8):
            segs.append(clean_audio[..., i * seg_len : (i + 1) * seg_len])
        clean_audio = torch.cat(segs, dim=0)
        if self.trainer.training:
            perm = torch.randperm(clean_audio.size(0))
            clean_audio = clean_audio[perm]

        eq = batch["label"]
        eq = eq.repeat(8, 1)

        # log scale x-axis eq data to linear scale x-axis
        # eq = interp(
        #     torch.linspace(1, eq.size(-1), eq.size(-1), device=eq.device)
        #     .unsqueeze(0)
        #     .repeat(eq.size(0), 1),
        #     torch.logspace(0, np.log10(eq.size(-1)), eq.size(-1), device=eq.device)
        #     .unsqueeze(0)
        #     .repeat(eq.size(0), 1),
        #     eq,
        # )

        if self.trainer.training:
            perm = torch.randperm(clean_audio.size(0))
            eq = eq[perm]

        clean_spec = torch.stft(
            clean_audio, window=self.window, return_complex=True, **self.stft_params
        )
        noisy_spec = self.apply_eq(clean_spec, eq)
        noisy_audio = torch.istft(
            noisy_spec, window=self.window, return_complex=False, **self.stft_params
        )

        batch["clean_audio"] = clean_audio
        batch["clean_spec"] = clean_spec
        batch["noisy_spec"] = noisy_spec
        batch["noisy_audio"] = noisy_audio
        batch["label"] = -eq / 20

        return batch

    # def on_fit_start(self) -> None:
    #     # Note that self.logger is set by the Trainer.fit()
    #     # self.logger is None at self.__init__
    #     self.is_wandb = isinstance(self.logger, WandbLogger)
    #     self.vis_per_batch = self.vis_per_batch if self.is_wandb else 0

    def on_train_epoch_start(self):
        self.sum = None
        self.sq_sum = None
        self.len = 0

    def training_step(self, batch, batch_idx):
        specs = batch["noisy_spec"]
        specs = torch.abs(specs).float()
        specs = specs**self.compression
        labels = batch["label"]
        # preds = self(specs.unsqueeze(1))
        if self.sum is None:
            self.sum = specs.sum(dim=0)
            self.sq_sum = (specs**2).sum(dim=0)
        else:
            self.sum += specs.sum(dim=0)
            self.sq_sum += (specs**2).sum(dim=0)
        self.len += specs.size(0)

        # loss = self.criterion(preds, labels)
        # self.log("train/loss", loss.item(), prog_bar=True)

        return None

    def on_train_epoch_end(self):
        print((self.sum / self.len).mean())
        sqsum = (self.sq_sum / self.len).mean()
        print(torch.sqrt(sqsum - (self.sum / self.len).mean() ** 2))

    def validation_step(self, batch, batch_idx):
        return
        specs = batch["noisy_spec"]
        specs = torch.abs(specs).float()
        specs = specs**self.compression
        labels = batch["label"]
        preds = self(specs.unsqueeze(1))

        loss = self.criterion(preds, labels)

        # reconstruct audio
        recon_specs = self.apply_inv_eq(batch["noisy_spec"], preds)
        recon_audio = torch.istft(
            recon_specs, window=self.window, return_complex=False, **self.stft_params
        )

        # l1 spectrum metric
        self.l1_spec(torch.abs(recon_specs).float(), batch["clean_spec"])

        # Compute SDR and SI-SDR
        self.sdr(batch["clean_audio"][..., : recon_audio.shape[-1]], recon_audio)
        self.si_sdr(batch["clean_audio"][..., : recon_audio.shape[-1]], recon_audio)

        self.log_dict(
            {
                "val/loss": loss.item(),
                "val/sdr": self.sdr,
                "val/si-sdr": self.si_sdr,
                "val/l1_spec": self.l1_spec,
            },
            on_epoch=True,
            on_step=False,
        )

        return {
            "loss": loss,
            "recon_spec": recon_specs,
            "recon_audio": recon_audio,
            "pred": preds,
        }  # for visualization


def interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    dim: int = -1,
    extrapolate: str = "constant",
) -> torch.Tensor:
    """One-dimensional linear interpolation between monotonically increasing sample
    points, with extrapolation beyond sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: The :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: The :math:`x`-coordinates of the data points, must be increasing.
        fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
        dim: Dimension across which to interpolate.
        extrapolate: How to handle values outside the range of `xp`. Options are:
            - 'linear': Extrapolate linearly beyond range of xp values.
            - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

    Returns:
        The interpolated values, same size as `x`.
    """
    # Move the interpolation dimension to the last axis
    x = x.movedim(dim, -1)
    xp = xp.movedim(dim, -1)
    fp = fp.movedim(dim, -1)

    m = torch.diff(fp) / torch.diff(xp)  # slope
    b = fp[..., :-1] - m * xp[..., :-1]  # offset
    indices = torch.searchsorted(xp, x, right=False)

    if extrapolate == "constant":
        # Pad m and b to get constant values outside of xp range
        m = torch.cat(
            [torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1
        )
        b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
    else:  # extrapolate == 'linear'
        indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

    values = m.gather(-1, indices) * x + b.gather(-1, indices)

    return values.movedim(-1, dim)
