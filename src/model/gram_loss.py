from typing import Optional
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import nn

import lightning as L
from lightning.pytorch.loggers import WandbLogger


from .default import DefaultModel, STFTConfig

try:
    import wandb
finally:
    pass


class GramLossModel(DefaultModel):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        stft_params: STFTConfig,
        sr: Optional[int] = None,
        compression: Optional[float] = None,
        vis_per_batch: int = 0,
        vis_batches: Optional[int] = None,
    ):
        super().__init__(
            net, criterion, stft_params, sr, compression, vis_per_batch, vis_batches
        )

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        specs = batch["noisy_spec"]
        specs = torch.abs(specs).float()
        specs = specs**self.compression
        preds = self(specs.unsqueeze(1))

        recon = self.apply_inv_eq(torch.abs(batch["noisy_spec"]).float(), preds)

        loss = self.criterion(recon, torch.abs(batch["clean_spec"]).float())
        self.log("train/loss", loss.item(), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        specs = batch["noisy_spec"]
        specs = torch.abs(specs).float()
        specs = specs**self.compression
        labels = batch["label"]
        preds = self(specs.unsqueeze(1))
        recon = self.apply_inv_eq(torch.abs(batch["noisy_spec"]).float(), preds)

        loss = self.criterion(recon, torch.abs(batch["clean_spec"]).float())

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
        }


# def interp(
#     x: torch.Tensor,
#     xp: torch.Tensor,
#     fp: torch.Tensor,
#     dim: int = -1,
#     extrapolate: str = "constant",
# ) -> torch.Tensor:
#     """One-dimensional linear interpolation between monotonically increasing sample
#     points, with extrapolation beyond sample points.

#     Returns the one-dimensional piecewise linear interpolant to a function with
#     given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

#     Args:
#         x: The :math:`x`-coordinates at which to evaluate the interpolated
#             values.
#         xp: The :math:`x`-coordinates of the data points, must be increasing.
#         fp: The :math:`y`-coordinates of the data points, same shape as `xp`.
#         dim: Dimension across which to interpolate.
#         extrapolate: How to handle values outside the range of `xp`. Options are:
#             - 'linear': Extrapolate linearly beyond range of xp values.
#             - 'constant': Use the boundary value of `fp` for `x` values outside `xp`.

#     Returns:
#         The interpolated values, same size as `x`.
#     """
#     # Move the interpolation dimension to the last axis
#     x = x.movedim(dim, -1)
#     xp = xp.movedim(dim, -1)
#     fp = fp.movedim(dim, -1)

#     m = torch.diff(fp) / torch.diff(xp)  # slope
#     b = fp[..., :-1] - m * xp[..., :-1]  # offset
#     indices = torch.searchsorted(xp, x, right=False)

#     if extrapolate == "constant":
#         # Pad m and b to get constant values outside of xp range
#         m = torch.cat(
#             [torch.zeros_like(m)[..., :1], m, torch.zeros_like(m)[..., :1]], dim=-1
#         )
#         b = torch.cat([fp[..., :1], b, fp[..., -1:]], dim=-1)
#     else:  # extrapolate == 'linear'
#         indices = torch.clamp(indices - 1, 0, m.shape[-1] - 1)

#     values = m.gather(-1, indices) * x + b.gather(-1, indices)

#     return values.movedim(-1, dim)
