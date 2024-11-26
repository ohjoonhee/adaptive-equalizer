from typing import Optional
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt

import torch
from torch import nn

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from torchmetrics import Accuracy

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


class DefaultModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        vis_per_batch: int,
        stft_params: STFTConfig,
        vis_batches: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = net
        self.criterion = criterion

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

    def _apply_eq(self, spec, eq):
        eq = torch.pow(10, eq / 20)
        return spec * eq.unsqueeze(-1)

    def on_after_batch_transfer(
        self, batch: torch.Any, dataloader_idx: int
    ) -> torch.Any:
        clean_audio = batch["clean_audio"]
        eq = batch["label"]
        clean_spec = torch.stft(
            clean_audio, window=self.window, return_complex=True, **self.stft_params
        )
        noisy_spec = self._apply_eq(clean_spec, eq)
        noisy_audio = torch.istft(
            noisy_spec, window=self.window, return_complex=False, **self.stft_params
        )

        batch["clean_spec"] = clean_spec
        batch["noisy_spec"] = noisy_spec
        batch["noisy_audio"] = noisy_audio
        batch["label"] = -eq / 20

        return batch

    def on_fit_start(self) -> None:
        # Note that self.logger is set by the Trainer.fit()
        # self.logger is None at self.__init__
        self.is_wandb = isinstance(self.logger, WandbLogger)
        self.vis_per_batch = self.vis_per_batch if self.is_wandb else 0

    def training_step(self, batch, batch_idx):
        specs = batch["noisy_spec"]
        specs = torch.abs(specs).float()
        labels = batch["label"]
        preds = self(specs.unsqueeze(1))

        loss = self.criterion(preds, labels)
        self.log("train/loss", loss.item())

        return loss

    def on_validation_epoch_start(self) -> None:
        if self.vis_per_batch:
            self.table = wandb.Table(columns=["clean_audio", "noisy_audio", "pred"])
            # self.vis_examples = []

    def validation_step(self, batch, batch_idx):
        specs = batch["noisy_spec"]
        specs = torch.abs(specs).float()
        labels = batch["label"]
        preds = self(specs.unsqueeze(1))

        loss = self.criterion(preds, labels)

        self.log_dict(
            {
                "val/loss": loss.item(),
            },
            on_epoch=True,
            on_step=False,
        )

        if self.vis_per_batch and batch_idx < self.vis_batches:
            self.visualize_preds(
                specs, labels, preds, batch["clean_audio"], batch["noisy_audio"]
            )

    def visualize_preds(self, specs, labels, pred, clean_audio, noisy_audio):
        for i in range(min(len(specs), self.vis_per_batch)):
            plt.plot(labels[i].cpu().numpy(), label="label")
            plt.plot(pred[i].cpu().numpy(), label="pred")
            self.table.add_data(
                wandb.Audio(clean_audio[i].squeeze().cpu().numpy(), sample_rate=22050),
                wandb.Audio(noisy_audio[i].squeeze().cpu().numpy(), sample_rate=22050),
                wandb.Image(plt),
            )
            plt.close()

    def on_validation_epoch_end(self) -> None:
        if self.vis_per_batch:
            self.logger.experiment.log({"val/samples": self.table})

    # def test_step(self, batch, batch_idx):
    #     img, labels = batch
    #     pred = self(img)

    #     acc = self.accuracy(pred, labels)
    #     self.log("test_acc", acc)
