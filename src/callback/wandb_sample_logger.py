from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from lightning.pytorch import Callback

import numpy as np
import matplotlib.pyplot as plt

import wandb
from lightning.pytorch.utilities import rank_zero_only


class WandbSampleLoggerCallback(Callback):
    def __init__(
        self,
        train_log_interval: int = 1000,
        val_log_interval: int = 1,
        train_samples_per_batch: int = 10,
        val_samples_per_batch: int = 10,
        train_batches_per_epoch: int = 10,
        val_batches_per_epoch: int = 10,
    ) -> None:
        """
        Args:
            log_interval: Number of steps between logging. Default: ``1000``.
            scale_factor: Scale factor used for downsampling the high-res images. Default: ``4``.
            num_samples: Number of images of displayed in the grid. Default: ``5``.
        """
        super().__init__()
        self.train_log_interval = train_log_interval
        self.val_log_interval = val_log_interval
        self.train_samples_per_batch = train_samples_per_batch
        self.val_samples_per_batch = val_samples_per_batch
        self.train_batches_per_epoch = train_batches_per_epoch
        self.val_batches_per_epoch = val_batches_per_epoch

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.train_log_interval == 0:
            self.train_table = wandb.Table(
                columns=["clean_audio", "noisy_audio", "label"]
            )

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        labels = batch["label"]
        clean_audio = batch["clean_audio"]
        noisy_audio = batch["noisy_audio"]
        if (
            trainer.current_epoch % self.train_log_interval == 0
            and batch_idx < self.train_batches_per_epoch
        ):
            x_logscale = np.logspace(0, np.log10(pl_module.sr / 2), labels.size(-1))
            for i in range(min(len(clean_audio), self.train_samples_per_batch)):
                plt.plot(x_logscale, labels[i].cpu().numpy() * 20, label="label")
                plt.ylabel("Magnitude (dB)")
                plt.xlabel("Frequency (Hz)")
                plt.xscale("log")
                plt.legend()
                self.train_table.add_data(
                    wandb.Audio(
                        clean_audio[i].squeeze().cpu().numpy(), sample_rate=pl_module.sr
                    ),
                    wandb.Audio(
                        noisy_audio[i].squeeze().cpu().numpy(), sample_rate=pl_module.sr
                    ),
                    wandb.Image(plt),
                )
                plt.close()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.train_log_interval == 0:
            trainer.logger.experiment.log({"train/samples": self.train_table})

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch % self.val_log_interval == 0:
            self.val_table = wandb.Table(
                columns=["clean_audio", "noisy_audio", "recon_audio", "pred"]
            )

    @rank_zero_only
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (
            trainer.current_epoch % self.val_log_interval == 0
            and batch_idx < self.val_batches_per_epoch
        ):
            labels = batch["label"]
            pred = outputs["pred"]
            clean_audio = batch["clean_audio"]
            noisy_audio = batch["noisy_audio"]
            recon_audio = outputs["recon_audio"]
            x_logscale = np.logspace(0, np.log10(pl_module.sr / 2), labels.size(-1))
            for i in range(min(len(labels), self.val_samples_per_batch)):
                plt.plot(x_logscale, labels[i].cpu().numpy() * 20, label="label")
                plt.plot(x_logscale, pred[i].cpu().numpy() * 20, label="pred")
                plt.ylabel("Magnitude (dB)")
                plt.xlabel("Frequency (Hz)")
                plt.xscale("log")
                plt.legend()
                self.val_table.add_data(
                    wandb.Audio(
                        clean_audio[i].squeeze().cpu().numpy(), sample_rate=pl_module.sr
                    ),
                    wandb.Audio(
                        noisy_audio[i].squeeze().cpu().numpy(), sample_rate=pl_module.sr
                    ),
                    wandb.Audio(
                        recon_audio[i].squeeze().cpu().numpy(), sample_rate=pl_module.sr
                    ),
                    wandb.Image(plt),
                )
                plt.close()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.val_log_interval == 0:
            trainer.logger.experiment.log({"val/samples": self.val_table})

    @rank_zero_only
    def on_predict_epoch_start(self, trainer, pl_module):
        self.predict_table = wandb.Table(columns=["noisy_audio", "recon_audio", "pred"])

    @rank_zero_only
    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        # labels = batch["label"]
        pred = outputs["pred"]
        # clean_audio = batch["clean_audio"]
        noisy_audio = outputs["noisy_audio"]
        recon_audio = outputs["recon_audio"]
        x_logscale = np.logspace(0, np.log10(pl_module.sr / 2), pred.size(-1))
        for i in range(len(pred)):
            # plt.plot(x_logscale, labels[i].cpu().numpy() * 20, label="label")
            plt.plot(x_logscale, pred[i].cpu().numpy() * 20, label="pred")
            plt.ylabel("Magnitude (dB)")
            plt.xlabel("Frequency (Hz)")
            plt.xscale("log")
            plt.legend()
            self.predict_table.add_data(
                wandb.Audio(noisy_audio[i], sample_rate=pl_module.sr),
                wandb.Audio(recon_audio[i], sample_rate=pl_module.sr),
                wandb.Image(plt),
            )
            plt.close()

    @rank_zero_only
    def on_predict_epoch_end(self, trainer, pl_module):
        trainer.logger.experiment.log({"predict/samples": self.predict_table})
