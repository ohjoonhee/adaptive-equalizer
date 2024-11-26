from typing import Optional

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from torchmetrics import Accuracy

try:
    import wandb
finally:
    pass


class PredictInterpolatedEQModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        vis_per_batch: int,
        vis_batches: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "criterion"])

        self.net = net
        self.criterion = criterion
        self.vis_per_batch = vis_per_batch
        self.vis_batches = vis_batches if vis_batches is not None else float("inf")
        self.num_classes = num_classes

    def forward(self, x):
        return self.net(x)

    def on_fit_start(self) -> None:
        # Note that self.logger is set by the Trainer.fit()
        # self.logger is None at self.__init__
        self.is_wandb = isinstance(self.logger, WandbLogger)
        self.vis_per_batch = self.vis_per_batch if self.is_wandb else 0
        # if self.vis_per_batch:
        #     if hasattr(self.trainer.datamodule, "ID2CLS"):
        #         self.ID2CLS = self.trainer.datamodule.ID2CLS
        #     else:
        #         self.ID2CLS = list(range(self.num_classes))

    def training_step(self, batch, batch_idx):
        noisy_audio = batch["noisy_audio"]
        labels = batch["label"]  # [B, 1025]

        # labels_interp = F.interpolate(
        #     labels.unsqueeze(1),  # [B, 1, 1025]
        #     size=noisy_audio.shape[-1],
        #     mode="linear",
        #     align_corners=False,
        # )  # [B, 1, 66048]
        labels_interp = labels.unsqueeze(1)

        preds = self(noisy_audio.unsqueeze(1))

        loss = self.criterion(preds, labels_interp)
        self.log("train/loss", loss.item())

        return loss

    def on_validation_epoch_start(self) -> None:
        if self.vis_per_batch:
            self.table = wandb.Table(columns=["clean_audio", "noisy_audio", "pred"])

    def validation_step(self, batch, batch_idx):
        noisy_audio = batch["noisy_audio"]
        labels = batch["label"]  # [B, 1025]

        # labels_interp = F.interpolate(
        #     labels.unsqueeze(1),  # [B, 1, 1025]
        #     size=noisy_audio.shape[-1],
        #     mode="linear",
        #     align_corners=False,
        # )  # [B, 1, 66048]
        # # clean_audio = clean_audio[:, :, : noisy_audio.shape[-1]]
        labels_interp = labels.unsqueeze(1)

        preds = self(noisy_audio.unsqueeze(1))

        loss = self.criterion(preds, labels_interp)

        self.log_dict(
            {
                "val/loss": loss.item(),
            },
            on_epoch=True,
            on_step=False,
        )

        if self.vis_per_batch and batch_idx < self.vis_batches:
            self.visualize_preds(
                labels, preds, batch["clean_audio"], batch["noisy_audio"]
            )

    def visualize_preds(self, label, pred, clean_audio, noisy_audio):
        # downsample eqs to 1025 for quick visualization
        # label = F.interpolate(
        #     label, size=1025, mode="linear", align_corners=False
        # ).squeeze(
        #     1
        # )  # [B, 1025]

        # pred = F.interpolate(
        #     pred, size=1025, mode="linear", align_corners=False
        # ).squeeze(
        #     1
        # )  # [B, 1025]
        for i in range(min(len(clean_audio), self.vis_per_batch)):
            plt.plot(label[i].cpu().numpy(), label="label")
            plt.plot(pred[i].cpu().numpy(), label="pred")
            print(label[i], pred[i])
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
