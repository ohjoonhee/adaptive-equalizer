import torch
from torch import nn

from transformers import Swinv2ForImageClassification, Swinv2Config


class HFSwinV2(nn.Module):
    def __init__(
        self,
        pretrain_path: str,
        config: Swinv2Config,
        ignore_mismatched_sizes: bool = True,
    ):
        super(HFSwinV2, self).__init__()
        self.backbone = Swinv2ForImageClassification.from_pretrained(
            pretrain_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )

    def forward(self, *args, **kwargs):
        return self.backbone(*args, **kwargs).logits
