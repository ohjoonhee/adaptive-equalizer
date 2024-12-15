import torch
from torch import nn
from torchvision.models import efficientnet_b0


class EfficientNet(nn.Module):
    def __init__(self, output_dim: int = 1025, pretrained: bool = True) -> None:
        super(EfficientNet, self).__init__()
        self.backbone = efficientnet_b0(pretrained=pretrained)
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False
        )
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, output_dim)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    inputs = torch.randn(32, 1, 1025, 130)
    m = EfficientNet()
    outputs = m(inputs)
    print(outputs.shape)
