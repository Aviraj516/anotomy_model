import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)


class FoveaNet(nn.Module):

    def __init__(self):

        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT

        self.backbone = efficientnet_b0(
            weights=weights
        )

        num_features = (
            self.backbone.classifier[1].in_features
        )

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.backbone(x)


if __name__ == "__main__":

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("FOVEA NETWORK TEST")
    print("=" * 60)

    print("Device:", device)

    model = FoveaNet().to(device)

    dummy = torch.randn(
        2,
        3,
        512,
        512
    ).to(device)

    with torch.no_grad():

        output = model(dummy)

    print("Input shape :", dummy.shape)
    print("Output shape:", output.shape)

    print("\nExpected output: [batch, 2]")
    print("2 values = X, Y normalized coordinates")

    print("\nModel test completed successfully.")