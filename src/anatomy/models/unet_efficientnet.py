import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class VesselUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation=None
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    print("=" * 60)
    print("EFFICIENTNET-B0 U-NET TEST")
    print("=" * 60)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Device:", device)

    model = VesselUNet().to(device)

    dummy_input = torch.randn(
        2, 3, 512, 512
    ).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)

    print("\nModel test completed successfully.")