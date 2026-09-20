import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class OpticDiscUNet(nn.Module):

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
    print("OPTIC DISC U-NET TEST")
    print("=" * 60)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    model = OpticDiscUNet().to(device)

    dummy = torch.randn(
        1, 3, 512, 512
    ).to(device)

    with torch.no_grad():
        output = model(dummy)

    print("Input :", dummy.shape)
    print("Output:", output.shape)

    print("\nOptic Disc model test successful.")