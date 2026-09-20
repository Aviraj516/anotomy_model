import torch
import torch.nn as nn


# ============================================================
# DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):
    """
    Two consecutive convolution blocks.

    Conv -> BatchNorm -> ReLU
    Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# STANDARD U-NET
# ============================================================

class UNet(nn.Module):
    """
    Standard U-Net for 4-class retinal lesion segmentation.

    Input:
        [B, 3, H, W]

    Output:
        [B, 4, H, W]

    Channels:
        0 -> MA
        1 -> HE
        2 -> EX
        3 -> SE
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=4
    ):
        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # ====================================================
        # POOLING
        # ====================================================

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        self.output = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(
            self.pool(e4)
        )

        # ====================================================
        # DECODER
        # ====================================================

        d4 = self.up4(b)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # ====================================================
        # OUTPUT
        # ====================================================

        output = self.output(d1)

        return output


# ============================================================
# ATTENTION GATE
# ============================================================

class AttentionGate(nn.Module):

    def __init__(
        self,
        gate_channels,
        skip_channels,
        inter_channels
    ):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(
                gate_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(
                inter_channels
            )
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(
                skip_channels,
                inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(
                inter_channels
            )
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                inter_channels,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(self, g, x):

        g1 = self.W_g(g)

        x1 = self.W_x(x)

        psi = self.relu(
            g1 + x1
        )

        psi = self.psi(psi)

        return x * psi


# ============================================================
# ATTENTION U-NET
# ============================================================

class AttentionUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=4
    ):
        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.enc4 = DoubleConv(
            256,
            512
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = DoubleConv(
            512,
            1024
        )

        # ====================================================
        # DECODER 4
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )

        self.att4 = AttentionGate(
            gate_channels=512,
            skip_channels=512,
            inter_channels=256
        )

        self.dec4 = DoubleConv(
            1024,
            512
        )

        # ====================================================
        # DECODER 3
        # ====================================================

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionGate(
            gate_channels=256,
            skip_channels=256,
            inter_channels=128
        )

        self.dec3 = DoubleConv(
            512,
            256
        )

        # ====================================================
        # DECODER 2
        # ====================================================

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionGate(
            gate_channels=128,
            skip_channels=128,
            inter_channels=64
        )

        self.dec2 = DoubleConv(
            256,
            128
        )

        # ====================================================
        # DECODER 1
        # ====================================================

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionGate(
            gate_channels=64,
            skip_channels=64,
            inter_channels=32
        )

        self.dec1 = DoubleConv(
            128,
            64
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        self.final = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(
            self.pool(e4)
        )

        # ====================================================
        # DECODER 4
        # ====================================================

        d4 = self.up4(b)

        a4 = self.att4(
            d4,
            e4
        )

        d4 = self.dec4(
            torch.cat(
                [d4, a4],
                dim=1
            )
        )

        # ====================================================
        # DECODER 3
        # ====================================================

        d3 = self.up3(d4)

        a3 = self.att3(
            d3,
            e3
        )

        d3 = self.dec3(
            torch.cat(
                [d3, a3],
                dim=1
            )
        )

        # ====================================================
        # DECODER 2
        # ====================================================

        d2 = self.up2(d3)

        a2 = self.att2(
            d2,
            e2
        )

        d2 = self.dec2(
            torch.cat(
                [d2, a2],
                dim=1
            )
        )

        # ====================================================
        # DECODER 1
        # ====================================================

        d1 = self.up1(d2)

        a1 = self.att1(
            d1,
            e1
        )

        d1 = self.dec1(
            torch.cat(
                [d1, a1],
                dim=1
            )
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        return self.final(d1)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):
    """
    Lightweight Transformer block.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        hidden_dim = int(
            embed_dim * mlp_ratio
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                embed_dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                embed_dim
            ),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        y = self.norm1(x)

        attention_output, _ = (
            self.attention(
                y,
                y,
                y,
                need_weights=False
            )
        )

        x = x + attention_output

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# LIGHTWEIGHT TRANSFORMER U-NET
# ============================================================

class TransformerUNet(nn.Module):
    """
    Lightweight Transformer U-Net.

    Input:
        [B, 3, 512, 512]

    Output:
        [B, 4, 512, 512]

    Channels:
        0 = MA
        1 = HE
        2 = EX
        3 = SE
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=4,
        embed_dim=256,
        num_heads=8,
        num_transformer_blocks=4
    ):
        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = DoubleConv(
            in_channels,
            64
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.enc2 = DoubleConv(
            64,
            128
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.enc3 = DoubleConv(
            128,
            256
        )

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.enc4 = DoubleConv(
            256,
            256
        )

        self.pool4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # ====================================================
        # CNN TO TRANSFORMER
        # ====================================================

        self.projection = nn.Conv2d(
            256,
            embed_dim,
            kernel_size=1
        )

        # ====================================================
        # TRANSFORMER
        # ====================================================

        self.transformer = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    dropout=0.1
                )
                for _ in range(
                    num_transformer_blocks
                )
            ]
        )

        self.transformer_to_features = (
            nn.Conv2d(
                embed_dim,
                256,
                kernel_size=1
            )
        )

        # ====================================================
        # DECODER 4
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            256,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        # ====================================================
        # DECODER 3
        # ====================================================

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = DoubleConv(
            384,
            128
        )

        # ====================================================
        # DECODER 2
        # ====================================================

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = DoubleConv(
            192,
            64
        )

        # ====================================================
        # DECODER 1
        # ====================================================

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = DoubleConv(
            32,
            32
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        self.final = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        # ====================================================
        # 32 × 32 FEATURES
        # ====================================================

        features = self.pool4(e4)

        # ====================================================
        # CNN TO TRANSFORMER
        # ====================================================

        features = self.projection(
            features
        )

        batch_size, channels, height, width = (
            features.shape
        )

        # ====================================================
        # FEATURE MAP TO TOKENS
        # ====================================================

        tokens = features.flatten(
            2
        ).transpose(
            1,
            2
        )

        # ====================================================
        # TRANSFORMER
        # ====================================================

        for block in self.transformer:

            tokens = block(tokens)

        # ====================================================
        # TOKENS TO FEATURE MAP
        # ====================================================

        features = tokens.transpose(
            1,
            2
        ).reshape(
            batch_size,
            channels,
            height,
            width
        )

        features = (
            self.transformer_to_features(
                features
            )
        )

        # ====================================================
        # DECODER 4
        # ====================================================

        d4 = self.up4(features)

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(d4)

        # ====================================================
        # DECODER 3
        # ====================================================

        d3 = self.up3(d4)

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(d3)

        # ====================================================
        # DECODER 2
        # ====================================================

        d2 = self.up2(d3)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        # ====================================================
        # DECODER 1
        # ====================================================

        d1 = self.up1(d2)

        d1 = self.dec1(d1)

        # ====================================================
        # OUTPUT
        # ====================================================

        output = self.final(d1)

        return output


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING MODELS")
    print("=" * 60)

    x = torch.randn(
        1,
        3,
        512,
        512
    )

    # ========================================================
    # STANDARD U-NET
    # ========================================================

    print("\nStandard U-Net")

    standard_model = UNet(
        in_channels=3,
        out_channels=4
    )

    with torch.no_grad():
        standard_output = standard_model(x)

    print(
        "Input :",
        x.shape
    )

    print(
        "Output:",
        standard_output.shape
    )

    standard_params = sum(
        p.numel()
        for p in standard_model.parameters()
    )

    print(
        "Parameters:",
        f"{standard_params:,}"
    )

    # ========================================================
    # ATTENTION U-NET
    # ========================================================

    print("\nAttention U-Net")

    attention_model = AttentionUNet(
        in_channels=3,
        out_channels=4
    )

    with torch.no_grad():
        attention_output = attention_model(x)

    print(
        "Input :",
        x.shape
    )

    print(
        "Output:",
        attention_output.shape
    )

    attention_params = sum(
        p.numel()
        for p in attention_model.parameters()
    )

    print(
        "Parameters:",
        f"{attention_params:,}"
    )

    # ========================================================
    # TRANSFORMER U-NET
    # ========================================================

    print("\nTransformer U-Net")

    transformer_model = TransformerUNet(
        in_channels=3,
        out_channels=4
    )

    with torch.no_grad():
        transformer_output = (
            transformer_model(x)
        )

    print(
        "Input :",
        x.shape
    )

    print(
        "Output:",
        transformer_output.shape
    )

    transformer_params = sum(
        p.numel()
        for p in transformer_model.parameters()
    )

    print(
        "Parameters:",
        f"{transformer_params:,}"
    )

    print()
    print("=" * 60)
    print("MODEL TEST COMPLETED")
    print("=" * 60)