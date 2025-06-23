import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # Add BatchNorm after conv
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),  # Add BatchNorm again
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetKeypoint(nn.Module):
    def __init__(self, in_channels=3, num_joints=21):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Final layer: 1 channel per joint
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, num_joints, kernel_size=1),
            # nn.Sigmoid()  # Explicit sigmoid for BCELoss
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bottleneck = self.bottleneck(self.pool4(d4))

        # Decoder with skip connections
        x = self.up4(bottleneck)
        x = self.dec4(torch.cat([x, d4], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, d1], dim=1))

        return self.out_conv(x)  # Shape: [B, 21, 224, 224]


class DoubleConvLight(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetKeypointLight(nn.Module):
    def __init__(self, in_channels=3, num_joints=1):
        super().__init__()

        # Уменьшенное количество каналов
        self.down1 = DoubleConvLight(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConvLight(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConvLight(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConvLight(64, 128)

        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DoubleConvLight(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = DoubleConvLight(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = DoubleConvLight(32, 16)

        # Выход: 1 тепловая карта
        self.out_conv = nn.Conv2d(16, num_joints, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bottleneck = self.bottleneck(self.pool3(d3))

        # Decoder
        x = self.up3(bottleneck)
        x = self.dec3(torch.cat([x, d3], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, d2], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, d1], dim=1))

        return self.out_conv(x)  # [B, 1, H, W]
