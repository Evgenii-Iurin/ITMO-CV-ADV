import torch.nn as nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class HandKeypointModel(nn.Module):
    def __init__(self, architecture="simple", pretrained=False):
        super(HandKeypointModel, self).__init__()
        self.architecture = architecture.lower()

        if self.architecture == "simple":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.feature_dim = 256

        elif self.architecture == "resnet18":
            base = models.resnet18(pretrained=pretrained)
            layers = list(base.children())[:-1]  # remove fc layer
            self.backbone = nn.Sequential(*layers)  # output: [B, 512, 1, 1]
            self.feature_dim = 512

        elif self.architecture == "efficientnet_b0":
            base = efficientnet_b0(
                weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
            )
            self.backbone = base.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        # Общая голова
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 67),  # bbox(4) + keypoints(21*2) + visibility(21)
        )

    def forward(self, x):
        x = self.backbone(x)
        if self.architecture == "efficientnet_b0":
            x = self.pool(x)
        x = self.head(x)

        # Разбиваем выход
        bbox = x[:, :4]
        keypoints = x[:, 4:46].view(-1, 21, 2)
        visibility = x[:, 46:].view(-1, 21)

        return {"bbox": bbox, "keypoints": keypoints, "visibility": visibility}
