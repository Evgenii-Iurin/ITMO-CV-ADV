import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU
from torch.utils.data import DataLoader
from typing import Callable


class YOLOv1(nn.Module):
    def __init__(self, num_classes=2, bounding_boxes = 2):
        super().__init__()
        self.num_classes = num_classes
        self.bounding_boxes = bounding_boxes
        # Размерность выхода: S x S x (B*5 + C) = 7 x 7 x (3*5 + 2) = 7 x 7 x 17
        self.output_dim = 7 * 7 * (self.bounding_boxes*5 + num_classes)

        self.model = nn.Sequential(
            # 1. Первый сверточный слой
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            ReLU(),
            MaxPool2d(2, stride=2),

            # 2. Дальше сверточные блоки
            Conv2d(64, 192, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, stride=2),

            Conv2d(192, 128, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(128, 256, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, stride=2),

            # 3. Блоки с 4 повторами 1x1 → 3x3
            Conv2d(512, 256, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(),

            Conv2d(512, 256, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(),

            Conv2d(512, 256, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(),

            Conv2d(512, 256, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(256, 512, kernel_size=3, padding=1),
            ReLU(),

            Conv2d(512, 512, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(2, stride=2),

            # 4. Дальше глубокие сверточные слои
            Conv2d(1024, 512, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            ReLU(),

            Conv2d(1024, 512, kernel_size=1, padding=0),
            ReLU(),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            ReLU(),

            Conv2d(1024, 1024, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            ReLU(),

            # 5. Заключительные свертки
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            ReLU(),

            Flatten(),

            # 6. Полносвязные слои
            Linear(1024 * 7 * 7, 4096),
            ReLU(),
            Linear(4096, self.output_dim)  # Изменено на правильную размерность
        )

    def forward(self, x):
    #С СИГМОИДАМИ
        out = self.model(x)
        out = out.view(-1, 7, 7, self.bounding_boxes*5 + self.num_classes)

        for b in range(self.bounding_boxes):
            # Индексы параметров bbox
            x_idx, y_idx, w_idx, h_idx = b*5, b*5+1, b*5+2, b*5+3

            # Все координаты через sigmoid (в [0,1])
            out[..., x_idx] = torch.sigmoid(out[..., x_idx])  # x_center
            out[..., y_idx] = torch.sigmoid(out[..., y_idx])  # y_center
            out[..., w_idx] = torch.sigmoid(out[..., w_idx])  # width
            out[..., h_idx] = torch.sigmoid(out[..., h_idx])  # height

            # Confidence
            out[..., b*5+4] = torch.sigmoid(out[..., b*5+4])

        # Softmax для классов
        out[..., self.bounding_boxes*5:] = torch.softmax(out[..., self.bounding_boxes*5:], dim=-1)

        return out

    # def forward(self, x):
    #     out = self.model(x)
    #     out = out.view(-1, 7, 7, self.bounding_boxes*5 + self.num_classes)

    #     for b in range(self.bounding_boxes):
    #         # Confidence score проходит через sigmoid
    #         out[..., b*5+4] = torch.sigmoid(out[..., b*5+4])

    #     # Softmax для классов
    #     out[..., self.bounding_boxes*5:] = torch.softmax(out[..., self.bounding_boxes*5:], dim=-1)

    #     return out


    def train_model(self, dataloader: DataLoader, loss_fn: Callable, optimizer: torch.optim.Optimizer, epochs: int = 10):
        self.train()

        for epoch in range(epochs):
            epoch_loss = 0
            i = 0
            for imgs, labels in dataloader:
                optimizer.zero_grad()
                preds = self(imgs)
                loss_value = loss_fn(preds, labels)
                loss_value.backward()
                optimizer.step()

                epoch_loss += loss_value.item()
                # i += 1
                # if i == 16:
                #     print(f"Batch {epoch+1}/{epochs}, Loss: {loss_value.item():.4f}")
                #     i = 0

            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
