import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, LeakyReLU
from torch.utils.data import DataLoader
from utils import get_bboxes, non_max_suppression
from metrics import intersection_over_union, calculate_map
from typing import Callable
from torch.utils.tensorboard import SummaryWriter

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class YOLOv1(nn.Module):
    def __init__(self, num_classes=2, bounding_boxes=2):
        super().__init__()
        self.num_classes = num_classes
        self.bounding_boxes = bounding_boxes
        self.output_dim = 7 * 7 * (self.bounding_boxes * 5 + num_classes)

        self.model = nn.Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            LeakyReLU(0.1),
            MaxPool2d(2, stride=2),
            Conv2d(64, 192, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            MaxPool2d(2, stride=2),
            Conv2d(192, 128, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(128, 256, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(256, 256, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            MaxPool2d(2, stride=2),
            Conv2d(512, 256, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(512, 256, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(512, 256, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(512, 256, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(512, 512, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            MaxPool2d(2, stride=2),
            Conv2d(1024, 512, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 512, kernel_size=1, padding=0),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Flatten(),
            Linear(1024 * 7 * 7, 4096),
            LeakyReLU(0.1),
            Linear(4096, self.output_dim),
        )

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1, 7, 7, self.bounding_boxes * 5 + self.num_classes)
        return out

    def train_model(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        device: str,
        epochs: int = 10,
        validation_dataloader: DataLoader = None,
        validate_every: int = 1,
        writer: SummaryWriter = None,
    ):
        self.to(device)

        for epoch in range(epochs):
            epoch_loss = 0
            self.train()
            for imgs, labels in dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                preds = self(imgs)
                loss_value = loss_fn(preds, labels)
                loss_value.backward()
                optimizer.step()

                epoch_loss += loss_value.item()

            avg_loss = epoch_loss / len(dataloader)
            logging.info("Epoch %d/%d, Train Loss: %.4f", epoch + 1, epochs, avg_loss)

            if writer is not None:
                writer.add_scalar("Loss/Train", avg_loss, epoch)

            if validation_dataloader is not None and (epoch + 1) % validate_every == 0:
                self.validate_model(
                    validation_dataloader, loss_fn, device, epoch, writer
                )

    def validate_model(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        device: str,
        epoch: int = None,
        writer: SummaryWriter = None,
    ):
        self.eval()
        self.to(device)
        total_loss = 0
        with torch.no_grad():
            for imgs, labels in dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                preds = self(imgs)
                loss_value = loss_fn(preds, labels)
                total_loss += loss_value.item()

        avg_loss = total_loss / len(dataloader)
        pred_boxes, true_boxes = get_bboxes(dataloader=dataloader, model=self, threshold=0.6, max_batches=None)
        filtered_boxes = non_max_suppression(pred_boxes, iou_threshold=0.5)
        map_score = calculate_map(filtered_boxes, true_boxes, iou_threshold=0.5, num_classes=2)
        logging.info("Validation Loss: %.4f", avg_loss)
        logging.info("mAP: %.4f", map_score)

        if writer is not None and epoch is not None:
            writer.add_scalar("Loss/Validation", avg_loss, epoch)
            writer.add_scalar("mAP/Validation", map_score, epoch)
        return avg_loss
