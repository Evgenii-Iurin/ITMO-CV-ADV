import os
import random
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import timm

import fiftyone.zoo as foz

import wandb

from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.utils.data import Dataset
from PIL import Image
import torch

class TripletFODataset(Dataset):
    def __init__(self, samples, transform=None, label_to_idx=None):
        """
        Параметры:
            samples (list): Список кортежей (filepath, label)
            transform: torchvision трансформации
            label_to_idx (dict): словарь отображения строковой метки в индекс
        """
        self.transform = transform

        if label_to_idx is None:
            labels = sorted({label for _, label in samples})
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        self.samples = [
            (filepath, self.label_to_idx[label]) for filepath, label in samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        image = Image.open(filepath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)



class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name="resnet18", embedding_dim=128, pretrained=True):
        """
        Модель-эмбеддер, использующая бэкбон из timm и дополнительный FC слой.
        Параметры:
            backbone_name (str): Имя модели-бэкбона (например, "resnet18").
            embedding_dim (int): Размерность выходного эмбеддинга.
            pretrained (bool): Использовать ли предобученные веса.
        """
        super(EmbeddingNet, self).__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        backbone_features = self.backbone.num_features
        self.fc = nn.Linear(backbone_features, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


import torch
import torch.nn.functional as F

def range_loss(embeddings, labels, margin=1.0):
    unique_labels = torch.unique(labels)
    intra_loss = 0.0
    inter_loss = 0.0
    count = 0

    class_means = []

    for label in unique_labels:
        class_mask = labels == label
        class_embeddings = embeddings[class_mask]

        if class_embeddings.size(0) < 2:
            continue  # Skip if not enough samples

        # Intra-class: max pairwise distance
        pdists = torch.cdist(class_embeddings, class_embeddings, p=2)
        max_dist = pdists.max()
        intra_loss += max_dist
        count += 1

        class_mean = class_embeddings.mean(dim=0)
        class_means.append(class_mean)

    if count > 0:
        intra_loss = intra_loss / count

    # Inter-class: min distance between class centers
    if len(class_means) >= 2:
        class_means = torch.stack(class_means)
        center_dists = torch.cdist(class_means, class_means, p=2)
        eye = torch.eye(center_dists.size(0), device=embeddings.device).bool()
        center_dists = center_dists.masked_fill(eye, float('inf'))
        min_center_dist = center_dists.min()
        inter_loss = torch.relu(margin - min_center_dist)

    total_loss = intra_loss + inter_loss

    # Убедитесь, что total_loss является тензором
    return total_loss


def train_one_epoch(model, dataloader, optimizer, device, margin=1.0):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # Ожидаем, что батч состоит из множества (image, label)
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(images)

        loss = range_loss(embeddings, labels, margin)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}")
            wandb.log({"batch_loss": loss.item()})

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Получаем эмбеддинги из модели
            embeddings = model(images)

            # Рассчитываем потерю
            loss = criterion(embeddings, labels)  # Вычисляем loss для всех эмбеддингов
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss



def validate_recall_at_k(model, dataloader, k, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            # Получаем эмбеддинги из модели
            embeddings = model(images)
            embeddings_list.append(embeddings)
            labels_list.append(labels)

    embeddings_all = torch.cat(embeddings_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    # Рассчитываем расстояния между всеми эмбеддингами
    distances = torch.cdist(embeddings_all, embeddings_all, p=2)
    sorted_indices = torch.argsort(distances, dim=1)

    hits = 0
    N = embeddings_all.size(0)
    for i in range(N):
        neighbors = sorted_indices[i, 1 : k + 1]
        if (labels_all[neighbors] == labels_all[i]).any():  # Проверяем, есть ли хотя бы один правильный сосед
            hits += 1

    recall_at_k = hits / N
    return recall_at_k


def main():
    # Инициализация wandb
    wandb.init(project="metric-learning")

    # Гиперпараметры
    BATCH_SIZE = 64
    MARGIN = 0.572356502367154
    LR = 0.0005
    EMBEDDING_DIM = 64
    NUM_EPOCHS = 1

    USE_FIFTYONE = False

    # Загрузка данных
    val_df = pd.read_csv("/home/kb/CV/ITMO-CV-ADV/src/hw_metric_learning/homework/val.csv")
    val_filenames = set(val_df["filename"].tolist())

    train_samples = []
    val_samples = []

    if USE_FIFTYONE:
        # Загрузка датасета через FiftyOne
        dataset = foz.load_zoo_dataset("caltech256", overwrite=True)
        print(f"Загружен Caltech256: {len(dataset)} образцов")

        for sample in dataset:
            filename = os.path.basename(sample.filepath)
            if "ground_truth" in sample and sample["ground_truth"] is not None:
                label = sample["ground_truth"]["label"]
            else:
                label = sample.get("label", None)
            if label is None:
                continue
            if filename in val_filenames:
                val_samples.append((sample.filepath, label))
            else:
                train_samples.append((sample.filepath, label))

    else:
        # Загрузка данных с локального пути
        data_dir = Path("/home/kb/CV/ITMO-CV-ADV/src/hw_metric_learning/homework/256_ObjectCategories")
        for sample_folder in data_dir.iterdir():
            for sample_path in sample_folder.iterdir():
                if sample_path.suffix not in [".jpg", ".jpeg", ".png"]:
                    continue

                label = sample_path.parent.name
                filename = sample_path.name
                if filename in val_filenames:
                    val_samples.append((sample_path, label))
                else:
                    train_samples.append((sample_path, label))

    print(f"Обучающих сэмплов: {len(train_samples)}")
    print(f"Валидационных сэмплов: {len(val_samples)}")

    # Логируем количество сэмплов
    wandb.log({
        "train_samples": len(train_samples),
        "val_samples": len(val_samples)
    })

    # Создаем отображение меток в индексы
    all_labels = {label for _, label in (train_samples + val_samples)}
    labels_sorted = sorted(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}

    # Трансформации для изображений
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Создаем PyTorch датасеты
    train_dataset = TripletFODataset(
        train_samples, transform=transform, label_to_idx=label_to_idx
    )
    val_dataset = TripletFODataset(
        val_samples, transform=transform, label_to_idx=label_to_idx
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Используем GPU, если доступен
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    # Модель
    model = EmbeddingNet(
        backbone_name="levit_128", embedding_dim=EMBEDDING_DIM, pretrained=True
    )
    model.to(device)

    # Логируем модель
    wandb.watch(model)

    # Оптимизатор и планировщик
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, verbose=True)

    # Критерий потерь для Range Loss
    criterion = range_loss  # Убедитесь, что этот критерий правильно импортирован

    k = 1

    # Цикл обучения
    for epoch in range(NUM_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, margin=MARGIN
        )
        val_loss = validate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']  # Получаем текущий LR
        wandb.log({"learning_rate": lr, "epoch": epoch + 1})
        scheduler.step(val_loss)
        recall_at_k = validate_recall_at_k(model, val_loader, k, device)
        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Recall@{k}: {recall_at_k:.4f}"
        )

        # Логируем метрики после каждой эпохи
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            f"recall_at_{k}": recall_at_k
        })

        # Сохраняем модель после каждой эпохи
        os.makedirs("batch_hard", exist_ok=True)
        torch.save(model.state_dict(), f"batch_hard/model_epoch_{epoch + 1}.pth")

    wandb.finish()


if __name__ == "__main__":
    main()
