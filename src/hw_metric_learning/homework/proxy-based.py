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

from pytorch_metric_learning.losses import TripletMarginLoss

class TripletFODataset(Dataset):
    def __init__(self, samples, transform=None, label_to_idx=None):
        """
        Параметры:
            samples (list): Список кортежей (filepath, label) – путь к изображению и его строковая метка.
            transform: Трансформации для изображения.
            label_to_idx (dict): Словарь для отображения строковой метки в числовой индекс.
                            Если None, он будет вычислен по списку samples.
        """
        self.transform = transform
        # Если не передан mapping, вычисляем его из всех меток
        if label_to_idx is None:
            labels = sorted({label for _, label in samples})
            self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

        # Преобразуем метки в числовые индексы
        self.samples = [
            (filepath, self.label_to_idx[label]) for filepath, label in samples
        ]

        # Построим словарь: для каждого класса список индексов образцов данного класса
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Возвращает кортеж:
        (anchor_img, positive_img, negative_img, anchor_label, negative_label)
        """
        filepath, anchor_label = self.samples[index]
        anchor_img = Image.open(filepath).convert("RGB")
        if self.transform:
            anchor_img = self.transform(anchor_img)

        # Выбираем позитив: другое изображение того же класса
        positive_index = index
        while positive_index == index:
            positive_index = random.choice(self.class_to_indices[anchor_label])
        positive_filepath, _ = self.samples[positive_index]
        positive_img = Image.open(positive_filepath).convert("RGB")
        if self.transform:
            positive_img = self.transform(positive_img)

        # Выбираем негатив: изображение из другого класса
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = random.choice(list(self.class_to_indices.keys()))
        negative_index = random.choice(self.class_to_indices[negative_label])
        negative_filepath, negative_label = self.samples[negative_index]
        negative_img = Image.open(negative_filepath).convert("RGB")
        if self.transform:
            negative_img = self.transform(negative_img)

        # Приводим метки к тензорам
        return (
            anchor_img,
            positive_img,
            negative_img,
            torch.tensor(anchor_label),
            torch.tensor(negative_label),
        )


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
import torch.nn as nn

class ProxyNCALoss(nn.Module):
    def __init__(self, num_classes, embedding_size, scale=3.0):
        super(ProxyNCALoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, embeddings, labels):
        proxies = nn.functional.normalize(self.proxies, p=2, dim=1)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(embeddings, proxies.t()) * self.scale
        log_probs = nn.functional.log_softmax(similarity, dim=1)
        return nn.NLLLoss()(log_probs, labels)

def train_one_epoch(model, dataloader, optimizer, device, num_classes=257, margin=1.0, embedding_size=64):
    model.train()
    running_loss = 0.0
    criterion = ProxyNCALoss(num_classes=num_classes, 
                            embedding_size=embedding_size).to(device)

    for batch_idx, batch in enumerate(dataloader):
        anchor, _, _, anchor_label, _ = batch  # Используем только anchor и метки
        anchor = anchor.to(device)
        anchor_label = anchor_label.to(device)

        optimizer.zero_grad()
        anchor_out = model(anchor)
        loss = criterion(anchor_out, anchor_label)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.4f}")
            wandb.log({"batch_loss": loss.item()})

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            anchor, positive, negative, _, _ = batch

            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)

            loss = criterion(anchor_out, positive_out, negative_out)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate_recall_at_k(model, dataloader, k, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            # Из батча берём только anchor и его метку
            anchor, _, _, labels, _ = batch
            anchor = anchor.to(device)
            emb = model(anchor)
            embeddings_list.append(emb)
            labels_list.append(labels.to(device))

    embeddings_all = torch.cat(embeddings_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    distances = torch.cdist(embeddings_all, embeddings_all, p=2)
    sorted_indices = torch.argsort(distances, dim=1)

    hits = 0
    N = embeddings_all.size(0)
    for i in range(N):
        neighbors = sorted_indices[i, 1 : k + 1]
        if (labels_all[neighbors] == labels_all[i]).any():
            hits += 1

    recall_at_k = hits / N
    return recall_at_k


def main():
    wandb.init(project="metric-learning-distance")

    BATCH_SIZE = 64
    MARGIN = 0.572356502367154
    LR = 0.0005
    EMBEDDING_DIM = 64
    SEMI_HARD = True
    NUM_EPOCHS = 2

    USE_FIFTYONE = False

    val_df = pd.read_csv("/home/kb/CV/ITMO-CV-ADV/src/hw_metric_learning/homework/val.csv")
    val_filenames = set(val_df["filename"].tolist())

    train_samples = []
    val_samples = []

    if USE_FIFTYONE:
        # Загружаем датасет Caltech256 через FiftyOne
        dataset = foz.load_zoo_dataset("caltech256", overwrite=True)
        print(f"Загружен Caltech256: {len(dataset)} образцов")

        for sample in dataset:
            filename = os.path.basename(sample.filepath)
            # Предполагается, что метка хранится в поле ground_truth с ключом "label"
            if "ground_truth" in sample and sample["ground_truth"] is not None:
                label = sample["ground_truth"]["label"]
            else:
                # Если поле отсутствует, можно попробовать sample["label"]
                label = sample.get("label", None)
            if label is None:
                continue
            if filename in val_filenames:
                val_samples.append((sample.filepath, label))
            else:
                train_samples.append((sample.filepath, label))

    else:
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

    # Вычисляем общее отображение меток (label -> числовой индекс)
    all_labels = {label for _, label in (train_samples + val_samples)}
    labels_sorted = sorted(all_labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels_sorted)}

    # Определяем трансформации для изображений
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Создаем PyTorch-датасеты
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    model = EmbeddingNet(
        backbone_name="levit_128", embedding_dim=EMBEDDING_DIM, pretrained=True
    )
    model.to(device)

    wandb.watch(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=0, verbose=True)

    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    k = 1

    for epoch in range(NUM_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, margin=MARGIN, 
            # semi_hard=SEMI_HARD
        )
        val_loss = validate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']  # Получаем LR после завершения эпохи
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

        os.makedirs("batch_hard", exist_ok=True)
        torch.save(model.state_dict(), f"batch_hard/model_epoch_{epoch + 1}.pth")

    wandb.finish()


if __name__ == "__main__":
    main()
