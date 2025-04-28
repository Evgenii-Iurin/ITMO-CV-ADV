import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, B=2):
        """
        Args:
            root_dir (string): Путь к папке dataset_ready (с подпапками imgs и labels)
            transform (callable, optional): Трансформации для изображений
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, "img")
        self.label_dir = os.path.join(root_dir, "labels")
        self.B = B

        # Получаем список всех файлов изображений
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
      # Загружаем изображение
      img_name = self.img_files[idx]
      img_path = os.path.join(self.img_dir, img_name)
      image = Image.open(img_path).convert('RGB')

      # Загружаем разметку
      label_name = img_name.replace('.jpg', '.txt')
      label_path = os.path.join(self.label_dir, label_name)

      # Инициализация target'а для YOLO
      S = 7  # Размер сетки
      C = 2
      label_matrix = torch.zeros((S, S, self.B * 5 + C))

      if os.path.exists(label_path):
          with open(label_path, 'r') as f:
              lines = f.readlines()

          for line in lines:
              # Парсим строку с нормализованными координатами
              class_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
              class_id = int(class_id)

              # Определяем ячейку (i,j)
              i = int(S * y_center)
              j = int(S * x_center)
              i = min(max(i, 0), S - 1)
              j = min(max(j, 0), S - 1)

              # вычисляем координаты ОТНОСИТЕЛЬНО ячейки
              x_cell = S * x_center - j
              y_cell = S * y_center - i
              w_cell = box_w * S    # ширина в ячейках
              h_cell = box_h * S    # высота в ячейках

              for b in range(self.B):
                  if label_matrix[i, j, b * 5] == 0:
                      label_matrix[i, j, b * 5: (b + 1) * 5] = torch.tensor([
                          1,         # вероятность объекта
                          x_cell,    # x относительно ячейки
                          y_cell,    # y относительно ячейки
                          w_cell,    # ширина в размере ячеек
                          h_cell     # высота в размере ячеек
                      ])
                      label_matrix[i, j, self.B * 5 + class_id] = 1
                      break

      if self.transform:
          image = self.transform(image)

      return image, label_matrix

