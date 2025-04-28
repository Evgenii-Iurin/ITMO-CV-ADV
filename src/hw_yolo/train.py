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

            # Получаем размеры изображения для проверки
            img_width, img_height = image.size

            for line in lines:
                # Парсим строку с нормализованными координатами
                class_id, x_center, y_center, box_w, box_h = map(float, line.strip().split())
                class_id = int(class_id)  # У вас всегда 0

                # НЕ нормализуем координаты, так как они уже нормализованы!
                # x_center /= img_width
                # y_center /= img_height
                # box_w /= img_width
                # box_h /= img_height

                # Определяем grid-ячейку, куда попадает центр объекта
                i = int(S * y_center)
                j = int(S * x_center)
                i = min(max(i, 0), S - 1)  # Ограничиваем индексы
                j = min(max(j, 0), S - 1)

                # Если в ячейке есть свободное место для нового объекта
                # for b in range(self.B):
                #     if label_matrix[i, j, 4 + b * 5] == 0:  # Проверка на наличие объекта
                #         label_matrix[i, j, 4 + b * 5: 4 + (b + 1) * 5] = torch.tensor([1, x_center, y_center, box_w, box_h])  # Добавляем объект
                #         label_matrix[i, j, 5 + class_id] = 1  # Устанавливаем класс
                #         break  # Выходим из цикла, как только объект был добавлен в ячейку


                # Если в ячейке есть свободное место для нового объекта
                for b in range(self.B):
                    if label_matrix[i, j, b * 5] == 0:  # Проверка на наличие объекта (индекс 0 для каждого bbox)
                        # Записываем параметры bounding box (5 значений: p, x, y, w, h)
                        label_matrix[i, j, b * 5: (b + 1) * 5] = torch.tensor([
                            1,          # p (вероятность наличия объекта)
                            x_center,     # x (относительно ячейки)
                            y_center,     # y (относительно ячейки)
                            box_w,      # ширина (нормализованная)
                            box_h       # высота (нормализованная)
                        ])

                        # Устанавливаем класс (после всех B*5 параметров bbox)
                        label_matrix[i, j, self.B * 5 + class_id] = 1
                        break  # Выходим из цикла, как только объект был добавлен в ячейку

        # Применяем трансформации к изображению
        if self.transform:
            image = self.transform(image)

        return image, label_matrix

# Трансформации (оставляем как в оригинале)
transform = T.Compose([
    T.Resize((448, 448)),  # Размер входа для YOLO v1
    T.ToTensor(),
])

# Создаем датасет
dataset = CustomDataset(root_dir="dataset_ready", transform=transform)

# Проверка
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Пример вывода информации
print(f"Всего изображений в датасете: {len(dataset)}")
sample_img, sample_label = dataset[1]
print(f"Размер изображения после трансформации: {sample_img.shape}")
print(f"Размер label matrix: {sample_label.shape}")

from loss import yolo_loss
from model import YOLOv1

model = YOLOv1()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
model.train_model(dataloader, yolo_loss, optimizer, 1)

torch.save(model.state_dict(), "yolo_trained.pt")