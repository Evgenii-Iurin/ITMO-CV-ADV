import numpy as np
import torch
import torch.nn as nn


def yolo_loss(predictions, targets, lambda_coord=5.0, lambda_noobj=0.5):
    """
    Вычисляет функцию потерь YOLO v1
    
    Args:
        predictions: тензор формы [batch_size, S*S, B*5+C]
        targets: тензор формы [batch_size, S*S, B*5+C]
        lambda_coord: вес для потерь координат
        lambda_noobj: вес для потерь отсутствия объекта
    """
    batch_size = predictions.size(0)
    S = int(np.sqrt(predictions.size(1)))  # Размер сетки
    B = 2  # Количество bounding boxes на ячейку
    C = predictions.size(2) - B*5  # Количество классов

    # Преобразуем предсказания в удобный формат
    predictions = predictions.view(batch_size, S*S, -1)
    
    # Извлекаем координаты, уверенность и классы для каждого предсказанного бокса
    pred_boxes = predictions[:, :, :B*4].contiguous().view(batch_size, S*S, B, 4)  # [batch, S*S, B, 4]
    pred_conf = predictions[:, :, B*4:B*5].contiguous().view(batch_size, S*S, B)  # [batch, S*S, B]
    pred_classes = predictions[:, :, B*5:].contiguous().view(batch_size, S*S, C)  # [batch, S*S, C]

    targets = targets.view(batch_size, S*S, -1)
    # Извлекаем целевые значения
    target_boxes = targets[:, :, :B*4].contiguous().view(batch_size, S*S, B, 4)  # [batch, S*S, B, 4]
    target_conf = targets[:, :, B*4:B*5].contiguous().view(batch_size, S*S, B)  # [batch, S*S, B]
    target_classes = targets[:, :, B*5:].contiguous().view(batch_size, S*S, C)  # [batch, S*S, C]
    
    # Маска для ячеек, содержащих объекты
    obj_mask = target_conf > 0  # [batch, S*S, B]
    noobj_mask = target_conf == 0  # [batch, S*S, B]
    
    # Потери для координат центра ячейки (x, y)
    xy_loss = lambda_coord * torch.sum(
        obj_mask * ((pred_boxes[:, :, :, 0] - target_boxes[:, :, :, 0]) ** 2 + 
                   (pred_boxes[:, :, :, 1] - target_boxes[:, :, :, 1]) ** 2)
    )
    
    # Потери для размеров (w, h) - используем квадратный корень
    wh_loss = lambda_coord * torch.sum(
        obj_mask * ((torch.sign(pred_boxes[:, :, :, 2]) * torch.sqrt(torch.abs(pred_boxes[:, :, :, 2]) + 1e-8) - 
                    torch.sqrt(target_boxes[:, :, :, 2] + 1e-8)) ** 2 + 
                   (torch.sign(pred_boxes[:, :, :, 3]) * torch.sqrt(torch.abs(pred_boxes[:, :, :, 3]) + 1e-8) - 
                    torch.sqrt(target_boxes[:, :, :, 3] + 1e-8)) ** 2)
    )
    
    # Потери для уверенности (confidence) при наличии объекта
    conf_obj_loss = torch.sum(obj_mask * (pred_conf - target_conf) ** 2)
    
    # Потери для уверенности при отсутствии объекта
    conf_noobj_loss = lambda_noobj * torch.sum(noobj_mask * (pred_conf - target_conf) ** 2)
    
    # Потери для классификации
    # Учитываем только ячейки с объектами
    cell_has_obj = torch.max(obj_mask, dim=2)[0]  # [batch, S*S]
    class_loss = torch.sum(
        cell_has_obj.unsqueeze(-1) * (pred_classes - target_classes) ** 2
    )
    
    # Суммарные потери
    total_loss = xy_loss + wh_loss + conf_obj_loss + conf_noobj_loss + class_loss
    
    # Нормализуем по размеру батча
    total_loss = total_loss / batch_size
    
    return total_loss