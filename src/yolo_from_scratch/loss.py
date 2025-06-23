import torch
from src.yolo_from_scratch.utils import intersection_over_union, cell_to_image_coords


def yolo_loss(
    predictions,
    targets,
    lambda_coord=5.0,
    lambda_noobj=0.5,
    S: int = 7,
    B: int = 2,
    C: int = 2,
):
    """
    Вычисляет функцию потерь YOLO v1

    Args:
        predictions: тензор формы [B, S*S, B*5+C]
        targets: тензор формы [B, S*S, B*5+C]
        lambda_coord: вес для потерь координат
        lambda_noobj: вес для потерь отсутствия объекта
        S: количество ячеек сетки в одном измерении
        B: количество предсказываемых боксов для каждой ячейки
        C: количество классов
    """

    batch_size = predictions.size(0)

    # Преобразуем предсказания в удобный формат
    predictions = predictions.view(batch_size, S * S, -1)

    # Извлекаем координаты, уверенность и классы для каждого предсказанного бокса
    pred_boxes = (
        predictions[:, :, : B * 4].contiguous().view(batch_size, S * S, B, 4)
    )  # [batch, S*S, B, 4]
    pred_conf = (
        predictions[:, :, B * 4 : B * 5].contiguous().view(batch_size, S * S, B)
    )  # [batch, S*S, B]
    pred_classes = (
        predictions[:, :, B * 5 :].contiguous().view(batch_size, S * S, C)
    )  # [batch, S*S, C]

    targets = targets.view(batch_size, S * S, -1)
    # Извлекаем целевые значения
    target_boxes = (
        targets[:, :, : B * 4].contiguous().view(batch_size, S * S, B, 4)
    )  # [batch, S*S, B, 4]
    target_conf = (
        targets[:, :, B * 4 : B * 5].contiguous().view(batch_size, S * S, B)
    )  # [batch, S*S, B]
    target_classes = (
        targets[:, :, B * 5 :].contiguous().view(batch_size, S * S, C)
    )  # [batch, S*S, C]

    # Маска для ячеек, содержащих объекты
    obj_mask = target_conf > 0  # [batch, S*S, B]
    # noobj_mask = target_conf == 0  # [batch, S*S, B]

    # Находим бокс с максимальным IoU для каждого целевого бокса
    iou_scores = torch.zeros_like(pred_conf)
    responsible_mask = torch.zeros_like(obj_mask, dtype=torch.bool)

    # Преобразуем координаты из относительных координат ячейки в координаты изображения
    pred_boxes_img = torch.zeros_like(pred_boxes)
    target_boxes_img = torch.zeros_like(target_boxes)

    # Преобразуем координаты используя cell_to_image_coords
    for batch_idx in range(batch_size):
        for cell_idx in range(S * S):
            # Индексы ячейки в сетке
            row = cell_idx // S
            col = cell_idx % S

            for box_idx in range(B):
                # Применяем имеющуюся функцию для преобразования координат
                x, y, w, h = cell_to_image_coords(
                    row,
                    col,
                    pred_boxes[batch_idx, cell_idx, box_idx, 0].item(),
                    pred_boxes[batch_idx, cell_idx, box_idx, 1].item(),
                    pred_boxes[batch_idx, cell_idx, box_idx, 2].item(),
                    pred_boxes[batch_idx, cell_idx, box_idx, 3].item(),
                    S,
                )
                pred_boxes_img[batch_idx, cell_idx, box_idx] = torch.tensor(
                    [x, y, w, h]
                )

                # Делаем то же для целевых боксов
                x, y, w, h = cell_to_image_coords(
                    row,
                    col,
                    target_boxes[batch_idx, cell_idx, box_idx, 0].item(),
                    target_boxes[batch_idx, cell_idx, box_idx, 1].item(),
                    target_boxes[batch_idx, cell_idx, box_idx, 2].item(),
                    target_boxes[batch_idx, cell_idx, box_idx, 3].item(),
                    S,
                )
                target_boxes_img[batch_idx, cell_idx, box_idx] = torch.tensor(
                    [x, y, w, h]
                )

    # Находим предикторы с наивысшим IoU для каждого целевого бокса
    for batch_idx in range(batch_size):
        for cell_idx in range(S * S):
            for target_idx in range(B):
                if obj_mask[batch_idx, cell_idx, target_idx]:
                    # Для каждого целевого объекта находим предиктор с наибольшим IoU
                    target_box = target_boxes_img[
                        batch_idx, cell_idx, target_idx
                    ].unsqueeze(0)
                    max_iou = 0
                    best_box_idx = 0

                    for pred_idx in range(B):
                        pred_box = pred_boxes_img[
                            batch_idx, cell_idx, pred_idx
                        ].unsqueeze(0)
                        iou = intersection_over_union(pred_box, target_box)
                        iou_scores[batch_idx, cell_idx, pred_idx] = iou

                        if iou > max_iou:
                            max_iou = iou
                            best_box_idx = pred_idx

                    # Назначаем ответственность предиктору с наивысшим IoU
                    responsible_mask[batch_idx, cell_idx, best_box_idx] = True

    # Координатные потери только для ответственных предикторов
    xy_loss = lambda_coord * torch.sum(
        responsible_mask
        * (
            (pred_boxes[:, :, :, 0] - target_boxes[:, :, :, 0]) ** 2
            + (pred_boxes[:, :, :, 1] - target_boxes[:, :, :, 1]) ** 2
        )
    )

    # Потери для размеров (w, h) - только для ответственных предикторов
    wh_loss = lambda_coord * torch.sum(
        responsible_mask
        * (
            (
                torch.sign(pred_boxes[:, :, :, 2])
                * torch.sqrt(torch.abs(pred_boxes[:, :, :, 2]) + 1e-8)
                - torch.sqrt(target_boxes[:, :, :, 2] + 1e-8)
            )
            ** 2
            + (
                torch.sign(pred_boxes[:, :, :, 3])
                * torch.sqrt(torch.abs(pred_boxes[:, :, :, 3]) + 1e-8)
                - torch.sqrt(target_boxes[:, :, :, 3] + 1e-8)
            )
            ** 2
        )
    )

    # Потери для уверенности (confidence):
    # - Для ответственных предикторов: уверенность должна равняться IoU
    conf_responsible_loss = torch.sum(responsible_mask * (pred_conf - iou_scores) ** 2)

    # - Для всех остальных предикторов: уверенность должна быть близка к 0
    conf_noobj_loss = lambda_noobj * torch.sum((~responsible_mask) * (pred_conf**2))

    # Потери для классификации - только для ячеек с объектами
    cell_has_obj = torch.max(obj_mask, dim=2)[0]  # [batch, S*S]
    class_loss = torch.sum(
        cell_has_obj.unsqueeze(-1) * (pred_classes - target_classes) ** 2
    )

    # Суммарные потери
    total_loss = (
        xy_loss + wh_loss + conf_responsible_loss + conf_noobj_loss + class_loss
    )

    # Нормализуем по размеру батча
    total_loss = total_loss / batch_size

    return total_loss
