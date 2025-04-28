import torch
import numpy as np


def intersection_over_union(boxes_preds, boxes_labels):
    """
    IoU между предсказанными и реальными боксами.

    boxes_preds: [x_center, y_center, width, height]
    boxes_labels: [x_center, y_center, width, height]
    Все координаты нормализованы (0-1).
    """

    # Переводим из (x_center, y_center, w, h) в (x1, y1, x2, y2)
    box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

    box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    # координаты пересечения
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # площадь пересечения
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Площади боксов
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Площадь объединения
    union = box1_area + box2_area - intersection + 1e-6  # чтобы не делить на 0

    iou = intersection / union

    return iou


def non_max_suppression(pred_boxes, iou_threshold=0.5):
    """
    Применяет Non-Maximum Suppression к списку предсказанных bounding box'ов

    Args:
        pred_boxes: список [img_idx, confidence, class_pred, x_center, y_center, width, height]
        iou_threshold: порог для подавления дубликатов

    Returns:
        Отфильтрованный список pred_boxes после NMS
    """
    if not pred_boxes:
        return []

    # Конвертируем в numpy для удобства работы
    boxes_array = np.array(pred_boxes)

    # Если нет боксов, возвращаем пустой список
    if len(boxes_array) == 0:
        return []

    # Разделяем по изображениям и классам
    result_boxes = []
    unique_images = np.unique(boxes_array[:, 0])

    for img_idx in unique_images:
        img_boxes = boxes_array[boxes_array[:, 0] == img_idx]
        unique_classes = np.unique(img_boxes[:, 2])

        for cls in unique_classes:
            # Выбираем боксы текущего класса
            cls_mask = img_boxes[:, 2] == cls
            cls_boxes = img_boxes[cls_mask]

            # Сортируем по confidence в убывающем порядке
            sorted_indices = np.argsort(-cls_boxes[:, 1])
            cls_boxes = cls_boxes[sorted_indices]

            # Применяем NMS для текущего класса
            keep = []
            while len(cls_boxes) > 0:
                # Добавляем бокс с максимальным confidence
                keep.append(cls_boxes[0])

                if len(cls_boxes) == 1:
                    break

                # Вычисляем IoU с остальными боксами
                current_box = np.array(keep[-1][3:7])  # x, y, w, h
                other_boxes = np.array(cls_boxes[1:, 3:7])

                ious = []
                for other_box in other_boxes:
                    iou = intersection_over_union(
                        torch.tensor([current_box], dtype=torch.float32),
                        torch.tensor([other_box], dtype=torch.float32)
                    ).item()
                    ious.append(iou)

                # Оставляем только боксы с IoU < порога
                cls_boxes = cls_boxes[1:][np.array(ious) < iou_threshold]

            result_boxes.extend(keep)

    # Сортируем результат по img_idx и confidence
    result_boxes = sorted(result_boxes, key=lambda x: (x[0], -x[1]))
    return [list(box) for box in result_boxes]
