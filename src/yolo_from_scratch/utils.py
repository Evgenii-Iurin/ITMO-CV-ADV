import torch
import numpy as np


def cell_to_image_coords(i, j, x_cell, y_cell, w_cell, h_cell, S):
    """
    Преобразует координаты из клеточных (ячейка сетки) в относительные координаты всего изображения.

    Args:
        i (int): индекс строки ячейки
        j (int): индекс столбца ячейки
        x_cell (float): x внутри ячейки (0..1)
        y_cell (float): y внутри ячейки (0..1)
        w_cell (float): ширина в ячейках
        h_cell (float): высота в ячейках
        S (int): размер сетки (например, 7 для YOLOv1)

    Returns:
        x_center (float): центр по x относительно всего изображения (0..1)
        y_center (float): центр по y относительно всего изображения (0..1)
        w (float): ширина относительно всего изображения (0..1)
        h (float): высота относительно всего изображения (0..1)
    """
    x_center = (j + x_cell) / S
    y_center = (i + y_cell) / S
    w = w_cell / S
    h = h_cell / S

    return x_center, y_center, w, h


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
                        torch.tensor([other_box], dtype=torch.float32),
                    ).item()
                    ious.append(iou)

                # Оставляем только боксы с IoU < порога
                cls_boxes = cls_boxes[1:][np.array(ious) < iou_threshold]

            result_boxes.extend(keep)

    # Сортируем результат по img_idx и confidence
    result_boxes = sorted(result_boxes, key=lambda x: (x[0], -x[1]))
    return [list(box) for box in result_boxes]


def get_bboxes(
    dataloader, model, iou_threshold=0.5, threshold=0.4, device="cpu", max_batches=None
):
    """
    Собирает списки предсказанных и реальных боксов из даталоадера.

    Args:
        dataloader: DataLoader с данными
        model: Модель YOLO
        iou_threshold: Порог для NMS (не используется в этой функции)
        threshold: Порог уверенности для предсказаний
        device: Устройство для вычислений
        max_batches: Максимальное количество батчей для обработки

    Returns:
        pred_boxes: список [img_idx, confidence, class_pred, x_center, y_center, width, height]
        true_boxes: список [img_idx, class_true, x_center, y_center, width, height]
        !! возвращает метки с координатами, нормализованными относительно ВСЕЙ картинки, а не grid cells
    """
    model.eval()
    pred_boxes = []
    true_boxes = []
    img_idx = 0
    batch_count = 0
    S = 7  # Размер grid-сетки
    B = 2  # Количество bounding boxes на ячейку
    C = 2  # Количество классов

    for imgs, labels in dataloader:
        if max_batches and batch_count >= max_batches:
            break

        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(imgs)

        batch_size = imgs.shape[0]

        for batch_idx in range(batch_size):
            label = labels[batch_idx]
            prediction = preds[batch_idx]

            # Проверка размерности предсказания
            if prediction.shape != (S, S, B * 5 + C):
                raise ValueError(
                    f"Unexpected prediction shape {prediction.shape}, expected (7, 7, {B*5 + C})"
                )

                # Истинные боксы (из labels)
            for i in range(S):
                for j in range(S):
                    for b in range(B):
                        p_idx = b * 4
                        conf_idx = B * 4 + b
                        if label[i, j, conf_idx] > 0.5:  # если есть объект
                            x_cell, y_cell, w_cell, h_cell = label[
                                i, j, p_idx : p_idx + 4
                            ]
                            class_label = torch.argmax(label[i, j, B * 5 : B * 5 + C])

                            # Восстановление глобальных координат
                            x = (j + x_cell.item()) / S
                            y = (i + y_cell.item()) / S
                            w = w_cell.item() / S
                            h = h_cell.item() / S

                            true_boxes.append([img_idx, class_label.item(), x, y, w, h])

            # Предсказанные боксы (из prediction)
            for i in range(S):
                for j in range(S):
                    best_box = None
                    max_conf = 0

                    for b in range(B):
                        conf = prediction[i, j, B * 4 + b]  # confidence отдельно
                        if conf > max_conf:
                            max_conf = conf
                            best_box = b

                    if best_box is not None and max_conf > threshold:
                        box = prediction[
                            i, j, best_box * 4 : (best_box + 1) * 4
                        ]  # координаты
                        class_probs = prediction[i, j, B * 5 : B * 5 + C]
                        best_class = torch.argmax(class_probs)
                        best_class_score = class_probs[best_class] * max_conf

                        x, y, w, h = box
                        x = (j + x.item()) / S
                        y = (i + y.item()) / S
                        w = w.item() / S
                        h = h.item() / S

                        pred_boxes.append(
                            [
                                img_idx,
                                best_class_score.item(),
                                best_class.item(),
                                x,
                                y,
                                w,
                                h,
                            ]
                        )

            img_idx += 1
        batch_count += 1

    return pred_boxes, true_boxes
