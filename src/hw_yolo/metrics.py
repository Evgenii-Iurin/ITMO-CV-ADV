import torch
from src.hw_yolo.utils import intersection_over_union


def calculate_map(pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20):
    """
    Подсчет mean Average Precision (mAP) для всех классов.

    pred_boxes: список [img_idx, confidence, class_pred, x, y, w, h]
    true_boxes: список [img_idx, class_true, x, y, w, h]
    """
    average_precisions = []

    epsilon = 1e-6  # маленькая константа для стабильности

    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[2] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        amount_bboxes = {}

        # сколько реальных боксов на каждую картинку
        for gt in ground_truths:
            img_idx = gt[0]
            if img_idx not in amount_bboxes:
                amount_bboxes[img_idx] = 0
            amount_bboxes[img_idx] += 1

        # какие боксы уже нашли
        for key in amount_bboxes:
            amount_bboxes[key] = torch.zeros(amount_bboxes[key])

        # сортируем предсказания по уверенности
        detections.sort(key=lambda x: x[1], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue  # нет объектов этого класса

        for detection_idx, detection in enumerate(detections):
            img_idx = detection[0]
            best_iou = 0
            best_gt_idx = -1

            gts_for_img = [gt for gt in ground_truths if gt[0] == img_idx]

            for gt_idx, gt in enumerate(gts_for_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]), torch.tensor(gt[2:])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                if amount_bboxes[img_idx][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[img_idx][
                        best_gt_idx
                    ] = 1  # Помечаем как использованный
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # Precision и Recall
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # AUC Precision-Recall - по мтеоду трапеций
        average_precision = torch.trapz(precisions, recalls)
        average_precisions.append(average_precision)

    # Возвращаем среднее значение AP по всем классам
    return sum(average_precisions) / len(average_precisions)
