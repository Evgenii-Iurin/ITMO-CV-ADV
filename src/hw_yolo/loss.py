import numpy as np
import torch
import torch.nn as nn


def yolo_loss(predictions, targets, lambda_coord=5.0, lambda_noobj=0.5):

    S = int(np.sqrt(predictions.shape[0]))
    B = predictions.shape[1] 
    C = predictions.shape[2] - 5
    
    pred_boxes = predictions[:, :, :4]  # x, y, w, h для каждого bbox
    pred_conf = predictions[:, :, 4]
    pred_classes = predictions[:, :, 5:]
    
    target_boxes = targets[:, :, :4]
    target_conf = targets[:, :, 4]
    target_classes = targets[:, :, 5:]
  
    obj_mask = target_conf > 0
    noobj_mask = target_conf == 0
    
    xy_loss = lambda_coord * torch.sum(
        obj_mask * ((pred_boxes[:, :, 0] - target_boxes[:, :, 0]) ** 2 + 
                   (pred_boxes[:, :, 1] - target_boxes[:, :, 1]) ** 2)
    )
    #print(f"xy_loss={xy_loss}")
    
    # добавляем малые значения в torch.sqrt чтоб не падало на 0
    wh_loss = lambda_coord * torch.sum(
        obj_mask * ((torch.sqrt(pred_boxes[:, :, 2] + 1e-8) - torch.sqrt(target_boxes[:, :, 2] + 1e-8)) ** 2 + 
                   (torch.sqrt(pred_boxes[:, :, 3] + 1e-8) - torch.sqrt(target_boxes[:, :, 3] + 1e-8)) ** 2)
    )
    #print(f"wh_loss={wh_loss}")
    
    conf_obj_loss = torch.sum(obj_mask * (pred_conf - target_conf) ** 2)
    #print(f"conf_obj_loss={conf_obj_loss}")
    conf_noobj_loss = lambda_noobj * torch.sum(noobj_mask * (pred_conf - target_conf) ** 2)
    #print(f"conf_noobj_loss={conf_noobj_loss}")
    cell_has_obj = torch.max(obj_mask, dim=1)[0]  # [S*S]
    #print(f"cell_has_obj={cell_has_obj}")
    class_loss = torch.sum(
        torch.sum(cell_has_obj.unsqueeze(1) * (pred_classes[:, 0, :] - target_classes[:, 0, :]) ** 2, dim=1)
    )
    #print(f"class_loss={class_loss}")
    total_loss = xy_loss + wh_loss + conf_obj_loss + conf_noobj_loss + class_loss
    
    return total_loss
