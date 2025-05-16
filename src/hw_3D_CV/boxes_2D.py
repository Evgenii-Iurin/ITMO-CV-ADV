import numpy as np

def filter_lidar_within_masks(projected_points_dict, results, original_img_size=(900, 1600), mask_img_size=(384, 640)):
    """
    Filters projected LiDAR points to only keep those inside segmentation masks.

    Args:
        projected_points_dict: dict[int, np.ndarray] — each value is (M, 3) of [u, v, depth]
        results: list of YOLO result objects (from model inference)
        original_img_size: (H, W) of original images before YOLO resizing
        mask_img_size: (H, W) of mask shape from YOLO

    Returns:
        filtered_points_dict: dict[int, np.ndarray] — same keys, but only points inside masks
    """
    orig_h, orig_w = original_img_size
    mask_h, mask_w = mask_img_size

    scale_x = mask_w / orig_w
    scale_y = mask_h / orig_h

    filtered_points_dict = {}

    for frame_id, (points, result) in enumerate(zip(projected_points_dict.values(), results)):
        # Scale u, v to YOLO mask resolution
        u_scaled = (points[:, 0] * scale_x).astype(int)
        v_scaled = (points[:, 1] * scale_y).astype(int)

        # Clamp to avoid out-of-bounds
        u_scaled = np.clip(u_scaled, 0, mask_w - 1)
        v_scaled = np.clip(v_scaled, 0, mask_h - 1)

        # Create global mask from union of all instance masks
        if result.masks is None:
            print(f"No masks in frame {frame_id}, skipping.")
            filtered_points_dict[frame_id] = np.empty((0, 3))
            continue

        mask_tensor = result.masks.data  # shape: (num_objects, 384, 640)
        combined_mask = mask_tensor.any(dim=0).cpu().numpy()  # shape: (384, 640), bool

        # Check which points fall inside the mask
        inside_mask = combined_mask[v_scaled, u_scaled]  # boolean mask

        # Filter points
        filtered_points = points[inside_mask]
        filtered_points_dict[frame_id] = filtered_points

    return filtered_points_dict