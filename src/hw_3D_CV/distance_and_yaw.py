from sklearn.decomposition import PCA
import numpy as np

def estimate_object_distances(results, filtered_points_dict, model, original_image_shape=(900, 1600)):
    """
    For each object detected by YOLO, estimate distance to ego using LiDAR points inside the bounding box.
    Also estimates yaw using PCA (optional).

    Returns:
        distances_per_frame: dict[frame_id] = list of dicts:
            { 'class': str, 'confidence': float, 'distance': float, 'yaw_deg': float or None }
    """
    distances_per_frame = {}

    for frame_id, result in enumerate(results):
        frame_points = filtered_points_dict.get(frame_id)
        if frame_points is None or frame_points.shape[0] == 0:
            distances_per_frame[frame_id] = []
            continue

        detections = []
        boxes = result.boxes

        if boxes is None or boxes.xyxy is None:
            distances_per_frame[frame_id] = []
            continue

        # Unpack box data
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for i, (box, cls_id, conf) in enumerate(zip(boxes_xyxy, class_ids, confidences)):
            x1, y1, x2, y2 = box
            class_name = model.names[cls_id]

            # Select lidar points inside the bounding box
            u, v = frame_points[:, 0], frame_points[:, 1]
            mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            box_points = frame_points[mask]

            if box_points.shape[0] == 0:
                continue

            # Compute depth statistics (Z-axis or full 3D)
            depths = box_points[:, 2]
            median_depth = np.median(depths)

            # Estimate yaw using PCA on 2D lidar points
            yaw_deg = None
            if box_points.shape[0] >= 5:  # Need at least 5 points for stable PCA
                try:
                    # Use 2D lidar projection points (u, v)
                    pca = PCA(n_components=2)
                    pca.fit(box_points[:, :2])  # (u, v)

                    angle_rad = np.arctan2(pca.components_[0][1], pca.components_[0][0])
                    yaw_deg = np.round(np.degrees(angle_rad), 1)
                except Exception:
                    yaw_deg = None

            detections.append({
                'class': class_name,
                'confidence': float(conf),
                'distance': float(median_depth),
                'yaw_deg': float(yaw_deg) if yaw_deg is not None else None,
            })

        distances_per_frame[frame_id] = detections

    return distances_per_frame