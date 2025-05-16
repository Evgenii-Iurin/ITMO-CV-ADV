from sklearn.decomposition import PCA
import numpy as np

def compute_3d_bounding_boxes(results, filtered_points_dict, point_cloud_dict, camera_intrinsics):
    """
    Estimates 3D bounding boxes for each object using segmented and projected LiDAR points.

    Args:
        results: YOLO results list (masks + bboxes + class ids)
        filtered_points_dict: {frame_id: (N, 3)} with [u, v, depth]
        point_cloud_dict: {frame_id: (M, 4)} with [x, y, z, _] (in camera coords)
        camera_intrinsics: 3x3 numpy array

    Returns:
        dict[frame_id] = list of objects with fields:
            class, confidence, box_3d (x_min, x_max, y_min, y_max, z_min, z_max), yaw
    """
    inverse_intrinsics = np.linalg.inv(camera_intrinsics)
    boxes_3d_per_frame = {}

    for frame_id, result in enumerate(results):
        proj_points = filtered_points_dict.get(frame_id)
        if proj_points is None or proj_points.shape[0] == 0:
            boxes_3d_per_frame[frame_id] = []
            continue

        full_cloud = point_cloud_dict[frame_id][:, :3]  # x, y, z

        # Reproject u, v, depth back to 3D points
        u, v, d = proj_points[:, 0], proj_points[:, 1], proj_points[:, 2]
        pixels = np.stack([u, v, np.ones_like(u)], axis=1).T  # shape: (3, N)
        rays = inverse_intrinsics @ pixels
        pts_3d = rays.T * d[:, None]  # shape: (N, 3)

        detections = []

        # Process per object
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            boxes_3d_per_frame[frame_id] = []
            continue

        for i, box in enumerate(boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = box
            cls_id = int(boxes.cls[i].cpu().item())
            conf = float(boxes.conf[i].cpu().item())
            class_name = result.names[cls_id] if hasattr(result, "names") else str(cls_id)

            # Select 3D points inside the 2D box
            mask = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
            object_pts = pts_3d[mask]

            if object_pts.shape[0] < 5:
                continue

            # Axis-aligned 3D bounding box
            x_min, y_min, z_min = object_pts.min(axis=0)
            x_max, y_max, z_max = object_pts.max(axis=0)

            # Yaw estimation (PCA on x, z)
            pca = PCA(n_components=2)
            pca.fit(object_pts[:, [0, 2]])
            angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
            yaw_deg = np.degrees(angle_rad)

            detections.append({
                'box_3d': [x_min, x_max, y_min, y_max, z_min, z_max],
                'yaw_deg': yaw_deg,
            })

        boxes_3d_per_frame[frame_id] = detections

    return boxes_3d_per_frame

def project_3d_to_2d(points_3d, intrinsics):
    """ Project Nx3 3D points to 2D image space using intrinsics """
    points_3d = points_3d.T  # (3, N)
    points_2d = intrinsics @ points_3d  # (3, N)
    points_2d /= points_2d[2, :]  # Normalize by z
    return points_2d[:2].T  # Return (N, 2)