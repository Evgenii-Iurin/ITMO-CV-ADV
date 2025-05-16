import numpy as np
from enum import Enum
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)

def get_affine_transformation_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
  """
  :param rotation: np.ndarray(3, 3)
  :param translation: np.ndarray(3, 1)
  :return: np.ndarray(4, 4)
  """
  affine_matrix = np.zeros((4, 4))
  affine_matrix[:3, :3] = rotation
  affine_matrix[:3, 3] = translation
  affine_matrix[3, 3] = 1
  return affine_matrix

def get_inverse_matrix(matrix: np.ndarray):
  """Return inverse matrix using numpy"""
  return np.linalg.inv(matrix)


class CAMERAS(Enum):
  FRONT = 'CAM_FRONT'
  FRONT_RIGHT = 'CAM_FRONT_RIGHT'
  FRONT_LEFT = 'CAM_FRONT_LEFT'
  BACK = 'CAM_BACK'
  BACK_RIGHT = 'CAM_BACK_RIGHT'
  BACK_LEFT = 'CAM_BACK_LEFT'

def get_transform_matrix_from_ego_pose(ego_pos_token: str):
  ego_pose_data = nusc.get('ego_pose', ego_pos_token)
  return get_affine_transformation_matrix(
      Quaternion(ego_pose_data["rotation"]).rotation_matrix,
      ego_pose_data["translation"]
      )

def get_transform_matrix_from_calibrated_sensor(calibrated_sensor_token: str):
  calibrated_sensor_data = nusc.get('calibrated_sensor', calibrated_sensor_token)
  return get_affine_transformation_matrix(
      Quaternion(calibrated_sensor_data["rotation"]).rotation_matrix,
      calibrated_sensor_data["translation"]
      )

def get_camera_intrinsic(calibrated_sensor_token: str):
  calibrated_sensor_data = nusc.get('calibrated_sensor', calibrated_sensor_token)
  return np.array(calibrated_sensor_data["camera_intrinsic"])

def transform_points_using_transformation_matrix(pointcloud: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray:
    """
    :param pointcloud: np.ndarray(N, 4)
        Pointcloud, where each sample represents (x, y, z, distance)

    :param transformation_matrix: np.ndarray(4, 4)
        Affine transformation matrix

    :return: np.ndarray(N, 4)
        Transformed pointcloud in same format
    """

    # Transform points to homogeneous coordinates
    lidar_points_homogeneous = np.hstack((
        pointcloud[:, :3],
        np.ones((pointcloud.shape[0], 1))
    ))  # shape: (N, 4)

    # Apply transformation
    transformed_homogeneous = transformation_matrix @ lidar_points_homogeneous.T  # shape: (4, N)
    transformed_homogeneous = transformed_homogeneous.T  # shape: (N, 4)

    # Convert from homogeneous to Euclidean coordinates
    transformed_points = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3].reshape(-1, 1)

    # Re-attach the distance (from original pointcloud)
    transformed_point_cloud = np.hstack((
        transformed_points,
        pointcloud[:, 3].reshape(-1, 1)
    ))

    return transformed_point_cloud

def filter_pointcloud_to_project_on_image(points_cam, image_height, image_width, camera_intrinsic):
    """
    Projects 3D points in the camera frame onto the 2D image, retaining depth.

    Args:
        points_cam: (N, 3) or (3, N) array of [x, y, z] points in the camera coordinate frame
        image_height: Image height in pixels
        image_width: Image width in pixels
        camera_intrinsic: (3, 3) camera intrinsics matrix

    Returns:
        result: (M, 3) array of [u, v, depth] where (u,v) are image coordinates and depth is the Z value
    """
    # Make sure shape is (N, 3)
    if points_cam.shape[0] == 3 and points_cam.shape[1] != 3:
        points_cam = points_cam.T

    if points_cam.shape[1] != 3:
        raise ValueError("Input points_cam must have shape (N, 3) or (3, N)")

    # Filter points with Z > 0 (in front of camera)
    z_mask = points_cam[:, 2] > 0
    points_cam = points_cam[z_mask]

    if points_cam.shape[0] == 0:
        print("No valid points after Z filter.")
        return np.empty((0, 3))

    # Depth = distance from camera (Z in camera frame)
    depths = points_cam[:, 2]  # Use Z as depth

    # Project 3D to 2D
    points_homogeneous = (camera_intrinsic @ points_cam.T).T  # (N, 3)
    points_2d = points_homogeneous[:, :2] / points_homogeneous[:, 2:3]

    u, v = points_2d[:, 0], points_2d[:, 1]

    # Filter points inside image bounds
    in_bounds = (
        (u >= 0) & (u < image_width) &
        (v >= 0) & (v < image_height)
    )

    valid_uv = points_2d[in_bounds]
    valid_depths = depths[in_bounds]

    print(f"Remaining: {valid_uv.shape[0]} valid points after filtering.")

    result = np.hstack([valid_uv, valid_depths[:, np.newaxis]])  # (M, 3)

    return result


