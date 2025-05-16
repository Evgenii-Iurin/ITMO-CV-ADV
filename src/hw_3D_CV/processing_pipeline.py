import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from utils import read_image, show_image, read_lidar, show_lidar_pointcloud
from pyquaternion import Quaternion
from projection import get_transform_matrix_from_ego_pose, get_transform_matrix_from_calibrated_sensor, transform_points_using_transformation_matrix, CAMERAS, get_inverse_matrix, get_camera_intrinsic, filter_pointcloud_to_project_on_image
from nuscenes.nuscenes import NuScenes
import cv2

nusc = NuScenes(version='v1.0-mini', dataroot='data/sets/nuscenes', verbose=True)
my_scene = nusc.scene[0]
my_scene_samples_tokens = [sample['token'] for sample in nusc.sample if sample['scene_token'] == my_scene['token']]
sample_token = my_scene_samples_tokens[10]
sample_info = nusc.get('sample', sample_token)
camera_front_data = nusc.get('sample_data', sample_info['data']['CAM_FRONT'])
nuscenes_root_dir = 'data/sets/nuscenes/'
camera_calibration = nusc.get('calibrated_sensor', camera_front_data['calibrated_sensor_token'])

lidar_token = sample_info['data']['LIDAR_TOP']
lidar_top_data = nusc.get('sample_data', lidar_token)
ego_pose_token = lidar_top_data['ego_pose_token']
vehicle_position_lidar_data =  nusc.get('ego_pose', ego_pose_token)["translation"]

camera_token = sample_info['data']['CAM_FRONT']
camera_top_data = nusc.get('sample_data', camera_token)
ego_pose_token = camera_top_data['ego_pose_token']
vehicle_position_camera_data =  nusc.get('ego_pose', ego_pose_token)["translation"]

lidar_top_pointcloud = read_lidar(nuscenes_root_dir + lidar_top_data['filename'])
lidar_calibration = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])


projected_points_dict = {}
camera_files_dict = {}
lidar_files_dict = {}
point_cloud_dict = {}

for num, sample_token in enumerate(my_scene_samples_tokens):
  sample_info = nusc.get('sample', sample_token)

  lidar_token = sample_info['data']['LIDAR_TOP']
  lidar_top_data = nusc.get('sample_data', lidar_token)
  pointcloud_in_lidar_coord = read_lidar(nuscenes_root_dir + lidar_top_data['filename'])

  # Get vehicle's pose at the lidar timestamp
  ego_vehicle_to_world_transform_matrix = get_transform_matrix_from_ego_pose(
      lidar_top_data["ego_pose_token"]
      )

  # Get lidar's position at the lidar timestamp
  lidar_to_ego_vehicle_transform_matrix = get_transform_matrix_from_calibrated_sensor(
      lidar_top_data["calibrated_sensor_token"]
      )


  # Lidar coord --> ego vehicle
  pointcloud_in_ego_vehicle = transform_points_using_transformation_matrix(
      pointcloud_in_lidar_coord,
      lidar_to_ego_vehicle_transform_matrix
      )

  # Ego vehicle --> world coord
  pointcloud_in_world = transform_points_using_transformation_matrix(
      pointcloud_in_ego_vehicle,
      ego_vehicle_to_world_transform_matrix
      )


  # ограничимся только фронтальной камерой
  # for camera_name in [CAMERAS.FRONT]:
  camera_name = CAMERAS.FRONT
  camera_data = nusc.get('sample_data', sample_info['data'][camera_name.value])
  # World coord --> Ego vehicle (camera)
  ego_vehicle_to_world_transform_matrix = get_transform_matrix_from_ego_pose(
      camera_data["ego_pose_token"]
      )
  pointcloud_in_ego_vehicle = transform_points_using_transformation_matrix(
      pointcloud_in_world,
      get_inverse_matrix(ego_vehicle_to_world_transform_matrix)
      )
  # Ego vehicle (camera) --> Camera coord
  camera_to_ego_vehicle_transform_matrix = get_transform_matrix_from_calibrated_sensor(
      camera_data["calibrated_sensor_token"]
      )
  camera_intristics = get_camera_intrinsic(camera_data["calibrated_sensor_token"])
  pointcloud_in_camera_coord = transform_points_using_transformation_matrix(
      pointcloud_in_ego_vehicle,
      get_inverse_matrix(camera_to_ego_vehicle_transform_matrix)
      )
  projected_points = filter_pointcloud_to_project_on_image(
      pointcloud_in_camera_coord[:, :3],
      camera_data["height"],
      camera_data["width"],
      camera_intristics
  )

  camera_files_dict[num] = nuscenes_root_dir + camera_data['filename']
  lidar_files_dict[num] = nuscenes_root_dir + lidar_top_data['filename']
  projected_points_dict[num] = projected_points
  point_cloud_dict[num] = pointcloud_in_camera_coord