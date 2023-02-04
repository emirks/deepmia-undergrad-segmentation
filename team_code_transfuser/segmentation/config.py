import torch
import numpy as np

labels=[4,6,7,8,10,18]

SEM_COLORS = {
    0 : (0, 0, 0),
    1 : (70, 70, 70), #building
    2 : (100, 40, 40),
    3 : (55, 90, 80),
    4 : (220, 20, 60), #pedestrian
    5 : (153, 153, 153), #pole
    6 : (157, 234, 50), #road line
    7 : (128, 64, 128), #road
    8 : (244, 35, 232),  #sidewalk
    9 : (107, 142, 35), #vegetation
    10: (0, 0, 142), #vehicles
    11 : (102, 102, 156),
    12 : (220, 220, 0),
    13 : (70, 130, 180),
    14 : (81, 0, 81),
    15 : (150, 100, 100),
    16 : (230, 150, 140),
    17 : (180, 165, 180),
    18 : (250, 170, 30), #traffic-light
    19 : (110, 190, 160),
    20 : (170, 120, 50),
    21 : (45, 60, 150),
    22 : (145, 170, 100)
}
weights = {
    1: 0.866, 
    4: 0.9969, 
    5: 0.8786, 
    6: 0.9754,
    7: 0.9754, 
    8: 0.918, 
    9: 0.7786,
    10: 0.9843, 
    18: 0.9037, 
}

class_weights = torch.FloatTensor([0.8373] + [weights[i] for i in labels]).cuda()

SAVE_DIR = "/media/transfuser/1ee3aeb6-a6d1-40b8-bd75-87f69311f33b1/segmentation/carla"

scale = 1 # image pre-processing
img_resolution = (160, 960) # image pre-processing in H, W
img_width = 320 # important this should be consistent with scale, e.g. scale = 1, img_width 320, scale=2, image_width 640
img_height = 160

lidar_channels = 64.0
lidar_range = 100.0
lidar_points_per_second = 100000
lidar_lower_fov = -25.0
lidar_upper_fov = 30.0

lidar_resolution_width  = 256 # Width of the LiDAR grid that the point cloud is voxelized into.
lidar_resolution_height = 256 # Height of the LiDAR grid that the point cloud is voxelized into.
pixels_per_meter = 8.0 # How many pixels make up 1 meter. 1 / pixels_per_meter = size of pixel in meters
lidar_pos = [1.3,0.0,2.5] # x, y, z mounting position of the LiDAR
lidar_rot = [0.0, 0.0, -90.0] # Roll Pitch Yaw of LiDAR in degree


camera_pos = [1.3, 0.0, 2.3] #x, y, z mounting position of the camera
camera_width = 960 # Camera width in pixel
camera_height = 480 # Camera height in pixel
camera_fov = 120 #Camera FOV in degree
camera_rot_0 = [0.0, 0.0, 0.0] # Roll Pitch Yaw of camera 0 in degree
camera_rot_1 = [0.0, 0.0, -60.0] # Roll Pitch Yaw of camera 1 in degree
camera_rot_2 = [0.0, 0.0, 60.0] # Roll Pitch Yaw of camera 2 in degree
camera_rots = [camera_rot_1, camera_rot_0, camera_rot_2]

total_camera_fov = camera_fov
focal = camera_width / (2.0 * np.tan(total_camera_fov * np.pi / 360.0))

K = np.identity(3)
K[0, 0] = K[1, 1] = focal
K[0, 2] = camera_width / 2.0
K[1, 2] = camera_height / 2.0


towns = ["town-1", "town-2", "town-3", "town-4", "town-5", "town-7"]


disparity_smoothness = 1e-1

split_cameras = True
base_lr = 0.01
optim_momentum = 0.9
optim_wd = 0.0001