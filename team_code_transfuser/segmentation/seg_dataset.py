import glob
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

import random, math
import cv2
import h5py

import imgaug as ia
from imgaug import augmenters as iaa

import config

from utils import lidar_to_bev, get_transform_matrix, get_inverse_transform_matrix

from matplotlib import cm
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def augment(prob=0.2):
    
    augmenter = iaa.Sequential([
        iaa.Sometimes(prob, iaa.GaussianBlur((0, 0.5))),
        iaa.Sometimes(prob, iaa.AdditiveGaussianNoise(loc=0, scale=(0., 0.05*255), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Dropout((0.01, 0.1), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Multiply((1/1.2, 1.2), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.LinearContrast((1/1.2, 1.2), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Grayscale((0.0, 0.5))),
        # iaa.Sometimes(prob, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
    ], random_order=True)
    
    
    return augmenter

def filter_sem(sem, labels=config.labels):
    resem = np.zeros_like(sem)
    for i, label in enumerate(labels):
        resem[sem==label] = i+1
    
    return resem

class SegmentationDataset(Dataset): 
    def __init__(self, hdf5_file_name, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]):
        super(SegmentationDataset, self).__init__()
        self.size = 0

        hdf5_file_path = f"{config.SAVE_DIR}/datasets/{hdf5_file_name}.hdf5"
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.file_timestamps = self.hdf5_file['timestamps']['timestamps']

        self.size = len(self.file_timestamps)

        self.mean = mean
        self.std = std
        self.augmenter = augment(0.2)

    def __len__(self): 
        return self.size
    
    def lidar_projection_to_camera(self, lidar_data, rgb, lidar_transform, camera_transform): 
        intensity = np.array(lidar_data[:, 3])
        local_lidar_points = np.array(lidar_data[:, :3]).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = np.r_[local_lidar_points, [np.ones(local_lidar_points.shape[1])]]

        # This (4, 4) matrix transforms the points from lidar space to world space.
        lidar_2_world = get_transform_matrix(lidar_transform[:3], lidar_transform[3:])

        # Transform the points from lidar space to world space.
        world_points = np.dot(lidar_2_world, local_lidar_points)

        # This (4, 4) matrix transforms the points from world to sensor coordinates.
        world_2_camera = get_inverse_transform_matrix(camera_transform[:3], camera_transform[3:])

        # Transform the points from world space to camera space.
        sensor_points = np.dot(world_2_camera, world_points)

        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y

        # This can be achieved by multiplying by the following matrix:
        # [[ 0,  1,  0 ],
        #  [ 0,  0, -1 ],
        #  [ 1,  0,  0 ]]

        # Or, in this case, is the same as swapping:
        # (x, y ,z) -> (y, -z, x)
        point_in_camera_coords = np.array([
            sensor_points[1],
            sensor_points[2] * -1,
            sensor_points[0]])

        # Finally we can use our K matrix to do the actual 3D -> 2D.
        points_2d = np.dot(config.K, point_in_camera_coords)

        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]])

        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        intensity = intensity.T
        points_in_canvas_mask = \
            (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < config.camera_width) & \
            (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < config.camera_height) & \
            (points_2d[:, 2] > 0.0)
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]

        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(np.int)
        v_coord = points_2d[:, 1].astype(np.int)

        # # Since at the time of the creation of this script, the intensity function
        # # is returning high values, these are adjusted to be nicely visualized.
        intensity = 4 * intensity - 3
        # color_map = np.array([
        #     np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
        #     np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
        #     np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0]).astype(np.int).T
        color_map = np.array([
            np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0
        ]).astype(np.int).T

        # Draw the 2d points on the image as a single pixel using numpy.
        lidar_projection = np.zeros((rgb.shape[0], rgb.shape[1], 1))
        lidar_projection[v_coord, u_coord] = color_map
        rgb_with_lidar = np.concatenate([rgb, lidar_projection], axis=2)
        return rgb_with_lidar

    def rgb_transform(self, rgb): 
        rgb = rgb / 255.0
        rgb -= self.mean
        rgb /= self.std
        return rgb
    
    def gen_sample(self, image, label,
                   edge_pad=True, edge_size=4):        
        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        y_k_size, x_k_size = 6, 6
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0

        return image, label, edge

    def __getitem__(self, index):
        if index > self.size: 
            raise Exception("Index of the required dataset element is higher than size of the dataset")
        
        # height, width = rgb.shape[:2]
        # rgb = rgb[height//2 - config.img_resolution[0]//2:height//2 + config.img_resolution[0]//2, 
        #     width//2 - config.img_resolution[1]//2:width//2 + config.img_resolution[1]//2]
        # semantic = semantic[height//2 - config.img_resolution[0]//2:height//2 + config.img_resolution[0]//2, 
        #     width//2 - config.img_resolution[1]//2:width//2 + config.img_resolution[1]//2]

        time = str(self.file_timestamps[index])
        lidar = np.array(self.hdf5_file['lidar'][time])
        lidar_transform = np.array(self.hdf5_file["lidar-transform"][time])
        rgb = []
        semantic = []
        rgb_with_lidar = []
        for camera_id in range(3):
            rgb_cam_name = f"rgb_{camera_id}"
            semantic_cam_name = f"semantic_{camera_id}"
            rgb_transform = np.array(self.hdf5_file[f"{rgb_cam_name}-transform"][time])
            rgb_i = np.array(self.hdf5_file[rgb_cam_name][time])
            rgb_with_lidar_i = self.lidar_projection_to_camera(lidar, rgb_i, lidar_transform, rgb_transform)
            semantic_i = np.array(self.hdf5_file[semantic_cam_name][time])
            rgb_i = rgb_i[config.img_height:config.img_height*2, config.img_width:config.img_width*2]
            rgb_with_lidar_i = rgb_with_lidar_i[config.img_height:config.img_height*2, config.img_width:config.img_width*2]
            semantic_i = semantic_i[config.img_height:config.img_height*2, config.img_width:config.img_width*2]
            rgb.append(rgb_i)
            semantic.append(semantic_i)
            rgb_with_lidar.append(rgb_with_lidar_i)
        rgb = np.concatenate(rgb, axis=1)
        semantic = np.concatenate(semantic, axis=1)
        rgb_with_lidar = np.concatenate(rgb_with_lidar, axis=1)

        lidar_bev = lidar_to_bev(lidar)

        # rgb = self.augmenter(images=rgb[...,::-1][None])[0]
        semantic = filter_sem(semantic)
        rgb, semantic, edge = self.gen_sample(rgb, semantic)

        return rgb, semantic, edge, lidar_bev, rgb_with_lidar

if __name__ == '__main__':
    dataset = SegmentationDataset("deneme")
    item = dataset[2]
    # import tqdm
    # for t in tqdm.tqdm(range(len(dataset))):
    #     item = dataset[t]
    #     del item