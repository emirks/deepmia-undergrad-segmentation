import numpy as np
from os.path import join
import glob
import cv2
import json

from torch.utils.data import Dataset

import config
from utils import lidar_to_bev

from segment_anything.utils.transforms import ResizeLongestSide

def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image

def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('/')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + \
                  'label_' + \
                  file_name_semantic_label[2] + '_' + \
                  file_name_semantic_label[3] + '.png'
    
    return file_name_semantic_label

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
    
def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)
    
    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']  
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                        np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
                (1. - pixel_opacity) * \
                np.multiply(image[pixel_rows, pixel_cols, :], \
                colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)

class SegmentationDataset(Dataset): 
    def __init__(self, mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]):
        super(SegmentationDataset, self).__init__()

        file_path = f"{config.SAVE_DIR}/kitti/a2d2/"

        self.root_path = f'{file_path}/camera_lidar_semantic/'
        # get the list of files in lidar directory
        self.file_names = sorted(glob.glob(join(self.root_path, '*/lidar/cam_front_center/*.npz')))
        self.size = len(self.file_names)

        self.mean = mean
        self.std = std

        with open (f'{file_path}/cams_lidars.json', 'r') as f:
            self.dataset_config = json.load(f)
            
    def __len__(self): 
        return self.size
    
    def undistort_image(self, image, cam_name):
        if cam_name in ['front_left', 'front_center', \
                        'front_right', 'side_left', \
                        'side_right', 'rear_center']:
            # get parameters from config file
            intr_mat_undist = \
                    np.asarray(self.dataset_config['cameras'][cam_name]['CamMatrix'])
            intr_mat_dist = \
                    np.asarray(self.dataset_config['cameras'][cam_name]['CamMatrixOriginal'])
            dist_parms = \
                    np.asarray(self.dataset_config['cameras'][cam_name]['Distortion'])
            lens = self.dataset_config['cameras'][cam_name]['Lens']
            
            if (lens == 'Fisheye'):
                return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                        D=dist_parms, Knew=intr_mat_undist)
            elif (lens == 'Telecam'):
                return cv2.undistort(image, intr_mat_dist, \
                        distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
            else:
                return image
        else:
            return image
    
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
    
    def resize_for_sam(self, mat): 
        sam_image_size = 640
        sam_transform = ResizeLongestSide(sam_image_size)
        mat = sam_transform.apply_image(mat)
        return mat
    
    def __getitem__(self, index):
        if index > self.size: 
            raise Exception("Index of the required dataset element is higher than size of the dataset")

        # select the lidar point cloud
        file_name_lidar = self.file_names[index]

        # read the lidar data
        lidar_front_center = np.load(file_name_lidar)

        seq_name = file_name_lidar.split('/')[-4]
        file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
        file_name_image = join(self.root_path, seq_name, 'camera/cam_front_center/', file_name_image)
        file_name_semantic_label = extract_semantic_file_name_from_image_file_name(file_name_image)
        file_name_semantic_label = join(self.root_path, seq_name, 'label/cam_front_center/', file_name_semantic_label)

        image_front_center = cv2.cvtColor(cv2.imread(file_name_image), cv2.COLOR_BGR2RGB)
        undist_image_front_center = self.undistort_image(image_front_center, 'front_center')
        semantic_image_front_center = cv2.cvtColor(cv2.imread(file_name_semantic_label), cv2.COLOR_BGR2RGB)
        semantic_image_front_center_undistorted = self.undistort_image(semantic_image_front_center, 'front_center')

        image = map_lidar_points_onto_image(undist_image_front_center, lidar_front_center)

        lidar = lidar_front_center["points"]
        rgb = image_front_center
        rgb_with_lidar = image
        lidar_bev = lidar_to_bev(lidar)
        semantic = semantic_image_front_center_undistorted

        rgb, semantic, edge = self.gen_sample(rgb, semantic)
        rgb = self.resize_for_sam(rgb)
        semantic = self.resize_for_sam(semantic)
        rgb_with_lidar = self.resize_for_sam(rgb_with_lidar)
        # cv2.imshow("rgb", rgb)
        # cv2.imshow("lidar-projected", rgb_with_lidar[:, :, :])
        # cv2.imshow("semantic", semantic)
        # cv2.imshow("lidar-bev-1", lidar_bev[:, :, 0])
        # cv2.imshow("lidar-bev-2", lidar_bev[:, :, 1])
        # cv2.waitKey(0)

        return rgb, semantic, edge, lidar_bev, rgb_with_lidar

if __name__ == '__main__':
    dataset = SegmentationDataset("deneme")
    item = dataset[4]