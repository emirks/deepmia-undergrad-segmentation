import glob
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

import cv2
import h5py

import imgaug as ia
from imgaug import augmenters as iaa

import config

def augment(prob=0.2):
    
    augmenter = iaa.Sequential([
        iaa.Sometimes(prob, iaa.GaussianBlur((0, 0.5))),
        iaa.Sometimes(prob, iaa.AdditiveGaussianNoise(loc=0, scale=(0., 0.05*255), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Dropout((0.01, 0.1), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Multiply((1/1.2, 1.2), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.LinearContrast((1/1.2, 1.2), per_channel=0.5)),
        iaa.Sometimes(prob, iaa.Grayscale((0.0, 0.5))),
        iaa.Sometimes(prob, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
    ], random_order=True)
    
    
    return augmenter

def filter_sem(sem, labels=config.labels):
    resem = np.zeros_like(sem)
    for i, label in enumerate(labels):
        resem[sem==label] = i+1
    
    return resem

class SegmentationDataset(Dataset): 
    def __init__(self, hdf5_file_name, dataset_mode="train"):
        super(SegmentationDataset, self).__init__()
        self.size = 0
        self.path = "/home/transfuser/autonomous_car/transfuser-erkam/semantic-segmentation-dataset"

        hdf5_file_path = f"{self.path}/{hdf5_file_name}.hdf5"
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        if dataset_mode == "test": 
            # If dataset is for testing, then only take 50 images. 
            self.file_timestamps = self.hdf5_file['timestamps']['timestamps'][:50]
        else: 
            self.file_timestamps = self.hdf5_file['timestamps']['timestamps']

        self.size = len(self.file_timestamps)

        self.augmenter = augment(0.2)


    def __len__(self): 
        return self.size

    def __getitem__(self, index):
        if index > self.size: 
            raise Exception("Index of the required dataset element is higher than size of the dataset")
        
        time = self.file_timestamps[index]
        rgb = []
        semantic = []
        for camera_id in range(len(config.camera_rots)):
            rgb_cam_name = f"rgb_{camera_id}"
            semantic_cam_name = f"semantic_{camera_id}"
            rgb_pos = np.array(self.hdf5_file[rgb_cam_name][str(time)])
            rgb_pos = rgb_pos[config.img_width:config.img_width*2, config.img_height:config.img_height*2]
            semantic_pos = np.array(self.hdf5_file[semantic_cam_name][str(time)])
            semantic_pos = semantic_pos[config.img_width:config.img_width*2, config.img_height:config.img_height*2]
            rgb.append(rgb_pos)
            semantic.append(semantic_pos)
        rgb = np.concatenate(rgb, axis=1)
        semantic = np.concatenate(semantic, axis=1)
        height, width = rgb.shape[:2]
        rgb = rgb[height//2 - config.img_resolution[0]//2:height//2 + config.img_resolution[0]//2, 
            width//2 - config.img_resolution[1]//2:width//2 + config.img_resolution[1]//2]
        semantic = semantic[height//2 - config.img_resolution[0]//2:height//2 + config.img_resolution[0]//2, 
            width//2 - config.img_resolution[1]//2:width//2 + config.img_resolution[1]//2]

        #rgb_image = self.augmenter(images=rgb_image[...,::-1][None])[0]

        semantic = filter_sem(semantic)

        return rgb, semantic

if __name__ == '__main__':
    dataset = SegmentationDataset("town-1")

    import tqdm
    for t in tqdm.tqdm(range(len(dataset))):
        dataset[t]