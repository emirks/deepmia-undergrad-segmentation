import glob
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

from PIL import Image

def filter_sem(sem, labels=[4,6,7,8,10]):
    resem = np.zeros_like(sem)
    for i, label in enumerate(labels):
        resem[sem==label] = i+1
    
    return resem

class SegmentationDataset(Dataset): 
    def __init__(self):
        super(SegmentationDataset, self).__init__()
        self.path = "/home/transfuser/autonomous_car/transfuser-erkam/semantic-segmentation-dataset"

        self.rgb_path = f"{self.path}/rgb/*.jpg"
        self.semantic_path = f"{self.path}/semantic/*.jpg"

        self.rgb_image_paths = []
        self.semantic_image_paths = []
        for rgb_file in glob.glob(self.rgb_path):
            self.rgb_image_paths.append(rgb_file)
        for semantic_file in glob.glob(self.semantic_path): 
            self.semantic_image_paths.append(semantic_file)
        # to ensure that rgb paths and semantic paths are in the same order
        self.rgb_image_paths.sort()
        self.semantic_image_paths.sort()

        assert(len(self.rgb_image_paths) == len(self.semantic_image_paths))
        self.size = len(self.rgb_image_paths)

    def __len__(self): 
        return self.size

    def __getitem__(self, index):
        if index > self.size: 
            raise Exception("Index of the required dataset element is higher than size of the dataset")
        
        rgb_image_name = self.rgb_image_paths[index].split("/")[-1]
        sem_image_name = self.semantic_image_paths[index].split("/")[-1]
        if(rgb_image_name != sem_image_name): 
            raise Exception(f"Name of rgb and semantic file are not same for index {index}")

        rgb_image = Image.open(self.rgb_image_paths[index])
        sem_image = Image.open(self.semantic_image_paths[index])
        sem_image = filter_sem(sem_image)

        return rgb_image, sem_image

if __name__ == '__main__':
    dataset = SegmentationDataset()

    import tqdm
    for t in tqdm.tqdm(range(200)):
        dataset[t]