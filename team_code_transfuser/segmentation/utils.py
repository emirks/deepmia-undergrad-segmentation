import torch
import numpy as np
from matplotlib import pyplot as plt

SEM_COLORS = {
    4 : (220, 20, 60),
    5 : (153, 153, 153),
    6 : (157, 234, 50),
    7 : (128, 64, 128),
    8 : (244, 35, 232),
    10: (0, 0, 142),
    18: (220, 220, 0),
}

def visualize_semantic_processed(sem, labels=[4,6,7,8,10]):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for i,label in enumerate(labels):
        canvas[sem==i+1] = SEM_COLORS[label]

    return canvas


def log_train_info(seg_info): 
    rgb = seg_info.pop('rgb')
    sem = seg_info.pop('sem')
    pred_sem = seg_info.pop('pred_sem')

    f, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(12,4))

    ax1.imshow(rgb)
    ax2.imshow(visualize_semantic_processed(sem))
    ax3.imshow(visualize_semantic_processed(pred_sem))
    plt.show()

    plt.close('all')

def log_eval_info(seg_info): 
    rgb = seg_info.pop('rgb')
    pred_sem = seg_info.pop('pred_sem')

    f, [ax1, ax2] = plt.subplots(1,2,figsize=(12,4))

    ax1.imshow(rgb)
    ax2.imshow(visualize_semantic_processed(pred_sem))
    plt.show()

    plt.close('all')


def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap

class Colorize:

    def __init__(self, n=22):
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
