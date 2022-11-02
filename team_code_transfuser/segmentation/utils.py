import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2

#labels=[4,6,7,8,10]
labels = [*range(20)]

SEM_COLORS = {
    0 : (0, 0, 0),
    1 : (70, 70, 70),
    2 : (100, 40, 40),
    3 : (55, 90, 80),
    4 : (220, 20, 60), #pedestrian
    5 : (153, 153, 153), #pole
    6 : (157, 234, 50), #road line
    7 : (128, 64, 128), #road
    8 : (244, 35, 232),  #side walk
    9 : (107, 142, 35),
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
def visualize_semantic_processed(sem, labels=labels):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for i,label in enumerate(labels):
        canvas[sem==i+1] = SEM_COLORS[label]
    cv2.imshow("semantic", canvas)
    cv2.waitKey(0)

    return canvas


def log_train_info(seg_info, counter): 
    rgb = seg_info.pop('rgb')
    sem = seg_info.pop('sem')
    pred_sem = seg_info.pop('pred_sem')

    f, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(32, 10))

    ax1.imshow(rgb)
    ax2.imshow(visualize_semantic_processed(sem))
    ax3.imshow(visualize_semantic_processed(pred_sem))
    #plt.show()
    plt.savefig(f"./logs/log-{counter}.png")

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
