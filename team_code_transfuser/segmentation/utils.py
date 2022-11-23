import torch
import numpy as np
from matplotlib import pyplot as plt
import cv2

import config

def visualize_semantic_processed(sem, labels=config.labels):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for i,label in enumerate(labels):
        canvas[sem==i+1] = config.SEM_COLORS[label]

    return canvas


def log_train_info(seg_info, counter): 
    loss = seg_info.pop('loss')
    rgb = seg_info.pop('rgb')
    sem = seg_info.pop('sem')
    pred_sem = seg_info.pop('pred_sem')

    f, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(32, 10))
    f.text(.01, .99, f"loss: {loss}", size = 20, ha='left', va='top')

    ax1.imshow(rgb)
    ax2.imshow(visualize_semantic_processed(sem))
    ax3.imshow(visualize_semantic_processed(pred_sem))
    #plt.show()
    plt.savefig(f"./logs/log-{counter}.png")
    del rgb, sem, pred_sem

    plt.close('all')

def log_eval_info(seg_info): 
    rgb = seg_info.pop('rgb')
    pred_sem = seg_info.pop('pred_sem')

    f, [ax1, ax2] = plt.subplots(1,2,figsize=(12,4))

    ax1.imshow(rgb)
    ax2.imshow(visualize_semantic_processed(pred_sem))
    plt.show()

    plt.close('all')

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()