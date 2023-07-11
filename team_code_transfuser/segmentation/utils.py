import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

# from config import labels, SEM_COLORS
import config


def load_pretrained(model, pretrained_path):
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} 
    # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print(msg)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

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
    plt.savefig(f"{config.SAVE_DIR}/carla/logs/log-{counter}.png")
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

def smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()
    return smooth_loss * config.disparity_smoothness

def weighted_bce(bd_pre, target):
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.reshape(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss

def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

def get_confusion_matrix(label, pred):
    """
        Calcute the confusion matrix by given label and pred
    """
    flat_label = label.flatten()
    flat_pred = pred.flatten()

    cm = confusion_matrix(flat_label, flat_pred)
    return cm

def calculate_IoU_from_cm(confusion_mat): 
    pos = confusion_mat.sum(1)
    res = confusion_mat.sum(0)
    tp = np.diag(confusion_mat)
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    mean_IoU = IoU_array.mean()
    return mean_IoU


def visualize_cm(confusion_mat, labels): 
    class_names = {
        1 : "building",
        4 : "pedestrian",
        5 : "pole",
        6 : "line",
        7 : "road",
        8 : "sidewalk",
        9 : "vegetation",
        10: "vehicles",
        18 : "traffic-light"
    }
    classes = ["empty"]
    for label in labels: 
        if label in class_names: 
            classes.append(class_names[label])
        else: 
            classes.append("unknown class")
    df_cm = pd.DataFrame(confusion_mat/np.sum(confusion_mat) * 10, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()

def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-x_meters_max, x_meters_max, 2*x_meters_max*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, y_meters_max*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.3]
    above = lidar[lidar[...,2]>-2.3]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    features = np.rot90(features, -1, axes=(1,2)).copy()
    return features

def lidar_to_bev(lidar_data): 
    lidar_transformed = deepcopy(lidar_data) 
    lidar_transformed[:, 1] *= -1  # invert
    lidar_transformed = lidar_to_histogram_features(lidar_transformed)
    lidar_transformed_degrees = [lidar_transformed]
    lidar_bev = np.concatenate(lidar_transformed_degrees[::-1], axis=0)
    lidar_bev = np.transpose(lidar_bev, (1, 2, 0))
    # lidar_transformed = torch.from_numpy(lidar_transformed).unsqueeze(0)
    # lidar_transformed_degrees = [lidar_transformed.to('cuda', dtype=torch.float32)]
    # lidar_bev = torch.cat(lidar_transformed_degrees[::-1], dim=1)
    
    return lidar_bev

def get_transform_matrix(location, rotation): 
    """
        Code is taken from Carla source code: 
        https://github.com/carla-simulator/carla/blob/d23f3dc1340e47265eeea2b1b33b2d3a2d6d4f42/LibCarla/source/carla/geom/Transform.h
    """
    roll = rotation[0]
    cr = math.cos(math.radians(roll))
    sr = math.sin(math.radians(roll))

    pitch = rotation[1]
    cp = math.cos(math.radians(pitch))
    sp = math.sin(math.radians(pitch))

    yaw = rotation[2]
    cy = math.cos(math.radians(yaw))
    sy = math.sin(math.radians(yaw))

    transform = [
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location[0]], 
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location[1]], 
        [sp, -cp * sr, cp * cr, location[2]],
        [0.0, 0.0, 0.0, 1.0]
    ]
    transform = np.array(transform) 
    return transform

def get_inverse_transform_matrix(location, rotation): 
    """
        Code is taken from Carla source code: 
        https://github.com/carla-simulator/carla/blob/d23f3dc1340e47265eeea2b1b33b2d3a2d6d4f42/LibCarla/source/carla/geom/Transform.h
    """
    roll = rotation[0]
    cr = math.cos(math.radians(roll))
    sr = math.sin(math.radians(roll))

    pitch = rotation[1]
    cp = math.cos(math.radians(pitch))
    sp = math.sin(math.radians(pitch))

    yaw = rotation[2]
    cy = math.cos(math.radians(yaw))
    sy = math.sin(math.radians(yaw))

    a = [
        -location[0] * (cp * cy) + -location[1] * (cp * sy) + -location[2] * sp, 
        -location[0] * (cy * sp * sr - sy * cr) + -location[1] * (sy * sp * sr + cy * cr) + -location[2] * (-cp * sr), 
        -location[0] * (-cy * sp * cr - sy * sr) + -location[1] * (-sy * sp * cr + cy * sr) + -location[2] * (cp * cr) 
    ]
    transform = [
        [cp * cy, cp * sy, sp, a[0]], 
        [cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, -cp * sr, a[1]], 
        [-cy * sp * cr - sy * sr, -sy * sp * cr + cy * sr, cp * cr, a[2]], 
        [0.0, 0.0, 0.0, 1.0]
    ]
    transform = np.array(transform)
    return transform