# Written by Jiacong Xu (jiacong.xu@tamu.edu)
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

from .model_backbone import PIDNet
from .model_utils import SegDecoder

class SegmentationModel(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()

        self.encoder = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=True)
        self.decoder = SegDecoder(512, num_classes)    

    def forward(self, rgb, lidar, fused):
        image_features = self.encoder(rgb, lidar, fused) 
        # print(image_features.shape)
        out = self.decoder(image_features)
        # print(out.shape)
        return out
    
    @staticmethod
    def initialize(num_classes): 
        return SegmentationModel(num_classes)

    @staticmethod
    def pass_from_model(model, batch, sem_loss, smooth_loss, bd_loss, device = "cuda"):
        rgb, sem, edge, lidar_bev, fused = batch
        rgb = rgb.float().permute(0,3,1,2).to(device) 
        lidar_bev = lidar_bev.float().permute(0,3,1,2).to(device)
        fused = fused.float().permute(0,3,1,2).to(device)
        sem = sem.long().to(device) 
        edge = edge.float().to(device)

        pred_sem = model(rgb, lidar_bev, fused)
        resize = Resize(size = (rgb.shape[2], rgb.shape[3]))
        pred_sem = resize(pred_sem)
        disparity = nn.Sigmoid()(pred_sem)
        loss = sem_loss(pred_sem, sem)

        # calculate smoothness and add it to the loss
        # mean_disp = disparity.mean(2, True).mean(3, True)
        # norm_disp = disparity / (mean_disp + 1e-7)
        # loss += smooth_loss(norm_disp, rgb)
        loss = torch.unsqueeze(loss,0).mean()

        del batch, edge, lidar_bev, fused
        return rgb, sem, pred_sem, loss