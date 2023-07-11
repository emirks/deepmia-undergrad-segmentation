# Written by Jiacong Xu (jiacong.xu@tamu.edu)
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from torchvision.transforms import Resize

from .model_utils import SegmentHead, SegDecoder, DAPPM, PAPPM, \
    FusedLidarAttention, FusedCameraAttention, Pag, MultiSpatialTransformer

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import SamPredictor, sam_model_registry
from config import SAVE_DIR

bn_mom = 0.1
algc = False
batch_norm = nn.BatchNorm2d

def normalize_image(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    return x

def normalize_fused(x):
    x = x.clone()
    x[:, 0] = ((x[:, 0] / 255.0) - 0.485) / 0.229
    x[:, 1] = ((x[:, 1] / 255.0) - 0.456) / 0.224
    x[:, 2] = ((x[:, 2] / 255.0) - 0.406) / 0.225
    x[:, 3] = ((x[:, 3] / 255.0) - 0.485) / 0.229
    x[:, 4] = ((x[:, 4] / 255.0) - 0.456) / 0.224
    x[:, 5] = ((x[:, 5] / 255.0) - 0.406) / 0.225
    return x

class PIDNet(nn.Module): 
    def __init__(self, m, n, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment
        self.relu = nn.ReLU(inplace=True)
        self.upsample = lambda x, size : F.interpolate(x, size, mode="bilinear", align_corners=algc)
        self.img_anchors = (5, 22)
        self.lidar_anchors = (8, 8)
        self.sam_checkpoint_file = f"{SAVE_DIR}/sam_vit_b_01ec64.pth"

        # Fused Branch 
        # (B, 4, H, W) --> (B, C, H/4, W/4)
        # (B, C, H/4, W/4) --> (B, C, H/4, W/4)
        # (B, C, H/4, W/4) --> (B, 2*C, H/8, W/8)
        # (B, 2*C, H/8, W/8) --> (B, 4*C, H/16, W/16)
        # (B, 4*C, H/16, W/16) --> (B, 8*C, H/32, W/32)
        # (B, 8*C, H/32, W/32) --> (B, 16*C, H/32, W/32)

        # Camera Branch
        # (B, 3, H, W) --> (B, C, H/4, W/4)
        # (B, C, H/4, W/4) --> (B, C, H/4, W/4)
        # (B, C, H/4, W/4) --> (B, 2*C, H/8, W/8)
        # (B, 2*C, H/8, W/8) --> (B, 4*C, H/16, W/16)
        # (B, 4*C, H/16, W/16) --> (B, 8*C, H/32, W/32)
        # (B, 8*C, H/32, W/32) --> (B, 16*C, H/32, W/32)
        self.sam_model_for_image = sam_model_registry["vit_b"](self.sam_checkpoint_file)
        self.sam_model_for_fused = sam_model_registry["vit_b"](self.sam_checkpoint_file)
        self.image_encoder = self.sam_model_for_image.image_encoder
        self.fused_encoder = self.sam_model_for_fused.image_encoder
        # freeze the model's parameters
        for layer in self.image_encoder.parameters(): 
            layer.requires_grad = False
        for layer in self.fused_encoder.parameters(): 
            layer.requires_grad = False
        # self.fused_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        # self.fused_encoder.conv1 = torch.nn.Conv2d(6, 64, (7, 7), (2, 2), (3, 3), bias=False)

        # Attentions and branch combinations
        # Lidar Branch
        self.layer1_lidar_fused_att = FusedLidarAttention(planes, n_head=4, img_anchors=self.img_anchors, lidar_anchors=self.lidar_anchors)
        self.layer2_lidar_fused_att = FusedLidarAttention(planes * 2, n_head=4, img_anchors=self.img_anchors, lidar_anchors=self.lidar_anchors)
        self.layer3_lidar_fused_att = FusedLidarAttention(planes * 4, n_head=4, img_anchors=self.img_anchors, lidar_anchors=self.lidar_anchors)
        self.layer4_lidar_fused_att = FusedLidarAttention(planes * 8, n_head=4, img_anchors=self.img_anchors, lidar_anchors=self.lidar_anchors)

        self.layer1_camera_fused_att = MultiSpatialTransformer(planes, n_head=4, img_anchors=self.img_anchors)
        self.layer2_camera_fused_att = MultiSpatialTransformer(planes * 2, n_head=4, img_anchors=self.img_anchors)
        self.layer3_camera_fused_att = MultiSpatialTransformer(planes * 4, n_head=4, img_anchors=self.img_anchors)
        self.layer4_camera_fused_att = MultiSpatialTransformer(planes * 8, n_head=4, img_anchors=self.img_anchors) 

        # Prediction Head
        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)

        # (B, 8*C, H/32, W/32) --> (B, 4*C, H/32, W/32)
        if m == 2: 
            self.ppm = PAPPM(planes * 8, ppm_planes, planes * 8)
        elif m == 3: 
            self.ppm = DAPPM(planes * 8, ppm_planes, planes * 8)

        self.change_channel_conv_image = nn.Conv2d(planes * 8, 512, (1, 1))

    def forward(self, image_branch, lidar_branch, fused_branch): 
        # image_branch = normalize_image(image_branch)
        # fused_branch = normalize_image(fused_branch)

        image_branch = self.sam_model_for_image.preprocess(image_branch)
        fused_branch = self.sam_model_for_fused.preprocess(fused_branch)

        print(image_branch.shape)
        image_branch = self.image_encoder.patch_embed(image_branch)
        if self.image_encoder.pos_embed is not None: 
            image_branch = image_branch + self.image_encoder.pos_embed
        enc_block_ind = 0
        print(image_branch.shape)
        for block in self.image_encoder.blocks: 
            image_branch = block(image_branch) 
            print(image_branch.shape)

        fused_branch= self.fused_encoder.conv1(fused_branch)
        fused_branch= self.fused_encoder.bn1(fused_branch)
        fused_branch= self.fused_encoder.relu(fused_branch)
        fused_branch= self.fused_encoder.maxpool(fused_branch)

        for block in self.image_encoder.blocks[enc_block_ind:enc_block_ind+3]: 
            image_branch = block(image_branch)
        fused_branch = self.fused_encoder.layer1(fused_branch)
        image_branch, fused_branch = self.layer1_camera_fused_att(image_branch, fused_branch)
        enc_block_ind += 3

        for block in self.image_encoder.blocks[enc_block_ind:enc_block_ind+3]: 
            image_branch = block(image_branch)
        fused_branch = self.fused_encoder.layer2(fused_branch)
        image_branch, fused_branch = self.layer2_camera_fused_att(image_branch, fused_branch)
        enc_block_ind += 3

        for block in self.image_encoder.blocks[enc_block_ind:enc_block_ind+3]: 
            image_branch = block(image_branch)
        fused_branch = self.fused_encoder.layer3(fused_branch)
        image_branch, fused_branch = self.layer3_camera_fused_att(image_branch, fused_branch)
        enc_block_ind += 3

        for block in self.image_encoder.blocks[enc_block_ind:enc_block_ind+3]: 
            image_branch = block(image_branch)
        fused_branch = self.fused_encoder.layer4(fused_branch)
        image_branch, fused_branch = self.layer4_camera_fused_att(image_branch, fused_branch)

        out = self.ppm(image_branch)
        return out
    

def get_seg_model(cfg, imgnet_pretrained):
    if 's' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in cfg.MODEL.NAME:
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256, augment=True)
    
    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')['state_dict'] 
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)
    
    return model


def get_pred_model(name, num_classes):
    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)    
    return model


if __name__ == '__main__':
    device = torch.device('cuda')
    model = get_pred_model(name='pidnet_s', num_classes=19)
    model.eval()
    model.to(device)
    iterations = None
    
    rgb = torch.randn(1, 3, 160, 960).cuda()
    lidar = torch.randn(1, 2, 256, 256).cuda()
    fused = torch.randn(1, 3, 160, 960).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(rgb, lidar, fused)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(rgb, lidar, fused)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(rgb, lidar, fused)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)