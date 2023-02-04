# Written by Jiacong Xu (jiacong.xu@tamu.edu)
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import logging
from torchvision.transforms import Resize

from .model_utils import BasicBlock, Bottleneck, SegmentHead, DAPPM, PAPPM, Pag, Bag, LightBag, FusedToCameraAttention, FusedToLidarAttention

bn_mom = 0.1
algc = False
batch_norm = nn.BatchNorm2d

class PIDNet(nn.Module): 
    def __init__(self, m, n, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment
        self.relu = nn.ReLU(inplace=True)
        self.upsample = lambda x, size : F.interpolate(x, size, mode="bilinear", align_corners=algc)

        # Fused Branch 
        # (B, 3, H, W) --> (B, C, H/4, W/4)
        self.layer0_fused = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=planes, kernel_size=3, stride=2, padding=1),
            batch_norm(planes, momentum=bn_mom), 
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            batch_norm(planes, momentum=bn_mom), 
            nn.ReLU(inplace=True)
        )
        # (B, C, H, W) --> (B, C, H, W)
        self.layer1_fused = self._make_layer(BasicBlock, planes, planes, m)
        # (B, C, H, W) --> (B, 2*C, H, W)
        self.layer2_fused = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        # (B, 2*C, H, W) --> (B, 4*C, H, W)
        self.layer3_fused = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        # (B, 4*C, H, W) --> (B, 8*C, H, W)
        self.layer4_fused = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        # (B, 8*C, H, W) --> (B, 16*C, H, W)
        self.layer5_fused = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        # (B, 16*C, H, W) --> (B, 4*C, H, W)
        if m == 2: 
            self.ppm = PAPPM(planes * 16, ppm_planes, planes * 4)
        elif m == 3: 
            self.ppm = DAPPM(planes * 16, ppm_planes, planes * 4)


        # Lidar Branch
        # (B, 3, H, W) --> (B, C, H, W)
        self.layer0_lidar = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=planes, kernel_size=3, stride=2, padding=1),
            batch_norm(planes, momentum=bn_mom), 
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            batch_norm(planes, momentum=bn_mom), 
            nn.ReLU(inplace=True)
        )
        # (B, C, H, W) --> (B, C, H, W)
        self.layer1_lidar = self._make_layer(BasicBlock, planes, planes, m)
        # (B, C, H, W) --> (B, 2*C, H, W)
        self.layer2_lidar = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        # (B, 2*C, H, W) --> (B, 2*C, H, W)
        self.layer3_lidar = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        # (B, 2*C, H, W) --> (B, 2*C, H, W)
        self.layer4_lidar = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        # (B, 2*C, H, W) --> (B, 4*C, H, W)
        self.layer5_lidar = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)


        # Camera Branch
        # (B, 3, H, W) --> (B, C, H, W)
        self.layer0_camera = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=planes, kernel_size=3, stride=2, padding=1),
            batch_norm(planes, momentum=bn_mom), 
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            batch_norm(planes, momentum=bn_mom), 
            nn.ReLU(inplace=True)
        )
        # (B, C, H, W) --> (B, C, H, W)        
        self.layer1_camera = self._make_layer(BasicBlock, planes, planes, m)
        # (B, C, H, W) --> (B, 2*C, H, W)
        self.layer2_camera = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        # (B, 2*C, H, W) --> (B, 2*C, H, W)
        # (B, 2*C, H, W) --> (B, 2*C, H, W)
        if m == 2: 
            self.layer3_camera = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_camera = self._make_layer(Bottleneck, planes * 2, planes, 1)
        elif m == 3: 
            self.layer3_camera = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_camera = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
        # (B, 2*C, H, W) --> (B, 4*C, H, W)
        self.layer5_camera = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)
        

        # Attentions and branch combinations
        # Lidar Branch
        self.layer3_lidar_fused_att = FusedToLidarAttention(planes * 2, n_head=4, attn_pdrop=0.1, resid_pdrop=0.1) 
        self.layer4_lidar_fused_att = FusedToLidarAttention(planes * 2, n_head=4, attn_pdrop=0.1, resid_pdrop=0.1)     
        self.layer3_compression_for_lidar = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            batch_norm(planes * 2, momentum=bn_mom)
        )
        self.layer4_compression_for_lidar = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            batch_norm(planes * 2, momentum=bn_mom)
        )

        # Camera Branch
        self.layer3_camera_fused_att = FusedToCameraAttention(planes * 2, n_head=4, attn_pdrop=0.1, resid_pdrop=0.1) 
        self.layer4_camera_fused_att = FusedToCameraAttention(planes * 2, n_head=4, attn_pdrop=0.1, resid_pdrop=0.1) 
        self.layer3_compression_for_camera = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
            batch_norm(planes * 2, momentum=bn_mom)
        )
        self.layer4_compression_for_camera = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
            batch_norm(planes * 2, momentum=bn_mom)
        )

        # Combining branches
        if m == 2:        
            self.dfm = LightBag(planes * 4, planes * 4)
        elif m == 3: 
            self.dfm = Bag(planes * 4, planes * 4)

        # Prediction Head
        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)

        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)


        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    
    def _make_layer(self, block, inplanes, outplanes, num_of_blocks, stride=1):
        """
            If block is BasicBlock, shape transformation: (B, inplanes, H, W) --> (B, outplanes, H, W)
            If block is Bottleneck, shape transformation: (B, inplanes, H, W) --> (B, outplanes * expansion, H, W)
        
        """
        layers = []
        layers.append(block(inplanes, outplanes, stride))
        inplanes = outplanes * block.expansion
        for i in range(1, num_of_blocks): 
            if i == num_of_blocks - 1: 
                layers.append(block(inplanes, outplanes, stride=1, apply_relu=False))
            else: 
                layers.append(block(inplanes, outplanes, stride=1, apply_relu=True))
        
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, outplanes, stride=1): 
        layer = block(inplanes, outplanes, stride, apply_relu=False)
        return layer

    def forward(self, image_input, lidar_bev_input, fused_input): 
        input_width = image_input.shape[3]
        input_height = image_input.shape[2]
        output_size = [input_height//8, input_width//8]

        image_branch = image_input
        lidar_branch = lidar_bev_input
        fused_branch = fused_input

        lidar_branch = self.layer0_lidar(lidar_branch)
        image_branch = self.layer0_camera(image_branch)
        fused_branch = self.layer0_fused(fused_branch)

        lidar_branch = self.relu(self.layer1_lidar(lidar_branch))
        image_branch = self.relu(self.layer1_camera(image_branch))
        fused_branch = self.relu(self.layer1_fused(fused_branch))

        lidar_branch = self.relu(self.layer2_lidar(lidar_branch))
        image_branch = self.relu(self.layer2_camera(image_branch))
        fused_branch = self.relu(self.layer2_fused(fused_branch))

        lidar_branch = self.layer3_lidar(lidar_branch)
        image_branch = self.layer3_camera(image_branch)
        fused_branch = self.relu(self.layer3_fused(fused_branch))
        lidar_branch = self.relu(self.layer3_lidar_fused_att(lidar_branch, 
                                self.layer3_compression_for_lidar(fused_branch)))
        image_branch = self.relu(self.layer3_camera_fused_att(image_branch, 
                                self.layer3_compression_for_camera(fused_branch)))

        lidar_branch = self.layer4_lidar(lidar_branch)
        image_branch = self.layer4_camera(image_branch)
        fused_branch = self.relu(self.layer4_fused(fused_branch))
        lidar_branch = self.relu(self.layer4_lidar_fused_att(lidar_branch, 
                                self.layer4_compression_for_lidar(fused_branch)))
        image_branch = self.relu(self.layer4_camera_fused_att(image_branch, 
                                self.layer4_compression_for_camera(fused_branch)))

        lidar_branch = self.layer5_lidar(lidar_branch)
        image_branch = self.layer5_camera(image_branch)
        fused_branch = self.upsample(self.ppm(self.layer5_fused(fused_branch)), output_size)

        lidar_branch = self.upsample(lidar_branch, output_size)
        out = self.final_layer(self.dfm(lidar_branch, fused_branch, image_branch))
        return out

    @staticmethod
    def initialize(num_classes):
        return PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=True)
    
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
        mean_disp = disparity.mean(2, True).mean(3, True)
        norm_disp = disparity / (mean_disp + 1e-7)
        loss += smooth_loss(norm_disp, rgb)
        loss = torch.unsqueeze(loss,0).mean()

        return rgb, sem, pred_sem, loss


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