# Written by Jiacong Xu (jiacong.xu@tamu.edu)
import torch 
import torch.nn as nn
import torch.nn.functional as F
import time
import logging

from .model_utils import BasicBlock, Bottleneck, SegmentHead, DAPPM, PAPPM, Pag, Bag, LightBag, DisparityBlock

bn_mom = 0.1
algc = False
batch_norm = nn.BatchNorm2d

class PIDNet(nn.Module): 
    def __init__(self, m, n, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.augment = augment
        self.relu = nn.ReLU(inplace=True)
        self.upsample = lambda x, size : F.interpolate(x, size, mode="bilinear", align_corners=algc)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=planes, kernel_size=3, stride=2, padding=1),
            batch_norm(planes, momentum=bn_mom), 
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            batch_norm(planes, momentum=bn_mom), 
            nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)

        # I Branch 
        self.layer3_i = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4_i = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5_i = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)
        if m == 2: 
            self.ppm = PAPPM(planes * 16, ppm_planes, planes * 4)
        elif m == 3: 
            self.ppm = DAPPM(planes * 16, ppm_planes, planes * 4)
        
        # P Branch
        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        self.layer3_pag = Pag(planes * 2, planes)
        self.layer4_pag = Pag(planes * 2, planes)
        self.layer3_compression = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            batch_norm(planes * 2, momentum=bn_mom)
        )
        self.layer4_compression = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            batch_norm(planes * 2, momentum=bn_mom)
        )

        # D Branch
        if m == 2: 
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
        elif m == 3: 
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, planes * 2, planes * 2)
        self.layer3_diff = nn.Sequential(
            nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
            batch_norm(planes, momentum=bn_mom)
        )
        self.layer4_diff = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
            batch_norm(planes * 2, momentum=bn_mom)
        )
        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Combining P, I, D branches
        if m == 2:        
            self.dfm = LightBag(planes * 4, planes * 4)
        elif m == 3: 
            self.dfm = Bag(planes * 4, planes * 4)

        # Prediction Head
        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)           
            # self.disp_conv = DisparityBlock(num_classes, 1)

        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)


        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    
    def _make_layer(self, block, inplanes, outplanes, num_of_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes * block.expansion, 
                        kernel_size=1, stride=stride, bias=False), 
                batch_norm(outplanes * block.expansion, momentum=bn_mom)
            )
        
        layers = []
        layers.append(block(inplanes, outplanes, stride, downsample))
        inplanes = outplanes * block.expansion
        for i in range(1, num_of_blocks): 
            if i == num_of_blocks - 1: 
                layers.append(block(inplanes, outplanes, stride=1, apply_relu=False))
            else: 
                layers.append(block(inplanes, outplanes, stride=1, apply_relu=True))
        
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, outplanes, stride=1): 
        downsample = None
        if stride != 1 or inplanes != outplanes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes * block.expansion, 
                        kernel_size=1, stride=stride, bias=False), 
                batch_norm(outplanes * block.expansion, momentum=bn_mom)
            )

        layer = block(inplanes, outplanes, stride, downsample, apply_relu=False)
        return layer
        

    def forward(self, x): 
        input_width = x.shape[3]
        input_height = x.shape[2]
        output_size = [input_height//8, input_width//8]

        # layer 1 and 2
        out = x
        out = self.conv1(out) 
        out = self.layer1(out) 
        out = self.relu(self.layer2(self.relu(out)))
        
        # layer 3
        out_p = self.layer3_p(out) 
        out_d = self.layer3_d(out)
        out_i = self.relu(self.layer3_i(out)) 
        out_p = self.layer3_pag(out_p, self.layer3_compression(out_i))
        out_d = out_d + self.upsample(self.layer3_diff(out_i), output_size)
        if self.augment: 
            temp_p = out_p
        
        # layer 4
        out_p = self.layer4_p(self.relu(out_p)) 
        out_d = self.layer4_d(self.relu(out_d)) 
        out_i = self.relu(self.layer4_i(out_i))
        out_p = self.layer4_pag(out_p, self.layer4_compression(out_i))
        out_d = out_d + self.upsample(self.layer4_diff(out_i), output_size)
        if self.augment: 
            temp_d = out_d

        # layer 5
        out_p = self.layer5_p(self.relu(out_p))
        out_d = self.layer5_d(self.relu(out_d))
        out_i = self.upsample(self.ppm(self.layer5_i(out_i)), output_size)

        # combine P, I, D
        out = self.final_layer(self.dfm(out_p, out_i, out_d))

        if self.augment: # in training augment will be true, in interference it will be false
            out_p_loss = self.seghead_p(temp_p)
            out_d_loss = self.seghead_d(temp_d)
            return [out_p_loss, out, out_d_loss]
        else: 
            return out
    
    def pass_from_model(self, rgb, sem, edge, config_file):
        if config_file.split_cameras: 
            camera_count = len(config_file.camera_rots)
            rgbs = torch.tensor_split(rgb, camera_count, dim=3)
            sems = torch.tensor_split(sem, camera_count, dim=2)
            edges = torch.tensor_split(edge, camera_count, dim=2)
            pred_sems = []
            losses = []
            for i in range(camera_count):
                rgb_i, sem_i, edge_i = rgbs[i], sems[i], edges[i] 
                out_p_loss, pred_sem, out_d_loss = model(rgb_i)
                resize = Resize(size = (rgb_i.shape[2], rgb_i.shape[3]))
                pred_sem = resize(pred_sem)
                out_p_loss = resize(out_p_loss)
                out_d_loss = resize(out_d_loss)

                loss_s = config_file.sem_loss(pred_sem, sem_i)
                loss_b = config_file.bd_loss()(out_d_loss, edge_i)
                loss = loss_s + loss_b
                
                # calculate smoothness and add it to the loss
                disparity = nn.Sigmoid()(pred_sem)
                mean_disp = disparity.mean(2, True).mean(3, True)
                norm_disp = disparity / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, rgb_i)
                loss += config.disparity_smoothness * smooth_loss     
                loss = torch.unsqueeze(loss,0).mean()   
                
                losses.append(loss)
                pred_sems.append(pred_sem)
            pred_sem = torch.cat(pred_sems, dim=3) 
            loss = sum(losses)
        else: 
            out_p_loss, pred_sem, out_d_loss = model(rgb)
            resize = Resize(size = (rgb.shape[2], rgb.shape[3]))
            pred_sem = resize(pred_sem)
            out_p_loss = resize(out_p_loss)
            out_d_loss = resize(out_d_loss)
            disparity = nn.Sigmoid()(pred_sem)

            loss_s = sem_loss(pred_sem, sem)
            loss_b = BondaryLoss()(out_d_loss, edge)

            loss = loss_s + loss_b
            # calculate smoothness and add it to the loss
            mean_disp = disparity.mean(2, True).mean(3, True)
            norm_disp = disparity / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, rgb)
            loss += config.disparity_smoothness * smooth_loss
            loss = torch.unsqueeze(loss,0).mean()

        return pred_sem, loss        


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
    
    input = torch.randn(1, 3, 1024, 2048).cuda()
    with torch.no_grad():
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
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
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)