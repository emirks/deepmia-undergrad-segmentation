# Written by Jiacong Xu (jiacong.xu@tamu.edu)
import torch
import torch.nn as nn
import torch.nn.functional as F

bn_mom = 0.1
algc = False
upsample = lambda x, size : F.interpolate(x, size, mode="bilinear", align_corners=algc)

class BasicBlock(nn.Module): 
    expansion = 1 # No expansion
    def __init__(self, inplanes, outplanes, stride=1, downsample=None, apply_relu=True) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes, momentum=bn_mom)

        self.downsample = downsample
        self.apply_relu = apply_relu
    
    def forward(self, x): 
        out = x
        out = self.conv1(out) 
        out = self.bn1(out) 
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) 

        if self.downsample != None: 
            x = self.downsample(x) 
        
        out = out + x #residual
        if self.apply_relu: 
            out = self.relu(out) 
        return out


class Bottleneck(nn.Module): 
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, downsample=None, apply_relu=True) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(outplanes, outplanes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.apply_relu = apply_relu
    
    def forward(self, x): 
        out = x
        out = self.conv1(out) 
        out = self.bn1(out) 
        out = self.relu(out) 
        out = self.conv2(out) 
        out = self.bn2(out) 
        out = self.relu(out)
        out = self.conv3(out) 
        out = self.bn3(out) 

        if self.downsample != None: 
            x = self.downsample(x) 
        out = out + x #residual
        if self.apply_relu: 
            out = self.relu(out) 
        return out


class SegmentHead(nn.Module): 
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(SegmentHead, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=False)
        self.scale_factor = scale_factor
    
    def forward(self, x): 
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))

        if self.scale_factor != None: 
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = upsample(out, [height, width])

        return out
            


class DAPPM(nn.Module): 
    """

    """
    def scale(self, kernel_size, stride, padding):
        if kernel_size == -1: 
            avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else: 
            avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding), 
        return nn.Sequential(
            avg_pool,
            nn.BatchNorm2d(self.inplanes, momentum=self.bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.branch_planes, kernel_size=1, bias=False),
        )
    
    def process(self): 
        return nn.Sequential(
            nn.BatchNorm2d(self.branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.branch_planes, self.branch_planes, kernel_size=3, padding=1, bias=False),
        )

    def __init__(self, inplanes, branch_planes, outplanes) -> None:
        super(DAPPM, self).__init__()
        self.bn_mom = 0.1
        self.inplanes = inplanes
        self.branch_planes = branch_planes

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(self.inplanes, momentum=self.bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.branch_planes, kernel_size=1, bias=False),
        )
        self.scale1 = self.scale(kernel_size=5, stride=2, padding=2)
        self.scale2 = self.scale(kernel_size=9, stride=4, padding=4) 
        self.scale3 = self.scale(kernel_size=17, stride=8, padding=8)
        self.scale4 = self.scale(kernel_size=-1)

        self.process1 = self.process()
        self.process2 = self.process()
        self.process3 = self.process()
        self.process4 = self.process()

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )
    
    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]

        x_list = []
        x_list.append(self.scale0(x))
        x_list.append(self.process1(
            upsample(self.scale1(x), [height, width]) + x_list[0]))
        x_list.append(self.process2(
            upsample(self.scale2(x), [height, width]) + x_list[1]))
        x_list.append(self.process3(
            upsample(self.scale3(x), [height, width]) + x_list[2]))
        x_list.append(self.process4(
            upsample(self.scale4(x), [height, width]) + x_list[3]))
        
        out = self.compression(torch.cat(x_list, dim=1) + self.shortcut(x))
        return out
        

class PAPPM(nn.Module): 
    def scale(self, kernel_size, stride=1, padding=1):
        if kernel_size == -1: 
            avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else: 
            avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        return nn.Sequential(
            avg_pool,
            nn.BatchNorm2d(self.inplanes, momentum=self.bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.branch_planes, kernel_size=1, bias=False),
        )

    def __init__(self, inplanes, branch_planes, outplanes) -> None:
        super(PAPPM, self).__init__()
        self.bn_mom = 0.1
        self.inplanes = inplanes
        self.branch_planes = branch_planes

        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(self.inplanes, momentum=self.bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.branch_planes, kernel_size=1, bias=False),
        )
        self.scale1 = self.scale(kernel_size=5, stride=2, padding=2)
        self.scale2 = self.scale(kernel_size=9, stride=4, padding=4) 
        self.scale3 = self.scale(kernel_size=17, stride=8, padding=8)
        self.scale4 = self.scale(kernel_size=-1)

        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(self.branch_planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.branch_planes * 4, self.branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False),
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x): 
        width = x.shape[-1]
        height = x.shape[-2]

        scale_list = []
        x0 = self.scale0(x)
        scale_list.append(upsample(self.scale1(x), [height, width]) + x0)
        scale_list.append(upsample(self.scale2(x), [height, width]) + x0)
        scale_list.append(upsample(self.scale3(x), [height, width]) + x0)
        scale_list.append(upsample(self.scale4(x), [height, width]) + x0)

        scale_out = self.scale_process(torch.cat(scale_list, 1))
        
        out = self.compression(torch.cat([x0, scale_out], dim=1)) + self.shortcut(x)
        return out


class Pag(nn.Module): 
    def __init__(self, in_channels, mid_channels, apply_relu_first=False, with_channel=False) -> None:
        super(Pag, self).__init__()

        self.with_channel = with_channel
        self.apply_relu_first = apply_relu_first
        self.f_integral = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels)
        )
        self.f_proportional = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), 
            nn.BatchNorm2d(mid_channels)
        )

        if apply_relu_first: 
            self.relu = nn.ReLU(inplace=True)
        if self.with_channel: 
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels)
            )

    def forward(self, p, i): 
        p_input_size = [p.size()[2], p.size()[3]]
        if self.apply_relu_first:
            p = self.relu(p) 
            i = self.relu(i)

        i_q = self.f_integral(i)
        i_q_upsampled = upsample(i_q, p_input_size)
        i_upsampled = upsample(i, p_input_size)

        p_q = self.f_proportional(p)

        # I don't understand with_channel part
        if self.with_channel: 
            similarity_map = torch.sigmoid(self.up(p_q * i_q_upsampled))
        else: 
            similarity_map = torch.sigmoid(torch.sum(p_q * i_q_upsampled, dim=1).unsqueeze(1))
        out = similarity_map * i_upsampled + (1 - similarity_map) * p
        return out


class Bag(nn.Module): 
    """
    Bag(Balancing the Details and Contexts) module fuses the features provided by three branches. 
        Context branch(i) -> semantically rich and could present more accurate semantics but loses too much spatial details 
        especially for the boundary region and small object.
        Detailed branch(p) -> Preserves the spatial details better. 
        Boundary branch(d) -> 
    We force the model to trust to the detailed branch more along the boundary region and utilize the context features to fill
    the area inside object with Bag. 
    """
    def __init__(self, in_channels, out_channels) -> None:
        super(Bag, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
    
    def forward(self, p, i, d): 
        boundary_attr = torch.sigmoid(d)
        out = self.conv((1 - boundary_attr) * i + boundary_attr * p)
        return out

class LightBag(nn.Module): 
    """
    LightBag convert 3d convolutions to 1d convolutions in Bag to make it faster. Also it slightly changes
    the forward part to suit this new convolution. 
    """
    def __init__(self, in_channels, out_channels) -> None:
        super(LightBag, self).__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, p, i, d): 
        boundary_attr = torch.sigmoid(d) 
        p_add = self.conv_p(boundary_attr * p + i) 
        i_add = self.conv_i((1 - boundary_attr) * i + p)
        
        out = p_add + i_add
        return out

