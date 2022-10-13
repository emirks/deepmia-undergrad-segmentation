import torch
from torch import nn
import torch.nn.functional as F

class DownsamplerBlock(nn.Module): 
    """
    DownsamplerBlock is a combination of convolution and pooling layers.    
    ninput -> the channel size of the input image
    noutput -> the channel size we want the output image to have
    To have an output channel as noutput, the convolution layer should give an output with 
    noutput - ninput channel size.(the channel comes from pooling layer is ninput) 
    """
    def __init__(self, ninput, noutput) -> None:
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride = 2) 
        self.bn = nn.BatchNorm2d(noutput, eps = 1e-3)

    def forward(self, input): 
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

class non_bottleneck_1d(nn.Module): 
    def __init__(self, chann, dropprob, dilated) -> None:
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated, 0), bias=True, dilation=(dilated, 1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    
    def forward(self, input): 
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)


        if self.dropout.p != 0: 
            output = self.dropout(output) 
        
        return F.relu(output + input) #residual connection


class SemanticSegmentationEncoder(nn.Module): 
    """
    Encoder for the semantic segmentation task. The design is taken from ERFNet.
    num_classes : 
    """
    def __init__(self, num_classes) -> None:
        super().__init__()
        # initial block of the encoder : only a downsampler block
        self.initial_block = DownsamplerBlock(3, 16) 

        self.layers = nn.ModuleList()
        # first downsampling layer : downsampler + 5 non_bottleneck with no dilation
        self.layers.append(DownsamplerBlock(16, 64))
        for _ in range(5): 
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))
        
        # second downsampling layer : downsampler + 8 non_bottleneck with increasing dilations(parsed into 2)
        self.layers.append(DownsamplerBlock(64, 128))
        for _ in range(2): 
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))
        
        # only for encoder mode: 
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)
        
    def forward(self, input, predict=False): 
        output = self.initial_block(input) 

        for layer in self.layers: 
            output = layer(output) 
        
        if predict: 
            output = self.output_conv(output)
        
        return output