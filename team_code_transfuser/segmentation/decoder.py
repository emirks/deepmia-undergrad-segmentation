import torch
from torch import nn
import torch.nn.functional as F

from encoder import non_bottleneck_1d

class Upsampler(nn.Module): 
    """
    Upsampler is deconvolution layer with stride 2 (also known as transposed
    convolutions or full-convolutions).
    """
    def __init__(self, ninput, noutput) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
    
    def forward(self, input): 
        output = self.conv(input) 
        output = self.bn(output) 
        return F.relu(output)

    
class SemanticSegmentationDecoder(nn.Module): 
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        # first upsampling layer: Upsampler + 2 non_bottleneck with no dilation
        self.layers.append(Upsampler(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        # second upsampling layer: same as the first layer
        self.layers.append(Upsampler(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        
        # third upsampling layer: Upsampler layer with different paddings. 
        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)
    
    def forward(self, input): 
        output = input
        for layer in self.layers: 
            output = layer(output) 
        output = self.output_conv(output) 

        return output