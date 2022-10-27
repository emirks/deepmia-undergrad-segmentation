import torch
from torch import nn

from .encoder import SemanticSegmentationEncoder
from .decoder import SemanticSegmentationDecoder


class SemanticSegmentation(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()

        self.encoder = SemanticSegmentationEncoder(num_classes)
        self.decoder = SemanticSegmentationDecoder(num_classes)


    def forward(self, input):
        output = self.encoder(input)
        return self.decoder(output)
