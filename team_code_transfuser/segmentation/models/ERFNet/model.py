import torch
from torch import nn

from .encoder import SemanticSegmentationEncoder
from .decoder import SemanticSegmentationDecoder
from torchvision.transforms import Resize

class SemanticSegmentation(nn.Module): 
    def __init__(self, num_classes): 
        super().__init__()

        self.encoder = SemanticSegmentationEncoder(num_classes)
        self.decoder = SemanticSegmentationDecoder(num_classes)


    def forward(self, input):
        output = self.encoder(input)
        return self.decoder(output)
    
    @staticmethod
    def initialize(num_classes): 
        return SemanticSegmentation(num_classes)
    
    @staticmethod
    def pass_from_model(model, batch, sem_loss, smooth_loss, bd_loss, device = "cuda"): 
        rgb, sem, edge = batch[:3]
        rgb = rgb.float().permute(0,3,1,2).to(device) 
        sem = sem.long().to(device) 
        edge = edge.float().to(device)        

        pred_sem = model(rgb)
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