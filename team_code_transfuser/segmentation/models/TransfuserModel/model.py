import torch
from torch import nn
from torchvision.transforms import Resize

from .transfuser_backbone import TransfuserBackbone, SegDecoder

class Config: 
    num_class = None

    # Data
    seq_len = 1 # input timesteps
    # use different seq len for image and lidar
    img_seq_len = 1 
    lidar_seq_len = 1

    # Conv Encoder
    img_vert_anchors = 5
    img_horz_anchors = 20 + 2
    lidar_vert_anchors = 8
    lidar_horz_anchors = 8
    
    img_anchors = img_vert_anchors * img_horz_anchors
    lidar_anchors = lidar_vert_anchors * lidar_horz_anchors

    perception_output_features = 512 # Number of features outputted by the perception branch.
    bev_features_chanels = 64 # Number of channels for the BEV feature pyramid
    bev_upsample_factor = 2

    deconv_channel_num_1 = 128 # Number of channels at the first deconvolution layer
    deconv_channel_num_2 = 64 # Number of channels at the second deconvolution layer
    deconv_channel_num_3 = 32 # Number of channels at the third deconvolution layer

    deconv_scale_factor_1 = 8 # Scale factor, of how much the grid size will be interpolated after the first layer
    deconv_scale_factor_2 = 4 # Scale factor, of how much the grid size will be interpolated after the second layer
	
    # GPT Encoder
    n_embd = 512
    block_exp = 4
    n_layer = 8
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0 # Mean of the normal distribution with which the linear layers in the GPT are initialized
    gpt_linear_layer_init_std  = 0.02 # Std  of the normal distribution with which the linear layers in the GPT are initialized
    gpt_layer_norm_init_weight = 1.0 # Initial weight of the layer norms in the gpt.



class SegmentationModel(nn.Module): 
    def __init__(self, config): 
        super().__init__()

        self.encoder = TransfuserBackbone(config)
        self.decoder = SegDecoder(config)    

    def forward(self, rgb, lidar):
        image_features_grid = self.encoder(rgb, lidar) 
        out = self.decoder(image_features_grid)
        return out
    
    @staticmethod
    def initialize(num_classes): 
        config = Config()
        config.num_class = num_classes
        return SegmentationModel(config)

    @staticmethod
    def pass_from_model(model, batch, sem_loss, smooth_loss, bd_loss, device = "cuda"):
        rgb, sem, edge, lidar_bev = batch[:4]
        rgb = rgb.float().permute(0,3,1,2).to(device) 
        lidar_bev = lidar_bev.float().permute(0,3,1,2).to(device)
        sem = sem.long().to(device) 
        edge = edge.float().to(device)

        pred_sem = model(rgb, lidar_bev)
        resize = Resize(size = (rgb.shape[2], rgb.shape[3]))
        pred_sem = resize(pred_sem)
        disparity = nn.Sigmoid()(pred_sem)
        loss = sem_loss(pred_sem, sem)

        # calculate smoothness and add it to the loss
        # mean_disp = disparity.mean(2, True).mean(3, True)
        # norm_disp = disparity / (mean_disp + 1e-7)
        # loss += smooth_loss(norm_disp, rgb)
        loss = torch.unsqueeze(loss,0).mean()

        del batch, edge, lidar_bev
        return rgb, sem, pred_sem, loss