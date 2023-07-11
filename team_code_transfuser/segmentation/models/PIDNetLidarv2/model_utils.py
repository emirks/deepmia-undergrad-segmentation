# Written by Jiacong Xu (jiacong.xu@tamu.edu)
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

bn_mom = 0.1
algc = False
upsample = lambda x, size : F.interpolate(x, size, mode="bilinear", align_corners=algc)
batch_norm = nn.BatchNorm2d

class SegmentHead(nn.Module): 
    """
        args: inplanes, interplanes, outplanes

        forward shape transformation: 
            (B, inplanes, H, W) --> (B, outplanes, H, W)
    """
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(SegmentHead, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = batch_norm(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = batch_norm(interplanes, momentum=bn_mom)
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
        args: inplanes, branch_planes, outplanes

        forward shape transformation: 
            (B, inplanes, H, W) --> (B, outplanes, H, W) 
    """    
    def scale(self, kernel_size, stride=1, padding=1):
        if kernel_size == -1: 
            avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else: 
            avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding) 
        return nn.Sequential(
            avg_pool,
            batch_norm(self.inplanes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.branch_planes, kernel_size=1, bias=False)
        )
    
    def process(self): 
        return nn.Sequential(
            batch_norm(self.branch_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.branch_planes, self.branch_planes, kernel_size=3, padding=1, bias=False),
        )

    def __init__(self, inplanes, branch_planes, outplanes) -> None:
        super(DAPPM, self).__init__()
        self.inplanes = inplanes
        self.branch_planes = branch_planes

        self.scale0 = nn.Sequential(
            batch_norm(self.inplanes, momentum=bn_mom),
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
            batch_norm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            batch_norm(inplanes, momentum=bn_mom),
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
        
        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)
        return out
        

class PAPPM(nn.Module): 
    """
        args: inplanes, branch_planes, outplanes

        forward shape transformation: 
            (B, inplanes, H, W) --> (B, outplanes, H, W) 
    """
    def scale(self, kernel_size, stride=1, padding=1):
        if kernel_size == -1: 
            avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else: 
            avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        return nn.Sequential(
            avg_pool,
            batch_norm(self.inplanes, momentum=self.bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.branch_planes, kernel_size=1, bias=False),
        )

    def __init__(self, inplanes, branch_planes, outplanes) -> None:
        super(PAPPM, self).__init__()
        self.bn_mom = 0.1
        self.inplanes = inplanes
        self.branch_planes = branch_planes

        self.scale0 = nn.Sequential(
            batch_norm(self.inplanes, momentum=self.bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.branch_planes, kernel_size=1, bias=False),
        )
        self.scale1 = self.scale(kernel_size=5, stride=2, padding=2)
        self.scale2 = self.scale(kernel_size=9, stride=4, padding=4) 
        self.scale3 = self.scale(kernel_size=17, stride=8, padding=8)
        self.scale4 = self.scale(kernel_size=-1)

        self.scale_process = nn.Sequential(
            batch_norm(self.branch_planes * 4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.branch_planes * 4, self.branch_planes * 4, kernel_size=3, padding=1, groups=4, bias=False),
        )

        self.compression = nn.Sequential(
            batch_norm(branch_planes * 5, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            batch_norm(inplanes, momentum=bn_mom),
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
    """
        args: in_channels, mid_channels

        forward shape transformation: 
            (B, in_channels, H1, W1), (B, in_channels, H2, W2) --> (B, in_channels, H1, W1)
    """
    def __init__(self, in_channels, mid_channels, apply_relu_first=False, with_channel=False) -> None:
        super(Pag, self).__init__()

        self.with_channel = with_channel
        self.apply_relu_first = apply_relu_first
        self.f_integral = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            batch_norm(mid_channels)
        )
        self.f_proportional = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), 
            batch_norm(mid_channels)
        )

        if apply_relu_first: 
            self.relu = nn.ReLU(inplace=True)
        if self.with_channel: 
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                batch_norm(in_channels)
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

class MultiHeadAttention(nn.Module): 
    """
        args: 
        embed_dim -->  The channel size of attention(all q, k and v should have C as embed_dim)
        num_heads
        attn_pdrop
        resid_pdrop

        forward shape transformation: 
            (B, Q_T, C), (B, KV_T, C), (B, KV_T, C) --> (B, Q_T, C)
            or
            (B, T, C) --> (B, T, C)
    """
    def __init__(self, embed_dim, num_heads, attn_pdrop, resid_pdrop) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.n_head = num_heads
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def _reshape_to_batches(self, x): 
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim
        sub_dim = embed_dim // self.n_head
        return x.view(batch_size, seq_len, self.n_head, sub_dim).transpose(1, 2)
    
    def _reshape_from_batches(self, x): 
        batch_size, n_head, seq_len, sub_dim = x.size()
        assert n_head == self.n_head
        out_dim = sub_dim * self.n_head
        assert out_dim == self.embed_dim
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, out_dim)
        
    def forward(self, q, k = None, v = None): 
        if k == None and v == None: 
            # self-attention
            k = v = q
        # Initial sizes: 
        # q -> (B, q_seq_len, C)
        # k -> (B, kv_seq_len, C)
        # v -> (B, kv_seq_len, C)
        assert k.size() == v.size()

        q = self._reshape_to_batches(self.query(q)) # (B, nh, q_seq_len, sub_dim)
        k = self._reshape_to_batches(self.key(k)) # (B, nh, kv_seq_len, sub_dim)
        v = self._reshape_to_batches(self.value(v)) # (B, nh, kv_seq_len, sub_dim)

        # self-attend: (B, nh, q_seq_len, sub_dim) x (B, nh, sub_dim, kv_seq_len) -> (B, nh, q_seq_len, kv_seq_len)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, q_seq_len, kv_seq_len) x (B, nh, kv_seq_len, sub_dim) -> (B, nh, q_seq_len, sub_dim)
        y = self._reshape_from_batches(y) # re-assemble all head outputs side by side
        # y -> (B, q_seq_len, embed_dim)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class CrossAttention(nn.Module): 
    def __init__(self, n_embd, n_head, in_channels, block_exp=4, attn_pdrop=0.1, resid_pdrop=0.1) -> None:
        super().__init__()
        self.f_fused = nn.Linear(n_embd, n_embd)
        self.f_camera = nn.Linear(n_embd, n_embd)

        self.camera_cross_attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.fused_cross_attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop)

    def forward(self, camera, fused): 
        camera_processed = self.f_camera(camera)
        fused_processed = self.f_fused(fused)

        camera_out = self.camera_cross_attn(camera_processed, fused_processed, fused_processed)
        fused_out = self.fused_cross_attn(fused_processed, camera_processed, camera_processed)
        camera_out += camera
        fused_out += fused
        return camera_out, fused_out



class TransformerBlock(nn.Module): 
    def __init__(self, n_embd, n_head, block_exp=4, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, q, k=None):
        if k == None: 
            x = q + self.attn(self.ln1(q))
        else: 
            q = self.ln1(q)
            v = k
            x = q + self.attn(q, k, v)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, n_embd, n_head, first_branch_anchors, second_branch_anchors, embd_pdrop=0.1):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1
        
        self.n_layer = 2
        self.gpt_linear_layer_init_mean = 0.0
        self.gpt_linear_layer_init_std  = 0.02
        self.gpt_layer_norm_init_weight = 1.0

        self.first_branch_anchors = first_branch_anchors
        self.second_branch_anchors = second_branch_anchors

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len * first_branch_anchors[0] * first_branch_anchors[1] + 
            self.seq_len * second_branch_anchors[0] * second_branch_anchors[1], n_embd))
        
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd=n_embd, n_head=n_head) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.gpt_linear_layer_init_mean, std=self.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.gpt_layer_norm_init_weight)

    def forward(self, first_branch_tensor, second_branch_tensor):
        bz = second_branch_tensor.shape[0]
        second_branch_h, second_branch_w = second_branch_tensor.shape[2:4]
        first_branch_h, first_branch_w = first_branch_tensor.shape[2:4]
        
        assert self.seq_len == 1
        first_branch_tensor = first_branch_tensor.view(bz, self.seq_len, -1, first_branch_h, first_branch_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)
        second_branch_tensor = second_branch_tensor.view(bz, self.seq_len, -1, second_branch_h, second_branch_w).permute(0,1,3,4,2).contiguous().view(bz, -1, self.n_embd)

        token_embeddings = torch.cat((first_branch_tensor, second_branch_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        x = x.view(bz, self.seq_len*self.first_branch_anchors[0]*self.first_branch_anchors[1] + self.seq_len*self.second_branch_anchors[0]*self.second_branch_anchors[1], self.n_embd)

        first_branch_tensor_out = x[:, :self.seq_len*self.first_branch_anchors[0]*self.first_branch_anchors[1], :].contiguous().view(bz * self.seq_len, -1, first_branch_h, first_branch_w)
        second_branch_tensor_out = x[:, self.seq_len*self.first_branch_anchors[0]*self.first_branch_anchors[1]:, :].contiguous().view(bz * self.seq_len, -1, second_branch_h, second_branch_w)

        return first_branch_tensor_out, second_branch_tensor_out


class FusedLidarAttention(nn.Module): 
    """
        args: n_channels, n_head, attn_pdrop, resid_pdrop 

        forward shape transformation: 
            (B, C, H1, W1), (B, C, H2, W2) --> (B, C, H1, W1)
    """
    def __init__(self, n_channels, n_head, img_anchors, lidar_anchors) -> None:
        super(FusedLidarAttention, self).__init__()

        self.avgpool_img = nn.AdaptiveAvgPool2d(img_anchors)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(lidar_anchors)
        self.transformer = GPT(n_channels, n_head, first_branch_anchors=img_anchors, second_branch_anchors=lidar_anchors)

    def forward(self, lidar_data, fused_data): 
        lidar_input_data = lidar_data
        fused_input_data = fused_data
        lidar_data = self.avgpool_lidar(lidar_data)
        fused_data = self.avgpool_img(fused_data)

        fused_features, lidar_features = self.transformer(fused_data, lidar_data)
        fused_features = fused_input_data + upsample(fused_features, fused_input_data.shape[2:4])
        lidar_features = lidar_input_data + upsample(lidar_features, lidar_input_data.shape[2:4])
        return lidar_features, fused_features

# class FusedLidarAttention(nn.Module): 
#     """
#         args: n_channels, n_head, attn_pdrop, resid_pdrop 

#         forward shape transformation: 
#             (B, C, H1, W1), (B, C, H2, W2) --> (B, C, H1, W1)
#     """
#     def __init__(self, n_channels, n_head, img_anchors, lidar_anchors, embd_pdrop=0.1) -> None:
#         super(FusedLidarAttention, self).__init__()

#         self.avgpool_img = nn.AdaptiveAvgPool2d(img_anchors)
#         self.avgpool_lidar = nn.AdaptiveAvgPool2d(lidar_anchors)

#         # self.f_fused_data = nn.Sequential(
#         #     nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
#         #     batch_norm(n_channels)
#         # )
#         # self.f_lidar_data = nn.Sequential(
#         #     nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False), 
#         #     batch_norm(n_channels)
#         # )

#         self.n_channels = n_channels
#         self.gpt_linear_layer_init_mean = 0.0
#         self.gpt_linear_layer_init_std  = 0.02
#         self.gpt_layer_norm_init_weight = 1.0

#         # positional embedding parameter (learnable), image
#         self.pos_emb = nn.Parameter(torch.zeros(1, img_anchors[0] * img_anchors[1], n_channels))
#         self.drop = nn.Dropout(embd_pdrop)

#         self.cross_attn_block = TransformerBlock(n_channels, n_head)
#         self.ln_f = nn.LayerNorm(n_channels)
        
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=self.gpt_linear_layer_init_mean, std=self.gpt_linear_layer_init_std)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(self.gpt_layer_norm_init_weight)    

#     def forward(self, lidar_data, fused_data, lidar_as_query=True): 
#         lidar_input_data = lidar_data
#         fused_initial_data = fused_data
#         lidar_data = self.avgpool_lidar(lidar_data)
#         fused_data = self.avgpool_img(fused_data)

#         lidar_h, lidar_w = lidar_data.shape[2:4]
#         fused_h, fused_w = fused_data.shape[2:4]

#         # fused_data = self.f_fused_data(fused_data)
#         # lidar_data = self.f_lidar_data(lidar_data)

#         batch = fused_data.shape[0]
#         fused_data = fused_data.view(batch, -1, fused_h, fused_w).permute(0,2,3,1).contiguous().view(batch, -1, self.n_channels)
#         lidar_data = lidar_data.view(batch, -1, lidar_h, lidar_w).permute(0,2,3,1).contiguous().view(batch, -1, self.n_channels)

#         # if lidar_as_query: 
#         #     out = self.cross_attn_block(lidar_data, fused_data, fused_data)
#         #     out = out.view(batch, lidar_h*lidar_w, self.n_channels).contiguous().view(batch, self.n_channels, lidar_h, lidar_w)
#         #     out = upsample(out, lidar_input_data.shape[2:4])
#         #     out = out + lidar_input_data
#         # else: 

#         # out = self.drop(self.pos_emb + fused_data)
#         out = self.cross_attn_block(fused_data, lidar_data)
#         out = self.ln_f(out)

#         out = out.view(batch, fused_h*fused_w, self.n_channels).contiguous().view(batch, self.n_channels, fused_h, fused_w)
#         out = upsample(out, fused_initial_data.shape[2:4])
#         out = out + fused_initial_data
#         return out

# class CameraCameraAttention(nn.Module): 
#     """
#         args: n_channels, n_head, attn_pdrop, resid_pdrop 

#         forward shape transformation: 
#             (B, C, H1, W1), (B, C, H2, W2) --> (B, C, H1, W1)
#     """
#     def __init__(self, n_channels, n_head, img_anchors) -> None:
#         super(CameraCameraAttention, self).__init__()

#         self.avgpool_img = nn.AdaptiveAvgPool2d(img_anchors)
#         self.transformer = GPT(n_channels, n_head, first_branch_anchors=img_anchors, second_branch_anchors=img_anchors)

#     def forward(self, camera_data, fused_data): 
#         camera_initial_data = camera_data
#         fused_initial_data = fused_data
#         fused_data = self.avgpool_img(fused_data)
#         camera_data = self.avgpool_img(camera_data)

#         fused_features, camera_features = self.transformer(fused_data, camera_data)
#         fused_features = fused_initial_data + upsample(fused_features, fused_initial_data.shape[2:4])
#         camera_features = camera_initial_data + upsample(camera_features, camera_initial_data.shape[2:4])
#         return camera_features, fused_features

class FusedCameraAttention(nn.Module): 
    """
        args: n_channels, n_head, attn_pdrop, resid_pdrop 

        forward shape transformation: 
            (B, C, H1, W1), (B, C, H2, W2) --> (B, C, H1, W1)
    """
    def __init__(self, n_channels, n_head, img_anchors, embd_pdrop=0.1) -> None:
        super(FusedCameraAttention, self).__init__()

        self.avgpool_img = nn.AdaptiveAvgPool2d(img_anchors)
        self.n_channels = n_channels

        self.cross_attn_block = CrossAttention(n_channels, n_head, in_channels=img_anchors[0]*img_anchors[1])
        # self.cross_attn_block2 = CrossAttention(n_channels, n_head)
        self.ln_f = nn.LayerNorm(n_channels)
        self.ln_f2 = nn.LayerNorm(n_channels)
        
    def forward(self, camera_data, fused_data): 
        camera_initial_data = camera_data
        fused_initial_data = fused_data
        fused_data = self.avgpool_img(fused_data)
        camera_data = self.avgpool_img(camera_data)

        camera_h, camera_w = camera_data.shape[2:4]
        # fused_data = self.f_fused_data(fused_data)
        # camera_data = self.f_camera_data(camera_data)

        batch = fused_data.shape[0]
        fused_data = fused_data.view(batch, -1, camera_h, camera_w).permute(0,2,3,1).contiguous().view(batch, -1, self.n_channels)
        camera_data = camera_data.view(batch, -1, camera_h, camera_w).permute(0,2,3,1).contiguous().view(batch, -1, self.n_channels)

        camera_out, fused_out = self.cross_attn_block(fused_data, camera_data)
        # camera_out = self.cross_attn_block2(camera_data, fused_data)

        fused_out = self.ln_f(fused_out)
        camera_out = self.ln_f2(camera_out)

        fused_out = fused_out.view(batch, camera_h*camera_w, self.n_channels).contiguous().view(batch, self.n_channels, camera_h, camera_w)
        fused_out = upsample(fused_out, fused_initial_data.shape[2:4])
        fused_out = fused_out + fused_initial_data
        camera_out = camera_out.view(batch, camera_h*camera_w, self.n_channels).contiguous().view(batch, self.n_channels, camera_h, camera_w)
        camera_out = upsample(camera_out, camera_initial_data.shape[2:4])
        camera_out = camera_out + camera_initial_data

        return camera_out, fused_out

from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if context != None:
            context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in

class MultiSpatialTransformer(nn.Module): 
    def __init__(self, in_channels, n_head, img_anchors) -> None:
        super().__init__()
        self.avgpool_img = nn.AdaptiveAvgPool2d(img_anchors)
        self.camera_spatial_att = SpatialTransformer(in_channels=in_channels, n_heads=n_head, d_head=in_channels//n_head)
        self.fused_spatial_att = SpatialTransformer(in_channels=in_channels, n_heads=n_head, d_head=in_channels//n_head)

    def forward(self, camera, fused): 
        camera_out = self.avgpool_img(camera)
        fused_out = self.avgpool_img(fused)
        camera_out = self.camera_spatial_att(camera_out, context=fused)
        fused_out = self.fused_spatial_att(fused_out, context=camera)
        camera_out = upsample(camera_out, camera.shape[2:4])
        fused_out = upsample(fused_out, fused.shape[2:4])

        return camera_out, fused_out


class SegDecoder(nn.Module):
    def __init__(self, in_channels, num_class):
        super().__init__()
        self.in_channels = in_channels
        self.deconv_channel_num_1 = 128 # Number of channels at the first deconvolution layer
        self.deconv_channel_num_2 = 64 # Number of channels at the second deconvolution layer
        self.deconv_channel_num_3 = 32 # Number of channels at the third deconvolution layer
        self.deconv_scale_factor_1 = 8 # Scale factor, of how much the grid size will be interpolated after the first layer
        self.deconv_scale_factor_2 = 4 # Scale factor, of how much the grid size will be interpolated after the second layer
        self.num_class = num_class

        self.deconv1 = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.deconv_channel_num_1, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.deconv_channel_num_1, self.deconv_channel_num_2, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv2 = nn.Sequential(
                    nn.Conv2d(self.deconv_channel_num_2, self.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.deconv_channel_num_3, self.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    )
        self.deconv3 = nn.Sequential(
                    nn.Conv2d(self.deconv_channel_num_3, self.deconv_channel_num_3, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(self.deconv_channel_num_3, self.num_class, 3, 1, 1),
                    )

    def forward(self, x):
        x = self.deconv1(x)
        x = F.interpolate(x, scale_factor=self.deconv_scale_factor_1, mode='bilinear', align_corners=False)
        x = self.deconv2(x)
        x = F.interpolate(x, scale_factor=self.deconv_scale_factor_2, mode='bilinear', align_corners=False)
        x = self.deconv3(x)

        return x