# Written by Jiacong Xu (jiacong.xu@tamu.edu)
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

bn_mom = 0.1
algc = False
upsample = lambda x, size : F.interpolate(x, size, mode="bilinear", align_corners=algc)
batch_norm = nn.BatchNorm2d

class BasicBlock(nn.Module): 
    """
        args : inplanes, outplanes

        forward shape transformation: 
            (B, inplanes, H, W)  --->  (B, outplanes, H, W)
    """
    expansion = 1 # No expansion
    def __init__(self, inplanes, outplanes, stride=1, apply_relu=True) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = batch_norm(outplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = batch_norm(outplanes, momentum=bn_mom)

        self.downsample = None
        if stride != 1 or inplanes != outplanes * self.expansion: 
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes * self.expansion, kernel_size=1, stride=stride, bias=False), 
                batch_norm(outplanes * self.expansion, momentum=bn_mom)
            )
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
    """
        args: inplanes, outplanes

        forward shape transformation: 
            (B, inplanes, H, W) --> (B, outplanes * expansion, H, W)
    """
    expansion = 2

    def __init__(self, inplanes, outplanes, stride=1, apply_relu=False) -> None:
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = batch_norm(outplanes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = batch_norm(outplanes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(outplanes, outplanes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = batch_norm(outplanes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or inplanes != outplanes * self.expansion: 
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes * self.expansion, kernel_size=1, stride=stride, bias=False), 
                batch_norm(outplanes * self.expansion, momentum=bn_mom)
            )

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

class TransformerBlock(nn.Module): 
    def __init__(self, n_embd, n_head, block_exp=4, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        # positional embedding parameter (learnable), only for query
        # self.pos_emb = nn.Parameter(torch.zeros(1, query_size, n_embd))

        self.attn = MultiHeadAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),
            nn.Dropout(resid_pdrop),
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, q, k=None, v=None):
        x = self.ln1(self.attn(q, k, v) + q)
        x = self.ln2(self.mlp(x) + x)
        return x


class GPT(nn.Module):
    def __init__(self, n_embd, n_head, embd_pdrop, attn_pdrop, resid_pdrop,
                    first_branch_anchors, second_branch_anchors):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1
        
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
        self.block = TransformerBlock(n_embd=n_embd, n_head=n_head)

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
        x = self.block(x)

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
    def __init__(self, n_channels, n_head, img_anchors, lidar_anchors, 
                apply_relu_first=False) -> None:
        super(FusedLidarAttention, self).__init__()

        self.apply_relu_first = apply_relu_first

        self.avgpool_img = nn.AdaptiveAvgPool2d(img_anchors)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d(lidar_anchors)

        self.f_fused_data = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
            batch_norm(n_channels)
        )
        self.f_lidar_data = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False), 
            batch_norm(n_channels)
        )
        self.f_fused_collapsed = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=1, bias=False)
        )
        if apply_relu_first: 
            self.relu = nn.ReLU(inplace=True)
        
        self.n_channels = n_channels
        self.multihead_attn_block = TransformerBlock(n_channels, n_head)
        # self.transformer = GPT(n_channels, n_head, block_exp=4, embd_pdrop=attn_pdrop, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, 
        #     first_branch_anchors=img_anchors, second_branch_anchors=lidar_anchors)

    # def collapse_image_for_bev(self, image): 
    #     batch_size, num_channel, img_h, img_w = image.shape
    #     img_collapsed = image.max(2).values # [BS, C, W]
    #     img_collapsed = self.f_fused_collapsed(img_collapsed).view(batch_size, num_channel, img_w)
    #     return img_collapsed

    def forward(self, lidar_data, fused_data, lidar_as_query=True): 
        lidar_input_data = lidar_data
        fused_input_data = fused_data
        lidar_data = self.avgpool_lidar(lidar_data)
        fused_data = self.avgpool_img(fused_data)

        lidar_h, lidar_w = lidar_data.shape[2:4]
        fused_h, fused_w = fused_data.shape[2:4]
        if self.apply_relu_first:
            lidar_data = self.relu(lidar_data) 
            fused_data = self.relu(fused_data)

        fused_data = self.f_fused_data(fused_data)
        lidar_data = self.f_lidar_data(lidar_data)
        # fused_data = upsample(fused_data, [fused_h, lidar_w])
        # fused_data = self.collapse_image_for_bev(fused_data)

        batch = fused_data.shape[0]
        fused_data = fused_data.view(batch, -1, fused_h, fused_w).permute(0,2,3,1).contiguous().view(batch, -1, self.n_channels)
        lidar_data = lidar_data.view(batch, -1, lidar_h, lidar_w).permute(0,2,3,1).contiguous().view(batch, -1, self.n_channels)

        if lidar_as_query: 
            out = self.multihead_attn_block(lidar_data, fused_data, fused_data)
            out = out.view(batch, lidar_h*lidar_w, self.n_channels).contiguous().view(batch, self.n_channels, lidar_h, lidar_w)
            out = upsample(out, lidar_input_data.shape[2:4])
            out = out + lidar_input_data
        else: 
            out = self.multihead_attn_block(fused_data, lidar_data, lidar_data)
            out = out.view(batch, fused_h*fused_w, self.n_channels).contiguous().view(batch, self.n_channels, fused_h, fused_w)
            out = upsample(out, fused_input_data.shape[2:4])
            out = out + fused_input_data
        return out
        # fused_features, lidar_features = self.transformer(fused_data, lidar_data)
        # fused_features = fused_input_data + upsample(fused_features, fused_input_data.shape[2:4])
        # lidar_features = lidar_input_data + upsample(lidar_features, lidar_input_data.shape[2:4])
        # return lidar_features, fused_features

class CameraCameraAttention(nn.Module): 
    """
        args: n_channels, n_head, attn_pdrop, resid_pdrop 

        forward shape transformation: 
            (B, C, H1, W1), (B, C, H2, W2) --> (B, C, H1, W1)
    """
    def __init__(self, n_channels, n_head, img_anchors,
                apply_relu_first=False) -> None:
        super(CameraCameraAttention, self).__init__()

        self.avgpool_img = nn.AdaptiveAvgPool2d(img_anchors)

        self.apply_relu_first = apply_relu_first
        self.f_fused_data = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False),
            batch_norm(n_channels)
        )
        self.f_camera_data = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1, bias=False), 
            batch_norm(n_channels)
        )
        if apply_relu_first: 
            self.relu = nn.ReLU(inplace=True)

        self.n_channels = n_channels
        self.multihead_attn_block = TransformerBlock(n_channels, n_head)
        # self.transformer = GPT(n_channels, n_head, block_exp=4, embd_pdrop=attn_pdrop, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop, 
        #     first_branch_anchors=img_anchors, second_branch_anchors=img_anchors)

    def forward(self, camera_data, fused_data, camera_as_query = True): 
        camera_initial_data = camera_data
        fused_initial_data = fused_data
        fused_data = self.avgpool_img(fused_data)
        camera_data = self.avgpool_img(camera_data)

        camera_h, camera_w = camera_data.shape[2:4]
        if self.apply_relu_first:
            camera_data = self.relu(camera_data) 
            fused_data = self.relu(fused_data)

        fused_data = self.f_fused_data(fused_data)
        camera_data = self.f_camera_data(camera_data)

        batch = fused_data.shape[0]
        fused_data = fused_data.view(batch, -1, camera_h, camera_w).permute(0,2,3,1).contiguous().view(batch, -1, self.n_channels)
        camera_data = camera_data.view(batch, -1, camera_h, camera_w).permute(0,2,3,1).contiguous().view(batch, -1, self.n_channels)

        if camera_as_query: 
            out = self.multihead_attn_block(camera_data, fused_data, fused_data)
            out = out.view(batch, camera_h*camera_w, self.n_channels).contiguous().view(batch, self.n_channels, camera_h, camera_w)
            out = upsample(out, camera_initial_data.shape[2:4])
            out = out + camera_initial_data
        else: 
            out = self.multihead_attn_block(fused_data, camera_data, camera_data)
            out = out.view(batch, camera_h*camera_w, self.n_channels).contiguous().view(batch, self.n_channels, camera_h, camera_w)
            out = upsample(out, fused_initial_data.shape[2:4])
            out = out + fused_initial_data
        return out
        # fused_features, camera_features = self.transformer(fused_data, camera_data)
        # fused_features = fused_initial_data + upsample(fused_features, fused_initial_data.shape[2:4])
        # camera_features = camera_initial_data + upsample(camera_features, camera_initial_data.shape[2:4])
        # return camera_features, fused_features



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
            batch_norm(in_channels),
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
            batch_norm(out_channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            batch_norm(out_channels),
        )
        self.f_lidar_converted = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
        )

    
    def lidar_bev_to_image(self, lidar_bev, image_h): 
        lidar_collapsed = lidar_bev.max(2).values.unsqueeze(2) # [B, C, 1, W]
        lidar_extended = torch.cat([lidar_collapsed]*image_h, axis=2) # [B, C, H, W]
        lidar_extended = self.f_lidar_converted(lidar_extended)
        return lidar_extended


    def forward(self, p, i, d): 
        B, C, image_h, image_w = i.shape
        p = self.lidar_bev_to_image(p, image_h)

        boundary_attr = torch.sigmoid(d) 
        p_add = self.conv_p(boundary_attr * p + i) 
        i_add = self.conv_i((1 - boundary_attr) * i + p)
        
        out = p_add + i_add
        return out


class LightBagLidar(nn.Module): 
    """
    LightBag convert 3d convolutions to 1d convolutions in Bag to make it faster. Also it slightly changes
    the forward part to suit this new convolution. 
    """
    def __init__(self, in_channels, out_channels, img_anchors, lidar_anchors) -> None:
        super(LightBagLidar, self).__init__()
        self.conv_lidar = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            batch_norm(out_channels),
        )
        self.conv_camera = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            batch_norm(out_channels),
        )
        self.lidar_fused_attention = FusedLidarAttention(in_channels, n_head=4, img_anchors=img_anchors, lidar_anchors=lidar_anchors)
        self.camera_fused_attention = CameraCameraAttention(in_channels, n_head=4, img_anchors=img_anchors)


    def forward(self, lidar, fused, camera):         
        lidar_add = self.conv_lidar(self.lidar_fused_attention(lidar, fused, lidar_as_query=False)) 
        camera_add = self.conv_camera(self.camera_fused_attention(camera, fused, camera_as_query=False))
        
        out = lidar_add + camera_add
        return out

if __name__ == '__main__':
    # test models
    B, C, H1, W1, H2, W2 = 5, 8, 10, 15, 20, 24
    H, W = 480, 640
    n_embd = 16
    block_exp = 4
    n_head = 4
    resid_pdrop = 0.1
    attn_pdrop = 0.1


    x = torch.randn(B, C, H1, W1)
    y = torch.randn(B, C, H2, W2)
    z = torch.randn(B, 3, H, W)

    pag = Pag(C, 2*C)
    att1 = FusedLidarAttention(C, n_head, attn_pdrop, resid_pdrop)
    att2 = CameraCameraAttention(C, n_head, attn_pdrop, resid_pdrop)
    dappm = DAPPM(C, 96, 4)

    basic_block = BasicBlock(C, 2*C)
    bottleneck = Bottleneck(C, 2*C)

    initial_layer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
        batch_norm(64, momentum=bn_mom), 
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        batch_norm(64, momentum=bn_mom), 
        nn.ReLU(inplace=True)
    )

    print(z.shape)
    z = initial_layer(z) 
    print(z.shape)

    # x = att2(x, y)
    # x = dappm(x)
    # print(x.shape)