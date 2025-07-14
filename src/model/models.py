### datashape: [batch_size, window_length, n_fts, 137,121]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Combined_Spatial, TemporalExactor, PredictionHead, SpatialExactor2, TemporalExactorSTrans
from .stransformer import PatchEmbedding, PositionEmbedding, MHABlock, WindowMultiHeadAttention, UpsampleWithTransposedConv,SEResNet
import timm
from timm import create_model
from torchvision import transforms

class Model_Ver1(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if config.MODEL.SPATIAL.TYPE == 0:
            self.spatial_exactor = Combined_Spatial(in_channels=config.MODEL.IN_CHANNEL,
                                                out_channels=config.MODEL.SPATIAL.OUT_CHANNEL,
                                                kernel_sizes=config.MODEL.SPATIAL.KERNEL_SIZES,
                                                use_batch_norm=config.MODEL.SPATIAL.USE_BATCH_NORM)
            self.temporal_exactor = TemporalExactor(input_size= len(config.MODEL.SPATIAL.KERNEL_SIZES) * config.MODEL.SPATIAL.OUT_CHANNEL,
                                                hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                                                num_layers=config.MODEL.TEMPORAL.NUM_LAYERS)
        elif config.MODEL.SPATIAL.TYPE == 1:
            self.spatial_exactor = SpatialExactor2(in_channels=config.MODEL.IN_CHANNEL,
                                                   out_channels= config.MODEL.SPATIAL.OUT_CHANNEL,
                                                   kernel_size= 3,
                                                   use_batch_norm= config.MODEL.SPATIAL.USE_BATCH_NORM,
                                                   num_conv_layers= config.MODEL.SPATIAL.NUM_LAYERS)
            self.temporal_exactor = TemporalExactor(input_size= config.MODEL.SPATIAL.OUT_CHANNEL,
                                                    hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                                                    num_layers=config.MODEL.TEMPORAL.NUM_LAYERS)
            
    
        ### learnable params
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, config.MODEL.TEMPORAL.HIDDEN_DIM))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(config.MODEL.TEMPORAL.HIDDEN_DIM,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=config.MODEL.DROPOUT)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module)
        
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")
        
    def forward(self, x):
        """
        input shape: [batch size, n_ts, n_fts, 17, 17]
        """
        ncmwf = x[0]
        
        lead_time = x[1]
        spatial_embedding = self.spatial_exactor(ncmwf) ##[batch_size, window_length, n_kernels * out_channels, h, w]
        
        temporal_embedding = self.temporal_exactor(spatial_embedding) ## [batch_size, height, width, hidden_dim]
        # temporal_embedding =  
        output = self.add_prompt_vecs(temporal_embedding, lead_time) # B, H, W, D
        
        output = self.prediction_head(output)
        
        return output
    
class SwinTransformer(nn.Module):
    def __init__(self, config):
        super(SwinTransformer, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = config.MODEL.SWIN_TRANSFORMER.EMBED_DIM
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.hidden_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.hidden_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution

        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape
        
        # Combine time and feature dimensions
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)

        # Step 0: Pad the input to make h and w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 2: Position embedding
        x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 3: Reshape for window-based attention
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        x = x.view(batch_size * n_ts, h_patch, w_patch, self.embed_dim)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)

        # Step 4: Apply window-based multi-head attention
        x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
        
        ## Step 4.1 To-Do temporal-exactor 
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim)
        
        ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
        # Step 5: Upsample to original resolution
        x = self.upsample(x)  # (batch_size, h, w, embed_dim)
        x = x[:, :h, :w, :] # (batch_size, h, w, embed_dim)

        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)

        return x 
  
class Model_Ver2(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        if config.MODEL.SPATIAL.TYPE == 0:
            self.spatial_exactor = Combined_Spatial(in_channels=config.MODEL.IN_CHANNEL,
                                                out_channels=config.MODEL.SPATIAL.OUT_CHANNEL,
                                                kernel_sizes=config.MODEL.SPATIAL.KERNEL_SIZES,
                                                use_batch_norm=config.MODEL.SPATIAL.USE_BATCH_NORM)
            self.temporal_exactor = TemporalExactor(input_size= len(config.MODEL.SPATIAL.KERNEL_SIZES) * config.MODEL.SPATIAL.OUT_CHANNEL,
                                                hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                                                num_layers=config.MODEL.TEMPORAL.NUM_LAYERS)
        elif config.MODEL.SPATIAL.TYPE == 1:
            self.spatial_exactor = SpatialExactor2(in_channels=config.MODEL.IN_CHANNEL,
                                                   out_channels= config.MODEL.SPATIAL.OUT_CHANNEL,
                                                   kernel_size= 3,
                                                   use_batch_norm= config.MODEL.SPATIAL.USE_BATCH_NORM,
                                                   num_conv_layers= config.MODEL.SPATIAL.NUM_LAYERS)
            self.temporal_exactor = TemporalExactor(input_size= config.MODEL.SPATIAL.OUT_CHANNEL,
                                                    hidden_size=config.MODEL.TEMPORAL.HIDDEN_DIM,
                                                    num_layers=config.MODEL.TEMPORAL.NUM_LAYERS)
            
    
        ### learnable params
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, config.MODEL.TEMPORAL.HIDDEN_DIM))
        
        else:
            raise("Wrong prompt_type")
        
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)
        
        self.channel_attn2 = SEResNet(in_channels=192, out_channels=192, reduction_ratio=16)
        
        self.channel_attn3 = SEResNet(in_channels=128, out_channels=128, reduction_ratio=16)
        
        self.prediction_head = PredictionHead(config.MODEL.TEMPORAL.HIDDEN_DIM,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=config.MODEL.DROPOUT)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module)
        
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")
        
    def forward(self, x):
        """
        input shape: [batch size, n_ts, n_fts, 17, 17]
        """
        ncmwf = x[0]
        lead_time = x[1]
        
        batch_size, n_ts, n_ft, h, w = ncmwf.shape
        
        ncmwf = ncmwf.view(batch_size * n_ts, n_ft, h, w)
        ncmwf = self.channel_attn(ncmwf)
        ncmwf = ncmwf.reshape(batch_size, n_ts, n_ft, h, w)
        
        spatial_embedding = self.spatial_exactor(ncmwf) ##[batch_size, window_length, n_kernels * out_channels, h, w]
        
        # batch_size, window_length, n_out_channels, h, w = spatial_embedding.shape
        # X_reshaped = spatial_embedding.view(batch_size * window_length, n_out_channels, h, w)
        # X_reshaped = self.channel_attn2(X_reshaped)
        # X_reshaped = X_reshaped.view(batch_size, window_length, n_out_channels, h, w)
        
        temporal_embedding = self.temporal_exactor(spatial_embedding) ## [batch_size, height, width, hidden_dim]
        # temporal_embedding = self.temporal_exactor(X_reshaped) ## [batch_size, height, width, hidden_dim]
        
        # temporal_embedding =  
        output = self.add_prompt_vecs(temporal_embedding, lead_time) # B, H, W, D
        
        output = self.prediction_head(output)
        
        return output
    
class SwinTransformer_Ver2(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver2, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        self.embed_dim = config.MODEL.SWIN_TRANSFORMER.EMBED_DIM
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.hidden_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.hidden_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution
        
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)
        
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape

        # Combine time and feature dimensions
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)
        
        # print(x.shape, "--------------------")
        x = self.channel_attn(x)

        # Step 0: Pad the input to make h and w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 2: Position embedding
        x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 3: Reshape for window-based attention
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        x = x.view(batch_size * n_ts, h_patch, w_patch, self.embed_dim)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
       
        # Step 4: Apply window-based multi-head attention
        x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
        ## Step 4.1 To-Do temporal-exactor 
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim)
        
        ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
        # Step 5: Upsample to original resolution
        x = self.upsample(x)  # (batch_size, h, w, embed_dim)
        x = x[:, :h, :w, :] # (batch_size, h, w, embed_dim)

        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)

        return x
    
class SwinTransformer_Ver3(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver3, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        # self.embed_dim = config.MODEL.SWIN_TRANSFORMER.EMBED_DIM
        # self.embed_dim = 768
        self.embed_dim = 192
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        
        
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.hidden_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.hidden_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution
        
        # vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        self.spatial_encoder = vit.blocks
        for param in self.spatial_encoder.parameters():
                param.requires_grad = False
        for blk in self.spatial_encoder[-3:]:
            for param in blk.parameters():
                param.requires_grad = True

        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:    
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    # print(lt)
                    assert lt < len(self.delta_t), f"lead_time {lt} out of range"
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape
        
        # Combine time and feature dimensions
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)

        # Step 0: Pad the input to make h and w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 2: Position embedding
        x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 3: Reshape for window-based attention
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        # resize_transform = transforms.Resize((224, 224))
        # input_tensor_resized = resize_transform(x)
        x = self.spatial_encoder(x)
        # Step 4: Apply window-based multi-head attention
        # x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
        
        
        ## Step 4.1 To-Do temporal-exactor 
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim)
        
        ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
        # Step 5: Upsample to original resolution
        x = self.upsample(x)  # (batch_size, h, w, embed_dim)
        x = x[:, :h, :w, :] # (batch_size, h, w, embed_dim)

        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)

        return x 
    
class SwinTransformer_Ver4(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver4, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        # self.embed_dim = config.MODEL.SWIN_TRANSFORMER.EMBED_DIM
        # self.embed_dim = 768
        self.embed_dim = 192
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        
        
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.hidden_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.hidden_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)
        
        # vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        self.spatial_encoder = vit.blocks
        for param in self.spatial_encoder.parameters():
                param.requires_grad = False
        for blk in self.spatial_encoder[-3:]:
            for param in blk.parameters():
                param.requires_grad = True
        
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    lt = int(lt)
                    lt -= 7
                    # print(lt)
                    assert lt < len(self.delta_t), f"lead_time {lt} out of range"
                    
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return temporal_embedding + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")

    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape
        
        # Combine time and feature dimensions
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)
        
        x = self.channel_attn(x)

        # Step 0: Pad the input to make h and w divisible by patch_size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 2: Position embedding
        x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)

        # Step 3: Reshape for window-based attention
        h_patch = padded_h // self.patch_size
        w_patch = padded_w // self.patch_size
        # resize_transform = transforms.Resize((224, 224))
        # input_tensor_resized = resize_transform(x)
        x = self.spatial_encoder(x)
        # Step 4: Apply window-based multi-head attention
        # x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
        
        
        
        
        ## Step 4.1 To-Do temporal-exactor 
        x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim)
        
        ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
        # Step 5: Upsample to original resolution
        x = self.upsample(x)  # (batch_size, h, w, embed_dim)
        x = x[:, :h, :w, :] # (batch_size, h, w, embed_dim)

        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)

        return x 
  
class SwinTransformer_Ver5(nn.Module):
    def __init__(self, config):
        super(SwinTransformer_Ver5, self).__init__()
        self.config = config
        self.patch_size = config.MODEL.PATCH_SIZE
        # self.embed_dim = config.MODEL.SWIN_TRANSFORMER.EMBED_DIM
        # self.embed_dim = 768
        self.embed_dim = 32
        self.hidden_dim = config.MODEL.TEMPORAL.HIDDEN_DIM
        self.num_layers = config.MODEL.SWIN_TRANSFORMER.NUM_LAYERS
        self.dropout = config.MODEL.DROPOUT
        self.shift_window = self.patch_size // 2
        self.num_swin_block = 2
        self.patch_embed = PatchEmbedding(self.patch_size, config.MODEL.IN_CHANNEL, self.embed_dim)
        self.patch_embed1 = PatchEmbedding(self.patch_size, self.embed_dim, self.embed_dim)
        self.window_attention = WindowMultiHeadAttention(self.embed_dim, config.MODEL.SWIN_TRANSFORMER.WINDOW_SIZE, 
                                                         config.MODEL.SWIN_TRANSFORMER.NUM_HEADS,
                                                         self.num_layers, config.MODEL.SWIN_TRANSFORMER.FF_DIM, self.dropout)
        self.temporal_exactor = TemporalExactorSTrans(self.embed_dim, self.embed_dim, self.num_layers)
        num_patches = self.cal_num_patches([self.config.DATA.HEIGHT, self.config.DATA.WIDTH])
        
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(self.embed_dim, self.embed_dim, scale_factor=self.patch_size)  # Upsample with transposed convolution
        
        self.channel_attn = SEResNet(in_channels=config.MODEL.IN_CHANNEL, out_channels=config.MODEL.IN_CHANNEL, reduction_ratio=2)
        
        # vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        vit = timm.create_model("vit_tiny_patch16_224", pretrained=True)
        self.spatial_encoder = vit.blocks
        for param in self.spatial_encoder.parameters():
                param.requires_grad = False
        for blk in self.spatial_encoder[-3:]:
            for param in blk.parameters():
                param.requires_grad = True
        
        self.prompt_type = config.MODEL.PROMPT_TYPE
        self.add_type = config.MODEL.TEMPORAL.ADDING_TYPE
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(config.MODEL.TEMPORAL.MAX_DELTA_T, self.hidden_dim))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(self.embed_dim,
                                              use_layer_norm=config.MODEL.USE_LAYER_NORM,
                                              dropout=self.dropout)

    def cal_num_patches(self, img_size):
        h, w = img_size[0], img_size[1]
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        padded_h, padded_w = h + pad_h, w + pad_w
        num_patches = (padded_h // self.patch_size) * (padded_w // self.patch_size)
        return num_patches
    
    def add_prompt_vecs(self, temporal_embedding, lead_time):
        list_prompt = []
        device = temporal_embedding.device  # Lấy thiết bị của temporal_embedding

        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    lt = int(lt)
                    lt -= 7
                    assert lt < len(self.delta_t), f"lead_time {lt} out of range"
                    
                    corress_prompt = self.delta_t[lt].to(device)  # Đảm bảo delta_t nằm trên cùng một device với temporal_embedding
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
                    corress_prompt = corress_prompt.expand(H, W, -1)  # Thêm chiều H và W
                    
                    # Nếu corress_prompt có chiều cuối cùng khác D, ta cần thay đổi nó để khớp với D
                    if corress_prompt.size(-1) != D:
                        # Sử dụng Linear để thay đổi kích thước của corress_prompt từ hidden_dim thành D
                        linear_layer = nn.Linear(corress_prompt.size(-1), D).to(device)  # Chuyển Linear lên cùng thiết bị
                        corress_prompt = linear_layer(corress_prompt)

                    list_prompt.append(corress_prompt)

                add_prompt = torch.stack(list_prompt, 0).to(device)  # Đảm bảo add_prompt nằm trên cùng một device
                return temporal_embedding + add_prompt  # Cộng temporal_embedding và add_prompt
            elif self.add_type == 1:
                for lt in lead_time:
                    lt -= 7
                    corress_prompt = self.delta_t[lt].to(device)
                    B, H, W, D = temporal_embedding.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt, 0).to(device)

                return torch.concat([temporal_embedding, add_prompt], -1)
            else:
                raise("Wrong adding type value")
        else:
            raise("Wrong prompt type value")


    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        batch_size, n_ts, n_ft, h, w = x.shape
        
        # Combine time and feature dimensions
        x = x.view(batch_size * n_ts, n_ft, h, w)  # (batch_size * n_ts, n_ft, h, w)
        
        x = self.channel_attn(x)
        
        for i in range(self.num_swin_block):
            
            x = x.reshape(batch_size * n_ts, -1, h, w)  # (batch_size * n_ts, n_ft, h, w)
            # Step 0: Pad the input to make h and w divisible by patch_size
            pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
            pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
            x = torch.roll(x, (self.shift_window, self.shift_window), dims = (-2, -1))
            self.shift_window = -self.shift_window
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')  # Pad (left, right, top, bottom)
            padded_h, padded_w = h + pad_h, w + pad_w
            
            if i == 0:
            # Step 1: Patch embedding
                x = self.patch_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)
            else:
                x = self.patch_embed1(x)
            # Step 2: Position embedding
            x = self.pos_embed(x)  # (batch_size * n_ts, num_patches, embed_dim)
            
            # Step 3: Reshape for window-based attention
            h_patch = padded_h // self.patch_size
            w_patch = padded_w // self.patch_size
            # resize_transform = transforms.Resize((224, 224))
            # input_tensor_resized = resize_transform(x)
            
            # x = self.spatial_encoder(x)
            
            # Step 4: Apply window-based multi-head attention
            x = x.reshape(-1, h_patch, w_patch, self.embed_dim)
            x = self.window_attention(x)  # (batch_size * n_ts, h_patch, w_patch, embed_dim)
            
            
            
            
            ## Step 4.1 To-Do temporal-exactor 
            x = x.reshape(batch_size, n_ts, h_patch, w_patch, -1) # (batch_size, n_ts, h_patch, w_patch, embed_dim)
            #x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim)
            
            ## Step 4.2 To-do adding delta_t the expected output shape is : batch, h_patch, w_patch, embed_dim
            #x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        
            device = x.device
            x_large = torch.zeros(x.shape[0], x.shape[1], x.shape[2] * self.patch_size, x.shape[3] * self.patch_size, x.shape[4]).to(device)
            # Step 5: Upsample to original resolution
            
            for j in range(x.shape[1]):
                x_large[:,j,:,:,:] = self.upsample(x[:,j,:,:,:])
            
            x = x_large
            #x = self.upsample(x)  # (batch_size, n_ts, h, w, embed_dim)
            x = x[:, :, :h, :w, :] # (batch_size, n_ts, h, w, embed_dim)
            
        
        x = self.temporal_exactor(x) # (batch_size, h_patch, w_patch, embed_dim)
        
        x = self.add_prompt_vecs(x, lead_time) # (batch_size, h_patch, w_patch, embed_dim)
        # Step 6: To-Do add prediction head on it
        x = self.prediction_head(x) # (batch_size, h, w)

        return x 

#######################################################################################
import torch
import torch.nn as nn
import torch.fft
from .layers import PredictionHead





# Cập nhật: giữ lại cả thành phần phát hiện ngoại lệ nhưng vẫn trả ra output [B, H, W]
import torch
import torch.nn as nn


import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True, dropout=0.0):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(4 * hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.hidden_channels = hidden_channels

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.norm(self.conv(combined))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        h_next = self.dropout(h_next)
        return h_next, c_next


class GradientHighwayUnit(nn.Module):
    def __init__(self, x_channels, h_channels, kernel_size, dropout=0.0):
        super(GradientHighwayUnit, self).__init__()
        padding = kernel_size // 2
        self.z_gate = nn.Conv2d(x_channels + h_channels, h_channels, kernel_size, padding=padding)
        self.h_transform = nn.Conv2d(h_channels, h_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(h_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        z = torch.sigmoid(self.z_gate(torch.cat([x, h], dim=1)))
        h_tilde = torch.tanh(self.norm(self.h_transform(h)))
        h_tilde = self.dropout(h_tilde)
        h_next = (1 - z) * h + z * h_tilde
        return h_next


class AnomalyDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout=0.0):
        super(AnomalyDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(x)


class PredRNNPlusPlusEnhanced(nn.Module):
    def __init__(self, config):
        super(PredRNNPlusPlusEnhanced, self).__init__()
        self.num_layers = config.MODEL.PRED.NUM_LAYERS
        self.hidden_dim = config.MODEL.PRED.HIDDEN_DIM
        self.kernel_size = 3
        self.input_channels = config.MODEL.IN_CHANNEL
        self.dropout_rate = config.MODEL.DROPOUT

        self.lstm_cells = nn.ModuleList()
        self.ghus = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for i in range(self.num_layers):
            in_ch = self.input_channels if i == 0 else self.hidden_dim
            self.lstm_cells.append(ConvLSTMCell(in_ch, self.hidden_dim, self.kernel_size, dropout=self.dropout_rate))
            self.ghus.append(GradientHighwayUnit(in_ch, self.hidden_dim, self.kernel_size, dropout=self.dropout_rate))
            self.residuals.append(nn.Conv2d(in_ch, self.hidden_dim, kernel_size=1) if in_ch != self.hidden_dim else nn.Identity())

        self.decoder = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.anomaly_decoder = AnomalyDecoder(self.hidden_dim, self.hidden_dim // 2, dropout=self.dropout_rate)

    def forward(self, x):
        x = x[0]
        B, T, C, H, W = x.shape
        h, c, ghu = [], [], []

        for _ in range(self.num_layers):
            h.append(torch.zeros(B, self.hidden_dim, H, W, device=x.device))
            c.append(torch.zeros(B, self.hidden_dim, H, W, device=x.device))
            ghu.append(torch.zeros_like(h[-1]))

        for t in range(T):
            input_t = x[:, t]
            for i in range(self.num_layers):
                residual = self.residuals[i](input_t)
                h[i], c[i] = self.lstm_cells[i](input_t, h[i], c[i])
                ghu[i] = self.ghus[i](input_t, ghu[i])
                input_t = h[i] + ghu[i] + residual

        y_pred = self.decoder(h[-1])              # [B, 1, H, W]
        anomaly_map = self.anomaly_decoder(h[-1]) # [B, 1, H, W]
        final_output = y_pred + anomaly_map
        final_output = final_output.permute(0, 2, 3, 1)  # [B, H, W, 1]
        return final_output

