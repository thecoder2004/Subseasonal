import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TemporalExactorSTrans, PredictionHead

# 1. Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input shape: (batch_size, in_channels, h, w)
        x = self.proj(x)  # (batch_size, embed_dim, h_patch, w_patch)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

# 2. Position Embedding
class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        # Add position embeddings to patch embeddings
        return x + self.pos_embed

# 3. Multi-Head Self-Attention Block
class MHABlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(MHABlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Head Attention
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_output)

        # Feed-Forward Network
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout(ffn_output)

        return x

# 4. Window-Based Multi-Head Attention
class WindowMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, window_size, num_heads, num_layers, ff_dim, dropout=0.1):
        super(WindowMultiHeadAttention, self).__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Multi-Head Attention blocks
        self.mha_blocks = nn.ModuleList([
            MHABlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        batch_size, h, w, embed_dim = x.shape

        # Step 0: Pad the input if necessary
        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # Pad (left, right, top, bottom)
        padded_h, padded_w = h + pad_h, w + pad_w
        
        # Step 1: Divide into non-overlapping windows
        x = x.view(batch_size, padded_h // self.window_size, self.window_size, padded_w // self.window_size, self.window_size, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (batch_size, h//ws, w//ws, ws, ws, embed_dim)
        x = x.view(batch_size, -1, self.window_size * self.window_size, embed_dim)  # (batch_size, num_windows, ws*ws, embed_dim)

        # Step 2: Reshape for Multi-Head Attention
        x = x.view(-1, self.window_size * self.window_size, embed_dim)  # (batch_size * num_windows, ws*ws, embed_dim)

        # Step 3: Apply multiple MHA layers
        for mha_block in self.mha_blocks:
            x = mha_block(x)

        # Step 4: Reshape back
        x = x.view(batch_size, padded_h // self.window_size, padded_w // self.window_size, self.window_size, self.window_size, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (batch_size, h//ws, ws, w//ws, ws, embed_dim)
        x = x.view(batch_size, padded_h, padded_w, embed_dim)  # (batch_size, padded_h, padded_w, embed_dim)

        # Step 5: Remove padding (if any)
        x = x[:, :h, :w, :]  # Crop back to the original size

        return x

# Add channel Attention 
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2) # (batch_size, num_channels)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor)) # (batch_size, num_channels_reduced)
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1)) # (batch_size, num_channels)

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class SEResNet(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=2):
        super(SEResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = ChannelSELayer(out_channels, reduction_ratio)

    def forward(self, x):
        identity = x # (batch_size * n_ts, n_ft, h, w)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)  # Apply SE block here

        out += identity  # Residual connection (optional)
        out = self.relu(out)

        return out
    
# 5. Upsampling with Transposed Convolution
class UpsampleWithTransposedConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(UpsampleWithTransposedConv, self).__init__()
        self.scale_factor = scale_factor  # Upsampling factor (e.g., 8 for 16x16 -> 128x128)
        self.in_channels = in_channels  # Input channels
        self.out_channels = out_channels  # Output channels

        # Transposed convolution for upsampling
        self.transposed_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,  # Keep the same number of channels
            kernel_size=scale_factor,
            stride=scale_factor,
            padding=0,
        )

        # 1x1 convolution to adjust the number of channels
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Input shape: (batch_size, h, w, in_channels)
        x = x.permute(0, 3, 1, 2)  # (batch_size, in_channels, h, w)

        # Step 1: Apply transposed convolution
        x = self.transposed_conv(x)  # (batch_size, in_channels, h * scale_factor, w * scale_factor)

        # Step 2: Adjust the number of channels
        x = self.conv1x1(x)  # (batch_size, out_channels, h * scale_factor, w * scale_factor)

        # Step 3: Permute back to (batch_size, h * scale_factor, w * scale_factor, out_channels)
        x = x.permute(0, 2, 3, 1)
        return x

# 6. Swin Transformer
class SwinTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, ff_dim, window_size, prompt_type, add_type, max_delta_t, use_layer_norm, dropout=0.1):
        super(SwinTransformer, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.window_attention = WindowMultiHeadAttention(embed_dim, window_size, num_heads, num_layers, ff_dim, dropout)
        self.temporal_exactor = TemporalExactorSTrans(embed_dim, embed_dim, num_layers)
        num_patches = self.cal_num_patches(img_size)
        self.pos_embed = PositionEmbedding(num_patches, self.embed_dim)
        self.upsample = UpsampleWithTransposedConv(embed_dim, embed_dim, scale_factor=patch_size)  # Upsample with transposed convolution

        self.prompt_type = prompt_type
        self.add_type = add_type
        if prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(max_delta_t, embed_dim))
        
        else:
            raise("Wrong prompt_type")
        
        self.prediction_head = PredictionHead(embed_dim,
                                              use_layer_norm=use_layer_norm,
                                              dropout=dropout)

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

# # Example usage
# patch_size = 16  # Patch size
# in_channels = 13  # Input channels (e.g., RGB)
# embed_dim = 128  # Embedding dimension
# num_heads = 8  # Number of attention heads
# num_layers = 4  # Number of transformer layers
# ff_dim = 256  # Feed-forward dimension
# window_size = 7  # Window size (default in Swin Transformer)
# prompt_type = 0
# add_type = 0
# max_delta_t = 40
# use_layer_norm = True
    
# # Input tensor: (batch_size, n_ts, n_ft, h, w)
# x = torch.randn(32, 7, in_channels, 137, 121)  # h = 137, w = 121
# lead_time = torch.randint(7, 47, (32,))
# # Swin Transformer model
# model = SwinTransformer([137, 121], patch_size, in_channels, embed_dim, num_heads, num_layers, ff_dim, window_size, prompt_type, add_type, max_delta_t, use_layer_norm)
# out = model([x, lead_time])  # Output shape: (batch_size, n_ts, h, w, embed_dim)
# print(out.shape)  # Should print: torch.Size([32, 137, 121, 1])