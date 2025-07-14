import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

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
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)  # Apply SE block here

        out += identity  # Residual connection (optional)
        out = self.relu(out)

        return out
    
class CNN_LSTM(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=args.n_f, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #LSTM layers
        self.lstm_input_size = (64 * (args.height // 2) * (args.width // 2))  # After conv + pooling
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim, 
                            num_layers=args.lstm_layers, batch_first=True)
        self.fc = nn.Linear(in_features=args.hidden_dim, out_features=args.height * args.width)
        
        ### learnable params
        self.prompt_type = args.prompt_type
        self.add_type = args.adding_type
        
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(args.max_delta_t, args.temporal_hidden_size))
        
        else:
            raise("Wrong prompt_type")
        
    def add_prompt_vecs(self, out, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = out.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return out + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = out.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([out, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")    
        
    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        # Shape of x is (batch_size, n_t, n_f, h, w)
        batch_size, n_t, n_f, h, w = x.shape
        
        cnn_out = []
        for t in range(n_t):
            #Shape of x[:, t] is (batch_size, n_f, h, w)
            out = self.conv1(x[:, t]) # Shape: (batch_size, 32, h, w)
            out = F.relu(out)
            out = self.conv2(out)  # Shape: (batch_size, 64, h, w)
            out = F.relu(out)
            out = self.pool(out)  # Shape: (batch_size, 64, h//2, w//2)
            
            out = out.view(batch_size, -1)  # Flatten
            cnn_out.append(out)
            
        cnn_out = torch.stack(cnn_out, dim=1)  # Shape: (batch_size, n_t, flattened_size)
        
        lstm_out, _ = self.lstm(cnn_out)  # Shape: (batch_size, n_t, hidden_dim)
        lstm_out = torch.sum(lstm_out, dim=1)  # Sum over n_t (batch_size, hidden_dim)
        
        out = self.fc(lstm_out)  # Shape: (batch_size, h*w)
        out = out.view(batch_size, h, w)  # Reshape to (batch_size, h, w)
        out = out.unsqueeze(-1)
        output = self.add_prompt_vecs(out, lead_time)
        
        return output
    

class CNN_LSTM_SE(nn.Module):
    def __init__(self, args):
        super(CNN_LSTM_SE, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Squeeze-and-Excitation Module
        self.channel_attn = SEResNet(in_channels=13, out_channels=13, reduction_ratio=2)  # Apply SEModule after conv2
        
        #LSTM layers
        # self.lstm_input_size = (64 * (args.height // 2) * (args.width // 2))  # After conv + pooling
        self.lstm_input_size = (64 * (17 // 2) * (17 // 2))  # After conv + pooling
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=args.hidden_dim, 
                            num_layers=args.lstm_layers, batch_first=True)
        self.fc = nn.Linear(in_features=args.hidden_dim, out_features=17 * 17)
        
        ### learnable params
        self.prompt_type = args.prompt_type
        self.add_type = args.adding_type
        
        if self.prompt_type == 0:
            
            self.delta_t = nn.Parameter(torch.randn(args.max_delta_t, args.temporal_hidden_size))
        
        else:
            raise("Wrong prompt_type")
        
    def add_prompt_vecs(self, out, lead_time):
        list_prompt = []
        if self.prompt_type == 0:
            if self.add_type == 0:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = out.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return out + add_prompt
            

            elif self.add_type == 1:
                for lt in lead_time:
                    # lt = int(lt)
                    lt -= 7
                    corress_prompt = self.delta_t[lt]
                    B, H, W, D = out.shape
                    corress_prompt = corress_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, channels]
                    corress_prompt = corress_prompt.expand(H, W, -1)
                    list_prompt.append(corress_prompt)
                add_prompt = torch.stack(list_prompt,0)
                
                return torch.concat([out, add_prompt], -1)
            else:
                raise("Wrong adding type value")
            
        else:
            raise("Wrong prompt type value")    
        
    def forward(self, x):
        lead_time = x[1]
        x = x[0]
        # Shape of x is (batch_size, n_t, n_f, h, w)
        batch_size, n_t, n_f, h, w = x.shape
        
        x = x.view(batch_size * n_t, n_f, h, w)
        x = self.channel_attn(x)
        x = x.reshape(batch_size, n_t, n_f, h, w)
        
        cnn_out = []
        for t in range(n_t):
            #Shape of x[:, t] is (batch_size, n_f, h, w)
            out = self.conv1(x[:, t]) # Shape: (batch_size, 32, h, w)
            out = F.relu(out)
            out = self.conv2(out)  # Shape: (batch_size, 64, h, w)
            out = F.relu(out)
            out = self.pool(out)  # Shape: (batch_size, 64, h//2, w//2)
            
            # Apply SEModule for feature recalibration
            out = self.se_module(out)  # Apply attention to features
            
            out = out.view(batch_size, -1)  # Flatten
            cnn_out.append(out)
            
        cnn_out = torch.stack(cnn_out, dim=1)  # Shape: (batch_size, n_t, flattened_size)
        
        lstm_out, _ = self.lstm(cnn_out)  # Shape: (batch_size, n_t, hidden_dim)
        lstm_out = torch.sum(lstm_out, dim=1)  # Sum over n_t (batch_size, hidden_dim)
        
        out = self.fc(lstm_out)  # Shape: (batch_size, h*w)
        out = out.view(batch_size, h, w)  # Reshape to (batch_size, h, w)
        out = out.unsqueeze(-1)
        output = self.add_prompt_vecs(out, lead_time)
        
        return output