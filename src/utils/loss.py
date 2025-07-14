import torch
import torch.nn.functional as F
import torch.nn as nn

def get_station_from_grid(y_pred, y, config):
    # y_pred: (batch_size, h, w, 1) 
    # y: (batch_size, num_station, 3) 
    batch_size = y.shape[0]
    num_station = y.shape[1]
    lat_start = config.DATA.LAT_START 
    lon_start = config.DATA.LON_START
    step = 0.125  

    stations_lon = y[:, :, 1]  # (batch_size, num_station)
    stations_lat = y[:, :, 2]  # (batch_size, num_station)

    lat_idx = torch.floor((lat_start - stations_lat) / step).long()  # (batch_size, num_station)
    lon_idx = torch.floor((stations_lon - lon_start) / step).long()  # (batch_size, num_station)

    num_lat, num_lon = config.DATA.HEIGHT, config.DATA.WIDTH
    lat_idx = torch.clamp(lat_idx, 0, num_lat - 1)  # (batch_size, num_station)
    lon_idx = torch.clamp(lon_idx, 0, num_lon - 1)  # (batch_size, num_station)
    
    batch_idx = torch.arange(batch_size).view(-1, 1).expand(-1, num_station)  # (batch_size, num_station)
    
    pred_values = y_pred[batch_idx, lat_idx, lon_idx, 0]  # (batch_size, num_station)
    pred_values = pred_values.unsqueeze(-1)  # (batch_size, num_station, 1)
    
    return pred_values

class MagnitudeWeightedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(MagnitudeWeightedHuberLoss, self).__init__()
        self.delta = delta  # Threshold for Huber Loss

    def forward(self, y_pred, y_true):
        # Compute the error
        error = y_true - y_pred
        
        # Compute weights based on absolute target values
        weights = torch.abs(y_true)
        
        # Huber Loss calculation
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.full_like(abs_error, self.delta))
        linear = abs_error - quadratic
        huber_loss = 0.5 * quadratic**2 + self.delta * linear
        
        # Apply magnitude weighting
        weighted_loss = weights * huber_loss
        
        # Return mean loss
        return torch.mean(weighted_loss)

class ExpMagnitudeWeightedMAELoss(nn.Module):
    def __init__(self, k=0.1, reduction='mean'):
        super(ExpMagnitudeWeightedMAELoss, self).__init__()
        self.k = k  # Scaling factor for exponential weighting
        self.reduction = reduction  # 'mean', 'sum', or 'none'

    def forward(self, y_pred, y_true):
        # Expected shapes: y_pred, y_true = [batch, n_station]
        assert y_pred.shape == y_true.shape, "Prediction and target shapes must match"
        
        # Compute the absolute error
        error = torch.abs(y_true - y_pred)  # Shape: [batch, n_station]

        # Compute weights based on exponential of absolute target values
        weights = torch.exp(self.k * torch.abs(y_true))  # Shape: [batch, n_station]

        # Apply weighting to MAE
        weighted_loss = weights * error  # Shape: [batch, n_station]

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(weighted_loss)  # Scalar: mean over batch and stations
        elif self.reduction == 'sum':
            return torch.sum(weighted_loss)  # Scalar: sum over batch and stations
        elif self.reduction == 'none':
            return weighted_loss  # Shape: [batch, n_station]
        else:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")
        

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_func='square', reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.weight_func = weight_func  # 'abs', 'square', or custom callable
        self.reduction = reduction      # 'mean', 'sum', or 'none'

    def forward(self, y_pred, y_true):
        # Expected shapes: y_pred, y_true = [batch, n_station]
        assert y_pred.shape == y_true.shape, "Prediction and target shapes must match"
        
        # Compute the squared error
        error = (y_true - y_pred) ** 2  # Shape: [batch, n_station]

        # Compute weights based on target values
        if self.weight_func == 'abs':
            weights = torch.abs(y_true)  # w_ij = |y_ij|
        elif self.weight_func == 'square':
            weights = y_true ** 2        # w_ij = y_ij^2
        elif callable(self.weight_func):
            weights = self.weight_func(y_true)  # Custom function
        else:
            raise ValueError("weight_func must be 'abs', 'square', or a callable")

        # Apply weighting to MSE
        weighted_loss = weights * error  # Shape: [batch, n_station]

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(weighted_loss)  # Scalar: mean over batch and stations
        elif self.reduction == 'sum':
            return torch.sum(weighted_loss)   # Scalar: sum over batch and stations
        elif self.reduction == 'none':
            return weighted_loss              # Shape: [batch, n_station]
        else:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")



class WeightedThresholdMSE(nn.Module):
    def __init__(self, high_weight=10, low_weight=1, threshold=200):
        super().__init__()
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.threshold = threshold
    
    
    def forward(self, y_pred, y_true):
        # Compute squared error
        squared_error = (y_true - y_pred) ** 2
        
        # Create weight mask based on threshold
        weights = torch.where(y_true > self.threshold, 
                            torch.tensor(self.high_weight, dtype=y_true.dtype, device=y_true.device), 
                            torch.tensor(self.low_weight, dtype=y_true.dtype, device=y_true.device))
        
        # Apply weights to squared errors
        weighted_error = weights * squared_error
        
        # Return mean loss
        return weighted_error.mean()
    
class LogMagnitudeWeightedHuberLoss(nn.Module):
    def __init__(self, delta=10.0, alpha=0.05, reduction='mean'):
        super(LogMagnitudeWeightedHuberLoss, self).__init__()
        self.delta = delta        # Ngưỡng phân chia giữa MSE và MAE
        self.alpha = alpha        # Điều chỉnh độ mạnh của weight theo lượng mưa
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Shape: [batch, num_station]
        assert y_pred.shape == y_true.shape, "Prediction and target shapes must match"

        # Error
        error = y_true - y_pred
        abs_error = torch.abs(error)

        # Huber Loss
        quadratic = torch.minimum(abs_error, torch.full_like(abs_error, self.delta))
        linear = abs_error - quadratic
        huber = 0.5 * quadratic ** 2 + self.delta * linear

        # Trọng số log(|y|): log(1 + α * |y|)
        weights = torch.log1p(self.alpha * torch.abs(y_true))

        # Áp dụng trọng số
        weighted_loss = weights * huber

        # Giảm chiều
        if self.reduction == 'mean':
            return torch.mean(weighted_loss)
        elif self.reduction == 'sum':
            return torch.sum(weighted_loss)
        elif self.reduction == 'none':
            return weighted_loss
        else:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")

import torch
import torch.nn as nn

class FocalMSELoss(nn.Module):
    def __init__(self, gamma=0.5, beta=1.0, reduction='mean', eps=1e-6): # Giam gamma, tang beta
        """
        Focal MSE Loss for regression with robustness to outliers.
        gamma: focal modulation strength
        beta: non-linearity exponent
        reduction: 'mean', 'sum', or 'none'
        eps: small value to ensure numerical stability
        """
        super(FocalMSELoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "Shape mismatch"

        error = y_true - y_pred
        mse = error ** 2
        modulation = (1 - torch.exp(-self.gamma * torch.abs(error))).clamp(min=self.eps) ** self.beta
        focal_mse = mse * modulation

        if self.reduction == 'mean':
            return torch.mean(focal_mse)
        elif self.reduction == 'sum':
            return torch.sum(focal_mse)
        elif self.reduction == 'none':
            return focal_mse
        else:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")

import torch
import torch.nn as nn


