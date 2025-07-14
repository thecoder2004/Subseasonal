import numpy as np 
from src.utils.utils import load_model
from src.model.models import Model_Ver1
from src.utils.get_option import get_option
from src.utils.visualization import create_heatmap
import argparse

from torch.utils.data import DataLoader
import torch 
import torch.nn as nn 
import wandb
import os 

from src.utils.get_option import get_option
from src.model.models import Model_Ver1
from src.utils.dataloader import CustomDataset
from src.utils import utils, get_scaler, train_func, test_func
from src.model import models 
from src.utils.loss import ExpMagnitudeWeightedMAELoss, WeightedMSELoss


args = get_option()
input_scaler, output_scaler = get_scaler.get_scaler(args)

checkpoint_path = f"saved_checkpoints/tuning_ver1/checkpoint/SubseasonalModelV1_Lr-0.0001_LF-mse_10.0_Dr-0.1_LN-False_Seed-42.pt"
test_dataset = CustomDataset(mode='test', args=args, ecmwf_scaler=input_scaler, output_scaler= output_scaler)

samples = test_dataset[0]
input_data, lead_time, y_grt = torch.tensor(samples['x']).unsqueeze(0), torch.tensor(samples['lead_time']).unsqueeze(0), torch.tensor(samples['y']).unsqueeze(0)


breakpoint()
model = Model_Ver1(args)
model.eval()
model([input_data, lead_time])
load_model(model, checkpoint_path)

prompt = model.delta_t
create_heatmap((spatial_embedding[0,0,:,:,0].detach().numpy()), "spatial_embedding.png", cmap="hot")
create_heatmap((prompt.detach().numpy()), "prompt.png", "Prompt value distribution", "Order", "", cmap='hot')


create_heatmap((temporal_embedding[0,:,0,:].detach().numpy()), "temp_embedding.png", cmap="hot")

breakpoint()
# load_model()