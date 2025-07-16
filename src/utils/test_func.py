import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)

import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.utils.loss import get_station_from_grid
from src.utils import utils
import os


def to_float(x, device):
    if isinstance(x,list):
        list_x = []
        for x_i in x:
            x_i = x_i.to(device).float()
            list_x.append(x_i)
        x = list_x
    else:
        x = x.to(device).float()
        
    return x   

def cal_acc(y_prd, y_grt):
    """
    Hàm tính toán các chỉ số đánh giá, tương thích với các phiên bản sklearn cũ.
    """
    mae = mean_absolute_error(y_grt, y_prd)
    
    # Sửa ở đây:
    # 1. Tính MSE một cách bình thường, không có tham số 'squared'.
    #    Hàm này mặc định trả về MSE.
    mse = mean_squared_error(y_grt, y_prd)
    
    mape = mean_absolute_percentage_error(y_grt, y_prd)
    
    # 2. Tính RMSE bằng cách lấy căn bậc hai của MSE.
    rmse = np.sqrt(mse)
    
    corr = np.corrcoef(np.reshape(y_grt, (-1)), np.reshape(y_prd, (-1)))[0][1]
    r2 = r2_score(y_grt, y_prd)
    
    return mae, mse, mape, rmse, r2, corr


def test_func(model, test_dataset, criterion, config, output_scaler, device):
    model.eval() 
    list_prd = []
    list_grt = []
    list_ecmwf = []
    epoch_loss = 0
    model.to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)

    print("********** Starting testing process **********")

    with torch.no_grad():
        
        for data in tqdm(test_dataloader):
            input_data, lead_time, y_grt, ecmwf = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device), data['ecmwf'].to(device)
            ecmwf = ecmwf[:,:,-1,:,:]# (B,7, H, W)
            ecmwf = torch.mean(ecmwf, dim=1) # (B, H, W)
            ecmwf = torch.unsqueeze(ecmwf, dim=-1) # Sử dụng torch.unsqueeze và dim=-1
            
            y_prd = model([input_data, lead_time]) # (batch_size, 137, 121, 1)
            
            y_prd = get_station_from_grid(y_prd, y_grt, config) # (batch_size, num_station, 1)
            ecmwf = get_station_from_grid(ecmwf, y_grt, config)
            y_prd = y_prd[:,:,0] # (batch_size, num_station)
            y_grt = y_grt[:,:,0] # (batch_size, num_station)
            ecmwf = ecmwf[:,:,0] # (batch_size, num_station)
            batch_loss = criterion(torch.squeeze(y_prd), torch.squeeze(y_grt))
            y_prd = y_prd.cpu().detach().numpy()
            y_grt = y_grt.cpu().detach().numpy()
            ecmwf = ecmwf.cpu().detach().numpy()
            if config.TRAIN.OUTPUT_NORM:
                y_prd = output_scaler.inverse_transform(y_prd)
                y_grt = output_scaler.inverse_transform(y_grt)
            
            y_prd = np.squeeze(y_prd)
            y_grt = np.squeeze(y_grt)
            ecmwf = np.squeeze(ecmwf)
            list_prd.append(y_prd)
            list_grt.append(y_grt)
            list_ecmwf.append(ecmwf)
            epoch_loss += batch_loss.item()
            # breakpoint()
    list_prd = np.concatenate(list_prd, 0)
    list_grt = np.concatenate(list_grt,0)
    list_ecmwf = np.concatenate(list_ecmwf, 0)
    # breakpoint()
    mae, mse, mape, rmse, r2, corr_ = cal_acc(list_prd, list_grt)
    mae_ecm, mse_ecm, mape_ecm, rmse_ecm, r2_ecm, corr_ecm = cal_acc(list_ecmwf, list_grt)
    # data = [[pred, gt] for pred, gt in zip(list_prd, list_grt)]
    plot_idx = [i for i in range(10)]
    if config.WANDB.STATUS:
        wandb.log({"mae":mae, "mse":mse, "mape":mape, "rmse":rmse, "r2":r2, "corr":corr_})
        wandb.log({"mae_ecm":mae_ecm, "mse_ecm":mse_ecm, "mape_ecm":mape_ecm, "rmse_ecm":rmse_ecm, "r2_ecm":r2_ecm, "corr_ecm":corr_ecm})
        for i in plot_idx:
            plt.figure(figsize=(20, 5))
            plt.plot(list_prd[i], label='Predictions', marker='o')
            plt.plot(list_grt[i], label='Ground Truths', marker='x')
            plt.plot(list_ecmwf[i], label="ECMWF", marker = 's')
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.title('Predictions vs Ground Truths')
            plt.legend()
            plt.grid(True)

            # Log the plot to W&B
            wandb.log({f"Output/Image{i}": wandb.Image(plt)})
            plt.close()
        

        # Flatten both arrays to 1D
        flattened1 = list_prd.flatten()  # Shape: (64 * 169,)
        flattened2 = list_grt.flatten()  # Shape: (64 * 169,)
        flattened3 = list_ecmwf.flatten() # Shape: (64 * 169,)
        data1 = np.stack([flattened1, flattened2], 0)
        table1 = wandb.Table(data=data1.T, columns=["Prediction", "Groundtruth"] )
        wandb.log({"Output/Table1": table1})
        data2 = np.stack([flattened3, flattened2], 0)
        table2 = wandb.Table(data=data2.T, columns=["Prediction", "Groundtruth"] )
        wandb.log({"Output/Table2": table2})
        wandb.finish()


    print(f"MSE: {mse} MAE:{mae} MAPE:{mape} RMSE:{rmse} R2:{r2} Corr:{corr_}")  
    print(f"MSE_ecm: {mse_ecm} MAE_ecm:{mae_ecm} MAPE_ecm:{mape_ecm} RMSE_ecm:{rmse_ecm} R2_ecm:{r2_ecm} Corr_ecm:{corr_ecm}")           
    return 

import matplotlib.pyplot as plt


def test_func1(model, test_dataset, criterion, config, output_scaler, device, output_dir):
    model.eval() 
    model.to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=config.TRAIN.BATCH_SIZE, shuffle=False, num_workers=config.TRAIN.NUMBER_WORKERS, collate_fn=utils.custom_collate_fn)

    print("********** Starting testing process **********")

    # Tạo thư mục lưu ảnh nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        for idx, data in enumerate(test_dataloader):
            # input_data, lead_time, y_grt, y_enmwf = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device), data['x_leadtime'].to(device)
            input_data, lead_time, y_grt = data['x'].to(device), data['lead_time'].to(device), data['y'].to(device)
            print(input_data.shape)
            print(lead_time.shape)
            y_prd = model([input_data, lead_time])  # (batch_size, 137, 121, 1)
            # print(y_enmwf.shape)

            # Đảo ngược chuẩn hóa nếu có
            y_prd = y_prd.cpu().detach().numpy()
            y_grt = y_grt.cpu().detach().numpy() 
            if config.TRAIN.OUTPUT_NORM:
                y_prd = output_scaler.inverse_transform(y_prd)
                y_grt = output_scaler.inverse_transform(y_grt)
            
            y_prd = np.squeeze(y_prd)
            y_grt = np.squeeze(y_grt)

            # Vẽ lưới các hình ảnh cho 10 mẫu đầu tiên
            # plot_idx = [i for i in range(min(10, y_prd.shape[0]))]  # Giới hạn ở 10 mẫu
            # print(plot_idx)
            for i in range(1):
                plt.figure(figsize=(5, 5))

                # Vẽ lưới mưa cho giá trị dự đoán
                # plt.subplot(1, 2, 1)  # 1 hàng, 2 cột, ảnh đầu tiên
                # print(y_prd[i].shape)
                plt.imshow(y_prd[i], cmap='Blues', aspect='auto')  # Sử dụng màu xanh cho lưới mưa
                plt.colorbar(label='Predicted Rainfall')
                plt.title(f'Sample {i+1}')
                plt.xlabel('Longitude')
                plt.ylabel('Latitude')

                # Vẽ lưới mưa cho giá trị thực tế
                # plt.subplot(1, 2, 2)  # 1 hàng, 2 cột, ảnh thứ hai
                # plt.imshow(y_grt[i], cmap='Blues', aspect='auto')  # Sử dụng màu xanh cho lưới mưa
                # plt.colorbar(label='Ground Truth Rainfall')
                # plt.title(f'Sample {i+1} - Ground Truth Rainfall')
                # plt.xlabel('Longitude')
                # plt.ylabel('Latitude')

                # Lưu ảnh vào thư mục output
                plt.tight_layout()
                plt.savefig(f"{output_dir}/ps4/sample_{idx*config.TRAIN.BATCH_SIZE + i}.png")
                plt.close()
                

    print(f"Testing completed. Images saved to {output_dir}")