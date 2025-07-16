import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import torch
import os
from tqdm import tqdm
import xarray as xr

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)

def convert_wandb_csv_nowcast():
    input_csv = "/mnt/disk1/env_data/result/nowcast/cnn_lstm/nowcast_cnn_lstm.csv"  
    station_csv = "/mnt/disk1/env_data/Gauge_thay_Tan/Final_Data.csv"   
    time_csv = "/mnt/disk1/nxmanh/Hydrometeology/Nowcasting_Prediction/data789_seed52/test.csv"          
    output_csv = "/mnt/disk1/env_data/result/nowcast/cnn_lstm/prediction_groundtruth.csv"     

    df_pred = pd.read_csv(input_csv)  
    df_station = pd.read_csv(station_csv)  
    df_time = pd.read_csv(time_csv) 

    stations = df_station['Station'].unique()

    num_stations = len(stations) # 141
    num_lead_time = 5
    num_days = len(df_time) 

    data = []

    for day_idx in tqdm(range(num_days), desc="Processing"):
        year = df_time.iloc[day_idx]['year']
        month = df_time.iloc[day_idx]['month']
        day = df_time.iloc[day_idx]['day']
        
        start_idx = day_idx * num_stations * num_lead_time # 0, 1 * 141 * 5, ...
        end_idx = start_idx + num_stations * num_lead_time # 1 * 141 * 5, 2 * 141 * 5, ...
        group_data = df_pred.iloc[start_idx:end_idx]
        for i, station in enumerate(stations):
            for lead_time in range(num_lead_time):
                row = {
                    'Prediction': group_data.iloc[i*num_lead_time+lead_time]['Prediction'], 
                    'Groundtruth': group_data.iloc[i*num_lead_time+lead_time]['Groundtruth'], 
                    'station': station,
                    'lead_time': lead_time+1,
                    'year': year,
                    'month': month,
                    'day': day
                }
                data.append(row)

    df_output = pd.DataFrame(data)

    df_output.to_csv(output_csv, index=False)

    print(f"Đã tạo file {output_csv} thành công!")
    
def convert_wandb_csv_subseasonal():
    input_csv = "/mnt/disk3/tunm/Subseasonal_Forecasting/results/Strans/2205/stranv4_mae.csv"  
    station_csv = "/mnt/disk3/longnd/env_data/Gauge_thay_Tan/Final_Data_Region_1.csv"     
    time_csv = "/mnt/disk3/nxmanh/Subseasonal_Prediction/data/data6789_seed52/test.csv"          
    # output_csv = "/mnt/disk1/env_data/result/subseasonal/reg_1/strans/prediction_groundtruth_strans.csv"  
    output_csv = "/mnt/disk3/tunm/Subseasonal_Forecasting/results/Strans/2205/stranv4_mae-final.csv"     
   

    df_pred = pd.read_csv(input_csv)  
    df_station = pd.read_csv(station_csv)  
    df_time = pd.read_csv(time_csv) 

    stations = df_station['Station'].unique()
    
    num_stations = len(stations) 
    num_lead_time = 40
    num_days_possible = 49921 // (num_stations * num_lead_time)  
    num_days = int(num_days_possible) 

    total_values_used = num_days * num_lead_time * num_stations
    print(f"Số ngày: {num_days}, Tổng giá trị sử dụng: {total_values_used}")
    assert total_values_used <= 49921, "Số giá trị vượt quá 200,000"

    data = []

    for day_idx in tqdm(range(num_days * num_lead_time), desc="Processing"):
        lead_time = df_time.iloc[day_idx]['leadTime']
        year = df_time.iloc[day_idx]['year']
        month = df_time.iloc[day_idx]['month']
        day = df_time.iloc[day_idx]['day']
        
        start_idx = day_idx * num_stations
        end_idx = start_idx + num_stations
        group_data = df_pred.iloc[start_idx:end_idx]

        for i, station in enumerate(stations):
            if i >= len(group_data):
                print(f"[Warning] day_idx {day_idx}, station_idx {i} out of bounds with group_data len = {len(group_data)}")
                continue  # bỏ qua nếu không đủ dòng

            row = {
                'Prediction': group_data.iloc[i]['Prediction'],
                'Groundtruth': group_data.iloc[i]['Groundtruth'],
                'station': station,
                'lead_time': lead_time,
                'year': year,
                'month': month,
                'day': day
            }
            data.append(row)


    df_output = pd.DataFrame(data)

    df_output.to_csv(output_csv, index=False)

    print(f"Đã tạo file {output_csv} thành công!")
    
def plot():
    csv_file = "/mnt/disk1/tunm/Subseasional_Forecasting/results/Fno2d/2205/data2-region1-fno2d-final.csv"
    saved_dir = "/mnt/disk1/tunm/Subseasional_Forecasting/results/Strans/2205/plot/leadtime10"    
    df = pd.read_csv(csv_file)

    year = 2018
    month = 7
    day = 4
    # stations = 5

    filtered_df = df[(df['year'] == year) & (df['month'] == month) & (df['day'] == day)]
    stations = filtered_df['station'].unique()
    
    for station in stations:
        station_data = filtered_df[filtered_df['station'] == station]
        
        station_data = station_data.sort_values('lead_time')
        
        lead_times = station_data['lead_time']
        predictions = station_data['Prediction']
        groundtruths = station_data['Groundtruth']
        
        plt.figure(figsize=(10, 6))
        plt.plot(lead_times, predictions, label='Prediction ', color='blue', marker='o', alpha=0.8)
        plt.plot(lead_times, groundtruths, label='Groundtruth (Gauge)', color='red', marker='o', alpha=0.8)
        
        plt.title(f'ECMWF Comparison Station {station} (2018-07-04)')
        plt.xlabel('Lead Time (days)')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{saved_dir}/prediction_station_{station}.png")
        plt.close()

    print("Đã tạo biểu đồ cho từng station từ file CSV.")


def plot_test_1():
    lead_time_value = 8  # Lead time = 8
    csv_file = "/mnt/disk3/tunm/Subseasonal_Forecasting/results/Strans/2205/mae-final.csv"
    saved_dir = f"/mnt/disk3/tunm/Subseasonal_Forecasting/results/Strans/2205/plot/data2/leadtime{lead_time_value}_dr025-mae"    
    os.makedirs(saved_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    
    

    # Lọc dữ liệu theo lead_time = 8
    filtered_df = df[df['lead_time'] == lead_time_value]
    
    # Lấy danh sách các station trong dữ liệu đã lọc
    stations = filtered_df['station'].unique()

    # Lặp qua tất cả các station và vẽ biểu đồ cho từng station
    for station_value in stations:
        # Lọc dữ liệu theo station
        station_df = filtered_df[filtered_df['station'] == station_value]
        
        # Tạo cột ngày từ year, month, day và sắp xếp dữ liệu theo ngày
        station_df['date'] = pd.to_datetime(station_df[['year', 'month', 'day']])
        
        # Sắp xếp dữ liệu theo ngày
        station_df = station_df.sort_values('date')
        
        print(f"Filtered Data Shape for Station {station_value}: {station_df.shape}")
        
        # Trục X là index của dữ liệu đã lọc (số ngày)
        indices = range(len(station_df))  # Chỉ số của các dòng (tương ứng với số ngày)

        # Vẽ biểu đồ cho Prediction và Groundtruth
        plt.figure(figsize=(12, 6))

        # Lấy dữ liệu cho Prediction và Groundtruth
        predictions = station_df['Prediction']
        groundtruths = station_df['Groundtruth']
            
        # Vẽ đường cho Prediction (ModelV1) với nét đứt
        plt.plot(indices, predictions, color='blue', marker='o', markersize=4, linestyle='--', alpha=0.7, label='Prediction')
        
        # Vẽ đường cho Groundtruth
        plt.plot(indices, groundtruths, color='red', marker='x', markersize=4, linestyle='-', alpha=0.7, label='Groundtruth (Gauge)')

        # Cài đặt các thông số biểu đồ
        plt.title(f'ECMWF Comparison for Station {station_value} - Lead Time = {lead_time_value}', fontsize=14)
        plt.xlabel('Index (Days)', fontsize=12)
        plt.ylabel('Rainfall (mm)', fontsize=12)
        
        # Chú thích
        plt.legend(loc='upper left')

        # Bỏ lưới ô vuông (grid)
        plt.grid(False)
        
        plt.tight_layout()
        
        # Lưu biểu đồ vào file
        plt.savefig(f"{saved_dir}/prediction_comparison_station_{station_value}_lead_time_{lead_time_value}.png")
        plt.close()

        print(f"Đã tạo biểu đồ cho station {station_value} với lead time = {lead_time_value}.")

def plot_test():
    lead_time_value = 8  # Lead time = 8
    csv_file = "/mnt/disk3/tunm/Subseasonal_Forecasting/results/Strans/2205/stranv4_mae-final.csv"
    csv_file_v2 = "/mnt/disk3/tunm/Subseasonal_Forecasting/results/Strans/2205/ecmwf-final.csv"
    saved_dir = f"/mnt/disk3/tunm/Subseasonal_Forecasting/results/Strans/2205/plot/leadtime{lead_time_value}" 
    os.makedirs(saved_dir, exist_ok=True)   
    df = pd.read_csv(csv_file)
    df_ver2 = pd.read_csv(csv_file_v2)

    model = "Strans-v4"

    # Lọc dữ liệu theo lead_time = 8
    filtered_df = df[df['lead_time'] == lead_time_value]
    filtered_df_ver2 = df_ver2[df_ver2['lead_time'] == lead_time_value]
    
    # Lấy danh sách các station trong dữ liệu đã lọc
    stations = filtered_df['station'].unique()

    # Lặp qua tất cả các station và vẽ biểu đồ cho từng station
    for station_value in stations:
        # Lọc dữ liệu theo station
        station_df = filtered_df[filtered_df['station'] == station_value]
        station_df_ver2 = filtered_df_ver2[filtered_df_ver2['station'] == station_value]
        
        # Tạo cột ngày từ year, month, day và sắp xếp dữ liệu theo ngày
        station_df['date'] = pd.to_datetime(station_df[['year', 'month', 'day']])
        station_df_ver2['date'] = pd.to_datetime(station_df_ver2[['year', 'month', 'day']])
        
        # Sắp xếp dữ liệu theo ngày
        station_df = station_df.sort_values('date')
        station_df_ver2 = station_df_ver2.sort_values('date')

        print(f"Filtered Data Shape for Station {station_value}: {station_df.shape}")
        
        # Trục X là index của dữ liệu đã lọc (số ngày)
        indices = range(len(station_df))  # Chỉ số của các dòng (tương ứng với số ngày)

        # Vẽ biểu đồ cho Prediction và Groundtruth
        plt.figure(figsize=(12, 6))

        # Lấy dữ liệu cho Prediction và Groundtruth
        predictions = station_df['Prediction']
        groundtruths = station_df['Groundtruth']
        predictions_ver2 = station_df_ver2['Prediction']
        groundtruths_ver2 = station_df_ver2['Groundtruth']
        stran_mae, stran_mse, stran_mape, stran_rmse, stran_r2, stran_corr = cal_acc(predictions, groundtruths)
        ecmwf_mae, ecmfw_mse, ecmwf_mape, ecmwf_rmse, ecmwf_r2, ecmwf_corr = cal_acc(predictions_ver2, groundtruths)
        text_str = (
            f"{model}:\n"
            f"  MAE: {stran_mae:.4f}\n"
            f"  RMSE: {stran_rmse:.4f}\n"
            f"  R²: {stran_r2:.4f}\n"
            f"  Corr: {stran_corr:.4f}\n\n"
            f"ECMWF:\n"
            f"  MAE: {ecmwf_mae:.4f}\n"
            f"  RMSE: {ecmwf_rmse:.4f}\n"
            f"  R²: {ecmwf_r2:.4f}\n"
            f"  Corr: {ecmwf_corr:.4f}"
        )
            
        # Vẽ đường cho Prediction (ModelV1) với nét đứt
        plt.plot(indices, predictions, color='blue', marker='o', markersize=4, linestyle='--', alpha=0.7, label=model)
        # Vẽ đường cho Prediction (ModelV1 + ChannelAttn)
        plt.plot(indices, predictions_ver2, color='green', marker='s', markersize=4, linestyle=':', alpha=0.7, label='ECMWF')
        # Vẽ đường cho Groundtruth
        plt.plot(indices, groundtruths, color='red', marker='x', markersize=4, linestyle='-', alpha=0.7, label='Groundtruth (Gauge)')
        props = dict(boxstyle='round', facecolor='white', edgecolor='black', lw=1)
        ax = plt.gca()
        ax.text(0.95, 0.95, text_str, transform=ax.transAxes, fontsize=10,verticalalignment='top', bbox=props)
        # Cài đặt các thông số biểu đồ
        plt.title(f'ECMWF Comparison for Station {station_value} - Lead Time = {lead_time_value}', fontsize=14)
        plt.xlabel('Index (Days)', fontsize=12)
        plt.ylabel('Rainfall (mm)', fontsize=12)
        
        # Chú thích
        plt.legend(loc='upper left')

        # Bỏ lưới ô vuông (grid)
        plt.grid(False)
        
        plt.tight_layout()
        
        # Lưu biểu đồ vào file
        plt.savefig(f"{saved_dir}/prediction_comparison_station_{station_value}_lead_time_{lead_time_value}.png")
        plt.close()

        print(f"Đã tạo biểu đồ cho station {station_value} với lead time = {lead_time_value}.")

def cal_acc(y_prd, y_grt):
    mae = mean_absolute_error(y_grt, y_prd)
    mse = mean_squared_error(y_grt, y_prd)
    mape = mean_absolute_percentage_error(y_grt, y_prd)
    rmse = np.sqrt(mse)
    corr = np.corrcoef(np.reshape(y_grt, (-1)), np.reshape(y_prd, (-1)))[0][1]
    r2 = r2_score(y_grt, y_prd)
    # mdape_ = mdape(y_grt,y_prd)
    return mae, mse, mape, rmse, r2, corr
    
def cal_acc_reg_4():
    reg_4_file = "/mnt/disk1/env_data/result/subseasonal/reg_4/model_v1/prediction_groundtruth.csv"
    df = pd.read_csv(reg_4_file)
    stations = df['station'].unique()
    
    csv_file = "/mnt/disk1/env_data/result/subseasonal/no_reg/model_v1/prediction_groundtruth.csv"
    df = pd.read_csv(csv_file)
    
    df = df[df['station'].isin(stations)]
    
    prediction = df['Prediction']
    groundtruth = df['Groundtruth']

    mae, mse, mape, rmse, r2, corr_ = cal_acc(prediction, groundtruth)
    print(f"MSE: {mse} MAE:{mae} MAPE:{mape} RMSE:{rmse} R2:{r2} Corr:{corr_}")     
    
def cal_acc_reg(region):
    reg_file = f"/mnt/disk1/env_data/Gauge_thay_Tan/Final_Data_Region_{region}.csv"
    df = pd.read_csv(reg_file)
    stations = df['Station'].unique()
    
    csv_file = "/mnt/disk1/env_data/result/subseasonal/no_reg/model_v1/prediction_groundtruth.csv"
    df = pd.read_csv(csv_file)
    
    df = df[df['station'].isin(stations)]
    
    prediction = df['Prediction']
    groundtruth = df['Groundtruth']

    mae, mse, mape, rmse, r2, corr_ = cal_acc(prediction, groundtruth)
    print(f"MSE: {mse} MAE:{mae} MAPE:{mape} RMSE:{rmse} R2:{r2} Corr:{corr_}")         

convert_wandb_csv_subseasonal()
#plot_test_1()
# convert_wandb_csv()

plot_test()
# cal_acc_reg(1)