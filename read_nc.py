from netCDF4 import Dataset, num2date
import os
import numpy as np
from tqdm import tqdm

params_dict = {143: "cp",
               146:"sshf",
               147:"slhf",
               175:"strd",
               176:"ssr",
               177:"str",
               179:"ttr",
               174008:"sro",
               121: "mx2t6",
               122: "mn2t6",
               165: "u10",
               166: "v10",
               228: "tp"}

different_dim_param = [121, 122, 165, 166, 228]

def read_nc():
    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"

    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr"

    for day in tqdm(list_day):
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            core_data = variables[params_dict[param]][:]
            data_shape = core_data.shape
            reshaped_data = core_data.reshape(20,47,137,121)
            
            list_data.append(reshaped_data)
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0) 
        #Save data as npy
        os.makedirs(f"{saved_path}/Step24h", exist_ok=True)
        np.save(f"{saved_path}/Step24h/{day[:-3]}.npy", new_arr)
    
def read_nc_region_1():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 20.75, 22.75
    LON_MIN, LON_MAX = 102.75, 104.75

    folder_path = f"/mnt/disk3/longnd/env_data/S2S_0.125_old/S2S_0.125/preprocessed_data/"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk3/longnd/env_data/S2S_0.125_old/S2S_0.125/nparr_reg_1"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 17, 17)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_2():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 20.75, 23
    LON_MIN, LON_MAX = 103.75, 108

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_2"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 19, 35)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_3():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 20, 21.5
    LON_MIN, LON_MAX = 105, 107.75

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_3"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 13, 23)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_4():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 16, 20.5
    LON_MIN, LON_MAX = 104.25, 107.75

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_4"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 37, 29)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_5():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 10.5, 16.25
    LON_MIN, LON_MAX = 108, 109.5

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_5"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 47, 13)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_6():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 11.5, 14.75
    LON_MIN, LON_MAX = 106.75, 109

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_6"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 27, 19)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")
    
def read_nc_region_7():
    # Định nghĩa khoảng Lat và Lon cần lọc
    LAT_MIN, LAT_MAX = 8.5, 11.5
    LON_MIN, LON_MAX = 103.75, 112

    folder_path = f"/mnt/disk1/env_data/S2S_0.125/preprocessed_data/Step24h"
    list_day = sorted(os.listdir(f"{folder_path}/143"))
    saved_path = f"/mnt/disk1/env_data/S2S_0.125/nparr_reg_7"

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(f"{saved_path}/Step24h", exist_ok=True)

    for day in tqdm(list_day):
        # Đường dẫn file .npy sẽ lưu
        output_file = f"{saved_path}/Step24h/{day[:-3]}.npy"
        
        # Kiểm tra xem file đã tồn tại chưa
        if os.path.exists(output_file):
            print(f"File {output_file} đã tồn tại, bỏ qua...")
            continue  # Skip sang ngày tiếp theo
        
        list_data = []
        for param in params_dict.keys():
            data = Dataset(f"{folder_path}/{param}/{day}")
            variables = data.variables
            
            # Lấy tọa độ latitude và longitude từ file .nc
            latitudes = variables['latitude'][:]
            longitudes = variables['longitude'][:]
            
            # Tìm chỉ số tương ứng với Lat và Lon trong khoảng min/max
            lat_mask = (latitudes >= LAT_MIN) & (latitudes <= LAT_MAX)
            lon_mask = (longitudes >= LON_MIN) & (longitudes <= LON_MAX)
            
            lat_indices = np.where(lat_mask)[0]  # Chỉ số latitude thỏa mãn
            lon_indices = np.where(lon_mask)[0]  # Chỉ số longitude thỏa mãn
            
            # Lấy dữ liệu chính (core_data) và cắt theo Lat, Lon
            core_data = variables[params_dict[param]][:]
            if param in different_dim_param:
                filtered_data = core_data[lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1, :]
                filtered_data = np.transpose(filtered_data, axes=(2, 0, 1))
            else:
                filtered_data = core_data[:, lat_indices[0]:lat_indices[-1]+1, 
                                        lon_indices[0]:lon_indices[-1]+1]
            
            # Reshape dữ liệu đã lọc
            n_lat = len(lat_indices)
            n_lon = len(lon_indices)
            reshaped_data = filtered_data.reshape(20, 47, n_lat, n_lon) # (20, 47, 25, 67)
            
            list_data.append(reshaped_data)
        
        # Stack và xử lý dữ liệu
        new_arr = np.stack(list_data, 1)
        new_arr = np.ma.filled(new_arr, 0)
        
        # Lưu dữ liệu
        np.save(output_file, new_arr)

    print(f"Đã lọc dữ liệu theo Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX}")

read_nc_region_1()