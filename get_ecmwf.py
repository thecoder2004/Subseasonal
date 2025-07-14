import numpy as np

def get_ecmwf(ecmwf_path, year, lead_time):
    # Dữ liệu gốc có shape (20, 13, 47, 137, 121)
    ecmwf_data = np.load(ecmwf_path) 
    # Sau khi cắt, ecmwf_data có shape (13, 7, 137, 121)
    ecmwf_data = ecmwf_data[year-2002,:,lead_time-6:lead_time+1,:,:] 
    
    return ecmwf_data

# Đường dẫn đến file .npy gốc (chứa toàn bộ dữ liệu ECMWF thô)
ecmwf_path = '/mnt/disk1/env_data/S2S_0.125/nparr_reg_1/Step24h/2022-06-02.npy'

# Gọi hàm với các tham số cụ thể để lấy dữ liệu ECMWF thô cho một trường hợp
ecmwf_data_raw = get_ecmwf(ecmwf_path=ecmwf_path, year=2002, lead_time=7)
print(ecmwf_data_raw.shape)
print(f"Kích thước (shape) của ecmwf_data_raw: {ecmwf_data_raw.shape}")
print(f"Kiểu dữ liệu (dtype) của ecmwf_data_raw: {ecmwf_data_raw.dtype}")

# Tính giá trị min, max, mean cho MỖI FEATURE
# (trung bình/min/max hóa qua các chiều thời gian và không gian)

# Các axis để tính toán:
# ecmswf_data_raw.shape = (13, 7, 137, 121)
# Chiều feature là axis 0.
# Chúng ta muốn tính min/max/mean trên các axis còn lại: 1 (days), 2 (height), 3 (width)
axes_to_aggregate = (1, 2, 3)

min_for_each_raw_feature = np.min(ecmwf_data_raw, axis=axes_to_aggregate)
max_for_each_raw_feature = np.max(ecmwf_data_raw, axis=axes_to_aggregate)
mean_for_each_raw_feature = np.mean(ecmwf_data_raw, axis=axes_to_aggregate)

print(f"\nKích thước (shape) của mảng min/max/mean cho từng feature: {min_for_each_raw_feature.shape}")
print("\n--- Thống kê chi tiết cho từng Feature THÔ (chưa chuẩn hóa) ---")

# In ra min, max, mean cho từng feature
for i in range(13):
    print(f"Feature {i} (Index {i}):")
    print(f"  Min:  {min_for_each_raw_feature[i]:.4f}")
    print(f"  Max:  {max_for_each_raw_feature[i]:.4f}")
    print(f"  Mean: {mean_for_each_raw_feature[i]:.4f}")
    print("-" * 30)

# Bạn vẫn có thể in giá trị min/max/mean tổng thể của toàn bộ dữ liệu thô nếu muốn
print(f"\n--- Thống kê tổng thể của toàn bộ dữ liệu thô ---")
print(f"Giá trị nhỏ nhất tổng thể: {np.min(ecmwf_data_raw):.4f}")
print(f"Giá trị lớn nhất tổng thể: {np.max(ecmwf_data_raw):.4f}")
print(f"Giá trị trung bình tổng thể: {np.mean(ecmwf_data_raw):.4f}")