import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
from datetime import datetime, timedelta
import joblib

def get_scaler(config):
    # Path to data folder
    data_dir=config.DATA.NPYARR_DIR
    csv_path=config.DATA.GAUGE_DATA_PATH
    
    # Path to save scalers
    scaler_file= f"{config.DATA.DATA_IDX_DIR}/scalers.pkl"
    output_scaler_file=f"{config.DATA.DATA_IDX_DIR}/output_scaler.pkl"
    
    # Check if scaler files exist
    if os.path.exists(scaler_file) and os.path.exists(output_scaler_file):
        print(f"Load scalers from file: {scaler_file}")
        print(f"Load output_scaler from file: {output_scaler_file}")
        scalers = joblib.load(scaler_file)
        output_scaler = joblib.load(output_scaler_file)
        return scalers, output_scaler
    
    # If scaler files do not exist, create new scalers
    print("Creating scalers...")
    
    # Create a list of dates from 2022-01-03
    start_date = datetime(2022, 1, 3)
    end_date = datetime(2022, 12, 31)
    day_steps = [3, 4]
    step_index = 0
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=day_steps[step_index])
        step_index = 1 - step_index
    
    # Filter dates to only include July, August, and September
    # dates = [date for date in dates if int(date.split('-')[1]) in [6, 7, 8, 9]]
    dates = [date for date in dates if int(date.split('-')[1]) in [9, 10, 11]]
    
    # Initialize scalers for 13 features
    scalers = [MinMaxScaler(feature_range=(-1, 1)) for _ in range(13)]
    
    # Calculate scalers for each feature
    first_pass = True
    for date in dates:
        file_path = os.path.join(data_dir, f"{date}.npy")
        if os.path.exists(file_path):
            data = np.load(file_path)  # Shape: (20, 13, 47, 137, 121)
            for i in range(13):
                feature_data = data[:, i, :, :, :].reshape(-1, 1)
                if first_pass:
                    scalers[i].fit(feature_data)
                else:
                    scalers[i].partial_fit(feature_data)
            first_pass = False
    
    # Convert scalers to numpy array
    scalers_array = np.array(scalers)
    
    # Save scalers to file
    joblib.dump(scalers_array, scaler_file)
    print(f"Saved scalers into file: {scaler_file}")
    
    # Create output_scaler
    print("Creating output_scaler...")
    csv_data = pd.read_csv(csv_path)
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Reshape and fit output_scaler
    r_data = csv_data['R'].values.reshape(-1, 1)
    output_scaler.fit(r_data)
    
    # Save output_scaler
    joblib.dump(output_scaler, output_scaler_file)
    print(f"Saved output_scaler into file: {output_scaler_file}")
    
    return scalers_array, output_scaler