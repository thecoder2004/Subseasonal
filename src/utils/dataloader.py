import pandas as pd 
import numpy as np
import os 
import pickle
from torch.utils.data import Dataset
from datetime import datetime, timedelta
import time
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, mode='train', ecmwf_scaler=None, output_scaler=None, config=None, shuffle = False):
        self.config = config
        self.mode = mode
        self.shuffle = shuffle
        self.output_norm = config.TRAIN.OUTPUT_NORM
        # Load gauge data
        self.gauge_data = pd.read_csv(config.DATA.GAUGE_DATA_PATH)
        self.gauge_data['Day'] = pd.to_datetime(self.gauge_data['Day'])
        
        # Load station data
        station_data = self.gauge_data[['Station', 'Lon', 'Lat']].drop_duplicates('Station')
        self.stations = station_data['Station'].values
        
        # Load station coords
        self.station_coords = station_data[['Lon', 'Lat']].to_numpy()
        
        # Load idx
        self.idx_df = pd.read_csv(f'{config.DATA.DATA_IDX_DIR}/{mode}.csv').values
        if self.shuffle:
            np.random.shuffle(self.idx_df)
        # Load scaler
        self.ecmwf_scaler = ecmwf_scaler
        self.output_scaler = output_scaler
        
        # Path to store ecmwf data
        self.processed_ecmwf_dir = f'{config.DATA.PROCESSED_ECMWF_DIR}/processed_ecmwf/{mode}'
        os.makedirs(self.processed_ecmwf_dir, exist_ok=True)
        
        # Process and save ecmwf data
        self.process_and_save_ecmwf_data()
        
    def calculate_leadtime_date(self, year, month, day, lead_time):
        """
        Calculate the date after adding `lead_time` days to the given (year, month, day).
        """
        start_date = datetime(year, month, day)
        leadtime_date = start_date + timedelta(days=lead_time)
        return leadtime_date

        
    def process_and_save_ecmwf_data(self):
        for idx in tqdm(range(len(self.idx_df)), desc=f"Processing ecmwf_data for {self.mode}"):
            # Create path to store processed ecmwf data
            processed_ecmwf_path = f'{self.processed_ecmwf_dir}/ecmwf_data_{idx}.npy'
            
            # If processed ecmwf data not exist, process and save it
            if not os.path.exists(processed_ecmwf_path):
                ecmwf_path, lead_time, year, month, day = self.idx_df[idx]
                ecmwf_data = self.get_ecmwf(ecmwf_path, year, lead_time)  # (13, 7, 137, 121)
                ecmwf_data_scaled = self.transform_ecmwf(ecmwf_data)  # (13, 7, 137, 121)
                ecmwf_data_scaled = ecmwf_data_scaled.transpose(1, 0, 2, 3)  # (7, 13, 137, 121)
                np.save(processed_ecmwf_path, ecmwf_data_scaled)
    
    def get_ecmwf(self, ecmwf_path, year, lead_time):
        ecmwf_data = np.load(ecmwf_path) # (20, 13, 47, 137, 121)
        ecmwf_data = ecmwf_data[year-2002,:,lead_time-6:lead_time+1,:,:] # (13, 7, 137, 121) ecmwf_data[year-2002,:,:21,:,:] # 13, 21, 137, 121
        
        return ecmwf_data
    
    def get_ground_truth(self, year, month, day, lead_time):
        # Get start and end date
        start_date = datetime(year, month, day) + timedelta(days=lead_time - 6)
        end_date = datetime(year, month, day) + timedelta(days=lead_time)
        
        # Filter data by date
        mask = (self.gauge_data['Day'] >= start_date) & (self.gauge_data['Day'] <= end_date)
        period_data = self.gauge_data.loc[mask]
        
        # Init array to store ground truth
        num_station = len(self.stations)
        total_rain = np.zeros((num_station, 1))
        
        # Calculate total rain for each station
        for i, station in enumerate(self.stations):
            station_data = period_data[period_data['Station'] == station]
            total_rain[i, 0] = station_data['R'].sum()
           
        # Stack total rain and station coords 
        result = np.hstack((total_rain, self.station_coords))
        
        return result
    
    def transform_ecmwf(self, ecmwf_data):
        # Transform ecmwf_data
        ecmwf_data_scaled = np.zeros_like(ecmwf_data, dtype=np.float32)
        for i in range(13):
            feature_data = ecmwf_data[i].reshape(-1, 1)
            scaled_feature = self.ecmwf_scaler[i].transform(feature_data)
            ecmwf_data_scaled[i] = scaled_feature.reshape(7, self.config.DATA.HEIGHT, self.config.DATA.WIDTH)
        return ecmwf_data_scaled
    
    def transform_ground_truth(self, ground_truth):
        # Transform ground_truth
        ground_truth_scaled = ground_truth.copy()
        rain_data = ground_truth[:, 0].reshape(-1, 1)
        if self.output_norm:
            scaled_rain = self.output_scaler.transform(rain_data)
        else:
            scaled_rain = rain_data
            
        ground_truth_scaled[:, 0] = scaled_rain.flatten()
        return ground_truth_scaled
    
    def __getitem__(self, idx):
        ecmwf_path, lead_time, year, month, day = self.idx_df[idx]
        processed_ecmwf_path = f'{self.processed_ecmwf_dir}/ecmwf_data_{idx}.npy'
        ecmwf_data = np.load(processed_ecmwf_path)  # (7, 13, 137, 121)  # (7, 13, 137, 121)
        ground_truth = self.get_ground_truth(year, month, day, lead_time)  # (num_station, 3)
        ground_truth = self.transform_ground_truth(ground_truth)
        
        x_leadtime = self.calculate_leadtime_date(year, month, day, lead_time)
        
        #return {'x': ecmwf_data, 'lead_time': lead_time, 'y': ground_truth}
        
        return {'x': ecmwf_data, 'lead_time': lead_time, 'y': ground_truth, 'x_leadtime': x_leadtime, 'ecmwf': self.get_ecmwf(ecmwf_path=ecmwf_path, year=year, lead_time=lead_time).transpose(1, 0, 2, 3)}
    
    def __len__(self):
        return len(self.idx_df)
