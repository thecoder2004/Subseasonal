CUDA_VISIBLE_DEVICES=1 python main.py --cfg config/default.yaml  \
        --name strans-v4 \
        --in_channel 13 \
        --adding_type 0 \
        --dropout 0.5 \
        --height 17 \
        --width 17 \
        --data_idx_dir /mnt/disk1/tunn/Subseasonal_Prediction/data3/data6789_reg_1_seed52 \
        --gauge_data_path /mnt/disk1/env_data/Gauge_thay_Tan/Final_Data_Region_1.csv \
        --npyarr_dir /mnt/disk1/env_data/S2S_0.125/nparr_reg_1/Step24h \
        --processed_ecmwf_dir /mnt/disk1/env_data/S2S_0.125/data3_reg_1 \
        --lat_start 22.75 \
        --lon_start 102.75 \
        --use_layer_norm \
        --loss_func mse \
        --lr 5e-4 \
        --use_lrscheduler \
        --scheduler_type ReduceLROnPlateau \
        --plateau_patience 3 \
        --plateau_min_lr 1e-6 \
        --plateau_factor 0.5 --plateau_verbose \
        --group_name data3-r1-test \
        --batch_size 64