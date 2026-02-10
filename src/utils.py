# ===========================
import pandas as pd

import tqdm

import os
import sys
sys.path.append('../')
sys.path.append('../src')

import config as c
import lidar_utils as lu
# ===========================


def first_preprocessing():
    """
    Take all raw landscape and apply our first pre-processing step on them :
    - Add label column
    - Compute Cartesian coordinate x, y and z
    - Add frame index column
    - Keep only columns of interest
    Output a list of processed dataframe
    """
    landscape_dfs = []

    class_df = pd.DataFrame({
        "r": [38, 177, 129, 66],
        "g": [23, 132, 81, 132],
        "b": [180, 47, 97, 9],
        "label": list(c.LABEL_MAP.values())[:-1]
    })


    for file in os.listdir(os.path.join(c.OUT_DIR, c.RAW_DATA_PATH)):
        
        df = lu.load_h5_data(os.path.join(c.OUT_DIR, c.RAW_DATA_PATH, file))
        df = df.iloc[df['distance_cm'] != 0]
    
        df = df.merge(class_df, on = ["r", "g", "b"], how = "left")
        df["label"] = df["label"].fillna(c.LABEL_MAP['Other'])
        df['label'] = df['label'].astype('int32')

        df[['x', 'y', 'z']] = lu.spherical_to_local_cartesian(df)

        frames = lu.get_unique_poses(df)
        df = df.merge(frames, on = ["ego_x", "ego_y", "ego_z", "ego_yaw"], how = "left")

        df = df.drop(columns = ['distance_cm', 'azimuth_raw', 'elevation_raw', 'r', 'g', 'b', 'ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'num_points'])

        landscape_dfs.append(df)

    return landscape_dfs