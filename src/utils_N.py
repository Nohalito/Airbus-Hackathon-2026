# ===========================
import pandas as pd
import numpy as np
import csv

import h5py

import tqdm

import os
import sys
sys.path.append('../')
sys.path.append('../src')

import config as c
import lidar_utils as lu

import torch
# ===========================


# =========================
# == 01 Pre-processing : ==
# =========================



def world_coordinates(df):
    """
    Convert cartesian coordinate into the landscape global coordinate using the 'ego' position
    """
    yaw = np.deg2rad(df['ego_yaw'].to_numpy() / 100.0)

    x_world = np.cos(yaw)*df['x'].to_numpy() - np.sin(yaw)*df['y'].to_numpy() + df['ego_x'].to_numpy() /100.0
    y_world = np.sin(yaw)*df['x'].to_numpy() + np.cos(yaw)*df['y'].to_numpy() + df['ego_y'].to_numpy()/100.0
    z_world = df['z'].to_numpy() + df['ego_z'].to_numpy()/100.0

    return np.column_stack((x_world, y_world, z_world))



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

    path = os.path.join(c.OUT_DIR, c.RAW_DATA_PATH)

    for file in tqdm.tqdm(os.listdir(path), desc="Processing files"):
            
        df = lu.load_h5_data(os.path.join(c.OUT_DIR, c.RAW_DATA_PATH, file))
        df = df.iloc[df['distance_cm'] != 0]
    
        df = df.merge(class_df, on = ["r", "g", "b"], how = "left")
        df["label"] = df["label"].fillna(c.LABEL_MAP['Other'])
        df = df.iloc[df['label'] != c.LABEL_MAP['Other']]
        df['label'] = df['label'].astype('int32')
        df[['x', 'y', 'z']] = lu.spherical_to_local_cartesian(df)
        #df[['x', 'y', 'z']] =  world_coordinates(df)

        frames = lu.get_unique_poses(df)
        df = df.merge(frames, on = ["ego_x", "ego_y", "ego_z", "ego_yaw"], how = "left")

        df = df.drop(columns = ['distance_cm', 'azimuth_raw', 'elevation_raw', 'r', 'g', 'b']) # 'ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'num_points'

        landscape_dfs.append(df)

    return landscape_dfs



def subsample_frame(frame_df, n_points = 4096):
    """
    Take the dataframe of a single LIDAR scan and sample randomly a given amount of point
    """

    labels = frame_df["label"].to_numpy(dtype=np.uint8)
    points = frame_df[["x", "y", "z", "reflectivity"]].to_numpy(dtype=np.float32)

    unique_classes = np.unique(labels)
    per_class = max(1, n_points // len(unique_classes))

    idxs = []

    for c in unique_classes:
        cls_idx = np.where(labels == c)[0]
        if len(cls_idx) <= per_class:
            idxs.extend(cls_idx)
        else:
            idxs.extend(
                np.random.choice(cls_idx, per_class, replace=False)
            )

    if len(idxs) < n_points:
        extra = np.random.choice(idxs, n_points - len(idxs), replace=True)
        idxs.extend(extra)

    idxs = np.asarray(idxs, dtype = np.int64)

    return points[idxs], labels[idxs]



def write_processed_data_h5(h5_path, landscape_dfs, landscape_ids, n_points = 4096):
    """
    Save the subsampled & processed data in the designated folder as a ".h5" file
    """
    with h5py.File(h5_path, "w") as h5f:
        for lid in landscape_ids:
            df = landscape_dfs[lid]
            lg = h5f.create_group(f"landscape_{lid}")

            for pose_idx, frame_df in df.groupby("pose_index"):
                pts, lbls = subsample_frame(frame_df, n_points)

                fg = lg.create_group(f"frame_{pose_idx:04d}")

                pose = np.array([
                    frame_df["ego_x"].iloc[0],
                    frame_df["ego_y"].iloc[0],
                    frame_df["ego_z"].iloc[0],
                    frame_df["ego_yaw"].iloc[0]
                ], dtype=np.float32)

                fg.create_dataset(
                    "points",
                    data = pts.astype(np.float32),
                    compression = "gzip"
                )
                fg.create_dataset(
                    "labels",
                    data = lbls.astype(np.uint8),
                    compression = "gzip"
                )
                

                fg.create_dataset(
                    "pose",
                    data=pose,
                    compression = "gzip"
                )


# ===================
# == 02 Training : ==
# ===================

# Ty for this guy training loop
# https://github.com/priyavrat-misra/xrays-and-gradcam?tab=readme-ov-file

def fit_nn(epochs, model, criterion, optimizer, train_loader, val_loader, device):
    """
    Model training loop
    """

    valid_loss_min = np.inf
    fields = ['epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc']
    rows = []

    for epoch in range(epochs):
        model.train()
        len_train = 0
        train_loss, train_correct = 0, 0
        train_loader = tqdm.tqdm(train_loader, desc = "batch beepobed :")

        for batch in train_loader:
            points, labels = batch[0].to(device), batch[1].to(device)
            preds = model(points)

            optimizer.zero_grad()
            loss = criterion(preds.permute(0, 2, 1), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.numel()
            train_correct += preds.argmax(dim = -1).eq(labels).sum().item()

            len_train += labels.numel()

        train_loss = train_loss/len_train
        train_acc = train_correct/len_train

        model.eval()
        with torch.no_grad():
            len_valid = 0
            valid_loss, valid_correct = 0, 0

            for batch in val_loader:
                points, labels = batch[0].to(device), batch[1].to(device)

                preds = model(points)
                loss = criterion(preds.permute(0, 2, 1), labels)

                valid_loss += loss.item() * labels.numel()
                valid_correct += preds.argmax(dim = -1).eq(labels).sum().item()

                len_valid += labels.numel()

            valid_loss = valid_loss/len_valid
            valid_acc = valid_correct/len_valid

            rows.append([epoch, train_loss, train_acc, valid_loss, valid_acc])

            train_loader.write(f'\n\t\tAvg train acc: {train_acc:.6f}', end='\t')
            train_loader.write(f'Avg valid acc: {valid_acc:.6f}\n')
            train_loader.write(f'\n\t\tAvg train loss: {train_loss:.6f}', end='\t')
            train_loader.write(f'Avg valid loss: {valid_loss:.6f}\n')

            if valid_loss <= valid_loss_min:
                    train_loader.write('\t\tvalid_loss decreased', end=' ')
                    train_loader.write(f'({valid_loss_min:.6f} -> {valid_loss:.6f})')
                    train_loader.write('\t\tsaving model...\n')
                    torch.save(
                        model.state_dict(),
                        f'../models/PointNetSeg_{device}.pth'
                    )
                    valid_loss_min = valid_loss

    with open(os.path.join(c.OUT_DIR, c.CSV_PATH, f'PointNetSeg_{device}.csv'), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        csv_writer.writerows(rows)


# =====================
# == 03 Evaluation : ==
# =====================

