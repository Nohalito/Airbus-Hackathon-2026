# ===========================
import pandas as pd
import numpy as np

import h5py

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

    path = os.path.join(c.OUT_DIR, c.RAW_DATA_PATH)

    for file in tqdm.tqdm(os.listdir(path), desc="Processing files"):
            
        df = lu.load_h5_data(os.path.join(c.OUT_DIR, c.RAW_DATA_PATH, file))
        df = df.iloc[df['distance_cm'] != 0]
    
        df = df.merge(class_df, on = ["r", "g", "b"], how = "left")
        df["label"] = df["label"].fillna(c.LABEL_MAP['Other'])
        df = df.iloc[df['label'] != 4]
        df['label'] = df['label'].astype('int32')
        df[['x', 'y', 'z']] = lu.spherical_to_local_cartesian(df)

        frames = lu.get_unique_poses(df)
        df = df.merge(frames, on = ["ego_x", "ego_y", "ego_z", "ego_yaw"], how = "left")

        df = df.drop(columns = ['distance_cm', 'azimuth_raw', 'elevation_raw', 'r', 'g', 'b', 'ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'num_points'])

        landscape_dfs.append(df)

    return landscape_dfs



def subsample_frame(frame_df, n_points = 4096):
    """
    Take the dataframe of a single LIDAR scan and sample randomly a given amount of point
    """

    #frame_df = frame_df[frame_df["label"] != 33]

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

    idxs = np.asarray(idxs, dtype=np.int64)

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

                fg.create_dataset(
                    "points",
                    data=pts.astype(np.float32),
                    compression="gzip"
                )
                fg.create_dataset(
                    "labels",
                    data=lbls.astype(np.uint8),
                    compression="gzip"
                )



def fit_nn(epochs, model, criterion, optimizer, train_dl, valid_dl):
    model_name = c.SELECTED_MODEL
    valid_loss_min = np.inf
    len_train, len_valid = 13720, 980
    fields = [
        'epoch', 'train_loss', 'train_acc', 'valid_loss', 'valid_acc'
    ]
    rows = []

    for epoch in range(epochs):
        train_loss, train_correct = 0, 0
        train_loop = tqdm(train_dl)

        model.train()
        for batch in train_loop:
            images, labels = batch[0].to(device), batch[1].to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += get_num_correct(preds, labels)

            train_loop.set_description(f'Epoch [{epoch+1:2d}/{epochs}]')
            train_loop.set_postfix(
                loss = loss.item(), acc=train_correct/len_train
            )
        train_loss = train_loss/len_train
        train_acc = train_correct/len_train

        model.eval()
        with torch.no_grad():
            valid_loss, valid_correct = 0, 0
            for batch in valid_dl:
                images, labels = batch[0].to(device), batch[1].to(device)
                preds = model(images)
                loss = criterion(preds, labels)
                valid_loss += loss.item() * labels.size(0)
                valid_correct += get_num_correct(preds, labels)

            valid_loss = valid_loss/len_valid
            valid_acc = valid_correct/len_valid

            rows.append([epoch, train_loss, train_acc, valid_loss, valid_acc])

            train_loop.write(
                f'\n\t\tAvg train loss: {train_loss:.6f}', end='\t'
            )
            train_loop.write(f'Avg valid loss: {valid_loss:.6f}\n')

            # save model if validation loss has decreased
            # (sometimes also referred as "Early stopping")
            if valid_loss <= valid_loss_min:
                train_loop.write('\t\tvalid_loss decreased', end=' ')
                train_loop.write(f'({valid_loss_min:.6f} -> {valid_loss:.6f})')
                train_loop.write('\t\tsaving model...\n')
                torch.save(
                    model.state_dict(),
                    f'../models/lr3e-5_{model_name}_{device}.pth'
                )
                valid_loss_min = valid_loss

    # write running results for plots
    with open(os.path.join(c.OUT_DIR, c.CSV_PATH, f'{model_name}.csv'), 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        csv_writer.writerows(rows)