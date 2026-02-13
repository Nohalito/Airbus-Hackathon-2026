import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Turn off warning
import warnings
warnings.filterwarnings('ignore')



class PointCloudDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.index = []

        with h5py.File(h5_path, "r") as f:
            for landscape in f.keys():
                for frame in f[landscape].keys():
                    self.index.append((landscape, frame))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        landscape, frame = self.index[idx]

        with h5py.File(self.h5_path, "r") as f:
            grp = f[landscape][frame]
            points = grp["points"][:]
            labels = grp["labels"][:]

        return (
            torch.tensor(points, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
    


class PointCloudTestDataset(Dataset):

    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.index = []

        with h5py.File(h5_path, "r") as f:
            for landscape in f.keys():
                for frame in f[landscape].keys():
                    self.index.append((landscape, frame))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        landscape, frame = self.index[idx]

        with h5py.File(self.h5_path, "r") as f:
            grp = f[landscape][frame]
            points = grp["points"][:]
            labels = grp["labels"][:]
            pose = grp["pose"][:]

        return (
            torch.tensor(points, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(pose, dtype=torch.float32)
        )



class TNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k)
        x = x + identity.repeat(batch_size, 1)

        return x.view(-1, self.k, self.k)



class PointNetSeg(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()

        self.input_transform = TNet(k=4)
        self.feature_transform = TNet(k=64)

        self.conv1 = nn.Conv1d(4, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv6 = nn.Conv1d(1088, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, num_classes, 1)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, N, 4]
        B, N, _ = x.size()

        x = x.transpose(2, 1)  # [B, 4, N]

        T = self.input_transform(x)
        x = torch.bmm(T, x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        T_feat = self.feature_transform(x)
        x = torch.bmm(T_feat, x)

        pointfeat = x

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        global_feat = torch.max(x, 2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, N)

        #x = torch.cat([x, global_feat], 1)
        x = torch.cat([pointfeat, global_feat], 1)

        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)

        return x.transpose(2, 1)  # [B, N, num_classes]