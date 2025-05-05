import h5py
import hdf5plugin
import torch
from torch.utils.data import Dataset
class H5Dataset(Dataset):
    """HDF5 ファイル群から LiDAR スキャン、前回アクション、アクションを読み込む Dataset"""
    def __init__(self, h5_files):
        self.samples = []
        for fpath in sorted(h5_files):
            with h5py.File(fpath, 'r') as f:
                scans = f['scans'][:]
                prev = f['prev_actions'][:]
                actions = f['actions'][:]
            for scan, pr, act in zip(scans, prev, actions):
                self.samples.append((
                    scan.astype('float32'),
                    pr.astype('float32'),
                    act.astype('float32')
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scan, pr, act = self.samples[idx]
        return {
            'scan': torch.from_numpy(scan),
            'prev': torch.from_numpy(pr),
            'action': torch.from_numpy(act)
        }