import os
import glob
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
    
class H5ScansDataset(Dataset):
    """
    HDF5 フォルダ内の .h5 ファイルから 'scans' データセットのみを読み込み
    x: torch.Tensor shape=(num_beams,)
    """
    def __init__(self, root_dir):
        self.paths = glob.glob(os.path.join(root_dir, "*", "*.h5"))
        self.files = [h5py.File(p, 'r') for p in self.paths]
        self.indices = []
        # すべてのファイルを通してインデックスを構築
        for fid, f in enumerate(self.files):
            length = f['scans'].shape[0]
            for i in range(length):
                self.indices.append((fid, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fid, i = self.indices[idx]
        scan = self.files[fid]['scans'][i]  # numpy array shape=(num_beams,)
        scan = torch.from_numpy(scan).float()
        return scan

    def __del__(self):
        for f in self.files:
            f.close()