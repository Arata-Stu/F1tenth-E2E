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
                scans   = f['scans'][:]
                prev    = f['prev_actions'][:]
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
            'scan':   torch.from_numpy(scan),
            'prev':   torch.from_numpy(pr),
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
        for fid, f in enumerate(self.files):
            length = f['scans'].shape[0]
            for i in range(length):
                self.indices.append((fid, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fid, i = self.indices[idx]
        scan = self.files[fid]['scans'][i]
        return torch.from_numpy(scan).float()

    def __del__(self):
        for f in self.files:
            f.close()

class H5SequenceDataset(Dataset):
    """
    HDF5 ファイル群から時系列長 seq_len のシーケンスを
    スライディングウィンドウでサンプリングする Dataset

    waypoints は HDF5 上で shape=(L, N, 2) として保存されている想定で，
    __getitem__ で (T, N, 2)→(T, N*2) に flatten します。
    """
    def __init__(self, root_dir: str, seq_len: int):
        """
        root_dir: データが {run_id}/{map_name}/*.h5 に入っているフォルダ
        seq_len:  サンプルの時系列長 T
        """
        self.seq_len = seq_len
        self.files = sorted(glob.glob(os.path.join(root_dir, "*", "*.h5")))
        self.index_map = []
        for fi, path in enumerate(self.files):
            with h5py.File(path, 'r') as f:
                length = f['scans'].shape[0]
            for start in range(length - seq_len + 1):
                self.index_map.append((fi, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        fi, start = self.index_map[idx]
        path = self.files[fi]
        with h5py.File(path, 'r') as f:
            scans    = f['scans'][start:start+self.seq_len]        # (T, num_beams)
            way_raw  = f['waypoints'][start:start+self.seq_len]    # (T, N, 2)
            prev_act = f['prev_actions'][start:start+self.seq_len] # (T, 2)
            actions  = f['actions'][start:start+self.seq_len]      # (T, 2)

        # waypoints を flatten: (T, N, 2) → (T, N*2)
        N = way_raw.shape[1]
        waypts = way_raw.reshape(self.seq_len, N * 2)

        return (
            torch.from_numpy(scans).float(),   # (T, num_beams)
            torch.from_numpy(waypts).float(),  # (T, N*2)=>(T,20)
            torch.from_numpy(prev_act).float(),# (T, 2)
            torch.from_numpy(actions).float()  # (T, 2)
        )
