import torch
import numpy as np
from torch.utils.data import Dataset

class MahjongDataset(Dataset):
    def __init__(self, npz_path):
        super().__init__()
        print(f"Loading {npz_path} ...")
        self.data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.obs = self.data['obs']          # [N, 45, 4, 9]
        self.actions = self.data['actions']  # [N, ]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        action_dim = 235
        input_dict = {
            "observation": torch.from_numpy(self.obs[idx]).float(),
            "action_mask": torch.ones(action_dim, dtype=torch.float)
        }
        action_label = torch.tensor(self.actions[idx], dtype=torch.long)
        return input_dict, action_label