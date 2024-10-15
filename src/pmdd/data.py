import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MagnetismData2D(Dataset):
    def __init__(self, datapath, db_name) -> None:
        self.db = h5py.File(datapath / db_name, "r")["field"]
        self.len = self.db.shape[0]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.0, 0.0), (np.std(self.db), np.std(self.db))),
            ]
        )

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> torch.Tensor:
        # Move the field components to last dimension for transform function
        field = self.db[idx].transpose(1, 2, 0)
        return self.transform(field)
