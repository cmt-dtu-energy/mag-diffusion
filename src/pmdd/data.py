import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MagnetismData2D(Dataset):
    def __init__(self, datapath, db_name, max_=False, norm_=False) -> None:
        self.fields = h5py.File(datapath / db_name, "r")["field"]
        self.len = self.fields.shape[0]
        self.max_val = np.max(np.abs(self.fields))
        self.max_ = max_
        self.norm_ = norm_
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.0, 0.0), (np.std(self.fields), np.std(self.fields))
                ),
            ]
        )

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> torch.Tensor:
        if self.max_:
            field = self.fields[idx] / self.max_val
        else:
            field = self.transform(self.fields[idx].transpose(1, 2, 0))

        return field
