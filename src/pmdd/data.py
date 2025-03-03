import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MagnetismData2D(Dataset):
    def __init__(self, datapath, db_name, max_=False, norm_=False, max_observations = None) -> None:
        self.fields = h5py.File(f"{datapath}/{db_name}", "r")["field"]
        self.len = self.fields.shape[0]
        self.max_val = np.max(np.abs(self.fields))
        self.max_ = max_
        self.norm_ = norm_
        self.max_observations = max_observations
        self.tf_ = [
                    transforms.ToTensor()
                ]
        if max_:
            self.tf_.append(transforms.Normalize(
                        (0.0, 0.0), (self.max_val, self.max_val)
                    ))
        else:
            self.tf_.append(transforms.Normalize(
                        (0.0, 0.0), (np.std(self.fields), np.std(self.fields)))
                    )
        self.transform = transforms.Compose(self.tf_)

    def __len__(self) -> int:
        if self.max_observations == None:
            return self.len
        else:
            return min(self.max_observations, self.len)

    def __getitem__(self, idx) -> torch.Tensor:
        field = self.transform(self.fields[idx].transpose(1, 2, 0))

        return field
