## Code borrowed from:
## https://github.com/microsoft/MICO/blob/main/src/mico-competition/mico.py
## https://github.com/microsoft/MICO/blob/main/src/mico-competition/challenge_datasets.py

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
from typing import List, Optional, Union, Type, TypeVar
from torch.utils.data import Dataset

Y = TypeVar("Y", bound="MLP")

class MLP(nn.Module):
    """
    The fully-connected network architecture from Bao et al. (2022).
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(600, 128), nn.Tanh(),
            nn.Linear(128, 100)
        )
        ## apply Xavier to each Linear immediately:
        # for layer in self.mlp:
        #     if isinstance(layer, nn.Linear):
        #         init.xavier_uniform_(layer.weight)
        #         init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    @classmethod
    def load(cls: Type[Y], path: Union[str, os.PathLike]) -> Y:
        model = cls()
        state_dict = torch.load(path)
        new_state_dict = OrderedDict((k.replace('_module.', ''), v) for k, v in state_dict.items())
        model.load_state_dict(new_state_dict)
        model.eval()
        return model


def load_model(task: str, path: Union[str, os.PathLike]) -> nn.Module:
    return MLP.load(os.path.join(path, 'model.pt'))
    

class Purchase100(Dataset):
    """
    Purchase100 dataset pre-processed by Shokri et al.
    (https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz).
    We save the dataset in a .pickle version because it is much faster to load
    than the original file.
    """
    def __init__(self, dataset_dir: str) -> None:
        import pickle

        dataset_path = os.path.join(dataset_dir, 'purchase100', 'dataset_purchase')

        # Saving the dataset in pickle format because it is quicker to load.
        dataset_path_pickle = dataset_path + '.pickle'

        if not os.path.exists(dataset_path) and not os.path.exists(dataset_path_pickle):
            raise ValueError("Purchase-100 dataset not found.\n"
                             "You may download the dataset from https://www.comp.nus.edu.sg/~reza/files/datasets.html\n"
                            f"and unzip it in the {dataset_dir}/purchase100 directory")

        with open(dataset_path_pickle, 'rb') as f:
            dataset = pickle.load(f)['dataset']

        self.labels = list(dataset[:, 0] - 1)
        self.records = torch.FloatTensor(dataset[:, 1:])
        assert len(self.labels) == len(self.records), f'ERROR: {len(self.labels)} and {len(self.records)}'
        print('Successfully loaded the Purchase-100 dataset consisting of',
            f'{len(self.records)} records and {len(self.records[0])}', 'attributes.')

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        return self.records[idx], self.labels[idx]


def load_purchase100(dataset_dir: str = ".") -> Dataset:
    """Loads the Purchase-100 dataset.
    """
    return Purchase100(dataset_dir)
