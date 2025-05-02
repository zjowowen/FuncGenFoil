import random
from typing import List, Dict

import h5py
import numpy as np
import torch

from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, LazyMemmapStorage


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        std_cst_augmentation,
        num_perturbed_airfoils=10,
        dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
        max_size: int = 100000,
        folder_path="airfoil_generation/data",
    ):
        self.split = split
        self.std_cst_augmentation = std_cst_augmentation
        self.num_perturbed_airfoils = num_perturbed_airfoils
        self.key_list = []
        self.params = {}
        self.len = 0
        self.storage = LazyMemmapStorage(max_size=max_size)
        self.dataset_names = []
        self.load_data(dataset_names, folder_path=folder_path)
        self.get_min_max()

    def load_data(
        self,
        dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
        folder_path="airfoil_generation/data",
    ):
        for dataset_name in dataset_names:
            with open(
                f"{folder_path}/{dataset_name}/{dataset_name}_parsec_params.txt"
            ) as f:
                for line in f.readlines():
                    name_params = line.strip().split(",")
                    # 取出路径的最后一个文件名作为key
                    name = name_params[0]
                    self.params[name] = np.array(list(map(float, name_params[1:])))

            with open(
                f"{folder_path}/{dataset_name}/{dataset_name}_{self.split}.txt"
            ) as f:
                temp_key_list = [line.strip() for line in f.readlines()]

            with h5py.File(
                f"{folder_path}/{dataset_name}/{dataset_name}_airfoils.h5", "r"
            ) as f:
                for key in temp_key_list:
                    temp_data = torch.from_numpy(f[key][:])
                    params = torch.from_numpy(self.params[key])
                    temp_aug_data = torch.concat(
                        [
                            torch.from_numpy(
                                f[f"{key}_{self.std_cst_augmentation}_{i:02d}"][:]
                            ).unsqueeze(0)
                            for i in range(self.num_perturbed_airfoils)
                        ]
                    )
                    self.extend_data(
                        {
                            "gt": temp_data.unsqueeze(0),
                            "params": params.unsqueeze(0),
                            "augmentation": temp_aug_data.unsqueeze(0),
                        }
                    )

        self.dataset_names += dataset_names

    def get_min_max(self):
        all_data = self.storage.get(range(self.len))["gt"].reshape(-1, 2)

        xmin = all_data.min(0)[0][0]
        xmax = all_data.max(0)[0][0]
        ymin = all_data.min(0)[0][1]
        ymax = all_data.max(0)[0][1]

        self.min, self.max = torch.tensor((xmin, ymin)), torch.tensor((xmax, ymax))
        return self.min, self.max

    def extend_data(self, data: Dict):
        # keys = ["gt", "params"]

        len_after_extend = self.len + 1

        self.storage.set(
            range(self.len, len_after_extend),
            TensorDict(
                data,
                batch_size=[1],
            ),
        )
        self.len = len_after_extend

    def __getitem__(self, index):
        data = self.storage.get(index=index)
        return data

    def __len__(self):
        return self.len


class ThreeDimensionalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_file_path="training.dat",
        modes_file_path="modes.dat",
        max_size: int = 200000,
    ):
        self.len = 0
        self.storage = LazyMemmapStorage(max_size=max_size)
        self.dataset_names = []
        self.load_data(training_file_path, modes_file_path)
        self.get_min_max()

    def load_data(self, training_file_path="training.dat", modes_file_path="modes.dat"):
        training = np.loadtxt(training_file_path)
        modes = np.loadtxt(modes_file_path)
        x_training = training[:, 10:60]
        x_training_modes = np.matmul(x_training, modes)
        x_training_modes = x_training_modes[:, None, :]

        self.extend_data(x_training_modes)

    def get_min_max(self):
        all_data = self.storage.get(range(self.len))["gt"]

        xmin = torch.tensor(0)
        xmax = torch.tensor(1)
        ymin = all_data.min()
        ymax = all_data.max()

        self.min, self.max = torch.tensor((xmin, ymin)), torch.tensor((xmax, ymax))
        return self.min, self.max

    def extend_data(self, data):
        # keys = ["gt", "params"]

        self.len = data.shape[0]

        self.storage.set(
            range(data.shape[0]),
            TensorDict(
                {"gt": data},
                batch_size=[data.shape[0]],
            ),
        )

    def __getitem__(self, index):
        data = self.storage.get(index=index)
        return data

    def __len__(self):
        return self.len


class AF200KDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        split: str,
        dataset_names=[
            "beziergan_gen",
            "cst_gen",
            "cst_gen_a",
            "cst_gen_b",
            "diffusion_gen",
            "interpolated_uiuc",
            "naca_gen",
            "supercritical_airfoil_af200k",
        ],
        max_size: int = 250000,
        folder_path="airfoil_generation/data",
    ):
        self.split = split
        self.key_list = []
        self.params = {}
        self.len = 0
        self.storage = LazyMemmapStorage(max_size=max_size)
        self.dataset_names = []
        self.load_data(dataset_names, folder_path=folder_path)
        self.get_min_max()

    def load_data(
        self,
        dataset_names=[
            "beziergan_gen",
            "cst_gen",
            "cst_gen_a",
            "cst_gen_b",
            "diffusion_gen",
            "interpolated_uiuc",
            "naca_gen",
            "supercritical_airfoil_af200k",
        ],
        folder_path="airfoil_generation/data",
    ):

        for dataset_name in dataset_names:

            with open(
                f"{folder_path}/{dataset_name}/{dataset_name}_parsec_params_11.txt"
            ) as f:
                for line in f.readlines():
                    name_params = line.strip().split(",")
                    # 取出路径的最后一个文件名作为key
                    name = name_params[0]
                    self.params[name] = np.array(list(map(float, name_params[1:])))

            with open(
                f"{folder_path}/{dataset_name}/{dataset_name}_{self.split}.txt"
            ) as f:
                temp_key_list = [line.strip() for line in f.readlines()]

            with h5py.File(
                f"{folder_path}/{dataset_name}/{dataset_name}_airfoils.h5", "r"
            ) as f:
                for key in temp_key_list:
                    temp_data = torch.from_numpy(f[key][:])
                    params = torch.from_numpy(self.params[key])
                    self.extend_data(
                        {"gt": temp_data.unsqueeze(0), "params": params.unsqueeze(0)}
                    )

        self.dataset_names += dataset_names

    def get_min_max(self):

        all_data = self.storage.get(range(self.len))["gt"].reshape(-1, 2)

        xmin = all_data.min(0)[0][0]
        xmax = all_data.max(0)[0][0]
        ymin = all_data.min(0)[0][1]
        ymax = all_data.max(0)[0][1]

        self.min, self.max = torch.tensor((xmin, ymin)), torch.tensor((xmax, ymax))
        return self.min, self.max

    def extend_data(self, data: Dict):

        # keys = ["gt", "params"]

        len_after_extend = self.len + 1

        self.storage.set(
            range(self.len, len_after_extend),
            TensorDict(
                data,
                batch_size=[1],
            ),
        )
        self.len = len_after_extend

    def __getitem__(self, index):

        data = self.storage.get(index=index)
        return data

    def __len__(self):
        return self.len


if __name__ == "__main__":
    dataset = ThreeDimensionalDataset()
    # dataset = Dataset(split='train', std_cst_augmentation=0.08, num_perturbed_airfoils=10, dataset_names=['supercritical_airfoil', 'data_4000', 'r05', 'r06'], max_size=100000)
    b = 1
