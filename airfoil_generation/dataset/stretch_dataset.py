import os

import h5py
import numpy as np
import torch

from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage
from scipy.interpolate import splev, splprep
from scipy import optimize


class StretchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
        max_size=100000,
        folder_path="data",
        num_constraints=15,
        bpart_before=0.5,
        bpart_after=0.6,
        joint_derivative_order=2,
    ):
        self.split = split
        self.params = {}
        self.derivs = {}
        self.len = 0
        self.storage = LazyMemmapStorage(max_size=max_size)
        self.dataset_names = []
        self.num_constraints = num_constraints
        self.bpart_before = bpart_before
        self.apart_before = 1 - bpart_before
        self.bpart_after = bpart_after
        self.stretch_factor = bpart_after / bpart_before
        self.joint_derivative_order = joint_derivative_order
        N_X = 257
        xs = (np.cos(np.linspace(0, 2 * np.pi, N_X)) + 1) / 2
        self.keep_indices = (xs > self.apart_before - 1e-4)
        self.load_data(dataset_names, folder_path=folder_path)
        self.get_min_max()

    def load_data(
        self,
        dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
        folder_path="data",
    ):
        for dataset_name in dataset_names:
            parsec_params_file = os.path.join(
                folder_path,
                dataset_name,
                f"{dataset_name}_parsec_params_{self.num_constraints}.txt",
            )
            with open(parsec_params_file) as f:
                for line in f.readlines():
                    name_params = line.strip().split(",")
                    name = name_params[0]
                    self.params[name] = np.array(list(map(float, name_params[1:])))
            
            key_file = os.path.join(
                folder_path,
                dataset_name,
                f"{dataset_name}_{self.split}.txt"
            )
            with open(key_file) as f:
                temp_key_list = [line.strip() for line in f.readlines()]

            joint_derivs_file = os.path.join(
                folder_path,
                dataset_name,
                f"{dataset_name}_joint_derivs_bpart_before_{int(100*self.bpart_before)}_bpart_after_{int(100*self.bpart_after)}.txt",
            )
            h5_file = os.path.join(
                folder_path,
                dataset_name,
                f"{dataset_name}_airfoils.h5"
            )
            if not os.path.exists(joint_derivs_file):
                with h5py.File(h5_file, "r") as f:
                    key_list = list(f.keys())
                    len_list = list(map(len, key_list))
                    full_key_list = np.array(key_list)[np.array(len_list) == min(len_list)]
                    for key in full_key_list:
                        temp_data = torch.from_numpy(f[key][:] * (self.stretch_factor, 1))
                        tck, u = splprep(temp_data.T, s=0)
                        iLE = 128
                        def objective(u_tmp):
                            x_tmp, _ = splev(u_tmp, tck)
                            return (x_tmp - self.apart_before * self.stretch_factor)**2
                        uup = optimize.minimize_scalar(
                            objective, bounds=(0, u[iLE]), method="bounded"
                        ).x
                        ulo = optimize.minimize_scalar(
                            objective, bounds=(u[iLE], 1), method="bounded"
                        ).x
                        _, yup = splev(uup, tck)
                        dxduup, dyduup = splev(uup, tck, der=1)
                        d2xdu2up, d2ydu2up = splev(uup, tck, der=2)
                        dydxup = dyduup / dxduup
                        d2ydx2up = (d2ydu2up * dxduup - d2xdu2up * dyduup) / dxduup**3
                        _, ylo = splev(ulo, tck)
                        dxdulo, dydulo = splev(ulo, tck, der=1)
                        d2xdu2lo, d2ydu2lo = splev(ulo, tck, der=2)
                        dydxlo = dydulo / dxdulo
                        d2ydx2lo = (d2ydu2lo * dxdulo - d2xdu2lo * dydulo) / dxdulo**3
                        # 保存到 txt 文件
                        with open(joint_derivs_file, "a") as joint_f:
                            joint_f.write(
                                f"{key},{yup},{ylo},{dydxup},{dydxlo},{d2ydx2up},{d2ydx2lo}\n"
                            )
            with open(joint_derivs_file) as f:
                for line in f.readlines():
                    name_derivs = line.strip().split(",")
                    name = name_derivs[0]
                    self.derivs[name] = np.array(list(map(float, name_derivs[1:2*self.joint_derivative_order+3])))
            
            with h5py.File(h5_file, "r") as f:
                for key in temp_key_list:
                    temp_data = torch.from_numpy(f[key][self.keep_indices] * (self.stretch_factor, 1))
                    temp_data_ = torch.from_numpy(f[key][~self.keep_indices] * (self.stretch_factor, 1))
                    temp_params = self.params[key][[-6, -5, -4, -3]]
                    temp_params[0] = np.rad2deg(np.arctan(np.tan(np.deg2rad(temp_params[0]))/self.stretch_factor))
                    temp_params[2] = temp_params[2] * self.stretch_factor
                    params = torch.from_numpy(temp_params)
                    temp_derivs = torch.from_numpy(self.derivs[key])
                    self.extend_data(
                        {
                            "bpart": temp_data.unsqueeze(0),
                            "apart": temp_data_.unsqueeze(0),
                            "params": torch.cat([params.unsqueeze(0), temp_derivs.unsqueeze(0)], dim=-1),
                        }
                    )

        self.dataset_names += dataset_names

    def get_min_max(self):
        all_data = self.storage.get(range(self.len))["bpart"].reshape(-1, 2)

        xmin = all_data.min(0)[0][0]
        xmax = all_data.max(0)[0][0]
        ymin = all_data.min(0)[0][1]
        ymax = all_data.max(0)[0][1]

        self.min, self.max = torch.tensor((xmin, ymin)), torch.tensor((xmax, ymax))
        return self.min, self.max

    def extend_data(self, data: dict):
        # keys = ["bpart", "apart", "params"]

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
    dataset = StretchDataset('test')
