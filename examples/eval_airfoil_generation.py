import datetime
import argparse

import os
import torch.multiprocessing as mp


import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data import SliceSamplerWithoutReplacement, SliceSampler, RandomSampler

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

import wandb
from rich.progress import track
from easydict import EasyDict
from scipy.interpolate import splev, splprep
from scipy.optimize import minimize
from scipy import optimize

from airfoil_generation.neural_networks.dit import PointDiT, PointDiTForGRL
from airfoil_generation.dataset import Dataset, AF200KDataset
from airfoil_generation.utils import vis_airfoil2, de_norm, cst_fit
from airfoil_generation.dataset.parsec_direct_n15 import Fit_airfoil

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
)
from airfoil_generation.dataset.toy_dataset import MaternGaussianProcess
from airfoil_generation.neural_networks.neural_operator import FourierNeuralOperator
from scipy.spatial.distance import pdist, squareform
from scipy.stats import linregress


class Fit_airfoil_11:
    '''
    Fit airfoil by 3 order Bspline and extract Parsec features.
    airfoil (npoints,2)
    '''
    def __init__(self,airfoil,iLE=128):
        airfoil = airfoil[::((airfoil.shape[0]-1)//256)]
        self.iLE = int((airfoil.shape[0] - 1) / 2)
        self.tck, self.u = splprep(airfoil.T,s=0)

        # parsec features
        rle = self.get_rle()
        xup, yup, yxxup = self.get_up()
        xlo, ylo, yxxlo = self.get_lo()
        yteup = airfoil[0,1]
        ytelo = airfoil[-1,1]
        alphate, betate = self.get_te_angle(airfoil)

        self.parsec_features = np.array([rle,xup,yup,yxxup,xlo,ylo,yxxlo,
                                         (yteup+ytelo)/2,yteup-ytelo,alphate,betate])

    def get_rle(self):
        uLE = self.u[self.iLE]
        xu,yu = splev(uLE, self.tck,der=1) # dx/du
        xuu,yuu = splev(uLE, self.tck,der=2) # ddx/du^2
        K = abs(xu*yuu-xuu*yu)/(xu**2+yu**2)**1.5 # curvature
        return 1/K
    
    def get_up(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return -y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(0,self.u[self.iLE]),method='bounded')
        uup = res.x
        xup ,yup = splev(uup, self.tck)

        xu,yu = splev(uup, self.tck, der=1) # dx/du
        xuu,yuu = splev(uup, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xup, yup, yxx

    def get_lo(self):
        def f(u_tmp):
            x_tmp,y_tmp = splev(u_tmp, self.tck)
            return y_tmp
        
        res = optimize.minimize_scalar(f,bounds=(self.u[self.iLE],1),method='bounded')
        ulo = res.x
        xlo ,ylo = splev(ulo, self.tck)

        xu,yu = splev(ulo, self.tck, der=1) # dx/du
        xuu,yuu = splev(ulo, self.tck, der=2) # ddx/du^2
        # yx = yu/xu
        yxx = (yuu*xu-xuu*yu)/xu**3
        return xlo, ylo, yxx

    def get_te_angle(self, airfoil):
        # xu,yu = splev(0, self.tck, der=1)
        # yx = yu/xu
        # alphate = np.arctan(yx)

        # xu,yu = splev(1, self.tck, der=1)
        # yx = yu/xu
        # betate = np.arctan(yx)
        # alphate = np.arctan((airfoil[0,1]-airfoil[1,1])/(airfoil[0,0]-airfoil[1,0]))
        # betate = np.arctan((airfoil[-1,1]-airfoil[-2,1])/(airfoil[-1,0]-airfoil[-2,0]))

        n = int(0.02*airfoil.shape[0])
        k1 = linregress(airfoil[:n,0], airfoil[:n,1])[0]
        k2 = linregress(airfoil[-n:,0], airfoil[-n:,1])[0]
        alphate = np.arctan(k1)
        betate = np.arctan(k2)
        return alphate, betate

def calculate_smoothness(airfoil):
    smoothness = 0.0
    num_points = airfoil.shape[0]

    for i in range(num_points):
        p_idx = (i - 1) % num_points
        q_idx = (i + 1) % num_points

        p = airfoil[p_idx]
        q = airfoil[q_idx]

        if p[0] == q[0]:  # 处理垂直于x轴的线段
            distance = abs(airfoil[i, 0] - p[0])
        else:
            m = (q[1] - p[1]) / (q[0] - p[0])
            b = p[1] - m * p[0]

            distance = abs(m * airfoil[i, 0] - airfoil[i, 1] + b) / np.sqrt(m**2 + 1)

        smoothness += distance

    return smoothness

def cal_diversity_score(data, subset_size=10, sample_times=10):
    # Average log determinant
    N = data.shape[0]
    data = data.reshape(N, -1) 
    mean_logdet = 0
    for i in range(sample_times):
        ind = np.random.choice(N, size=subset_size, replace=False)
        subset = data[ind]
        D = squareform(pdist(subset, 'euclidean'))
        S = np.exp(-0.5*np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        mean_logdet += logdet
    return mean_logdet/sample_times

def cal_mean(arr):
    # 计算去掉最大和最小5%数据后的平均值
    percentile_5 = np.percentile(arr, 5)
    percentile_95 = np.percentile(arr, 95)

    # 过滤数据
    filtered_data = arr[(arr > percentile_5) & (arr < percentile_95)]

    # 计算剩余数据的平均值
    mean_value = filtered_data.mean()
    return mean_value

def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with=None, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index

    set_seed(seed=42+process_rank)

    print(f"Process rank: {process_rank}")

    project_name = "airfoil-evaluation"
    config = EasyDict(
        dict(
            device=device,
            flow_model=dict(
                device=device,
                gaussian_process=dict(
                    length_scale=0.03,
                    nu=2.5,
                    dims=[257],
                ),
                solver=dict(
                    type="ODESolver",
                    args=dict(
                        library="torchdiffeq",
                    ),
                ),
                path=dict(
                    sigma=1e-4,
                    device=device,
                ),
                model=dict(
                    type="velocity_function",
                    args=dict(
                        backbone=dict(
                            type="FourierNeuralOperator",
                            args=dict(
                                modes=48,
                                vis_channels=1,
                                hidden_channels=256,
                                proj_channels=128,
                                x_dim=1,
                                t_scaling=1,
                                n_layers=4,
                                n_conditions=15,
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                train_samples=20000 if args.dataset == 'supercritical' else 200000,
                batch_size=1024,
                learning_rate=5e-5 * accelerator.num_processes,
                iterations=20000 // 1024 * 2000 if args.dataset == 'supercritical' else 200000 // 1024 * 2000,
                warmup_steps=2000 if args.dataset == 'supercritical' else 20000 // 1024 * 2000,
                log_rate=100,
                eval_rate=20000 // 1024 * 500 if args.dataset == 'supercritical' else 200000 // 1024 * 500,
                checkpoint_rate=20000 // 1024 * 500 if args.dataset == 'supercritical' else 200000 // 1024 * 500,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=args.model_path,
            ),
        )
    )

    flow_model = OptimalTransportFunctionalFlow(
        config=config.flow_model,
    )

    if config.parameter.model_load_path is not None and os.path.exists(
        config.parameter.model_load_path
    ):
        # pop out _metadata key
        state_dict = torch.load(config.parameter.model_load_path, map_location="cpu", weights_only=False)
        state_dict.pop("_metadata", None)
        flow_model.model.load_state_dict(state_dict)

    flow_model.model = accelerator.prepare(flow_model.model)

    os.makedirs(config.parameter.model_save_path, exist_ok=True)

    train_dataset = Dataset(
        split="train",
        std_cst_augmentation=0.08,
        num_perturbed_airfoils=10,
        dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
        max_size=100000,
    ) if args.dataset == 'supercritical' else (
        AF200KDataset(split="train")
    )

    test_dataset = Dataset(
        split="test",
        std_cst_augmentation=0.08,
        num_perturbed_airfoils=10,
        dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
        max_size=100000,
    ) if args.dataset == 'supercritical' else (
        AF200KDataset(split="test")
    )

    data_matrix = torch.from_numpy(np.array(list(train_dataset.params.values())))
    train_dataset_std, train_dataset_mean = torch.std_mean(data_matrix, dim=0)
    train_dataset_std = torch.where(torch.isnan(train_dataset_std) | torch.isinf(train_dataset_std), torch.tensor(0.0), train_dataset_std)
    train_dataset_std = train_dataset_std.to(device)
    train_dataset_mean = train_dataset_mean.to(device)

    # acclerate wait for every process to be ready
    accelerator.wait_for_everyone()

    test_replay_buffer = TensorDictReplayBuffer(
        storage=test_dataset.storage,
        batch_size=1,
        sampler=SamplerWithoutReplacement(drop_last=False, shuffle=False),
        prefetch=10,
    )

    accelerator.init_trackers("airfoil-evaluation", config=None)
    accelerator.print("✨ Start training ...")

    # breakpoint()
    flow_model.eval()
    resolution = config.flow_model.gaussian_process.dims[0]
    rs = [resolution]
    l = len(rs)
    label_error = np.zeros((len(test_replay_buffer),l,11))
    smoothness = np.zeros((len(test_replay_buffer),l,))
    diversity = np.zeros((len(test_replay_buffer),l,))

    with torch.no_grad():
        for i, data in enumerate(test_replay_buffer):
            data = data.to(device)
            y = (data['params'][:,:] - train_dataset_mean[None,:]) / (train_dataset_std[None,:] + 1e-8)  # (b,15)

            priors = []
            for r in rs:
                priors.append(flow_model.gaussian_process.sample(dims=[r], n_samples=20, n_channels=1))

            sample_trajectorys = []
            for r, prior in zip(rs, priors):
                sample_trajectorys.append(flow_model.sample_process(
                    n_dims=[r],
                    n_channels=1,
                    t_span=torch.linspace(0.0, 1.0, 10),
                    batch_size=1,
                    x_0=prior,
                    condition=y.repeat(20,1)
                ))

            data_list_list = []
            for sample_trajectory in sample_trajectorys:
                data_list_list.append([
                    x.squeeze(0).cpu().numpy() for x in torch.split(sample_trajectory, split_size_or_sections=1, dim=0)
                ])
            
            label_error_ = np.zeros((l,11))
            smoothness_ = np.zeros((l,))

            for j, data_list in enumerate(data_list_list):
                data_list_ = []
                airfoils = data_list[-1][0,:,0,:]

                for airfoil in airfoils:
                    airfoil = (airfoil + 1) / 2 * (train_dataset.max.cpu().numpy()[1] - train_dataset.min.cpu().numpy()[1]) + train_dataset.min.cpu().numpy()[1]
                    data_list_.append(airfoil)
                    xs = (np.cos(np.linspace(0, 2*np.pi, airfoil.shape[-1])) + 1) / 2
                    parsec_params = Fit_airfoil_11(np.stack([xs,airfoil], axis=-1)).parsec_features
                    label_error_[j] += np.abs(parsec_params - data['params'].reshape(-1).cpu().numpy())
                    smoothness_[j] += calculate_smoothness(np.stack([xs,airfoil], axis=-1))

                label_error_[j] /= len(airfoils)
                smoothness_[j] /= len(airfoils)
                label_error[i,j] = label_error_[j]
                smoothness[i,j] = smoothness_[j]
                data_list_ = np.array(data_list_)
                diversity[i,j] = cal_diversity_score(data_list_)
                print(np.mean(label_error[i,j]), label_error[i,j], smoothness[i,j], diversity[i,j])

        np.save(f"output/{project_name}/label_error.npy", label_error)
        np.save(f"output/{project_name}/smoothness.npy", smoothness)
        np.save(f"output/{project_name}/diversity.npy", diversity)

        for i, r in enumerate(rs):
            print('Resolution: ', r)
            for arr in label_error[:,i,:].T:
                print(cal_mean(arr))
            print(cal_mean(np.mean(label_error[:,i,:], axis=-1)))
            print(cal_mean(smoothness[:,i]))
            print(cal_mean(diversity[:,i]))

        accelerator.wait_for_everyone()

    accelerator.print("✨ Training complete!")
    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description='train_parser')
    argparser.add_argument('--dataset', '-d', default='supercritical', type=str, choices=['supercritical', 'af200k'], help="Choose a dataset.")
    argparser.add_argument('--model_path', '-p', type=str, help="Model load path.")
    args = argparser.parse_args()
    main(args)
