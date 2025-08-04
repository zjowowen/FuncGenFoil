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
from torch.optim.optimizer import Optimizer, required

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data import SliceSamplerWithoutReplacement, SliceSampler, RandomSampler
from torchrl.data import LazyTensorStorage, LazyMemmapStorage

import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

import wandb
from rich.progress import track
from easydict import EasyDict

import copy

from airfoil_generation.neural_networks.dit import PointDiT, PointDiTForGRL
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset import Dataset

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
    OptimalTransportFunctionalFlowForRegression,
)
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset.toy_dataset import MaternGaussianProcess
from airfoil_generation.neural_networks.neural_operator import FourierNeuralOperator
from airfoil_generation.utils import find_parameters


def calculate_smoothness(airfoil):
    smoothness = 0.0
    num_points = airfoil.shape[0]

    for i in range(num_points):
        p_idx = (i - 1) % num_points
        q_idx = (i + 1) % num_points

        p = airfoil[p_idx]
        q = airfoil[q_idx]

        if p[0] == q[0]:  # 处理垂直于x轴的线段
            distance = torch.abs(airfoil[i, 0] - p[0])
        else:
            m = (q[1] - p[1]) / (q[0] - p[0])
            b = p[1] - m * p[0]

            distance = torch.abs(m * airfoil[i, 0] - airfoil[i, 1] + b) / torch.sqrt(
                m**2 + 1
            )

        smoothness += distance

    return smoothness


def plot_airfoil(
    y,
    save_path,
    train_dataset_max=None,
    train_dataset_min=None,
    scatter_x=None,
    scatter_y=None,
    scatter_y_noise=None,
    dpi=300,
):
    xs = (np.cos(np.linspace(0, 2 * np.pi, 257)) + 1) / 2
    if scatter_x is not None:
        scatter_x_cos = (np.cos(scatter_x * 2 * np.pi) + 1) / 2

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    if train_dataset_max is not None and train_dataset_min is not None:
        ax.plot(
            xs,
            (
                (y + 1) / 2 * (train_dataset_max[1] - train_dataset_min[1])
                + train_dataset_min[1]
            ),
            lw=0.1,
        )
        ax.scatter(
            xs,
            (
                (y + 1) / 2 * (train_dataset_max[1] - train_dataset_min[1])
                + train_dataset_min[1]
            ),
            s=0.01,
            c="b",
        )
        if scatter_x is not None:
            ax.scatter(scatter_x_cos, scatter_y, s=0.02, c="r")
            if scatter_y_noise is not None:
                ax.scatter(scatter_x_cos, scatter_y_noise, s=0.02, c="g")
    else:
        ax.plot(xs, y, lw=0.1)
        ax.scatter(xs, y, s=0.01, c="b")
        if scatter_x is not None:
            ax.scatter(scatter_x_cos, scatter_y, s=0.02, c="r")
            if scatter_y_noise is not None:
                ax.scatter(scatter_x_cos, scatter_y_noise, s=0.02, c="g")

    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, 0.1])
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)
    plt.clf()


def render_video_3x3(
    data_list,
    video_save_path,
    iteration,
    train_dataset_max,
    train_dataset_min,
    fps=100,
    dpi=100,
):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)

    xs = (np.cos(np.linspace(0, 2 * np.pi, 257)) + 1) / 2

    # Number of frames in the animation is just len(data_list) = 1000
    frames = len(data_list)

    # Create figure with 3x3 subplots
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    def update(frame_idx):
        data = data_list[frame_idx].squeeze()
        for j, ax in enumerate(axs.flat):
            ax.clear()
            ax.set_xlim([0, 1])
            ax.set_ylim([-0.1, 0.1])
            # ax.set_title(f"Subplot {j+1}")
            ax.plot(
                xs,
                (
                    (data[j, :] + 1) / 2 * (train_dataset_max[1] - train_dataset_min[1])
                    + train_dataset_min[1]
                ),
                lw=0.1,
            )
            ax.scatter(
                xs,
                (
                    (data[j, :] + 1) / 2 * (train_dataset_max[1] - train_dataset_min[1])
                    + train_dataset_min[1]
                ),
                s=0.01,
                c="b",
            )

        return []

    ani = animation.FuncAnimation(
        fig, update, frames=range(frames), interval=1000 / fps, blit=False
    )

    # Save animation as MP4
    save_path = os.path.join(video_save_path, f"iteration_{iteration}.mp4")
    ani.save(save_path, fps=fps, dpi=dpi)

    # Clean up
    plt.close(fig)
    plt.clf()
    print(f"Saved video to {save_path}")


def render_last_fig_3x3(
    data_list,
    video_save_path,
    iteration,
    train_dataset_max,
    train_dataset_min,
    scatter_x=None,
    scatter_y=None,
    scatter_y_noise=None,
    fps=100,
    dpi=100,
):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)

    xs = (np.cos(np.linspace(0, 2 * np.pi, 257)) + 1) / 2

    if scatter_x is not None:
        scatter_x_cos = (np.cos(scatter_x * 2 * np.pi) + 1) / 2

    # Number of frames in the animation is just len(data_list) = 1000
    frames = len(data_list)

    # Create figure with 3x3 subplots
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    data = data_list[-1].squeeze()
    for j, ax in enumerate(axs.flat):
        ax.clear()
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.1, 0.1])
        ax.plot(
            xs,
            (
                (data[j, :] + 1) / 2 * (train_dataset_max[1] - train_dataset_min[1])
                + train_dataset_min[1]
            ),
            lw=0.1,
        )
        ax.scatter(
            xs,
            (
                (data[j, :] + 1) / 2 * (train_dataset_max[1] - train_dataset_min[1])
                + train_dataset_min[1]
            ),
            s=0.01,
            c="b",
        )
        if scatter_x is not None:
            ax.scatter(scatter_x_cos, scatter_y, s=0.02, c="r")
            if scatter_y_noise is not None:
                ax.scatter(scatter_x_cos, scatter_y_noise, s=0.02, c="g")

    save_path = os.path.join(video_save_path, f"iteration_last_fig_{iteration}.png")
    plt.savefig(save_path, dpi=300)

    # Clean up
    plt.close(fig)
    plt.clf()
    print(f"Saved image to {save_path}")


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb" if args.wandb else None, kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index

    # set_seed(seed=42+process_rank)

    print(f"Process rank: {process_rank}")

    project_name = args.project_name  # "airfoil-editing-with-inverse-prior"
    config = EasyDict(
        dict(
            device=device,
            flow_model=dict(
                device=device,
                gaussian_process=dict(
                    type=args.kernel_type,
                    args={
                        "matern": dict(
                            device=device,
                            length_scale=args.length_scale,
                            nu=args.nu,
                            dims=[257],
                        ),
                        "rbf": dict(
                            device=device,
                            length_scale=args.length_scale,
                            dims=[257],
                        ),
                        "white": dict(
                            device=device,
                            noise_level=args.noise_level,
                            dims=[257],
                        ),
                    }.get(args.kernel_type, None),
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
                                modes=64,
                                vis_channels=1,
                                hidden_channels=256,
                                proj_channels=128,
                                x_dim=1,
                                t_scaling=1,
                                n_layers=6,
                                n_conditions=args.num_constraints,
                            ),
                        ),
                    ),
                ),
            ),
            flow_model_regression=dict(
                device=device,
                gaussian_process=dict(
                    type=args.kernel_type,
                    args={
                        "matern": dict(
                            device=device,
                            length_scale=args.length_scale,
                            nu=args.nu,
                            dims=[257],
                        ),
                        "rbf": dict(
                            device=device,
                            length_scale=args.length_scale,
                            dims=[257],
                        ),
                        "white": dict(
                            device=device,
                            noise_level=args.noise_level,
                            dims=[257],
                        ),
                    }.get(args.kernel_type, None),
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
                                modes=64,
                                vis_channels=1,
                                hidden_channels=256,
                                proj_channels=128,
                                x_dim=1,
                                t_scaling=1,
                                n_layers=6,
                                n_conditions=args.num_constraints,
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                noise_level=0.000003,
                batch_size=1024,
                learning_rate=5e-6 * accelerator.num_processes,
                iterations=11,
                warmup_steps=5,
                log_rate=1,
                eval_rate=10,
                model_save_rate=1000,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=args.model_path,
            ),
        )
    )

    flow_model = OptimalTransportFunctionalFlow(
        config=config.flow_model,
    )

    # pop out _metadata key
    state_dict = torch.load(
        config.parameter.model_load_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict.pop("_metadata", None)

    # Create a new dictionary with updated keys
    prefix = "_orig_mod."
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith(prefix):
            # Remove the prefix from the key
            new_key = key[len(prefix) :]
        else:
            new_key = key
        new_state_dict[new_key] = value

    flow_model.model.load_state_dict(new_state_dict)
    print("Model loaded from: ", config.parameter.model_load_path)

    train_dataset = Dataset(
        split="train",
        std_cst_augmentation=0.08,
        num_perturbed_airfoils=10,
        dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
        max_size=100000,
        folder_path=args.data_path,
        num_constraints=args.num_constraints,
    )
    test_dataset = Dataset(
        split="test",
        std_cst_augmentation=0.08,
        num_perturbed_airfoils=10,
        dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
        max_size=100000,
        folder_path=args.data_path,
        num_constraints=args.num_constraints,
    )
    print(f"Data number: {len(train_dataset)}")
    data_matrix = torch.from_numpy(np.array(list(train_dataset.params.values())))
    train_dataset_std, train_dataset_mean = torch.std_mean(data_matrix, dim=0)
    train_dataset_std = torch.where(
        torch.isnan(train_dataset_std) | torch.isinf(train_dataset_std),
        torch.tensor(0.0),
        train_dataset_std,
    )
    train_dataset_std = train_dataset_std.to(device)
    train_dataset_mean = train_dataset_mean.to(device)

    if not os.path.exists(f"output/{project_name}"):
        os.makedirs(f"output/{project_name}")

    # dataset_for_edit = test_dataset.storage[:300].clone().cpu()

    number_of_edit = 6

    edit_scale_sqrt = 0.0001
    edit_scale_sqrt = 0.0002
    edit_scale_sqrt = 0.0004

    edit_scale = edit_scale_sqrt * edit_scale_sqrt

    # idx_mask_list = []
    # for i in range(300):
    #     if i<100:
    #         number_of_edit = 2
    #     elif i<200:
    #         number_of_edit = 3
    #     else:
    #         number_of_edit = 4
    #     idx_mask = torch.zeros(config.flow_model.gaussian_process.args.dims[0])
    #     idx = torch.tensor(np.random.choice(config.flow_model.gaussian_process.args.dims[0], number_of_edit, replace=False))
    #     idx_mask[idx] = 1
    #     idx_mask = idx_mask == 1
    #     idx_mask_list.append(idx_mask)

    # idx_tensor = torch.stack(idx_mask_list, dim=0)

    # noise_pattern = torch.randn((300, config.flow_model.gaussian_process.args.dims[0]))

    # dataset_for_edit["pos_mask"]=idx_tensor
    # dataset_for_edit["noise_pattern"]=noise_pattern

    # torch.save(dataset_for_edit, f"output/{project_name}/test_dataset_for_editing.pth")

    dataset_for_edit = torch.load(
        f"output/{project_name}/test_dataset_for_editing.pth", map_location="cpu"
    )

    new_storage = LazyMemmapStorage(max_size=1000)
    new_storage.set(
        range(len(dataset_for_edit)),
        dataset_for_edit,
    )

    flow_model = flow_model.to(device)

    mse_list = []
    smooth_list = []

    flow_model_for_regression = OptimalTransportFunctionalFlowForRegression(
        config=config.flow_model_regression,
        model=copy.deepcopy(flow_model.model),
        prior=torch.zeros(1, 1, config.flow_model.gaussian_process.args.dims[0]),
    ).to(device)

    for model_index in track(range(300)):
        model_index_inverse = 299 - model_index

        x_test = (
            dataset_for_edit[model_index_inverse]["gt"][:, 1]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        noise_pattern = (
            dataset_for_edit[model_index_inverse]["noise_pattern"]
            .unsqueeze(0)
            .to(device)
        )
        pos_mask = dataset_for_edit[model_index_inverse]["pos_mask"].to(device)

        inverse_prior = flow_model.inverse_sample(
            n_dims=config.flow_model.gaussian_process.args.dims,
            n_channels=1,
            t_span=torch.linspace(0.0, 1.0, 1000),
            x_0=(x_test - train_dataset.min.to(device)[1])
            / (train_dataset.max.to(device)[1] - train_dataset.min.to(device)[1])
            * 2
            - 1,
        )

        # deep copy the flow model for regression
        state_dict = flow_model.model.state_dict()
        state_dict.pop("_metadata", None)
        flow_model_for_regression.model.load_state_dict(state_dict)
        flow_model_for_regression.prior = inverse_prior

        optimizer = torch.optim.Adam(
            find_parameters(flow_model_for_regression),
            lr=config.parameter.learning_rate,
        )

        scheduler = CosineAnnealingWarmupLR(
            optimizer,
            T_max=config.parameter.iterations,
            eta_min=2e-6,
            warmup_steps=config.parameter.warmup_steps,
        )

        # flow_model_for_regression.model, flow_model_for_regression.prior, optimizer = accelerator.prepare(flow_model_for_regression.model, flow_model_for_regression.prior, optimizer)

        x_range = torch.linspace(0, 1, config.flow_model.gaussian_process.args.dims[0])

        x_pos_mask = x_range[pos_mask.cpu()]

        x_partial_obs = x_test[-1, :, pos_mask].to(device)

        x_partial_obs_with_noise = x_partial_obs + noise_pattern[
            :, pos_mask
        ] * torch.sqrt(torch.tensor(edit_scale, device=device))

        # plot the 6 random observations and x_test, save to file

        # fig, ax = plt.subplots()
        # # ax.set_xlim([0, 257])
        # # ax.set_ylim([-3, 3])
        # x_test_np = x_test[-1, 0, :].cpu().numpy()
        # x_partial_obs_np = x_partial_obs[0].cpu().numpy()
        # x_partial_obs_with_noise_np = x_partial_obs_with_noise[0].cpu().numpy()
        # ax.scatter(np.arange(config.flow_model.gaussian_process.args.dims[0])[pos_mask.cpu()], x_partial_obs_np, s=3, color="red", label="partial observation")
        # ax.scatter(np.arange(config.flow_model.gaussian_process.args.dims[0])[pos_mask.cpu()], x_partial_obs_with_noise_np, s=3, color="green", label="partial observation with noise")
        # ax.scatter(np.arange(config.flow_model.gaussian_process.args.dims[0]), x_test_np, s=1, color="blue", label="full observation")
        # ax.legend()
        # if not os.path.exists(f"output/{project_name}"):
        #     os.makedirs(f"output/{project_name}")
        # plt.savefig(f"output/{project_name}/observations.png", dpi=300)
        # plt.close(fig)
        # plt.clf()
        # plot_airfoil(x_test[-1, 0, :].cpu().numpy(), f"output/{project_name}/airfoil_observation_train.png", scatter_x=x_pos_mask.cpu().numpy(), scatter_y=x_partial_obs_np, scatter_y_noise=x_partial_obs_with_noise_np, dpi=300)

        t_span = torch.linspace(0.0, 1.0, 10).to(device)
        x_partial_obs_norm = (x_partial_obs - train_dataset.min.to(device)[1]) / (
            train_dataset.max.to(device)[1] - train_dataset.min.to(device)[1]
        ) * 2 - 1
        x_partial_obs_with_noise_norm = (
            x_partial_obs_with_noise - train_dataset.min.to(device)[1]
        ) / (train_dataset.max.to(device)[1] - train_dataset.min.to(device)[1]) * 2 - 1

        xs = (torch.cos(torch.linspace(0, 2 * torch.pi, 257)) + 1) / 2
        xs = xs.to(device)

        mp_list = []

        for iteration in range(config.parameter.iterations):
            if False and iteration == 0:
                flow_model_for_regression.eval()
                with torch.no_grad():
                    sample_trajectory = flow_model_for_regression.sample_process(
                        n_dims=config.flow_model_regression.gaussian_process.args.dims,
                        n_channels=1,
                        t_span=torch.linspace(0.0, 1.0, 100),
                        x_0=flow_model_for_regression.prior.repeat(9, 1, 1),
                        # batch_size=9,
                    )
                    # sample_trajectory is of shape (T, B, C, D)

                    data_list = [
                        x.squeeze(0).cpu().numpy()
                        for x in torch.split(
                            sample_trajectory, split_size_or_sections=1, dim=0
                        )
                    ]

                    # render_last_fig_3x3(data_list, f"output/{project_name}", -1, train_dataset.max.cpu().numpy(), train_dataset.min.cpu().numpy(), scatter_x=x_pos_mask.cpu().numpy(), scatter_y=x_partial_obs_np, scatter_y_noise=x_partial_obs_with_noise_np)
                    # p = mp.Process(target=render_video_3x3, args=(data_list, f"output/{project_name}", iteration, train_dataset.max.cpu().numpy(), train_dataset.min.cpu().numpy()))
                    # p.start()
                    # mp_list.append(p)

            flow_model_for_regression.train()
            with accelerator.autocast():
                with accelerator.accumulate(flow_model_for_regression.model):
                    x_0_repeat = flow_model_for_regression.prior.repeat(32, 1, 1)
                    (
                        x_1_repeat,
                        logp_1_repeat,
                        logp_x1_minus_logp_x0,
                    ) = flow_model_for_regression.sample_with_log_prob(
                        t_span=t_span,
                        x_0=x_0_repeat,
                        using_Hutchinson_trace_estimator=True,
                    )

                    loss_1 = torch.mean(
                        0.5
                        * torch.sum(
                            (
                                (
                                    x_partial_obs_with_noise_norm
                                    - x_1_repeat[:, -1, pos_mask]
                                )
                                ** 2
                            ),
                            dim=1,
                        )
                        / config.parameter.noise_level
                    )
                    loss_2 = -logp_1_repeat.mean()

                    loss = loss_1 + loss_2

                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            find_parameters(flow_model_for_regression), max_norm=1.0
                        )
                    optimizer.step()
                    scheduler.step()

            loss = accelerator.gather(loss)
            loss_1 = accelerator.gather(loss_1)
            loss_2 = accelerator.gather(loss_2)

            if iteration % config.parameter.log_rate == 0:
                acc_train_loss = loss.mean().item()
                print(
                    f"iteration: {iteration}, train_loss: {acc_train_loss:.5f}, lr: {scheduler.get_last_lr()[0]:.7f}"
                )

            if False and iteration % config.parameter.eval_rate == 0:
                flow_model_for_regression.eval()
                with torch.no_grad():
                    sample_trajectory = flow_model_for_regression.sample_process(
                        n_dims=config.flow_model_regression.gaussian_process.args.dims,
                        n_channels=1,
                        t_span=torch.linspace(0.0, 1.0, 100),
                        x_0=flow_model_for_regression.prior.repeat(9, 1, 1),
                        # batch_size=9,
                    )
                    # sample_trajectory is of shape (T, B, C, D)

                    data_list = [
                        x.squeeze(0).cpu().numpy()
                        for x in torch.split(
                            sample_trajectory, split_size_or_sections=1, dim=0
                        )
                    ]

                    # render_video_3x3(data_list, "output", iteration)
                    render_last_fig_3x3(
                        data_list,
                        f"output/{project_name}",
                        iteration,
                        train_dataset.max.cpu().numpy(),
                        train_dataset.min.cpu().numpy(),
                        scatter_x=x_pos_mask.cpu().numpy(),
                        scatter_y=x_partial_obs_np,
                        scatter_y_noise=x_partial_obs_with_noise_np,
                    )
                    # p = mp.Process(target=render_video_3x3, args=(data_list, f"output/{project_name}", iteration, train_dataset.max.cpu().numpy(), train_dataset.min.cpu().numpy()))
                    # p.start()
                    # mp_list.append(p)

            accelerator.wait_for_everyone()

        # compute editting score
        with torch.no_grad():
            airfoil_generated = flow_model_for_regression.sample(
                n_dims=config.flow_model_regression.gaussian_process.args.dims,
                n_channels=1,
                t_span=torch.linspace(0.0, 1.0, 100),
                x_0=flow_model_for_regression.prior,
            )
            airfoil_generated = (airfoil_generated + 1) / 2 * (
                train_dataset.max[1] - train_dataset.min[1]
            ) + train_dataset.min[1]
            mse = torch.mean(
                (airfoil_generated[-1, 0, pos_mask] - x_partial_obs_with_noise) ** 2
            )
            smoothness = calculate_smoothness(
                torch.stack([xs, airfoil_generated.squeeze()], axis=-1)
            )
            print(
                f"iteration: {iteration}, mse: {mse:.5f}, smoothness: {smoothness:.5f}"
            )

            mse_list.append(mse.cpu())
            smooth_list.append(smoothness.cpu())

        for p in mp_list:
            p.join()

        # Remove all references:
        del flow_model_for_regression
        del optimizer

        # (Optional) If you were using GPU memory, you may want to clear the cache:
        torch.cuda.empty_cache()

    mse_tensor = torch.stack(mse_list)
    smooth_tensor = torch.stack(smooth_list)

    print(
        f"mse_tensor mean: {mse_tensor.mean().item()}, smooth_tensor mean: {smooth_tensor.mean().item()}"
    )

    torch.save(mse_tensor, f"output/{project_name}/mse_tensor.pth")
    torch.save(smooth_tensor, f"output/{project_name}/smooth_tensor.pth")


if __name__ == "__main__":
    main()
