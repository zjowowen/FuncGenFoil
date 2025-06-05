import datetime
import argparse

import os
import torch.multiprocessing as mp


import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm

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

from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset import StretchDataset
from airfoil_generation.dataset.parsec_direct_n15 import Fit_airfoil_15

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
)
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset.toy_dataset import MaternGaussianProcess
from airfoil_generation.neural_networks.neural_operator import FourierNeuralOperator


def render_video_3x3_polish(
    data_list,
    video_save_path,
    iteration,
    train_dataset_max,
    train_dataset_min,
    fps=30,  # Lower FPS for artistic, slower animation
    dpi=150,  # Higher DPI for crispness
):
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path, exist_ok=True)

    # Use a modern matplotlib style
    plt.style.use("seaborn-v0_8-dark-palette")

    xs = 1.2 * (np.cos(np.linspace(0, 2 * np.pi, 257)) + 1) / 2
    frames = len(data_list)

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    fig.patch.set_facecolor("#11131b")  # deep dark background

    # Choose a beautiful colormap
    color_map = cm.get_cmap("magma", 9)
    scatter_map = cm.get_cmap("cool", 9)

    def update(frame_idx):
        data = data_list[frame_idx].squeeze()
        for j, ax in enumerate(axs.flat):
            ax.clear()
            ax.set_xlim([0, 1])
            ax.set_ylim([-0.1, 0.1])
            ax.set_facecolor("#1b1d2b")

            # Remove axis ticks and spines for minimalism
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Calculate y values
            y = (data[j, :] + 1) / 2 * (
                train_dataset_max[1] - train_dataset_min[1]
            ) + train_dataset_min[1]

            # Plot with smooth, thick, semi-transparent line
            ax.plot(
                xs,
                y,
                lw=2.5,
                color=color_map(j),
                alpha=0.85,
                solid_capstyle="round",
            )
            # Scatter with lower alpha, slightly larger size for glow effect
            ax.scatter(
                xs,
                y,
                s=12,
                c=[scatter_map(j)],
                edgecolors="none",
                alpha=0.28,
                zorder=3,
            )

            # Optional: Add a soft grid for depth
            ax.grid(
                visible=True, color="#2a2e42", linestyle="--", linewidth=0.5, alpha=0.3
            )

        return []

    ani = animation.FuncAnimation(
        fig, update, frames=range(frames), interval=1000 / fps, blit=False
    )

    save_path = os.path.join(video_save_path, f"iteration_{iteration}.mp4")
    ani.save(save_path, fps=fps, dpi=dpi, writer="ffmpeg")

    plt.close(fig)
    plt.clf()
    print(f"Saved video to {save_path}")


def main(args):

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with="wandb" if args.wandb else None, kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index

    set_seed(seed=42 + process_rank)

    print(f"Process rank: {process_rank}")

    project_name = args.project_name
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
                            dims=[130],
                        ),
                        "rbf": dict(
                            device=device,
                            length_scale=args.length_scale,
                            dims=[130],
                        ),
                        "white": dict(
                            device=device,
                            noise_level=args.noise_level,
                            dims=[130],
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
                                modes=args.modes,
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
                train_samples=20000 if args.dataset == "supercritical" else 200000,
                batch_size=1024,
                learning_rate=5e-5 * accelerator.num_processes,
                iterations=(
                    args.iterations
                    if args.iterations is not None
                    else (
                        args.epoch * 20000 // 1024
                        if args.epoch is not None and args.dataset == "supercritical"
                        else (
                            (
                                args.epoch * 200000 // 1024
                                if args.dataset_names != ['interpolated_uiuc']
                                else args.epoch * 2048 // 1024
                            )
                            if args.epoch is not None and args.dataset == "af200k"
                            else (
                                20000 // 1024 * 2000
                                if args.dataset == "supercritical"
                                else (
                                    200000 // 1024 * 2000
                                    if args.dataset_names != ['interpolated_uiuc']
                                    else 2048 // 1024 * 2000
                                )
                            )
                        )
                    )
                ),
                warmup_steps=2000,
                log_rate=100,
                eval_rate=(
                    20000 // 1024 * 500
                    if args.dataset == "supercritical"
                    else (
                        200000 // 1024 * 500
                        if args.dataset_names != ['interpolated_uiuc']
                        else 2048 // 1024 * 500
                    )
                ),
                checkpoint_rate=(
                    20000 // 1024 * 500
                    if args.dataset == "supercritical"
                    else (
                        200000 // 1024 * 500
                        if args.dataset_names != ['interpolated_uiuc']
                        else 2048 // 1024 * 500
                    )
                ),
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=None,
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
        print(f"Load model from {config.parameter.model_load_path} successfully!")

    optimizer = torch.optim.Adam(
        flow_model.model.parameters(), lr=config.parameter.learning_rate
    )

    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        T_max=config.parameter.iterations,
        eta_min=2e-6,
        warmup_steps=config.parameter.warmup_steps,
        last_epoch=-1,
    )

    flow_model.model, optimizer = accelerator.prepare(flow_model.model, optimizer)

    os.makedirs(config.parameter.model_save_path, exist_ok=True)

    batch_size = config.parameter.batch_size

    train_dataset = StretchDataset(split="train", folder_path=args.data_path)

    test_dataset = StretchDataset(split="test", folder_path=args.data_path)

    print(f"Data number: {len(train_dataset)}")

    # # save train_dataset_min and train_dataset_max using safetensors
    # from safetensors.torch import save_file
    # # Create a dictionary
    # tensors_to_save = {
    #     "train_dataset_min": train_dataset.min,
    #     "train_dataset_max": train_dataset.max,
    # }

    # # Save using safetensors
    # save_file(tensors_to_save, f"output/{project_name}/train_datasets.safetensors")

    data_matrix = train_dataset["params"]
    train_dataset_std, train_dataset_mean = torch.std_mean(data_matrix, dim=0)
    train_dataset_std = torch.where(
        torch.isnan(train_dataset_std) | torch.isinf(train_dataset_std),
        torch.tensor(0.0),
        train_dataset_std,
    )
    train_dataset_std = train_dataset_std.to(device)
    train_dataset_mean = train_dataset_mean.to(device)

    # # save train_dataset_mean and train_dataset_std using torch.save
    # stats = {
    #     "mean": train_dataset_mean,
    #     "std": train_dataset_std,
    # }
    # torch.save(stats, f"output/{project_name}/mean_std.pt")
    # # load train_dataset_min and train_dataset_max using safetensors
    # stats = torch.load('mean_std.pt')
    # train_dataset_mean, train_dataset_std = stats['mean'], stats['std']

    # acclerate wait for every process to be ready
    accelerator.wait_for_everyone()

    train_replay_buffer = TensorDictReplayBuffer(
        storage=train_dataset.storage,
        batch_size=batch_size,
        # sampler=RandomSampler(),
        sampler=SamplerWithoutReplacement(drop_last=False, shuffle=True),
        prefetch=10,
    )

    test_replay_buffer = TensorDictReplayBuffer(
        storage=test_dataset.storage,
        batch_size=1,
        sampler=SamplerWithoutReplacement(drop_last=False, shuffle=False),
        prefetch=10,
    )

    iteration_per_epoch = len(train_dataset.storage) // batch_size + 1

    accelerator.init_trackers(project_name, config=config)
    accelerator.print("✨ Start training ...")

    mp_list = []

    for iteration in track(
        range(1, config.parameter.iterations + 1),
        description="Training",
        disable=not accelerator.is_local_main_process,
    ):
        flow_model.train()
        with accelerator.autocast():
            with accelerator.accumulate(flow_model.model):

                data = train_replay_buffer.sample()
                data = data.to(device)
                data["bpart"] = (data["bpart"] - train_dataset.min.to(device)) / (
                    train_dataset.max.to(device) - train_dataset.min.to(device)
                ) * 2 - 1

                gt = data["bpart"][:, :, 1:2]  # (b,130,1)
                y = (data["params"][:, :] - train_dataset_mean[None, :]) / (
                    train_dataset_std[None, :] + 1e-8
                )  # (b,15)
                gt = gt.to(device).to(torch.float32)
                y = y.to(device).to(torch.float32)

                gt = gt.reshape(-1, 1, 130)  # (b,1,130)
                gaussian_prior = flow_model.gaussian_process.sample_from_prior(
                    dims=config.flow_model.gaussian_process.args.dims,
                    n_samples=gt.shape[0],
                    n_channels=gt.shape[1],
                )
                loss = flow_model.optimal_transport_functional_flow_matching_loss(
                    x0=gaussian_prior, x1=gt, condition=y
                )
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                scheduler.step()

        loss = accelerator.gather(loss)
        if iteration % config.parameter.log_rate == 0:
            if accelerator.is_local_main_process:
                to_log = {
                    "loss/mean": loss.mean().item(),
                    "iteration": iteration,
                    "epoch": iteration // iteration_per_epoch,
                    "lr": scheduler.get_last_lr()[0],
                }

                if len(loss.shape) == 0:
                    to_log["loss/std"] = 0
                    to_log["loss/0"] = loss.item()
                elif loss.shape[0] > 1:
                    to_log["loss/std"] = loss.std().item()
                    for i in range(loss.shape[0]):
                        to_log[f"loss/{i}"] = loss[i].item()
                accelerator.log(
                    to_log,
                    step=iteration,
                )
                acc_train_loss = loss.mean().item()
                print(
                    f"iteration: {iteration}, train_loss: {acc_train_loss:.5f}, lr: {scheduler.get_last_lr()[0]:.7f}"
                )

        if iteration % config.parameter.eval_rate == 0:
            # breakpoint()
            flow_model.eval()
            with torch.no_grad():
                data = test_replay_buffer.sample()
                data = data.to(device)
                y = (data["params"][:, :] - train_dataset_mean[None, :]) / (
                    train_dataset_std[None, :] + 1e-8
                )  # (b,15)
                sample_trajectory = flow_model.sample_process(
                    n_dims=config.flow_model.gaussian_process.args.dims,
                    n_channels=1,
                    t_span=torch.linspace(0.0, 1.0, 1000),
                    batch_size=1,
                    condition=y.repeat(3*3, 1),
                )
                # sample_trajectory is of shape (T, B, C, D)
                data_list = [
                    torch.cat([x.squeeze()[:65].cpu(),
                               ((data["apart"].squeeze() - train_dataset.min.to(device)) / (
                                    train_dataset.max.to(device) - train_dataset.min.to(device)
                                ) * 2 - 1)[:, 1].cpu().repeat(3*3, 1),
                               x.squeeze()[65:].cpu()], dim=1).numpy()
                    for x in torch.split(
                        sample_trajectory, split_size_or_sections=1, dim=0
                    )
                ]

                p = mp.Process(
                    target=render_video_3x3_polish,
                    args=(
                        data_list,
                        "output",
                        iteration,
                        train_dataset.max.cpu().numpy(),
                        train_dataset.min.cpu().numpy(),
                    ),
                    daemon=True,
                )
                p.start()
                mp_list.append(p)

        if iteration % config.parameter.checkpoint_rate == 0:

            if accelerator.is_local_main_process:
                if not os.path.exists(config.parameter.model_save_path):
                    os.makedirs(config.parameter.model_save_path)
                torch.save(
                    accelerator.unwrap_model(flow_model.model).state_dict(),
                    f"{config.parameter.model_save_path}/model_{iteration}.pth",
                )

        accelerator.wait_for_everyone()

    for p in mp_list:
        p.join()

    accelerator.print("✨ Training complete!")
    accelerator.end_training()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description="train_parser")
    argparser.add_argument(
        "--dataset",
        "-d",
        default="supercritical",
        type=str,
        choices=["supercritical", "af200k"],
        help="Choose a dataset.",
    )
    argparser.add_argument(
        "--dataset_names",
        type=lambda s: s.split(","),
        default=[],
        help="Type of the data to be used, default is all the data in the dataset.",
    )
    argparser.add_argument(
        "--data_path", "-dp", default="data", type=str, help="Dataset path."
    )
    argparser.add_argument(
        "--num_constraints", "-nc", default=15, type=int, help="Number of constraints."
    )
    argparser.add_argument(
        "--wandb",
        action="store_true",
        help="Whether to use wandb",
    )
    argparser.add_argument(
        "--project_name",
        type=str,
        default="airfoil-conditional-training",
        help="Project name",
    )
    argparser.add_argument(
        "--iterations",
        "-i",
        default=None,
        type=int,
        help="Number of training iterations.",
    )
    argparser.add_argument(
        "--epoch",
        "-e",
        default=None,
        type=int,
        help="Number of training epochs.",
    )
    argparser.add_argument(
        "--length_scale",
        "-l",
        default=0.03,
        type=float,
        help="length_scale of Matérn kernel and rbf kernel default = 1 if rbf else 0.03 ",
    )

    argparser.add_argument(
        "--nu",
        default=2.5,
        type=float,
        help="Matérn kernel nu",
    )

    argparser.add_argument(
        "--noise_level",
        default=1.0,
        type=float,
        help="noise_level of white kernel ",
    )

    argparser.add_argument(
        "--kernel_type",
        default="matern",
        type=str,
        help="which gausssian kernel to use, you can use matern, rbf, white curruntly",
    )

    argparser.add_argument(
        "--modes",
        default=64,
        type=int,
        help="Number of modes in Fourier Neural Operator",
    )

    args = argparser.parse_args()
    main(args)
