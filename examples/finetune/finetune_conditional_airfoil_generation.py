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
from airfoil_generation.dataset import Dataset, AF200KDataset
from airfoil_generation.dataset.parsec_direct_n15 import Fit_airfoil_15

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
)
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset.toy_dataset import MaternGaussianProcess
from airfoil_generation.neural_networks.neural_operator import FourierNeuralOperator

from airfoil_generation.dataset.airfoil_metric_batchwise import (
    calculate_airfoil_metric_n15_batch,
)


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

    xs = (np.cos(np.linspace(0, 2 * np.pi, 257)) + 1) / 2
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
        os.makedirs(video_save_path, exist_ok=True)

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
                batch_size=args.batch_size,
                learning_rate=5e-5 * accelerator.num_processes,
                gradient_accumulation_steps=args.batch_size_accumulation
                // args.batch_size,
                iterations=(
                    args.iterations
                    if args.iterations is not None
                    else (
                        args.epoch * 20000 // args.batch_size
                        if args.epoch is not None and args.dataset == "supercritical"
                        else (
                            (
                                args.epoch * 200000 // args.batch_size
                                if args.dataset_names != ["interpolated_uiuc"]
                                else args.epoch * 2048 // args.batch_size
                            )
                            if args.epoch is not None and args.dataset == "af200k"
                            else (
                                20000 // args.batch_size * 2000
                                if args.dataset == "supercritical"
                                else (
                                    200000 // args.batch_size * 2000
                                    if args.dataset_names != ["interpolated_uiuc"]
                                    else 2048 // args.batch_size * 2000
                                )
                            )
                        )
                    )
                ),
                warmup_steps=args.warmup_steps,
                log_rate=100,
                eval_rate=(
                    20000 // args.batch_size * 500 // 1000
                    if args.dataset == "supercritical"
                    else (
                        200000 // args.batch_size * 500
                        if args.dataset_names != ["interpolated_uiuc"]
                        else 2048 // args.batch_size * 500
                    )
                ),
                checkpoint_rate=(
                    20000 // args.batch_size * 500 // 1000
                    if args.dataset == "supercritical"
                    else (
                        200000 // args.batch_size * 500
                        if args.dataset_names != ["interpolated_uiuc"]
                        else 2048 // args.batch_size * 500
                    )
                ),
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

    train_dataset = (
        Dataset(
            split="train",
            std_cst_augmentation=0.08,
            num_perturbed_airfoils=10,
            dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
            max_size=100000,
            folder_path=args.data_path,
            num_constraints=args.num_constraints,
        )
        if args.dataset == "supercritical"
        else (
            AF200KDataset(
                split="train",
                dataset_names=(
                    [
                        "beziergan_gen",
                        "cst_gen",
                        "cst_gen_a",
                        "cst_gen_b",
                        "diffusion_gen",
                        "interpolated_uiuc",
                        "naca_gen",
                        "supercritical_airfoil_af200k",
                    ]
                    if len(args.dataset_names) == 0
                    else args.dataset_names
                ),
                folder_path=args.data_path,
                num_constraints=args.num_constraints,
            )
        )
    )

    test_dataset = (
        Dataset(
            split="test",
            std_cst_augmentation=0.08,
            num_perturbed_airfoils=10,
            dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
            max_size=100000,
            folder_path=args.data_path,
            num_constraints=args.num_constraints,
        )
        if args.dataset == "supercritical"
        else (
            AF200KDataset(
                split="test",
                dataset_names=(
                    [
                        "beziergan_gen",
                        "cst_gen",
                        "cst_gen_a",
                        "cst_gen_b",
                        "diffusion_gen",
                        "interpolated_uiuc",
                        "naca_gen",
                        "supercritical_airfoil_af200k",
                    ]
                    if len(args.dataset_names) == 0
                    else args.dataset_names
                ),
                folder_path=args.data_path,
                num_constraints=args.num_constraints,
            )
        )
    )

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

    data_matrix = torch.from_numpy(np.array(list(train_dataset.params.values())))
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
    # load train_dataset_min and train_dataset_max using safetensors
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

    t_span = torch.linspace(0.0, 1.0, 50).to(device)

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
                x_ = (
                    torch.tensor((np.cos(np.linspace(0, 2 * np.pi, 257)) + 1) / 2)
                    .to(device)
                    .repeat(data.shape[0], 1)
                )
                data["gt"] = (data["gt"] - train_dataset.min.to(device)) / (
                    train_dataset.max.to(device) - train_dataset.min.to(device)
                ) * 2 - 1

                gt = data["gt"][:, :, 1:2]  # (b,257,1)
                y = (data["params"][:, :] - train_dataset_mean[None, :]) / (
                    train_dataset_std[None, :] + 1e-8
                )  # (b,15)
                gt = gt.to(device).to(torch.float32)
                y = y.to(device).to(torch.float32)

                gt = gt.reshape(-1, 1, 257)  # (b,1,257)

                noised_y = y + torch.randn_like(y) * args.perturbation_level

                (
                    x1,
                    logp_x1,
                    logp_x1_minus_logp_x0,
                ) = flow_model.sample_with_log_prob(
                    n_dims=config.flow_model.gaussian_process.args.dims,
                    n_channels=1,
                    t_span=t_span,
                    condition=noised_y,
                    using_Hutchinson_trace_estimator=True,
                )

                x1_denormed = (x1 + 1) / 2.0 * (
                    train_dataset.max[1] - train_dataset.min[1]
                ) + train_dataset.min[1]

                y_calculated = calculate_airfoil_metric_n15_batch(
                    x=x_, y=x1_denormed[:, 0], stacked=True
                )

                y_calculated_normed = (y_calculated - train_dataset_mean[None, :]) / (
                    train_dataset_std[None, :] + 1e-8
                )

                loss_1 = torch.mean(
                    0.5 * torch.sum((y_calculated_normed - noised_y) ** 2) / 0.0001
                )
                loss_2 = -logp_x1.mean()
                loss = loss_1 + loss_2

                # accumulate gradients for distributed training every config.parameter.gradient_accumulation_steps
                accelerator.backward(loss)
                if iteration % config.parameter.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

        loss = accelerator.gather(loss)
        loss_1 = accelerator.gather(loss_1)
        loss_2 = accelerator.gather(loss_2)
        if iteration % config.parameter.log_rate == 0:
            if accelerator.is_local_main_process:
                to_log = {
                    "loss/mean": loss.mean().item(),
                    "loss_1/mean": loss_1.mean().item(),
                    "loss_2/mean": loss_2.mean().item(),
                    "iteration": iteration,
                    "epoch": iteration // iteration_per_epoch,
                    "lr": scheduler.get_last_lr()[0],
                }

                if len(loss.shape) == 0:
                    to_log["loss/std"] = 0
                    to_log["loss_1/std"] = 0
                    to_log["loss_2/std"] = 0
                    to_log["loss/0"] = loss.item()
                    to_log["loss_1/0"] = loss_1.item()
                    to_log["loss_2/0"] = loss_2.item()
                elif loss.shape[0] > 1:
                    to_log["loss/std"] = loss.std().item()
                    to_log["loss_1/std"] = loss_1.std().item()
                    to_log["loss_2/std"] = loss_2.std().item()
                    for i in range(loss.shape[0]):
                        to_log[f"loss/{i}"] = loss[i].item()
                        to_log[f"loss_1/{i}"] = loss_1[i].item()
                        to_log[f"loss_2/{i}"] = loss_2[i].item()

                if True:
                    # test a specific design generation error
                    specific_design_params = (
                        torch.tensor(
                            [
                                0.012627,
                                0.031974,
                                -0.024484,
                                0.4013,
                                0.065599,
                                0.3096,
                                -0.046061,
                                0.061031,
                                -0.045374,
                                -30.572049,
                                0.001439,
                                0.8863,
                                0.01305,
                                0.060057,
                                -0.029317,
                            ]
                        )
                        .to(device)
                        .float()
                        .unsqueeze(0)
                    )
                    specific_design_params_normed = (
                        specific_design_params - train_dataset_mean[None, :]
                    ) / (train_dataset_std[None, :] + 1e-8)
                    sample_trajectory = flow_model.sample_process(
                        n_dims=config.flow_model.gaussian_process.args.dims,
                        n_channels=1,
                        t_span=torch.linspace(0.0, 1.0, 1000),
                        condition=specific_design_params_normed,
                    )
                    sample_denormed = (sample_trajectory[-1] + 1) / 2.0 * (
                        train_dataset.max[1] - train_dataset.min[1]
                    ) + train_dataset.min[1]
                    y_calculated = calculate_airfoil_metric_n15_batch(
                        x=torch.tensor((np.cos(np.linspace(0, 2 * np.pi, 257)) + 1) / 2)
                        .to(device)
                        .repeat(1, 1),
                        y=sample_denormed[:, 0],
                        stacked=True,
                    )
                    label_error = (
                        (y_calculated - specific_design_params.repeat(10, 1))
                        .abs()
                        .mean(dim=0)
                    )
                    to_log["specific_label_error/mean"] = label_error.mean().item()
                    for index in range(label_error.shape[0]):
                        to_log[f"specific_label_error_{index}/mean"] = (
                            label_error[index].mean().item()
                        )

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
                    condition=y.repeat(10, 1),
                )
                # sample_trajectory is of shape (T, B, C, D)

                data_list = [
                    x.squeeze(0).cpu().numpy()
                    for x in torch.split(
                        sample_trajectory, split_size_or_sections=1, dim=0
                    )
                ]

                if accelerator.is_local_main_process:
                    # render_video_3x3(data_list, "output", iteration)
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
    argparser.add_argument("--model_path", "-p", type=str, help="Model load path.")
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
        default="airfoil-conditional-finetune",
        help="Project name",
    )
    argparser.add_argument(
        "--batch_size",
        "-bs",
        default=16,
        type=int,
        help="Batch size for training.",
    )
    argparser.add_argument(
        "--batch_size_accumulation",
        "-bsa",
        default=1024,
        type=int,
        help="Batch size for gradient accumulation.",
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
        "--warmup_steps",
        "-ws",
        default=2000,
        type=int,
        help="Number of warmup steps for the learning rate scheduler.",
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

    argparser.add_argument(
        "--perturbation_level",
        default=0.125,
        type=float,
        help="Perturbation level for the input parameters.",
    )

    args = argparser.parse_args()
    main(args)
