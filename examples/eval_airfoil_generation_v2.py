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
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

from rich.progress import track
from tqdm import tqdm
from easydict import EasyDict

from airfoil_generation.dataset import Dataset, AF200KDataset
from airfoil_generation.dataset.parsec_direct_n15 import Fit_airfoil_15
from airfoil_generation.dataset.parsec_direct_n11 import Fit_airfoil_11

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
)

from airfoil_generation.dataset.airfoil_metric_batchwise import (
    calculate_airfoil_metric_n15_batch,
)

from scipy.spatial.distance import pdist, squareform


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
        D = squareform(pdist(subset, "euclidean"))
        S = np.exp(-0.5 * np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        mean_logdet += logdet
    return mean_logdet / sample_times


def cal_mean(arr, remove_max_percent=100, remove_min_percent=0):
    # 计算去掉最大和最小5%数据后的平均值
    percentile_min = np.percentile(arr, remove_min_percent)
    percentile_max = np.percentile(arr, remove_max_percent)

    # 过滤数据
    filtered_data = arr[(arr >= percentile_min) & (arr <= percentile_max)]

    # 计算剩余数据的平均值
    mean_value = filtered_data.mean()
    return mean_value


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
                        ode_solver=args.ode_solver,
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
                            args.epoch * 200000 // 1024
                            if args.epoch is not None and args.dataset == "af200k"
                            else (
                                20000 // 1024 * 2000
                                if args.dataset == "supercritical"
                                else 200000 // 1024 * 2000
                            )
                        )
                    )
                ),
                warmup_steps=(
                    2000 if args.dataset == "supercritical" else 20000 // 1024 * 2000
                ),
                log_rate=100,
                eval_rate=(
                    20000 // 1024 * 500
                    if args.dataset == "supercritical"
                    else 200000 // 1024 * 500
                ),
                checkpoint_rate=(
                    20000 // 1024 * 500
                    if args.dataset == "supercritical"
                    else 200000 // 1024 * 500
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
    flow_model.model = accelerator.prepare(flow_model.model)

    os.makedirs(config.parameter.model_save_path, exist_ok=True)

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

    data_matrix = torch.from_numpy(np.array(list(train_dataset.params.values())))
    train_dataset_std, train_dataset_mean = torch.std_mean(data_matrix, dim=0)
    train_dataset_std = torch.where(
        torch.isnan(train_dataset_std) | torch.isinf(train_dataset_std),
        torch.tensor(0.0),
        train_dataset_std,
    )
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

    accelerator.init_trackers(project_name, config=config)
    accelerator.print("✨ Start evaluation ...")

    flow_model.eval()
    resolution = config.flow_model.gaussian_process.args.dims[0]
    rs = [resolution]
    l = len(rs)
    label_error = np.zeros((len(test_replay_buffer), l, args.num_constraints))
    smoothness = np.zeros(
        (
            len(test_replay_buffer),
            l,
        )
    )
    diversity = np.zeros(
        (
            len(test_replay_buffer),
            l,
        )
    )

    mp_list = []
    for i in track(
        range(len(test_dataset.storage)),
        description="Evaluating",
        disable=not accelerator.is_local_main_process,
    ):

        data = test_replay_buffer.sample().to(device)
        y = (data["params"][:, :] - train_dataset_mean[None, :]) / (
            train_dataset_std[None, :] + 1e-8
        )  # (b,15)

        priors = []
        for r in rs:
            priors.append(
                flow_model.gaussian_process.sample(dims=[r], n_samples=20, n_channels=1)
            )

        sample_trajectorys = []
        for r, prior in zip(rs, priors):
            sample_trajectory = flow_model.sample_process(
                n_dims=[r],
                n_channels=1,
                t_span=torch.linspace(0.0, args.t_end, args.t_span),
                batch_size=1,
                x_0=prior,
                condition=y.repeat(20, 1),
                with_grad=False,
            )

            if args.render:
                figure_list = [
                    x.squeeze(0).cpu().numpy()
                    for x in torch.split(
                        sample_trajectory, split_size_or_sections=1, dim=0
                    )
                ]

                # render_video_3x3(figure_list, args.project_name, i, train_dataset.max.cpu().numpy(), train_dataset.min.cpu().numpy())
                # render_video_3x3_polish(figure_list, args.project_name, i, train_dataset.max.cpu().numpy(), train_dataset.min.cpu().numpy())

                p = mp.Process(
                    target=render_video_3x3_polish,
                    args=(
                        figure_list,
                        args.project_name,
                        i,
                        train_dataset.max.cpu().numpy(),
                        train_dataset.min.cpu().numpy(),
                    ),
                    daemon=True,
                )
                p.start()
                mp_list.append(p)

            sample_trajectorys.append(sample_trajectory)

        data_list_list = []
        for sample_trajectory in sample_trajectorys:
            data_list_list.append(
                [
                    x.squeeze(0).detach()
                    for x in torch.split(
                        sample_trajectory, split_size_or_sections=1, dim=0
                    )
                ]
            )

        label_error_ = np.zeros((l, args.num_constraints))
        smoothness_ = np.zeros((l,))

        for j, data_list in enumerate(data_list_list):
            data_list_ = []
            airfoils = data_list[-1][0, :, 0, :]

            for airfoil in airfoils:
                airfoil = (airfoil + 1) / 2 * (
                    train_dataset.max[1]
                    - train_dataset.min[1]
                ) + train_dataset.min[1]
                data_list_.append(airfoil.cpu().numpy())
                xs = (np.cos(np.linspace(0, 2 * np.pi, airfoil.shape[-1])) + 1) / 2
                if args.num_constraints == 11:
                    raise ValueError(
                        "num_constraints 11 is not supported in evaluation, please use 15 instead."
                    )
                elif args.num_constraints == 15:
                    parsec_params = calculate_airfoil_metric_n15_batch(
                        x=torch.tensor((np.cos(np.linspace(0, 2 * np.pi, 257)) + 1) / 2).to(device).unsqueeze(0), 
                        y=airfoil.unsqueeze(0).to(device), 
                        stacked=True
                    ).squeeze(0)
                else:
                    raise ValueError(
                        f"num_constraints {args.num_constraints} not supported"
                    )
                label_error_[j] += np.abs(
                    parsec_params.cpu().numpy() - data["params"].reshape(-1).cpu().numpy()
                )
                smoothness_[j] += calculate_smoothness(np.stack([xs, airfoil.cpu().numpy()], axis=-1))

            label_error_[j] /= len(airfoils)
            smoothness_[j] /= len(airfoils)
            label_error[i, j] = label_error_[j]
            smoothness[i, j] = smoothness_[j]
            data_list_ = np.array(data_list_)
            diversity[i, j] = cal_diversity_score(data_list_)
            # print(np.mean(label_error[i,j]), label_error[i,j], smoothness[i,j], diversity[i,j])

    np.save(f"output/{project_name}/label_error.npy", label_error)
    np.save(f"output/{project_name}/smoothness.npy", smoothness)
    np.save(f"output/{project_name}/diversity.npy", diversity)

    log_msg = {}

    for i, r in enumerate(rs):
        print("Resolution: ", r)
        index = 0

        mean_label_error_list = []
        mean_label_error_filtered_list = []
        for arr in label_error[:, i, :].T:
            index += 1

            label_error_i = np.mean(arr)
            mean_label_error_list.append(label_error_i)
            label_error_filtered_i = cal_mean(arr, remove_max_percent=args.remove_max_percent)
            mean_label_error_filtered_list.append(label_error_filtered_i)

            print(f"label error {index}: {label_error_i}")
            log_msg[f"label error {i}-{index}"] = label_error_i
            print(f"label error {index} Filtered: {label_error_filtered_i}")
            log_msg[f"label error {i}-{index} Filtered"] = label_error_filtered_i

            # print the maxium error of each label
            max_index = np.argsort(arr)[-9:]
            print(f"label error {index} max: {arr[max_index]}, index: {max_index}")


        # compute arithmetic mean of label error
        arithmetic_mean_error = np.mean(mean_label_error_list)
        print(f"label error (arithmetic mean) : {arithmetic_mean_error}")
        log_msg[f"label error (arithmetic mean) {i}"] = arithmetic_mean_error

        arithmetic_mean_error_filtered = np.mean(mean_label_error_filtered_list)
        print(f"label error (arithmetic mean) Filtered: {arithmetic_mean_error_filtered}")
        log_msg[f"label error (arithmetic mean) {i} Filtered"] = arithmetic_mean_error_filtered


        # compute geometric mean of label error
        geometric_mean_error = np.abs(np.prod(mean_label_error_list)) ** (
            1 / label_error.shape[2]
        )
        print(f"label error (geometric mean) : {geometric_mean_error}")
        log_msg[f"label error (geometric mean) {i}"] = geometric_mean_error

        geometric_mean_error_filtered = np.abs(np.prod(mean_label_error_filtered_list)) ** (
            1 / label_error.shape[2]
        )
        print(f"label error (geometric mean) Filtered: {geometric_mean_error_filtered}")
        log_msg[f"label error (geometric mean) {i} Filtered"] = geometric_mean_error_filtered

        print(f"mean smoothness: {np.mean(smoothness[:,i])}")
        print(f"mean diversity: {np.mean(diversity[:,i])}")

        log_msg[f"mean smoothness {i}"] = np.mean(smoothness[:,i])
        log_msg[f"mean diversity {i}"] = np.mean(diversity[:, i])

    if args.wandb:
        accelerator.log(log_msg, step=0)

    accelerator.wait_for_everyone()

    accelerator.print("✨ Evaluation complete!")
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
    argparser.add_argument("--model_path", "-p", type=str, help="Model load path.")
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
        default="airfoil-evaluation",
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
        "--render",
        action="store_true",
        help="Whether to render the video",
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
        "--t_span",
        default=10,
        type=int,
        help="number of time steps to sample",
    )
    argparser.add_argument(
        "--t_end",
        default=1.0,
        type=float,
        help="time end for ODE solver",
    )
    argparser.add_argument(
        "--modes",
        default=64,
        type=int,
        help="Number of modes in Fourier Neural Operator",
    )

    argparser.add_argument(
        "--remove_max_percent",
        default=100,
        type=float,
        help="remove max percent of data when calculating mean",
    )
    argparser.add_argument(
        "--remove_min_percent",
        default=0,
        type=float,
        help="remove min percent of data when calculating mean",
    )
    argparser.add_argument(
        "--ode_solver",
        default="euler",
        type=str,
        help="ODE solver to use, euler, rk4, midpoint",
    )

    args = argparser.parse_args()
    main(args)
