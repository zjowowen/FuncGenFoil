import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

from easydict import EasyDict

from airfoil_generation.dataset import Dataset

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
)


def render_last_fig_3x3_super_resolution(
    data_list,
    resolution,
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

    xs = (np.cos(np.linspace(0, 2 * np.pi, resolution)) + 1) / 2

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

    save_path = os.path.join(
        video_save_path, f"iteration_last_fig_{iteration}_{resolution}.png"
    )
    plt.savefig(save_path, dpi=300)

    # Clean up
    plt.close(fig)
    plt.clf()
    print(f"Saved image to {save_path}")


def render_last_fig_3x3_super_resolution_all(
    data_list,
    resolution,
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

    # Number of frames in the animation is just len(data_list) = 1000
    frames = len(data_list)

    # Create figure with 3x3 subplots
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    for j, ax in enumerate(axs.flat):
        for data in data_list:
            data = data[-1].squeeze()
            xs = (np.cos(np.linspace(0, 2 * np.pi, data.shape[-1])) + 1) / 2
            if scatter_x is not None:
                scatter_x_cos = (np.cos(scatter_x * 2 * np.pi) + 1) / 2

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

    save_path = os.path.join(
        video_save_path, f"iteration_last_fig_{iteration}_{resolution}_all.png"
    )
    plt.savefig(save_path, dpi=300)

    # Clean up
    plt.close(fig)
    plt.clf()
    print(f"Saved image to {save_path}")


def main():
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index

    # set_seed(seed=42+process_rank)

    print(f"Process rank: {process_rank}")

    project_name = "airfoil-generation-super-resolution"
    config = EasyDict(
        dict(
            device=device,
            flow_model=dict(
                device=device,
                gaussian_process=dict(
                    length_scale=0.01,
                    nu=1.5,
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
                                modes=32,
                                vis_channels=1,
                                hidden_channels=256,
                                proj_channels=128,
                                x_dim=1,
                                t_scaling=1,
                            ),
                        ),
                    ),
                ),
            ),
            flow_model_regression=dict(
                device=device,
                gaussian_process=dict(
                    length_scale=0.01,
                    nu=1.5,
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
                                modes=32,
                                vis_channels=1,
                                hidden_channels=256,
                                proj_channels=128,
                                x_dim=1,
                                t_scaling=1,
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                noise_level=0.000003,
                batch_size=1024,
                learning_rate=5e-6 * accelerator.num_processes,
                iterations=51,
                warmup_steps=5,
                log_rate=1,
                eval_rate=10,
                model_save_rate=1000,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=f"output/airfoil-generation/models/model_700000.pth",
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
        state_dict = torch.load(config.parameter.model_load_path, map_location="cpu")
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
    )

    # test model
    if True:
        flow_model.eval()
        flow_model = flow_model.to(device)
        with torch.no_grad():
            resolution_4x = (config.flow_model.gaussian_process.dims[0] - 1) * 4 + 1
            resolution_2x = (config.flow_model.gaussian_process.dims[0] - 1) * 2 + 1
            resolution = config.flow_model.gaussian_process.dims[0]
            resolution_05x = (config.flow_model.gaussian_process.dims[0] - 1) // 2 + 1
            resolution_025x = (config.flow_model.gaussian_process.dims[0] - 1) // 4 + 1
            prior_4x = flow_model.gaussian_process.sample(
                dims=[resolution_4x], n_samples=9, n_channels=1
            )
            prior_2x = prior_4x[:, :, ::2]
            prior_x = prior_2x[:, :, ::2]
            prior_05x = prior_x[:, :, ::2]
            prior_025x = prior_05x[:, :, ::2]
            sample_trajectory_4x = flow_model.sample_process(
                n_dims=[resolution_4x],
                n_channels=1,
                t_span=torch.linspace(0.0, 1.0, 1000),
                x_0=prior_4x,
            )
            sample_trajectory_2x = flow_model.sample_process(
                n_dims=[resolution_2x],
                n_channels=1,
                t_span=torch.linspace(0.0, 1.0, 1000),
                x_0=prior_2x,
            )
            sample_trajectory_x = flow_model.sample_process(
                n_dims=[resolution],
                n_channels=1,
                t_span=torch.linspace(0.0, 1.0, 1000),
                x_0=prior_x,
            )
            sample_trajectory_05x = flow_model.sample_process(
                n_dims=[resolution_05x],
                n_channels=1,
                t_span=torch.linspace(0.0, 1.0, 1000),
                x_0=prior_05x,
            )
            sample_trajectory_025x = flow_model.sample_process(
                n_dims=[resolution_025x],
                n_channels=1,
                t_span=torch.linspace(0.0, 1.0, 1000),
                x_0=prior_025x,
            )

            # sample_trajectory is of shape (T, B, C, D)
            data_list_4x = [
                x.squeeze(0).cpu().numpy()
                for x in torch.split(
                    sample_trajectory_4x, split_size_or_sections=1, dim=0
                )
            ]

            data_list_2x = [
                x.squeeze(0).cpu().numpy()
                for x in torch.split(
                    sample_trajectory_2x, split_size_or_sections=1, dim=0
                )
            ]

            data_list_x = [
                x.squeeze(0).cpu().numpy()
                for x in torch.split(
                    sample_trajectory_x, split_size_or_sections=1, dim=0
                )
            ]

            data_list_05x = [
                x.squeeze(0).cpu().numpy()
                for x in torch.split(
                    sample_trajectory_05x, split_size_or_sections=1, dim=0
                )
            ]

            data_list_025x = [
                x.squeeze(0).cpu().numpy()
                for x in torch.split(
                    sample_trajectory_025x, split_size_or_sections=1, dim=0
                )
            ]

            render_last_fig_3x3_super_resolution(
                data_list_4x,
                resolution_4x,
                f"output/{project_name}/",
                -1,
                train_dataset.max.cpu().numpy(),
                train_dataset.min.cpu().numpy(),
            )
            render_last_fig_3x3_super_resolution(
                data_list_2x,
                resolution_2x,
                f"output/{project_name}/",
                -1,
                train_dataset.max.cpu().numpy(),
                train_dataset.min.cpu().numpy(),
            )
            render_last_fig_3x3_super_resolution(
                data_list_x,
                resolution,
                f"output/{project_name}/",
                -1,
                train_dataset.max.cpu().numpy(),
                train_dataset.min.cpu().numpy(),
            )
            render_last_fig_3x3_super_resolution(
                data_list_05x,
                resolution_05x,
                f"output/{project_name}/",
                -1,
                train_dataset.max.cpu().numpy(),
                train_dataset.min.cpu().numpy(),
            )
            render_last_fig_3x3_super_resolution(
                data_list_025x,
                resolution_025x,
                f"output/{project_name}/",
                -1,
                train_dataset.max.cpu().numpy(),
                train_dataset.min.cpu().numpy(),
            )
            render_last_fig_3x3_super_resolution_all(
                [
                    data_list_4x,
                    data_list_2x,
                    data_list_x,
                    data_list_05x,
                    data_list_025x,
                ],
                resolution_4x,
                f"output/{project_name}/",
                -1,
                train_dataset.max.cpu().numpy(),
                train_dataset.min.cpu().numpy(),
            )


if __name__ == "__main__":
    main()
