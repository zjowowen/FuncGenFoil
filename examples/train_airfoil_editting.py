import os
import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

from torchrl.data import LazyMemmapStorage

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

from rich.progress import track
from easydict import EasyDict

import copy

from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset import Dataset, AF200KDataset

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
    OptimalTransportFunctionalFlowForRegression,
)
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
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


def main(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index

    # set_seed(seed=42+process_rank)

    print(f"Process rank: {process_rank}")

    project_name = "airfoil-editing"
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
                iterations=11,
                warmup_steps=5,
                log_rate=1,
                eval_rate=10,
                model_save_rate=1000,
                video_save_path=f"output/{project_name}/videos",
                model_save_path=f"output/{project_name}/models",
                model_load_path=f"output/generation/models/model_700000.pth",
                dataset_for_edit_path=f"data/dataset_for_editing.pth",
                edit_scale_sqrt=0.0001,  # 0.0001, 0.0002, 0.0004
            ),
        )
    )

    flow_model = OptimalTransportFunctionalFlow(
        config=config.flow_model,
    )

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

    train_dataset = (
        Dataset(
            split="train",
            std_cst_augmentation=0.08,
            num_perturbed_airfoils=10,
            dataset_names=["supercritical_airfoil", "data_4000", "r05", "r06"],
            max_size=100000,
        )
        if args.dataset == "supercritical"
        else AF200KDataset(
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

    data_matrix = torch.from_numpy(np.array(list(train_dataset.params.values())))
    train_dataset_std, train_dataset_mean = torch.std_mean(data_matrix, dim=0)

    train_dataset_std = train_dataset_std.to(device)
    train_dataset_mean = train_dataset_mean.to(device)

    if not os.path.exists(f"output/{project_name}"):
        os.makedirs(f"output/{project_name}")

    edit_scale_sqrt = config.parameter.edit_scale_sqrt

    edit_scale = edit_scale_sqrt * edit_scale_sqrt

    dataset_for_edit = torch.load(
        config.parameter.dataset_for_edit_path, map_location="cpu"
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

        # deep copy the flow model for regression
        state_dict = flow_model.model.state_dict()
        state_dict.pop("_metadata", None)
        flow_model_for_regression.model.load_state_dict(state_dict)

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

        x_range = torch.linspace(0, 1, config.flow_model.gaussian_process.args.dims[0])

        x_pos_mask = x_range[pos_mask.cpu()]

        x_partial_obs = x_test[-1, :, pos_mask].to(device)

        x_partial_obs_with_noise = x_partial_obs + noise_pattern[
            :, pos_mask
        ] * torch.sqrt(torch.tensor(edit_scale, device=device))

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

            accelerator.wait_for_everyone()

        # compute editting score
        with torch.no_grad():
            airfoil_generated = flow_model_for_regression.sample(
                n_dims=config.flow_model_regression.gaussian_process.dims,
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

        # (Optional) If you were using GPU memory, you may want to clear the cache:
        torch.cuda.empty_cache()

    mse_tensor = torch.stack(mse_list)
    smooth_tensor = torch.stack(smooth_list)

    print(
        f"mse_tensor mean: {mse_tensor.mean().item()}, smooth_tensor mean: {smooth_tensor.mean().item()}"
    )


if __name__ == "__main__":
    main()
