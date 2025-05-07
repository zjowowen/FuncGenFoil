import os
import torch.multiprocessing as mp

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed

from rich.progress import track
from easydict import EasyDict

from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset import Dataset, AF200KDataset

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
)
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR

from airfoil_generation.neural_networks.point_cloud_neural_operator import (
    compute_Fourier_modes,
)


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

    ndims = 1
    kx_max = 64
    Lx = 1.0
    modes = compute_Fourier_modes(ndims, [kx_max], [Lx])
    modes = torch.tensor(modes, dtype=torch.float).to(device)

    project_name = args.project_name
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
                            type="PointCloudNeuralOperator",
                            args=dict(
                                ndims=ndims,
                                modes=modes,
                                nmeasures=1,
                                layers=[256, 256, 256, 256, 256, 256],
                                fc_dim=128,
                                in_dim=4 + args.num_constraints,
                                out_dim=1,
                                train_sp_L="independently",
                                act="gelu",
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                equal_weights=args.equal_weights,
                train_samples=20000 if args.dataset == "supercritical" else 200000,
                batch_size=10,
                learning_rate=5e-5 * accelerator.num_processes,
                iterations=(
                    20000 // 1024 * 2000
                    if args.dataset == "supercritical"
                    else 200000 // 1024 * 2000
                ),
                warmup_steps=2000 if args.dataset == "supercritical" else 20000,
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

    from airfoil_generation.dataset.point_cloud_data_preprocess import (
        convert_structured_data_1D,
        preprocess_data,
        compute_node_weights,
        data_preparition_with_tensordict,
    )

    if args.preprocess_data and accelerator.is_local_main_process:

        def preprocess(dataset, file_name="pcno_data.npz"):
            coordx = np.linspace(0, 2 * np.pi, 257)[None, :].repeat(
                len(dataset), axis=0
            )  # of shape (b,257)
            features = (
                dataset.storage["gt"].cpu().numpy()[:, :, 1:2]
            )  # of shape (b,257,1)

            nodes_list, elems_list, features_list = convert_structured_data_1D(
                [
                    coordx,
                ],
                features,
                nnodes_per_elem=2,
                feature_include_coords=False,
            )

            (
                nnodes,
                node_mask,
                nodes,
                node_measures_raw,
                features,
                directed_edges,
                edge_gradient_weights,
            ) = preprocess_data(nodes_list, elems_list, features_list)
            node_measures, node_weights = compute_node_weights(
                nnodes, node_measures_raw, equal_measure=False
            )
            node_equal_measures, node_equal_weights = compute_node_weights(
                nnodes, node_measures_raw, equal_measure=True
            )
            np.savez_compressed(
                os.path.join("./", file_name),
                nnodes=nnodes,
                node_mask=node_mask,
                nodes=nodes,
                node_measures_raw=node_measures_raw,
                node_measures=node_measures,
                node_weights=node_weights,
                node_equal_measures=node_equal_measures,
                node_equal_weights=node_equal_weights,
                features=features,
                directed_edges=directed_edges,
                edge_gradient_weights=edge_gradient_weights,
            )

        preprocess(train_dataset, file_name="pcno_data_train.npz")
        preprocess(test_dataset, file_name="pcno_data_test.npz")
        print("Preprocess data done!")

    accelerator.wait_for_everyone()

    def load_data(data_path, dataset, file_name="pcno_data.npz", equal_weights=True):

        data = np.load(os.path.join(data_path, file_name))
        nnodes, node_mask, nodes = data["nnodes"], data["node_mask"], data["nodes"]
        node_weights = (
            data["node_equal_weights"] if equal_weights else data["node_weights"]
        )
        node_measures = data["node_measures"]
        directed_edges, edge_gradient_weights = (
            data["directed_edges"],
            data["edge_gradient_weights"],
        )
        features = data["features"]

        node_measures_raw = data["node_measures_raw"]
        indices = np.isfinite(node_measures_raw)
        node_rhos = np.copy(node_weights)
        node_rhos[indices] = node_rhos[indices] / node_measures[indices]

        point_cloud_dataset = data_preparition_with_tensordict(
            nnodes,
            node_mask,
            nodes,
            node_weights,
            node_rhos,
            features,
            directed_edges,
            edge_gradient_weights,
            params=dataset.storage["params"],
        )

        return point_cloud_dataset

    point_cloud_train_dataset = load_data(
        "./",
        dataset=train_dataset,
        file_name="pcno_data_train.npz",
        equal_weights=config.parameter.equal_weights,
    )
    point_cloud_test_dataset = load_data(
        "./",
        dataset=test_dataset,
        file_name="pcno_data_test.npz",
        equal_weights=config.parameter.equal_weights,
    )

    train_replay_buffer = TensorDictReplayBuffer(
        storage=point_cloud_train_dataset.storage,
        batch_size=batch_size,
        # sampler=RandomSampler(),
        sampler=SamplerWithoutReplacement(drop_last=False, shuffle=True),
        prefetch=10,
    )

    test_replay_buffer = TensorDictReplayBuffer(
        storage=point_cloud_test_dataset.storage,
        batch_size=9,
        sampler=SamplerWithoutReplacement(drop_last=False, shuffle=False),
        prefetch=10,
    )

    iteration_per_epoch = len(point_cloud_train_dataset.storage) // batch_size + 1

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
                gt = (data["y"] - train_dataset.min.to(device)[1]) / (
                    train_dataset.max.to(device)[1] - train_dataset.min.to(device)[1]
                ) * 2 - 1  # (b,257,1)
                data["condition"]["params"] = (
                    (data["condition"]["params"] - train_dataset_mean[None, :])
                    / (train_dataset_std[None, :] + 1e-8)
                ).to(
                    torch.float32
                )  # (b,15)

                gt = gt.to(device).to(torch.float32)

                gt_reshape = gt.reshape(-1, 1, 257)  # (b,1,257)
                gaussian_prior = flow_model.gaussian_process.sample_from_prior(
                    dims=config.flow_model.gaussian_process.dims,
                    n_samples=gt_reshape.shape[0],
                    n_channels=gt_reshape.shape[1],
                )
                loss = flow_model.functional_flow_matching_loss(
                    x0=gaussian_prior,
                    x1=gt_reshape,
                    condition=data["condition"],
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
            flow_model.eval()
            with torch.no_grad():
                data = test_replay_buffer.sample()
                data = data.to(device)
                data["condition"]["params"] = (
                    (data["condition"]["params"][:, :] - train_dataset_mean[None, :])
                    / (train_dataset_std[None, :] + 1e-8)
                ).to(
                    torch.float32
                )  # (b,15)
                sample_trajectory = flow_model.sample_process(
                    n_dims=config.flow_model.gaussian_process.dims,
                    n_channels=1,
                    t_span=torch.linspace(0.0, 1.0, 100),
                    condition=data["condition"],
                )
                # sample_trajectory is of shape (T, B, C, D)

                data_list = [
                    x.squeeze(0).cpu().numpy()
                    for x in torch.split(
                        sample_trajectory, split_size_or_sections=1, dim=0
                    )
                ]

                # render_video_3x3(data_list, "output", iteration)
                p = mp.Process(
                    target=render_video_3x3,
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
        default="airfoil-unconditional-training-with-PCNO",
        help="Project name",
    )
    argparser.add_argument(
        "--preprocess_data",
        action="store_true",
        help="Whether to preprocess data",
    )
    argparser.add_argument(
        "--equal_weights",
        action="store_true",
        help="Whether to use equal weights for different nodes",
    )

    args = argparser.parse_args()
    main(args)
