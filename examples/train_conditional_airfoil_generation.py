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

from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset import Dataset, AF200KDataset
from airfoil_generation.dataset.parsec_direct_n15 import Fit_airfoil

from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
)
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.dataset.toy_dataset import MaternGaussianProcess
from airfoil_generation.neural_networks.neural_operator import FourierNeuralOperator


def render_video_3x3(data_list, video_save_path, iteration, train_dataset_max, train_dataset_min, fps=100, dpi=100):

    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path, exist_ok=True)
    
    xs = (np.cos(np.linspace(0, 2*np.pi, 257)) + 1) / 2

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
            ax.plot(xs, ((data[j,:] + 1) / 2 * (train_dataset_max[1] - train_dataset_min[1]) + train_dataset_min[1]), lw=0.1)
            ax.scatter(xs, ((data[j,:] + 1) / 2 * (train_dataset_max[1] - train_dataset_min[1]) + train_dataset_min[1]), s=0.01, c='b')

        return []

    ani = animation.FuncAnimation(fig, update, frames=range(frames), interval=1000/fps, blit=False)

    # Save animation as MP4
    save_path = os.path.join(video_save_path, f"iteration_{iteration}.mp4")
    ani.save(save_path, fps=fps, dpi=dpi)

    # Clean up
    plt.close(fig)
    plt.clf()
    print(f"Saved video to {save_path}")

def main(args):

    # breakpoint()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with=None, kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    state = AcceleratorState()

    # Get the process rank
    process_rank = state.process_index

    set_seed(seed=42+process_rank)

    print(f"Process rank: {process_rank}")

    project_name = "airfoil-generation"
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
                            type="FourierNeuralOperatorConditional",
                            args=dict(
                                modes=64,
                                vis_channels=1,
                                hidden_channels=256,
                                proj_channels=128,
                                x_dim=1,
                                t_scaling=1,
                                n_layers=6,
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
        state_dict = torch.load(config.parameter.model_load_path, map_location="cpu")
        state_dict.pop("_metadata", None)
        flow_model.model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(
        flow_model.model.parameters(), lr=config.parameter.learning_rate
    )

    scheduler = CosineAnnealingWarmupLR(optimizer,
                                        T_max=config.parameter.iterations,
                                        eta_min=2e-6,
                                        warmup_steps=config.parameter.warmup_steps,
                                        last_epoch=-1)
    
    flow_model.model, optimizer = accelerator.prepare(flow_model.model, optimizer)

    os.makedirs(config.parameter.model_save_path, exist_ok=True)

    batch_size = config.parameter.batch_size

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

    accelerator.init_trackers("airfoil-generation", config=None)
    accelerator.print("✨ Start training ...")

    mp_list = []

    for iteration in track(
        range(config.parameter.iterations),
        description="Training",
        disable=not accelerator.is_local_main_process,
    ):
        flow_model.train()
        with accelerator.autocast():
            with accelerator.accumulate(flow_model.model):
                
                data = train_replay_buffer.sample()
                data = data.to(device)
                data['gt'] = (data['gt'] - train_dataset.min.to(device)) / (train_dataset.max.to(device) - train_dataset.min.to(device)) * 2 - 1

                gt = data['gt'][:,:,1:2]  # (b,257,1)
                y = (data['params'][:,:] - train_dataset_mean[None,:]) / (train_dataset_std[None,:] + 1e-8)  # (b,15)
                gt = gt.to(device).to(torch.float32)
                y = y.to(device).to(torch.float32)

                gt = gt.reshape(-1, 1, 257) # (b,1,257)
                gaussian_prior = flow_model.gaussian_process.sample_from_prior(dims=config.flow_model.gaussian_process.dims, n_samples=gt.shape[0], n_channels=gt.shape[1])
                loss = flow_model.optimal_transport_functional_flow_matching_loss(x0=gaussian_prior, x1=gt, condition=y)
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
                print(f"iteration: {iteration}, train_loss: {acc_train_loss:.5f}, lr: {scheduler.get_last_lr()[0]:.7f}")

        if iteration % config.parameter.eval_rate == 0:
            # breakpoint()
            flow_model.eval()
            with torch.no_grad():
                data = test_replay_buffer.sample()
                data = data.to(device)
                y = (data['params'][:,:] - train_dataset_mean[None,:]) / (train_dataset_std[None,:] + 1e-8)  # (b,15)
                sample_trajectory=flow_model.sample_process(
                    n_dims=config.flow_model.gaussian_process.dims,
                    n_channels=1,
                    t_span=torch.linspace(0.0, 1.0, 1000),
                    batch_size=1,
                    condition=y.repeat(10,1),
                )
                # sample_trajectory is of shape (T, B, C, D)

                data_list =[
                    x.squeeze(0).cpu().numpy() for x in torch.split(sample_trajectory, split_size_or_sections=1, dim=0)
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
    argparser = argparse.ArgumentParser(description='train_parser')
    argparser.add_argument('--dataset', '-d', default='supercritical', type=str, choices=['supercritical', 'af200k'], help="Choose a dataset.")
    args = argparser.parse_args()
    main(args)
