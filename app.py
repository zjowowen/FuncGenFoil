import gradio as gr
import plotly.graph_objects as go
import json
import os
import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import random
from easydict import EasyDict
from safetensors.torch import load_file
import copy
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.model.optimal_transport_functional_flow_model import (
    OptimalTransportFunctionalFlow,
    OptimalTransportFunctionalFlowForRegression,
)
from airfoil_generation.training.optimizer import CosineAnnealingWarmupLR
from airfoil_generation.utils import find_parameters
from airfoil_generation.dataset.toy_dataset import MaternGaussianProcess
from airfoil_generation.dataset.parsec_direct_n15 import Fit_airfoil_15
from airfoil_generation.dataset.airfoil_metric import calculate_airfoil_metric_n15


def render_fig(
    xs,
    ys,
    xs_controlled=None,
    ys_controlled=None,
    ys_controlled_edited=None,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="black", width=1),
            name="Airfoil Curve",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            marker=dict(size=3, color="black"),
            name="Airfoil Points",
        )
    )

    if xs_controlled is not None:
        fig.add_trace(
            go.Scatter(
                x=xs_controlled,
                y=ys_controlled,
                mode="markers",
                marker=dict(size=5, color="red", symbol="x"),
                name="Airfoil Points (Original)",
            )
        )

        if ys_controlled_edited is not None:
            fig.add_trace(
                go.Scatter(
                    x=xs_controlled,
                    y=ys_controlled_edited,
                    mode="markers",
                    marker=dict(size=5, color="green", symbol="x"),
                    name="Airfoil Points (Edited)",
                )
            )

    fig.update_layout(
        xaxis=dict(range=[0, 1], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-0.1, 0.1], showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        width=800,
        height=800,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    return fig, xs, ys


def init_unconditional_flow_model(config, device):
    flow_model = OptimalTransportFunctionalFlow(
        config=config.unconditional_flow_model
    ).to(device)

    assert os.path.exists(
        config.parameter.unconditional_model_load_path
    ) and os.path.isfile(
        config.parameter.unconditional_model_load_path
    ), f"Model file not found at {config.parameter.unconditional_model_load_path}"
    # pop out _metadata key
    state_dict = torch.load(
        config.parameter.unconditional_model_load_path,
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
    print("Model loaded from: ", config.parameter.unconditional_model_load_path)

    return flow_model


def init_conditional_flow_model(config, device):
    flow_model = OptimalTransportFunctionalFlow(
        config=config.conditional_flow_model
    ).to(device)

    assert os.path.exists(
        config.parameter.conditional_model_load_path
    ) and os.path.isfile(
        config.parameter.conditional_model_load_path
    ), f"Model file not found at {config.parameter.conditional_model_load_path}"
    # pop out _metadata key
    state_dict = torch.load(
        config.parameter.conditional_model_load_path,
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
    print("Model loaded from: ", config.parameter.conditional_model_load_path)

    return flow_model


def generate_airfoil_curve(
    resolution, seed=None, prior_x=None, select_last_prior=False
):
    if seed is not None:
        print("Setting random seed to: ", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if select_last_prior and prior_x is not None:
        assert (
            prior_x is not None
        ), "prior_x should not be None when select_last_prior is True"
        if prior_x.shape[-1] != resolution:
            prior_x = torch.nn.functional.interpolate(
                prior_x,
                size=(resolution,),
                mode="linear",  # "cubic",
                align_corners=True,
            )
    else:
        prior_x = unconditional_flow_model.gaussian_process.sample(
            dims=[resolution], n_samples=1, n_channels=1
        )

    sample_trajectory_x = unconditional_flow_model.sample_process(
        n_dims=[resolution],
        n_channels=1,
        t_span=torch.linspace(0.0, 1.0, 1000),
        x_0=prior_x,
    )

    airfoil_generated_normed = sample_trajectory_x[-1, 0, 0, :]

    airfoil_generated = (airfoil_generated_normed + 1) / 2.0 * (
        train_dataset_max[1] - train_dataset_min[1]
    ) + train_dataset_min[1]

    xs = (np.cos(np.linspace(0, 2 * np.pi, resolution)) + 1) / 2

    ys = airfoil_generated.squeeze().cpu().numpy()

    fig, xs, ys = render_fig(
        xs=xs,
        ys=ys,
    )

    curve_data = [
        {"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(xs, ys)
    ]
    curve_json = json.dumps(curve_data, indent=2)
    print(airfoil_generated_normed.cpu().numpy().shape)
    return fig, curve_json, airfoil_generated_normed.squeeze().cpu().numpy(), prior_x


def generate_constraints(
    resolution: int,
    random_scale: float = 0.0,
    random_points_number: int = 0,
    control_points_range: str = "",
    airfoil_for_editing_numpy=None,
):
    """
    Overview:
        Generate random constraints for airfoil editing, and recognize the manually set constraints by checking if some points locates in the control points range.
    Arguments:
        resolution: int - The resolution of the airfoil curve.
        random_scale: float - The scale of random constraints.
        random_points_number: int - The number of random constraints.
        control_points_range: str - The range of control points. Box-ranged Points Constraints by JSON (e.g., [{"x_min": -0.1, "x_max": 0.2, "y_min": -0.5, "y_max": 0.5, "delta": 0.0}, {"x_min": 0.3, "x_max": 0.5, "y_min": -0.5, "y_max": 0.0, "delta": 0.000003}])
        airfoil_for_editing_numpy: np.ndarray - The airfoil curve for editing.
    """

    # Load/Generate airfoil data
    if airfoil_for_editing_numpy is None:
        prior_x = unconditional_flow_model.gaussian_process.sample(
            dims=[resolution], n_samples=1, n_channels=1
        )
        sample_trajectory_x = unconditional_flow_model.sample_process(
            n_dims=[resolution],
            n_channels=1,
            t_span=torch.linspace(0.0, 1.0, 1000),
            x_0=prior_x,
        )
        airfoil_for_editing_normed = (
            sample_trajectory_x[-1, 0, 0, :].unsqueeze(0).unsqueeze(0).to(device)
        )
    else:
        airfoil_for_editing_normed = (
            torch.tensor(airfoil_for_editing_numpy, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

    # Rescale airfoil data
    airfoil_for_editing = (airfoil_for_editing_normed + 1) / 2.0 * (
        train_dataset_max[1] - train_dataset_min[1]
    ) + train_dataset_min[1]

    xs = (np.cos(np.linspace(0, 2 * np.pi, resolution)) + 1) / 2
    ys = airfoil_for_editing.squeeze().cpu().numpy()

    # 1. generate random constraints

    # Initialize Gaussian process
    gp = MaternGaussianProcess(
        device=device,
        length_scale=0.2,
        nu=2.5,
        dims=[resolution],
    )

    points_idx_random = np.random.choice(
        resolution, random_points_number, replace=False
    )
    points_idx_random = torch.tensor(points_idx_random)
    points_mask_random = torch.zeros(resolution, dtype=bool)
    points_mask_random[points_idx_random] = True

    # 2. recognize the manually set constraints
    if control_points_range == "":
        constraints_json = []
    else:
        constraints_json = json.loads(control_points_range)
    points_id_constraints_for_editing_list = []
    delta_list = []

    for constraint in constraints_json:
        x_min = constraint["x_min"]
        x_max = constraint["x_max"]
        y_min = constraint["y_min"]
        y_max = constraint["y_max"]
        delta = constraint["delta"]
        points_id_constraints_for_editing_list.append(
            np.where((xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max))[0]
        )
        delta_list.append(delta)

    # 3. combine the random constraints and manually set constraints

    points_delta = torch.zeros([resolution]).to(device)

    # Sample noise
    noise_pattern = (
        gp.sample(dims=[resolution], n_samples=1, n_channels=1).squeeze(0).to(device)
    )
    noise_pattern_copy = noise_pattern.clone()
    noise_pattern.fill_(0)
    noise_pattern[:, points_mask_random] = noise_pattern_copy[:, points_mask_random]

    points_delta[points_mask_random] = noise_pattern[
        :, points_mask_random
    ] * torch.sqrt(torch.tensor(random_scale, device=device))

    # Add noise to the manually set constraints
    for points_id_constraints_for_editing, delta in zip(
        points_id_constraints_for_editing_list, delta_list
    ):
        points_delta[points_id_constraints_for_editing] = torch.tensor(
            delta, device=device
        )

    ys_delta = ys + points_delta.cpu().numpy()

    # merge the random constraints and manually set constraints index
    points_id_constraints_for_editing_all = []
    for temp_id in points_id_constraints_for_editing_list:
        points_id_constraints_for_editing_all = list(
            set(points_id_constraints_for_editing_all) | set(temp_id)
        )
    points_id_constraints_for_editing_all = list(
        set(points_id_constraints_for_editing_all)
        | set(points_idx_random.cpu().numpy().tolist())
    )

    xs_controlled = xs[points_id_constraints_for_editing_all]
    ys_controlled = ys[points_id_constraints_for_editing_all]
    ys_controlled_edited = ys_delta[points_id_constraints_for_editing_all]

    # Render figure
    fig, xs, ys = render_fig(
        xs=xs,
        ys=ys,
        xs_controlled=xs_controlled,
        ys_controlled=ys_controlled,
        ys_controlled_edited=ys_controlled_edited,
    )

    # Convert curve to JSON
    curve_data = [
        {"x": float(x), "y": float(y)}
        for x, y in zip(xs_controlled, ys_controlled_edited)
    ]
    curve_json = json.dumps(curve_data, indent=2)

    return (
        fig,
        curve_json,
        points_id_constraints_for_editing_all,
        xs_controlled,
        ys_controlled,
        ys_controlled_edited,
    )


def generate_editing_airfoil_curve(
    resolution,
    airfoil_for_editing_numpy,
    finetune_iterations,
    points_id_constraints_for_editing_all,
    xs_controlled,
    ys_controlled,
    ys_controlled_edited,
    t_steps=100,
    latent_initialization="Zero latent function initialization",
    progress=gr.Progress(),
):
    progress(0, desc="Starting...")

    t_span = torch.linspace(0.0, 1.0, t_steps).to(device)
    airfoil_for_editing_normed = (
        torch.tensor(airfoil_for_editing_numpy, device=device).unsqueeze(0).unsqueeze(0)
    )

    if latent_initialization == "Random latent function initialization":
        inverse_prior = unconditional_flow_model.gaussian_process.sample(
            dims=[resolution], n_samples=1, n_channels=1
        )
    elif latent_initialization == "Zero latent function initialization":
        inverse_prior = torch.zeros(1, 1, resolution).to(device)
    elif latent_initialization == "Inverse prior initialization":
        inverse_prior = unconditional_flow_model.inverse_sample(
            n_dims=[resolution],
            n_channels=1,
            t_span=t_span,
            x_0=airfoil_for_editing_normed,
        )
    else:
        inverse_prior = torch.zeros(1, 1, resolution).to(device)

    # deep copy the flow model for regression
    flow_model_for_regression = OptimalTransportFunctionalFlowForRegression(
        config=config.flow_model_regression,
        model=copy.deepcopy(unconditional_flow_model.model),
        prior=inverse_prior,
    ).to(device)

    optimizer = torch.optim.Adam(
        find_parameters(flow_model_for_regression),
        lr=config.parameter.learning_rate,
    )

    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        T_max=finetune_iterations,
        eta_min=2e-6,
        warmup_steps=config.parameter.warmup_steps,
    )

    ys_controlled_norm = (
        torch.tensor(ys_controlled, device=device) - train_dataset_min.to(device)[1]
    ) / (train_dataset_max.to(device)[1] - train_dataset_min.to(device)[1]) * 2 - 1
    ys_controlled_edited_norm = (
        torch.tensor(ys_controlled_edited, device=device)
        - train_dataset_min.to(device)[1]
    ) / (train_dataset_max.to(device)[1] - train_dataset_min.to(device)[1]) * 2 - 1

    for iteration in progress.tqdm(range(finetune_iterations)):
        flow_model_for_regression.train()

        x_0_repeat = flow_model_for_regression.prior.repeat(4, 1, 1)
        (
            x_1_repeat,
            logp_1_repeat,
            logp_x1_minus_logp_x0,
        ) = flow_model_for_regression.sample_with_log_prob(
            t_span=t_span, x_0=x_0_repeat, using_Hutchinson_trace_estimator=True
        )

        loss_1 = torch.mean(
            0.5
            * torch.sum(
                (
                    (
                        ys_controlled_edited_norm
                        - x_1_repeat[:, -1, points_id_constraints_for_editing_all]
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
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            find_parameters(flow_model_for_regression), 1.0
        )

        optimizer.step()
        scheduler.step()

        acc_train_loss = loss.mean().item()
        print(
            f"iteration: {iteration}, train_loss: {acc_train_loss:.5f}, loss 1: {loss_1.mean().item():.5f}, loss 2: {loss_2.mean().item():.5f}, lr: {scheduler.get_last_lr()[0]:.7f}"
        )

    flow_model_for_regression.eval()
    with torch.no_grad():
        sample_trajectory = flow_model_for_regression.sample_process(
            n_dims=[resolution],
            n_channels=1,
            t_span=t_span,
            x_0=flow_model_for_regression.prior,
        )

        airfoil_for_editing_sampled = (
            sample_trajectory[-1, 0, 0, :].unsqueeze(0).unsqueeze(0).to(device)
        )

        airfoil_for_editing_sampled = (airfoil_for_editing_sampled + 1) / 2.0 * (
            train_dataset_max[1] - train_dataset_min[1]
        ) + train_dataset_min[1]

        xs = (np.cos(np.linspace(0, 2 * np.pi, resolution)) + 1) / 2

        ys = airfoil_for_editing_sampled.squeeze().cpu().numpy()

        fig, xs, ys = render_fig(
            xs=xs,
            ys=ys,
            xs_controlled=xs_controlled,
            ys_controlled=ys_controlled,
            ys_controlled_edited=ys_controlled_edited,
        )

        curve_data = [
            {"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(xs, ys)
        ]
        curve_json = json.dumps(curve_data, indent=2)

        return fig, curve_json


def generate_airfoil_from_geometry_params(resolution, *args):
    prior_x = conditional_flow_model.gaussian_process.sample(
        dims=[resolution], n_samples=1, n_channels=1
    )
    y = (
        torch.tensor(args).reshape(1, 15).to(device)
        - train_dataset_mean.reshape(1, 15).to(device)
    ) / (train_dataset_std.reshape(1, 15).to(device) + 1e-8)
    sample_x, logp_x, logp_x1_minus_logp_x0 = (
        conditional_flow_model.sample_with_log_prob(
            n_dims=[resolution],
            n_channels=1,
            t_span=torch.linspace(0.0, 1.0, 1000),
            batch_size=1,
            x_0=prior_x,
            condition=y.to(device).repeat(1, 1),
        )
    )

    airfoil_generated_normed = sample_x.detach()[0, 0, :]

    airfoil_generated = (airfoil_generated_normed + 1) / 2.0 * (
        train_dataset_max[1] - train_dataset_min[1]
    ) + train_dataset_min[1]

    xs = (np.cos(np.linspace(0, 2 * np.pi, resolution)) + 1) / 2

    ys = airfoil_generated.cpu().numpy()

    coordination_xy = np.concatenate([xs[:, None], ys[:, None]], axis=-1)

    Fit = Fit_airfoil_15(data=coordination_xy)
    parsec_features = Fit.parsec_features
    (
        rf_real,
        t4u_real,
        t4l_real,
        xumax_real,
        yumax_real,
        xlmax_real,
        ylmax_real,
        t25u_real,
        t25l_real,
        angle_real,
        te1_real,
        xr_real,
        yr_real,
        t60u_real,
        t60l_real,
    ) = parsec_features

    fig, xs, ys = render_fig(
        xs=xs,
        ys=ys,
    )

    curve_data = [
        {"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(xs, ys)
    ]
    curve_json = json.dumps(curve_data, indent=2)
    return (
        fig,
        curve_json,
        airfoil_generated_normed.cpu().numpy(),
        rf_real,
        t4u_real,
        t4l_real,
        xumax_real,
        yumax_real,
        xlmax_real,
        ylmax_real,
        t25u_real,
        t25l_real,
        angle_real,
        te1_real,
        xr_real,
        yr_real,
        t60u_real,
        t60l_real,
        logp_x.item(),
    )


def generate_airfoil_from_geometry_params_with_finetuning(
    resolution,
    rf,
    t4u,
    t4l,
    xumax,
    yumax,
    xlmax,
    ylmax,
    t25u,
    t25l,
    angle,
    te1,
    xr,
    yr,
    t60u,
    t60l,
    finetune_iterations=20,
    t_steps=1000,
    latent_initialization="Inverse prior initialization",
    progress=gr.Progress(),
):
    prior_x = conditional_flow_model.gaussian_process.sample(
        dims=[resolution], n_samples=1, n_channels=1
    )
    y = (
        torch.tensor(
            (
                rf,
                t4u,
                t4l,
                xumax,
                yumax,
                xlmax,
                ylmax,
                t25u,
                t25l,
                angle,
                te1,
                xr,
                yr,
                t60u,
                t60l,
            )
        )
        .reshape(1, 15)
        .to(device)
        - train_dataset_mean.reshape(1, 15).to(device)
    ) / (train_dataset_std.reshape(1, 15).to(device) + 1e-8)

    # test_code
    t_span = torch.linspace(0.0, 1.0, t_steps).to(device)
    (
        sample_x,
        logp_x,
        logp_x1_minus_logp_x0,
    ) = conditional_flow_model.sample_with_log_prob(
        n_dims=[resolution],
        n_channels=1,
        t_span=t_span,
        x_0=prior_x,
        condition=y.to(device).repeat(1, 1),
        with_grad=False,
    )

    if latent_initialization == "Random latent function initialization":
        inverse_prior = unconditional_flow_model.gaussian_process.sample(
            dims=[resolution], n_samples=1, n_channels=1
        )
    elif latent_initialization == "Zero latent function initialization":
        inverse_prior = torch.zeros(1, 1, resolution).to(device)
    elif latent_initialization == "Inverse prior initialization":
        inverse_prior = unconditional_flow_model.inverse_sample(
            n_dims=[resolution],
            n_channels=1,
            t_span=(
                torch.linspace(0.0, 1.0, 100).to(device) if t_steps < 100 else t_span
            ),
            x_0=sample_x,
        )
        if inverse_prior.abs().max() > 1000:
            print(
                "Inverse prior is too large, please check the input parameters. Turning to zero initialization."
            )
            inverse_prior = torch.zeros(1, 1, resolution).to(device)
    else:
        inverse_prior = torch.zeros(1, 1, resolution).to(device)

    # deep copy the flow model for regression
    flow_model_for_regression = OptimalTransportFunctionalFlowForRegression(
        config=config.flow_model_regression,
        model=copy.deepcopy(unconditional_flow_model.model),
        prior=inverse_prior,
    ).to(device)

    optimizer = torch.optim.Adam(
        find_parameters(flow_model_for_regression),
        lr=config.parameter.learning_rate,
    )

    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        T_max=finetune_iterations,
        eta_min=2e-6,
        warmup_steps=config.parameter.warmup_steps,
    )

    for iteration in progress.tqdm(range(finetune_iterations)):
        flow_model_for_regression.train()

        x_0_repeat = flow_model_for_regression.prior.repeat(1, 1, 1)
        (
            x_1_repeat,
            logp_1_repeat,
            logp_x1_minus_logp_x0,
        ) = flow_model_for_regression.sample_with_log_prob(
            t_span=t_span, x_0=x_0_repeat, using_Hutchinson_trace_estimator=True
        )

        airfoil_generated_normed = x_1_repeat[0, 0, :]

        airfoil_generated = (airfoil_generated_normed + 1) / 2.0 * (
            train_dataset_max[1] - train_dataset_min[1]
        ) + train_dataset_min[1]

        xs = (
            torch.cos(
                torch.linspace(
                    0.0,
                    2.0 * torch.pi,
                    resolution,
                    device=airfoil_generated.device,  # keep on same device
                    dtype=airfoil_generated.dtype,
                )  # keep same precision
            )
            + 1.0
        ) / 2.0

        ys = airfoil_generated.squeeze()

        coordination_xy = torch.stack((xs, ys), dim=1)

        airfoil_metric = calculate_airfoil_metric_n15(x=xs, y=ys, n_inner_steps=20000)

        (
            rf_real,
            t4u_real,
            t4l_real,
            xumax_real,
            yumax_real,
            xlmax_real,
            ylmax_real,
            t25u_real,
            t25l_real,
            angle_real,
            te1_real,
            xr_real,
            yr_real,
            t60u_real,
            t60l_real,
        ) = airfoil_metric
        metric = torch.stack(
            (
                rf_real,
                t4u_real,
                t4l_real,
                xumax_real,
                yumax_real,
                xlmax_real,
                ylmax_real,
                t25u_real,
                t25l_real,
                angle_real,
                te1_real,
                xr_real,
                yr_real,
                t60u_real,
                t60l_real,
            ),
            dim=0,
        )

        loss_1 = torch.mean(
            0.5
            * torch.sum(
                (
                    (
                        (
                            metric
                            - torch.tensor(
                                (
                                    rf,
                                    t4u,
                                    t4l,
                                    xumax,
                                    yumax,
                                    xlmax,
                                    ylmax,
                                    t25u,
                                    t25l,
                                    angle,
                                    te1,
                                    xr,
                                    yr,
                                    t60u,
                                    t60l,
                                )
                            ).to(device)
                        )
                        / train_dataset_std.to(device)
                    )
                    ** 2
                )
            )
            / 0.0001
        )
        loss_2 = -logp_1_repeat.mean()

        loss = loss_1 + loss_2

        optimizer.zero_grad()
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            find_parameters(flow_model_for_regression), 1.0
        )

        optimizer.step()
        scheduler.step()

        acc_train_loss = loss.mean().item()
        print(
            f"iteration: {iteration}, train_loss: {acc_train_loss:.5f}, loss 1: {loss_1.mean().item():.5f}, loss 2: {loss_2.mean().item():.5f}, lr: {scheduler.get_last_lr()[0]:.7f}"
        )

    flow_model_for_regression.eval()
    with torch.no_grad():
        (
            sample_x_ft,
            logp_x_ft,
            logp_x1_minus_logp_x0,
        ) = flow_model_for_regression.sample_with_log_prob(
            n_dims=[resolution],
            n_channels=1,
            t_span=t_span,
            x_0=flow_model_for_regression.prior,
        )

        airfoil_generated_normed = sample_x_ft[0, 0, :]

        airfoil_generated = (airfoil_generated_normed + 1) / 2.0 * (
            train_dataset_max[1] - train_dataset_min[1]
        ) + train_dataset_min[1]

        xs = (
            torch.cos(
                torch.linspace(
                    0.0,
                    2.0 * torch.pi,
                    resolution,
                    device=airfoil_generated.device,  # keep on same device
                    dtype=airfoil_generated.dtype,
                )  # keep same precision
            )
            + 1.0
        ) / 2.0

        ys = airfoil_generated.squeeze()

        coordination_xy = torch.stack((xs, ys), dim=1)

        Fit = Fit_airfoil_15(data=coordination_xy.cpu().numpy())
        parsec_features = Fit.parsec_features
        (
            rf_real,
            t4u_real,
            t4l_real,
            xumax_real,
            yumax_real,
            xlmax_real,
            ylmax_real,
            t25u_real,
            t25l_real,
            angle_real,
            te1_real,
            xr_real,
            yr_real,
            t60u_real,
            t60l_real,
        ) = parsec_features

        # airfoil_metric = calculate_airfoil_metric_n15(x=xs, y=ys, n_inner_steps=20000)
        # rf_real, t4u_real, t4l_real, xumax_real, yumax_real, xlmax_real, ylmax_real, t25u_real, t25l_real, angle_real, te1_real, xr_real, yr_real, t60u_real, t60l_real = airfoil_metric

        fig, xs, ys = render_fig(
            xs=xs.cpu().numpy(),
            ys=ys.cpu().numpy(),
        )

        curve_data = [
            {"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(xs, ys)
        ]
        curve_json = json.dumps(curve_data, indent=2)
        return (
            fig,
            curve_json,
            airfoil_generated_normed.squeeze().cpu().numpy(),
            rf_real,
            t4u_real,
            t4l_real,
            xumax_real,
            yumax_real,
            xlmax_real,
            ylmax_real,
            t25u_real,
            t25l_real,
            angle_real,
            te1_real,
            xr_real,
            yr_real,
            t60u_real,
            t60l_real,
            logp_x.item(),
            logp_x_ft.item(),
        )


with gr.Blocks() as demo:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = "airfoil-editing-gradio"
    config = EasyDict(
        dict(
            device=device,
            unconditional_flow_model=dict(
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
                                n_layers=4,
                                n_conditions=0,
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
                                n_layers=4,
                                n_conditions=0,
                            ),
                        ),
                    ),
                ),
            ),
            conditional_flow_model=dict(
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
                                modes=64,
                                vis_channels=1,
                                hidden_channels=256,
                                proj_channels=128,
                                x_dim=1,
                                t_scaling=1,
                                n_layers=6,
                                n_conditions=15,
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                noise_level=0.000003,
                learning_rate=5e-6,
                warmup_steps=0,
                unconditional_model_load_path="unconditional.pth",
                conditional_model_load_path="conditional.pth",
            ),
        )
    )

    loaded_tensors = load_file(f"train_datasets.safetensors")
    train_dataset_min = loaded_tensors["train_dataset_min"]
    train_dataset_max = loaded_tensors["train_dataset_max"]
    stats = torch.load("mean_std.pt")
    train_dataset_mean, train_dataset_std = stats["mean"], stats["std"]

    unconditional_flow_model = init_unconditional_flow_model(config, device)
    conditional_flow_model = init_conditional_flow_model(config, device)

    prior_x = gr.State()
    airfoil_for_editing = gr.State()
    constraints_for_editing = gr.State()
    points_id_constraints_for_editing_all = gr.State()
    xs_controlled = gr.State()
    ys_controlled = gr.State()
    ys_controlled_edited = gr.State()

    # Add a title
    gr.Markdown(
        "# FuncGenFoil: Airfoil Generation and Editing Model in Function Space "
    )

    gr.Markdown("## Set Resolution")

    with gr.Row():
        resolution = gr.Number(value=257, label="Resolution")
        seed = gr.Number(value=None, label="Random Seed")
        select_last_prior = gr.Checkbox(value=False, label="Select last prior")

    gr.Markdown("## Airfoil Generation (Unconditional)")

    btn_1 = gr.Button("Generate airfoil")

    with gr.Row():
        plot_1 = gr.Plot(label="Airfoil Generation (Unconditional)")
        text_1 = gr.Textbox(label="Airfoil Curve Data (JSON Format)", interactive=False)

    gr.Markdown("## Airfoil Generation (Conditional)")

    gr.Markdown("### Input geometry parameters for conditional generation")

    with gr.Row():
        rf = gr.Number(value=0.01, label="Leading edge radius (Design)")
        t4u = gr.Number(
            value=0.035, label="Upper surface thickness at 4% chord length (Design)"
        )
        t4l = gr.Number(
            value=-0.02, label="Lower surface thickness at 4% chord length (Design)"
        )

    with gr.Row():
        rf_real = gr.Number(value=None, label="Leading edge radius (Generated)")
        t4u_real = gr.Number(
            value=None, label="Upper surface thickness at 4% chord length (Generated)"
        )
        t4l_real = gr.Number(
            value=None, label="Lower surface thickness at 4% chord length (Generated)"
        )

    with gr.Row():
        xumax = gr.Number(
            value=0.47, label="X-coordinate of maximum upper surface thickness (Design)"
        )
        yumax = gr.Number(
            value=0.07, label="Y-coordinate of maximum upper surface thickness (Design)"
        )
        xlmax = gr.Number(
            value=0.36, label="X-coordinate of maximum lower surface thickness (Design)"
        )

    with gr.Row():
        xumax_real = gr.Number(
            value=None,
            label="X-coordinate of maximum upper surface thickness (Generated)",
        )
        yumax_real = gr.Number(
            value=None,
            label="Y-coordinate of maximum upper surface thickness (Generated)",
        )
        xlmax_real = gr.Number(
            value=None,
            label="X-coordinate of maximum lower surface thickness (Generated)",
        )

    with gr.Row():
        ylmax = gr.Number(
            value=-0.05,
            label="Y-coordinate of maximum lower surface thickness (Design)",
        )
        t25u = gr.Number(
            value=0.064, label="Upper surface thickness at 25% chord length (Design)"
        )
        t25l = gr.Number(
            value=-0.048, label="Lower surface thickness at 25% chord length (Design)"
        )

    with gr.Row():
        ylmax_real = gr.Number(
            value=None,
            label="Y-coordinate of maximum lower surface thickness (Generated)",
        )
        t25u_real = gr.Number(
            value=None, label="Upper surface thickness at 25% chord length (Generated)"
        )
        t25l_real = gr.Number(
            value=None, label="Lower surface thickness at 25% chord length (Generated)"
        )

    with gr.Row():
        angle = gr.Number(
            value=-17.74, label="Upper surface trailing edge angle (Design)"
        )
        te1 = gr.Number(value=0.002, label="Trailing edge thickness (Design)")
        xr = gr.Number(value=0.9, label="Rear loading X-coordinate (Design)")

    with gr.Row():
        angle_real = gr.Number(
            value=None, label="Upper surface trailing edge angle (Generated)"
        )
        te1_real = gr.Number(value=None, label="Trailing edge thickness (Generated)")
        xr_real = gr.Number(value=None, label="Rear loading X-coordinate (Generated)")

    with gr.Row():
        yr = gr.Number(value=0.0098, label="Rear loading Y-coordinate (Design)")
        t60u = gr.Number(
            value=0.068, label="Upper surface thickness at 60% chord length (Design)"
        )
        t60l = gr.Number(
            value=-0.036, label="Lower surface thickness at 60% chord length (Design)"
        )

    with gr.Row():
        yr_real = gr.Number(value=None, label="Rear loading Y-coordinate (Generated)")
        t60u_real = gr.Number(
            value=None, label="Upper surface thickness at 60% chord length (Generated)"
        )
        t60l_real = gr.Number(
            value=None, label="Lower surface thickness at 60% chord length (Generated)"
        )

    btn_conditional = gr.Button("Generate airfoil with geometry constraints")
    with gr.Row():
        logp_x = gr.Number(value=None, label="Log probability of generated airfoil")
        finetune_iterations_for_geometry = gr.Number(
            value=20, label="Finetune iterations [5~1000]"
        )
        t_steps_for_finetune = gr.Number(value=1000, label="t steps [10~1000]")
    btn_conditional_ft = gr.Button(
        "Generate airfoil with geometry constraints with finetuning"
    )
    with gr.Row():
        logp_x_ft = gr.Number(
            value=None, label="Log probability of generated airfoil with finetuning"
        )

    with gr.Row():
        plot_conditional = gr.Plot(label="Airfoil Generation (Conditional)")
        text_conditional = gr.Textbox(
            label="Airfoil Curve Data (JSON Format)", interactive=False
        )

    gr.Markdown("## Airfoil Editing (by fixing and modifying certain points)")

    gr.Markdown("### Set random constraints")
    with gr.Row():
        random_scale = gr.Number(value=0.000003, label="Random scale")
        random_points_number = gr.Number(value=6, label="Random points number")

    gr.Markdown("### Set constraints manually")

    with gr.Row():
        control_points_range = gr.Textbox(
            label='Enter Box-ranged Point Constraints by JSON (e.g., [{"x_min": -0.1, "x_max": 0.2, "y_min": -0.5, "y_max": 0.5, "delta": 0.0}, {"x_min": 0.3, "x_max": 0.5, "y_min": -0.5, "y_max": 0.0, "delta": 0.000003}])'
        )

    btn_2 = gr.Button("Set constraints")

    with gr.Row():
        # plot_2 = gr.Plot(label="Airfoil Editing (Original)")
        text_2 = gr.Textbox(label="Constraint Points (JSON Format)", interactive=False)

    gr.Markdown("### Finetune the airfoil curve")
    with gr.Row():
        finetune_iterations = gr.Number(value=20, label="Finetune iterations [5~1000]")
        t_steps = gr.Number(value=20, label="t steps [10~100]")
        # add choice for random_latent_initialization or zero_latent_initialization or inverse_prior_initialization
        latent_initialization = gr.Dropdown(
            choices=[
                "Zero latent function initialization",
                "Inverse prior initialization",
                "Random latent function initialization",
            ],
            value="Zero latent function initialization",
            label="Latent Function Initialization",
        )

    btn_3 = gr.Button("Generate random constraints")

    with gr.Row():
        # plot_3 = gr.Plot(label="Airfoil Editing (Edited)")
        text_3 = gr.Textbox(label="Airfoil Curve Data (JSON Format)", interactive=False)

    with gr.Row():
        plot_2 = gr.Plot(label="Airfoil Editing (Original)")
        plot_3 = gr.Plot(label="Airfoil Editing (Edited)")

    btn_1.click(
        generate_airfoil_curve,
        inputs=[resolution, seed, prior_x, select_last_prior],
        outputs=[plot_1, text_1, airfoil_for_editing, prior_x],
    )

    btn_conditional.click(
        generate_airfoil_from_geometry_params,
        inputs=[
            resolution,
            rf,
            t4u,
            t4l,
            xumax,
            yumax,
            xlmax,
            ylmax,
            t25u,
            t25l,
            angle,
            te1,
            xr,
            yr,
            t60u,
            t60l,
        ],
        outputs=[
            plot_conditional,
            text_conditional,
            airfoil_for_editing,
            rf_real,
            t4u_real,
            t4l_real,
            xumax_real,
            yumax_real,
            xlmax_real,
            ylmax_real,
            t25u_real,
            t25l_real,
            angle_real,
            te1_real,
            xr_real,
            yr_real,
            t60u_real,
            t60l_real,
            logp_x,
        ],
    )

    btn_conditional_ft.click(
        generate_airfoil_from_geometry_params_with_finetuning,
        inputs=[
            resolution,
            rf,
            t4u,
            t4l,
            xumax,
            yumax,
            xlmax,
            ylmax,
            t25u,
            t25l,
            angle,
            te1,
            xr,
            yr,
            t60u,
            t60l,
            finetune_iterations_for_geometry,
            t_steps_for_finetune,
        ],
        outputs=[
            plot_conditional,
            text_conditional,
            airfoil_for_editing,
            rf_real,
            t4u_real,
            t4l_real,
            xumax_real,
            yumax_real,
            xlmax_real,
            ylmax_real,
            t25u_real,
            t25l_real,
            angle_real,
            te1_real,
            xr_real,
            yr_real,
            t60u_real,
            t60l_real,
            logp_x,
            logp_x_ft,
        ],
    )

    btn_2.click(
        generate_constraints,
        inputs=[
            resolution,
            random_scale,
            random_points_number,
            control_points_range,
            airfoil_for_editing,
        ],
        outputs=[
            plot_2,
            text_2,
            points_id_constraints_for_editing_all,
            xs_controlled,
            ys_controlled,
            ys_controlled_edited,
        ],
    )

    btn_3.click(
        generate_editing_airfoil_curve,
        inputs=[
            resolution,
            airfoil_for_editing,
            finetune_iterations,
            points_id_constraints_for_editing_all,
            xs_controlled,
            ys_controlled,
            ys_controlled_edited,
            t_steps,
            latent_initialization,
        ],
        outputs=[plot_3, text_3],
    )

demo.launch()
