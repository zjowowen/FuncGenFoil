import gradio as gr
import plotly.graph_objects as go
import json
import os
import matplotlib

# matplotlib.use("Agg")
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


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = "airfoil-editing-gradio"
    config = EasyDict(
        dict(
            device=device,
            unconditional_flow_model=dict(
                device=device,
                gaussian_process=dict(
                    type="matern",
                    args=dict(
                        device=device,
                        length_scale=0.01,
                        nu=1.5,
                        dims=[257],
                    ),
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
                    type="matern",
                    args=dict(
                        device=device,
                        length_scale=0.01,
                        nu=1.5,
                        dims=[257],
                    ),
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
                    type="matern",
                    args=dict(
                        device=device,
                        length_scale=0.03,
                        nu=2.5,
                        dims=[257],
                    ),
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

    # For an example, let's set the resolution to 257 and other parameters

    resolution = 257
    rf = 0.01
    t4u = 0.035
    t4l = -0.02
    xumax = 0.47
    yumax = 0.07
    xlmax = 0.36
    ylmax = -0.05
    t25u = 0.064
    t25l = -0.048
    angle = -17.74
    te1 = 0.002
    xr = 0.9
    yr = 0.0098
    t60u = 0.068
    t60l = -0.036

    (
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
    ) = generate_airfoil_from_geometry_params(
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
    )

    # plot_conditional.show()
    print("airfoil output :", airfoil_for_editing.shape)
    # geometry parameters recomputed for the generated airfoil
    print("rf_real output :", rf_real)
    print("t4u_real output :", t4u_real)
    print("t4l_real output :", t4l_real)
    print("xumax_real output :", xumax_real)
    print("yumax_real output :", yumax_real)
    print("xlmax_real output :", xlmax_real)
    print("ylmax_real output :", ylmax_real)
    print("t25u_real output :", t25u_real)
    print("t25l_real output :", t25l_real)
    print("angle_real output :", angle_real)
    print("te1_real output :", te1_real)
    print("xr_real output :", xr_real)
    print("yr_real output :", yr_real)
    print("t60u_real output :", t60u_real)
    print("t60l_real output :", t60l_real)

    # the likelihood of the generated airfoil
    print("logp_x output :", logp_x)
