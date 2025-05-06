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
from airfoil_generation.dataset.parsec_direct_n15 import Fit_airfoil
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




    def generate_airfoil_curve(
        resolution, seed=None, prior_x=None, select_last_prior=False
    ):
        """
        Overview:
            Generate a random airfoil curve using the unconditional flow model.
        Arguments:
            resolution (int): The number of points in the airfoil curve.
            seed (int, optional): Random seed for reproducibility. Default is None.
            prior_x (torch.Tensor, optional): Prior input for the flow model. Default is None.
            select_last_prior (bool, optional): If True, use the last prior input. Default is False.
        Returns:
            fig (plotly.graph_objects.Figure): The generated airfoil curve figure.
            curve_json (str): JSON string of the airfoil curve data points.
            airfoil_generated_normed (numpy.ndarray): The generated airfoil curve in normalized form.
            prior_x (torch.Tensor): The prior input used for generation.
        """
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

    # For an example, let's set the resolution to 257 and seed to 42
    resolution = 257
    seed = 42
    prior_x = None
    select_last_prior = False

    plot_1, text_1, airfoil_for_editing, prior_x = generate_airfoil_curve(resolution, seed, prior_x, select_last_prior)

    # plot_1.show()

    print("airfoil output :", airfoil_for_editing.shape)
    print("prior_x output :", prior_x.shape)

