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
from airfoil_generation.dataset.stretch_dataset import StretchDataset


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

    return fig


def init_stretch_flow_model(config, device):
    flow_model = OptimalTransportFunctionalFlow(
        config=config.conditional_flow_model
    ).to(device)

    assert os.path.exists(
        config.parameter.stretch_model_load_path
    ) and os.path.isfile(
        config.parameter.stretch_model_load_path
    ), f"Model file not found at {config.parameter.stretch_model_load_path}"

    # pop out _metadata key
    state_dict = torch.load(
        config.parameter.stretch_model_load_path,
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
            new_key = key[len(prefix):]
        else:
            new_key = key
        new_state_dict[new_key] = value

    flow_model.model.load_state_dict(new_state_dict)
    print("Model loaded from: ", config.parameter.stretch_model_load_path)

    return flow_model


def sample_from_dataset(seed=None):
    test_dataset = StretchDataset(split="test", folder_path="../../data")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    idx = random.randint(0, len(test_dataset) - 1)
    data = test_dataset[idx]
    


with gr.Blocks() as demo:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_name = "airfoil-stretch-gradio"
    config = EasyDict(
        dict(
            device=device,
            conditional_flow_model=dict(
                device=device,
                gaussian_process=dict(
                    type="matern",
                    args=dict(
                        device=device,
                        length_scale=0.03,
                        nu=2.5,
                        dims=[130],
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
                                n_conditions=10,
                            ),
                        ),
                    ),
                ),
            ),
            parameter=dict(
                noise_level=0.000003,
                learning_rate=5e-6,
                warmup_steps=0,
                stretch_model_load_path="stretch.pth",
            ),
        )
    )

    loaded_tensors = load_file(f"train_datasets.safetensors")
    train_dataset_min = loaded_tensors["train_dataset_min"]
    train_dataset_max = loaded_tensors["train_dataset_max"]
    stats = torch.load("mean_std.pt", map_location=torch.device('cpu'))
    train_dataset_mean, train_dataset_std = stats["mean"], stats["std"]

    strecth_flow_model = init_stretch_flow_model(config, device)

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

    gr.Markdown("## Choose an apart from dataset")

    with gr.Row():
        seed = gr.Number(value=None, label="Random Seed")

    btn_sample = gr.Button("Random Sample")

    with gr.Row():
        plot_1 = gr.Plot(label="Selected Airfoil")
        text_1 = gr.Textbox(label="Airfoil Curve Data (JSON Format)", interactive=False)

    gr.Markdown("## Stretch Airfoil")

    gr.Markdown("### Input geometry parameters for stretching")

    with gr.Row():
        joint_y_up_req = gr.Number(value=None, label="Joint Y-coordinate of upper surface (Required)")
        joint_y_down_req = gr.Number(value=None, label="Joint Y-coordinate of lower surface (Required)")
    
    with gr.Row():
        joint_y_up_gen = gr.Number(value=None, label="Joint Y-coordinate of upper surface (Generated)")
        joint_y_down_gen = gr.Number(value=None, label="Joint Y-coordinate of lower surface (Generated)")

    with gr.Row():
        joint_dydx_up_req = gr.Number(value=None, label="Joint upper surface slope (Required)")
        joint_dydx_down_req = gr.Number(value=None, label="Joint lower surface slope (Required)")

    with gr.Row():
        joint_dydx_up_gen = gr.Number(value=None, label="Joint upper surface slope (Generated)")
        joint_dydx_down_gen = gr.Number(value=None, label="Joint lower surface slope (Generated)")
    
    with gr.Row():
        joint_d2ydx2_up_req = gr.Number(value=None, label="Joint upper surface second derivative (Required)")
        joint_d2ydx2_down_req = gr.Number(value=None, label="Joint lower surface second derivative (Required)")

    with gr.Row():
        joint_d2ydx2_up_gen = gr.Number(value=None, label="Joint upper surface second derivative (Generated)")
        joint_d2ydx2_down_gen = gr.Number(value=None, label="Joint lower surface second derivative (Generated)")

    with gr.Row():
        angle_req = gr.Number(value=None, label="Upper surface trailing edge angle (Required)")
        thickness_req = gr.Number(value=None, label="Trailing edge thickness (Required)")

    with gr.Row():
        angle_gen = gr.Number(value=None, label="Upper surface trailing edge angle (Generated)")
        thickness_gen = gr.Number(value=None, label="Trailing edge thickness (Generated)")

    with gr.Row():
        xr_req = gr.Number(value=None, label="Rear loading X-coordinate (Required)")
        yr_req = gr.Number(value=None, label="Rear loading Y-coordinate (Required)")

    with gr.Row():
        xr_gen = gr.Number(value=None, label="Rear loading X-coordinate (Generated)")
        yr_gen = gr.Number(value=None, label="Rear loading Y-coordinate (Generated)")

    btn_stretch = gr.Button("Stretch Airfoil")

    with gr.Row():
        plot_2 = gr.Plot(label="Airfoil Generation (Conditional)")
        text_2 = gr.Textbox(label="Airfoil Curve Data (JSON Format)", interactive=False)


    btn_sample.click(
        sample_from_dataset,
        inputs=[seed],
        outputs=[plot_1, text_1, joint_y_up_req, joint_y_down_req, joint_dydx_up_req,
                 joint_dydx_down_req, joint_d2ydx2_up_req, joint_d2ydx2_down_req,
                 angle_req, thickness_req, xr_req, yr_req],
    )

demo.launch()
