import gradio as gr
import plotly.graph_objects as go
import json
import os
import matplotlib
import math

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
from scipy.interpolate import splev, splprep
from scipy import optimize
from airfoil_generation.dataset.stretch_dataset import StretchDataset


seed_flag = False


def render_fig(
    xs_apart,
    ys_apart,
    xs_bpart,
    ys_bpart,
    ys_bpart_original=None,
):
    fig = go.Figure()

    # A部分
    fig.add_trace(
        go.Scatter(
            x=xs_apart,
            y=ys_apart,
            mode="lines+markers",
            line=dict(color="blue", width=2),
            marker=dict(size=4, color="blue"),
            name="A Part",
        )
    )

    # B部分(上半部分)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([xs_bpart[:len(xs_bpart) // 2], [xs_apart[0]]]),
            y=np.concatenate([ys_bpart[:len(ys_bpart) // 2], [ys_apart[0]]]),
            mode="lines+markers",
            line=dict(color="orange", width=2),
            marker=dict(size=4, color="orange"),
            name="B Part",
        )
    )

    # B部分(下半部分)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([[xs_apart[-1]], xs_bpart[len(xs_bpart) // 2:]]),
            y=np.concatenate([[ys_apart[-1]], ys_bpart[len(ys_bpart) // 2:]]),
            mode="lines+markers",
            line=dict(color="orange", width=2),
            marker=dict(size=4, color="orange"),
            name="B Part",
            showlegend=False,
        )
    )

    # 原始B部分(上半部分)
    if ys_bpart_original is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xs_bpart[:len(xs_bpart) // 2], [xs_apart[0]]]),
                y=np.concatenate([ys_bpart_original[:len(ys_bpart_original) // 2], [ys_apart[0]]]),
                mode="markers",
                marker=dict(size=4, color="red"),
                name="B Part (Original)",
            )
        )
    
    # 原始B部分(下半部分)
    if ys_bpart_original is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([[xs_apart[-1]], xs_bpart[len(xs_bpart) // 2:]]),
                y=np.concatenate([[ys_apart[-1]], ys_bpart_original[len(ys_bpart_original) // 2:]]),
                mode="markers",
                marker=dict(size=4, color="red"),
                name="B Part (Original)",
                showlegend=False,
            )
        )

    # 图像布局
    fig.update_layout(
        xaxis=dict(range=[0, 1.2], showgrid=False, zeroline=False, visible=False),
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
    global seed_flag
    test_dataset = StretchDataset(split="test", folder_path="data")
    if seed is not None and not seed_flag:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        seed_flag = True
    idx = random.randint(0, len(test_dataset) - 1)
    data = test_dataset[idx]
    apart = data["apart"]
    bpart = data["bpart"]
    params = data["params"]
    fig = render_fig(
        xs_apart=apart[:, 0].numpy(),
        ys_apart=apart[:, 1].numpy(),
        xs_bpart=bpart[:, 0].numpy(),
        ys_bpart=bpart[:, 1].numpy(),
    )
    curve_data = {
        "A-Part": [
            {"x": float(x_val), "y": float(y_val)} for x_val, y_val in apart
        ],
        "B-Part": [
            {"x": float(x_val), "y": float(y_val)} for x_val, y_val in bpart
        ],
    }
    curve_json = json.dumps(curve_data, indent=2)
    return fig, curve_json, *params.tolist(), data


def stretch_airfoil(data):
    strecth_flow_model.eval()
    with torch.no_grad():
        data = data.to(device)
        y = (data["params"] - train_dataset_mean.to(device)) / (
            train_dataset_std.to(device) + 1e-8
        )
        sample = strecth_flow_model.sample_process(
            n_dims=config.conditional_flow_model.gaussian_process.args.dims,
            n_channels=1,
            t_span=torch.linspace(0.0, 1.0, 1000),
            batch_size=1,
            condition=y.unsqueeze(0),
        )[-1]
        # sample_trajectory is of shape (T, B, C, D)
        ys_bpart_ = sample.squeeze().cpu().numpy()
        ys_bpart_ = (ys_bpart_ + 1) / 2 * (
                        train_dataset_max.numpy()[1] - train_dataset_min.numpy()[1]
                    ) + train_dataset_min.numpy()[1]
        apart = data["apart"]
        bpart = data["bpart"]
        fig = render_fig(
            xs_apart=apart[:, 0].cpu().numpy(),
            ys_apart=apart[:, 1].cpu().numpy(),
            xs_bpart=bpart[:, 0].cpu().numpy(),
            ys_bpart=ys_bpart_,
            ys_bpart_original=bpart[:, 1].cpu().numpy()
        )
        curve_data = {
            "A-Part": [
                {"x": float(x_val), "y": float(y_val)} for x_val, y_val in apart
            ],
            "B-Part": [
                {"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(bpart[:, 0].cpu().numpy(), ys_bpart_)
            ],
        }
        curve_json = json.dumps(curve_data, indent=2)

        # Calculate the metrics
        a = (ys_bpart_[0] - ys_bpart_[9]) / (bpart[0, 0] - bpart[9, 0])
        theta_radians = math.atan(a)
        theta_degrees = math.degrees(theta_radians)
        angle = theta_degrees

        te = (ys_bpart_[0] - ys_bpart_[-1])/2
        yr = ys_bpart_[65:].max()
        xr = bpart[65:, 0].cpu().numpy()[np.argmax(ys_bpart_[65:])]

        # Calculate the joint derivatives
        temp_data = np.stack([np.concatenate([bpart[:, 0].cpu().numpy()[:65],
                                              apart[:, 0].cpu().numpy(),
                                              bpart[:, 0].cpu().numpy()[65:]]),
                              np.concatenate([ys_bpart_[:65],
                                              apart[:, 1].cpu().numpy(),
                                              ys_bpart_[65:]])], axis=1)
        tck, u = splprep(temp_data.T, s=0)
        iLE = 128
        def objective(u_tmp):
            x_tmp, _ = splev(u_tmp, tck)
            return (x_tmp - 0.6)**2
        uup = optimize.minimize_scalar(
            objective, bounds=(0, u[iLE]), method="bounded"
        ).x
        ulo = optimize.minimize_scalar(
            objective, bounds=(u[iLE], 1), method="bounded"
        ).x
        _, yup = splev(uup, tck)
        dxduup, dyduup = splev(uup, tck, der=1)
        d2xdu2up, d2ydu2up = splev(uup, tck, der=2)
        dydxup = dyduup / dxduup
        d2ydx2up = (d2ydu2up * dxduup - d2xdu2up * dyduup) / dxduup**3
        _, ylo = splev(ulo, tck)
        dxdulo, dydulo = splev(ulo, tck, der=1)
        d2xdu2lo, d2ydu2lo = splev(ulo, tck, der=2)
        dydxlo = dydulo / dxdulo
        d2ydx2lo = (d2ydu2lo * dxdulo - d2xdu2lo * dydulo) / dxdulo**3

        return fig, curve_json, angle, te, xr, yr, \
               yup, ylo, dydxup, dydxlo, d2ydx2up, d2ydx2lo
    
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
                stretch_model_load_path="examples\\stretch\\stretch_model.pth",
            ),
        )
    )

    loaded_tensors = load_file("examples\\stretch\\train_datasets.safetensors")
    train_dataset_min = loaded_tensors["train_dataset_min"]
    train_dataset_max = loaded_tensors["train_dataset_max"]
    stats = torch.load("examples\\stretch\\mean_std.pt", map_location=torch.device('cpu'))
    train_dataset_mean, train_dataset_std = stats["mean"], stats["std"]

    strecth_flow_model = init_stretch_flow_model(config, device)

    # Add a title
    gr.Markdown(
        "# FuncGenFoil: Airfoil Generation and Editing Model in Function Space "
    )

    gr.Markdown("## Choose an airfoil from dataset")

    with gr.Row():
        seed = gr.Number(value=42, label="Random Seed")

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
        plot_2 = gr.Plot(label="Stretched Airfoil")
        text_2 = gr.Textbox(label="Airfoil Curve Data (JSON Format)", interactive=False)

    condition = gr.State()

    btn_sample.click(
        sample_from_dataset,
        inputs=[seed],
        outputs=[plot_1, text_1, angle_req, thickness_req, xr_req, yr_req,
                 joint_y_up_req, joint_y_down_req, joint_dydx_up_req,
                 joint_dydx_down_req, joint_d2ydx2_up_req, joint_d2ydx2_down_req, condition],
    )

    btn_stretch.click(
        stretch_airfoil,
        inputs=[condition],
        outputs=[plot_2, text_2, angle_gen, thickness_gen, xr_gen, yr_gen,
                 joint_y_up_gen, joint_y_down_gen, joint_dydx_up_gen,
                 joint_dydx_down_gen, joint_d2ydx2_up_gen, joint_d2ydx2_down_gen],
    )

demo.launch()
