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
    ):

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

        for iteration in range(finetune_iterations):
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

            return fig, curve_json, airfoil_for_editing_sampled.squeeze().cpu().numpy()



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


    # For an example, let's set the resolution to 257 and seed to 42
    resolution = 257
    seed = 42
    prior_x = None
    select_last_prior = False

    plot_1, text_1, airfoil_for_editing, prior_x = generate_airfoil_curve(resolution, seed, prior_x, select_last_prior)

    # plot_1.show()

    # so we generate a random airfoil curve
    print("airfoil output :", airfoil_for_editing.shape)
    print("prior_x output :", prior_x.shape)

    # We then set the constraints for the airfoil curve, suppose we set the constraints as:
    random_scale = 0.000003
    random_points_number = 5
    control_points_range = '[{"x_min": -0.1, "x_max": 0.2, "y_min": -0.5, "y_max": 0.5, "delta": 0.000003}, {"x_min": 0.3, "x_max": 0.5, "y_min": -0.5, "y_max": 0.0, "delta": 0.000003}]'
    
    (
        plot_2,
        text_2,
        points_id_constraints_for_editing_all,
        xs_controlled,
        ys_controlled,
        ys_controlled_edited
    ) = generate_constraints(
            resolution,
            random_scale,
            random_points_number,
            control_points_range,
            airfoil_for_editing,
    )

    # plot_2.show()
    print("points_id_constraints_for_editing_all output :", points_id_constraints_for_editing_all)
    print("xs_controlled output :", xs_controlled.shape)
    print("ys_controlled output :", ys_controlled.shape)
    print("ys_controlled_edited output :", ys_controlled_edited.shape)


    # Now we can edit the airfoil curve using the generated constraints
    finetune_iterations = 20
    t_steps = 20
    latent_initialization = "Inverse prior initialization"

    plot_3, text_3, airfoil_after_editing = generate_editing_airfoil_curve(
        resolution,
        airfoil_for_editing,
        finetune_iterations,
        points_id_constraints_for_editing_all,
        xs_controlled,
        ys_controlled,
        ys_controlled_edited,
        t_steps,
        latent_initialization
    )

    # plot_3.show()
    print("Editing airfoil curve generated successfully.")
    print("airfoil_after_editing output :", airfoil_after_editing.shape)
