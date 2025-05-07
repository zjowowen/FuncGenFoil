from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import ot
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from easydict import EasyDict

from airfoil_generation.model.probability_path import ConditionalProbabilityPath
from airfoil_generation.model.intrinsic_model import DeterministicIntrinsicModel
from airfoil_generation.model.stochastic_process import StochasticProcess
from airfoil_generation.dataset.toy_dataset import MaternGaussianProcess
from airfoil_generation.numerical_solvers import ODESolver
from airfoil_generation.numerical_solvers import get_solver
from airfoil_generation.utils import find_parameters


class FunctionalDeterministicModel(nn.Module):
    """
    Overview:
        A functional flow model.
    Interface:
        "__init__", "get_type", "forward", "sample", "loss"
    """

    def __init__(
        self,
        config: EasyDict,
        model: nn.Module = None,
    ):
        """
        Overview:
            Initialize the functional flow model.
        Arguments:
            - config (EasyDict): configuration for the model
            - model (nn.Module): intrinsic model
        """
        super().__init__()

        self.config = config
        self.device = config.device

        self.model = (
            DeterministicIntrinsicModel(config.model.args) if model is None else model
        )

        self.gaussian_process = MaternGaussianProcess(
            device=self.device, **config.gaussian_process
        )

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

    def get_type(self):
        return "FunctionalDeterministicModel"

    def forward(
        self,
    ):
        pass

    def sample(
        self,
        n_dims: List[int],
        n_channels: int,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: torch.Tensor = None,
        condition: torch.Tensor = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the functional flow model.
        Arguments:
            - n_dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_channels (int): number of independent channels to draw samples for
            - t_span (tensor): time span to sample over
            - batch_size (int, tuple, list): batch size for sampling
            - x_0 (tensor): initial condition for sampling
            - condition (tensor): condition for sampling
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
        Returns:
            - samples (tensor): samples from the functional flow model; tensor (T, B, N, D)
        """
        return self.sample_process(
            n_dims=n_dims,
            n_channels=n_channels,
            t_span=t_span,
            batch_size=batch_size,
            x_0=x_0,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
        )[-1]

    def loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        condition: torch.Tensor = None,
        average: bool = True,
        sum_all_elements: bool = True,
    ):
        """
        Overview:
            Compute the functional flow matching loss.
        Arguments:
            - x0 (tensor): initial condition for sampling
            - x1 (tensor): target condition for sampling
            - condition (tensor): condition for sampling
            - average (bool): whether to average the loss
            - sum_all_elements (bool): whether to sum all elements of the loss
        Returns:
            - loss (tensor): functional flow matching loss
        """

        def get_loss(velocity_value, velocity):
            if average:
                return torch.mean(
                    torch.sum(0.5 * (velocity_value - velocity) ** 2, dim=(1,))
                )
            else:
                if sum_all_elements:
                    return torch.sum(0.5 * (velocity_value - velocity) ** 2, dim=(1,))
                else:
                    return 0.5 * (velocity_value - velocity) ** 2

        batch_size = x0.shape[0]
        t_random = (
            torch.rand(batch_size, device=self.device) * self.stochastic_process.t_max
        )
        x_t = self.stochastic_process.direct_sample(t_random, x0, x1)

        velocity_value = self.model(t_random, x_t, condition=condition)
        velocity = self.stochastic_process.velocity(t_random, x0, x1)
        loss = get_loss(velocity_value, velocity)
        return loss
