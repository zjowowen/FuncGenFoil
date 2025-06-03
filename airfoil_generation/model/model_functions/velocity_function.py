from typing import Callable, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict

from airfoil_generation.model.diffusion_process import DiffusionProcess


class VelocityFunction:
    """
    Overview:
        Velocity function in diffusion model.
    Interfaces:
        ``__init__``, ``forward``, ``flow_matching_loss``
    """

    def __init__(
        self,
        model_type: str,
        process: DiffusionProcess,
    ):
        """
        Overview:
            Initialize the velocity function.
        Arguments:
            - model_type (:obj:`str`): The type of the model.
            - process (:obj:`DiffusionProcess`): The process.
        """
        self.model_type = model_type
        self.process = process

    def forward(
        self,
        model: Union[Callable, nn.Module],
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return velocity of the model at time t given the initial state.
            .. math::
                v_{\theta}(t, x)
        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The model.
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input state at time t.
            - condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        if self.model_type == "noise_function":
            # TODO: check if this is correct
            return self.process.drift(t, x) + 0.5 * self.process.diffusion_squared(
                t, x
            ) / self.process.std(t, x) * model(t, x, condition)
        elif self.model_type == "score_function":
            # TODO: check if this is correct
            return self.process.drift(t, x) - 0.5 * self.process.diffusion_squared(
                t, x
            ) * model(t, x, condition)
        elif self.model_type == "velocity_function":
            return model(t, x, condition)
        elif self.model_type == "data_prediction_function":
            # TODO: check if this is correct
            D = (
                0.5
                * self.process.diffusion_squared(t, x)
                / self.process.covariance(t, x)
            )
            return (self.process.drift_coefficient(t) + D) - D * self.process.scale(
                t
            ) * model(t, x, condition)
        else:
            raise NotImplementedError(
                "Unknown type of Velocity Function {}".format(type)
            )
