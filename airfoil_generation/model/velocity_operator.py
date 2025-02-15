from typing import Callable, Union

import torch
import torch.nn as nn
from easydict import EasyDict


class VelocityOperator:
    """
    Overview:
        Velocity operator in functional generative model.
    Interfaces:
        ``__init__``, ``forward``, ``flow_matching_loss``
    """

    def __init__(
        self,
        process: object,
    ):
        """
        Overview:
            Initialize the velocity function.
        Arguments:
            - model_type (:obj:`str`): The type of the model.
            - process (:obj:`object`): The process.
        """
        self.process = process

    def forward(
        self,
        model: Union[Callable, nn.Module],
        t: torch.Tensor,
        x: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return velocity of the model at time t given the initial state.

            .. math::
                v_{\theta}(t, x)

        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The model.
            - t (:obj:`torch.Tensor`): The input time.
            - x (:obj:`torch.Tensor`): The input state at time t.
            - condition (:obj:`torch.Tensor`): The input condition.
        """

        return model(t, x, condition)

    def flow_matching_loss(
        self,
        model: Union[Callable, nn.Module],
        x: torch.Tensor,
        condition: torch.Tensor = None,
        gaussian_generator: Callable = None,
        average: bool = True,
    ) -> torch.Tensor:
        """
        Overview:
            Return the flow matching loss.

            .. math::
                \mathcal{L}_{\text{FM}} = \mathbb{E}_{t \sim U(0, T)} \left[ \mathbb{E}_{x_0,x_1} \left[ \left\| v_{\theta}(t, x_t) - v(t, x, \epsilon) \right\|^2 \right] \right]

        Arguments:
            - model (:obj:`Union[Callable, nn.Module]`): The model.
            - x (:obj:`torch.Tensor`): The input state at time t.
            - condition (:obj:`torch.Tensor`): The input condition.
            - gaussian_generator (:obj:`Callable`): The Gaussian generator.
            - average (:obj:`bool`): The average flag.

        Returns:
            - loss (:obj:`torch.Tensor`): The flow matching loss.
        """

        def get_batch_size_and_device(x):
            return x.shape[0], x.device

        def get_loss(velocity_value, velocity):
            if average:
                return torch.mean(
                    torch.sum(0.5 * (velocity_value - velocity) ** 2, dim=(1,))
                )
            else:
                return torch.sum(0.5 * (velocity_value - velocity) ** 2, dim=(1,))

        eps = 1e-5
        batch_size, device = get_batch_size_and_device(x)
        t_random = (
            torch.rand(batch_size, device=device) * (self.process.t_max - eps) + eps
        )
        if gaussian_generator is None:
            noise = torch.randn_like(x).to(device)
        else:
            noise = gaussian_generator(batch_size)
        std = self.process.std(t_random, x)
        x_t = self.process.scale(t_random, x) * x + std * noise
        velocity_value = model(t_random, x_t, condition=condition)
        velocity = self.process.velocity(t_random, x, noise=noise)
        loss = get_loss(velocity_value, velocity)
        return loss
