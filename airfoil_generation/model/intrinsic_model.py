from typing import Union

import torch
import torch.nn as nn
from easydict import EasyDict
from tensordict import TensorDict

from airfoil_generation.neural_networks import get_module


class IntrinsicModel(nn.Module):
    """
    Overview:
        Intrinsic model of generative model, which is the backbone of many continuous-time generative models.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialize the model.
        Arguments:
            - config (EasyDict): configuration for the model
        """
        # TODO

        super().__init__()

        self.config = config
        assert hasattr(config, "backbone"), "backbone must be specified in config"

        self.model = torch.nn.ModuleDict()
        if hasattr(config, "t_encoder"):
            self.model["t_encoder"] = get_module(config.t_encoder.type)(
                **config.t_encoder.args
            )
        else:
            self.model["t_encoder"] = torch.nn.Identity()
        if hasattr(config, "x_encoder"):
            self.model["x_encoder"] = get_module(config.x_encoder.type)(
                **config.x_encoder.args
            )
        else:
            self.model["x_encoder"] = torch.nn.Identity()
        if hasattr(config, "condition_encoder"):
            self.model["condition_encoder"] = get_module(config.condition_encoder.type)(
                **config.condition_encoder.args
            )
        else:
            self.model["condition_encoder"] = torch.nn.Identity()

        # TODO
        # specific backbone network
        self.model["backbone"] = get_module(config.backbone.type)(
            **config.backbone.args
        )

    def forward(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict],
        condition: Union[torch.Tensor, TensorDict] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the output of the model at time t given the initial state.
        Arguments:
            - t (torch.Tensor): time tensor
            - x (torch.Tensor or TensorDict): initial state tensor
            - condition (torch.Tensor or TensorDict): condition tensor
        """

        if condition is not None:
            t = self.model["t_encoder"](t)
            x = self.model["x_encoder"](x)
            condition = self.model["condition_encoder"](condition)
            output = self.model["backbone"](t, x, condition)
        else:
            t = self.model["t_encoder"](t)
            x = self.model["x_encoder"](x)
            output = self.model["backbone"](t, x)

        return output
