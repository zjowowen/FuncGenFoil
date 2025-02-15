from typing import Union

import torch
from easydict import EasyDict


class ConditionalProbabilityPath:
    """
    Overview:
        Conditional probability path for general continuous-time normalizing flow.

    """

    def __init__(self, config: EasyDict) -> None:
        """
        Overview:
            Initialize the conditional probability path.
        Arguments:
            - config (EasyDict): configuration for the conditional probability path
        """
        self.config = config
        self.sigma = torch.tensor(config.sigma, device=config.device)

    def std(self, t: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the standard deviation of the conditional probability path at time t.
        Arguments:
            - t (torch.Tensor): time tensor
        """

        return self.sigma
