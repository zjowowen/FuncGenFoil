from typing import Union

import torch


from airfoil_generation.model.probability_path import ConditionalProbabilityPath


class StochasticProcess:
    """
    Overview:
        Class for describing a stochastic process for generative models.
    Interfaces:
        ``__init__``, ``mean``, ``std``, ``velocity``, ``direct_sample``, ``direct_sample_with_noise``, ``velocity_SchrodingerBridge``, ``score_SchrodingerBridge``
    """

    def __init__(
        self, path: ConditionalProbabilityPath, gaussian_process, t_max: float = 1.0
    ) -> None:
        super().__init__()
        self.path = path
        self.gaussian_process = gaussian_process
        self.t_max = t_max

    def mean(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the mean of the state at time t given the initial state x0 and the final state x1.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`torch.Tensor`): The input state at time 0.
            x1 (:obj:`torch.Tensor`): The input state at time 1.
            condition (:obj:`torch.Tensor`): The input condition.
        """

        if x0 is not None and len(x0.shape) > len(t.shape):
            t = t[(...,) + (None,) * (len(x0.shape) - len(t.shape))].expand(x0.shape)
            return x0 * (1 - t) + x1 * t
        else:
            return x0 * (1 - t) + x1 * t

    def std(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor = None,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the standard deviation of the state at time t given the initial state x0 and the final state x1.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`torch.Tensor`): The input state at time 0.
            x1 (:obj:`torch.Tensor`): The input state at time 1.
            condition (:obj:`torch.Tensor`): The input condition.
        """

        if x0 is not None and len(x0.shape) > len(t.shape):
            return self.path.std(t)[
                (...,) + (None,) * (len(x0.shape) - len(t.shape))
            ].expand(x0.shape)
        else:
            return self.path.std(t)

    def velocity(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the velocity of the state at time t given the state x.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`torch.Tensor`): The input state at time 0.
            x1 (:obj:`torch.Tensor`): The input state at time 1.
            condition (:obj:`torch.Tensor`): The input condition.
        """

        return x1 - x0

    def direct_sample(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return the sample of the state at time t given the initial state x0 and the final state x1.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`torch.Tensor`): The input state at time 0.
            x1 (:obj:`torch.Tensor`): The input state at time 1.
            condition (:obj:`torch.Tensor`): The input condition.
        """

        # TODO: make it compatible with TensorDict

        return self.mean(t, x0, x1, condition) + self.std(
            t, x0, x1, condition
        ) * self.gaussian_process.sample_from_prior(
            dims=x1.shape[2:], n_samples=x1.shape[0], n_channels=x1.shape[1]
        )

    def direct_sample_with_noise(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
        condition: torch.Tensor = None,
        noise: torch.Tensor = None,
    ):
        """
        Overview:
            Return the sample of the state at time t given the initial state x0 and the final state x1 with noise.
        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x0 (:obj:`torch.Tensor`): The input state at time 0.
            x1 (:obj:`torch.Tensor`): The input state at time 1.
            condition (:obj:`torch.Tensor`): The input condition.
            noise (:obj:`torch.Tensor`): The input noise.
        """
        return self.mean(t, x0, x1, condition) + self.std(
            t, x0, x1, condition
        ) * noise.to(x0.device)
