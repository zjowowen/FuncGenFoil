from typing import Any, Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import treetensor
from easydict import EasyDict
from tensordict import TensorDict

from airfoil_generation.model.diffusion_process import DiffusionProcess
from airfoil_generation.model.intrinsic_model import IntrinsicModel
from airfoil_generation.model.stochastic_process import StochasticProcess
from airfoil_generation.model.probability_path import GaussianConditionalProbabilityPath

from airfoil_generation.model.model_functions.data_prediction_function import (
    DataPredictionFunction,
)
from airfoil_generation.model.model_functions.noise_function import NoiseFunction
from airfoil_generation.model.model_functions.score_function import ScoreFunction
from airfoil_generation.model.model_functions.velocity_function import VelocityFunction

from airfoil_generation.dataset.toy_dataset import get_gaussian_process
from airfoil_generation.numerical_solvers import ODESolver
from airfoil_generation.numerical_solvers import get_solver
from airfoil_generation.utils import find_parameters


class EnergyGuidance(nn.Module):
    """
    Overview:
        Energy Guidance for Energy Conditional Diffusion Model.
    Interfaces:
        ``__init__``, ``forward``, ``calculate_energy_guidance``
    """

    def __init__(self, config: EasyDict):
        """
        Overview:
            Initialization of Energy Guidance.

        Arguments:
            config (:obj:`EasyDict`): The configuration.
        """
        super().__init__()
        self.config = config
        self.model = IntrinsicModel(self.config)

    def forward(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Overview:
            Return output of Energy Guidance.

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
        """

        return self.model(t, x, condition)

    def calculate_energy_guidance(
        self,
        t: torch.Tensor,
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]:
        """
        Overview:
            Calculate the guidance for sampling.

        Arguments:
            t (:obj:`torch.Tensor`): The input time.
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.
            guidance_scale (:obj:`float`): The scale of guidance.
        Returns:
            guidance (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The guidance for sampling.
        """

        # TODO: make it compatible with TensorDict
        with torch.enable_grad():
            x.requires_grad_(True)
            x_t = self.forward(t, x, condition)
            guidance = guidance_scale * torch.autograd.grad(torch.sum(x_t), x)[0]
        return guidance.detach()


class FunctionalDiffusion(nn.Module):
    """
    Overview:
        A functional diffusion model.
    Interface:
        "__init__", "get_type", "forward", "sample", "sample_process", "inverse_sample", "inverse_sample_process", "inverse_sample_with_log_prob", "inverse_sample_process_with_log_prob", "functional_flow_matching_loss"
    """

    def __init__(
        self,
        config: EasyDict,
        model: nn.Module = None,
    ):
        """
        Overview:
            Initialize the functional diffusion model.
        Arguments:
            - config (EasyDict): configuration for the model
            - model (nn.Module): intrinsic model
        """
        super().__init__()

        self.config = config
        self.device = config.device
        self.path = GaussianConditionalProbabilityPath(config.path)
        self.diffusion_process = DiffusionProcess(self.path)
        self.reverse_path = config.get("reverse_path", None)
        if self.reverse_path is not None:
            self.reverse_diffusion_process = DiffusionProcess(self.reverse_path)
        else:
            self.reverse_diffusion_process = None
        self.model = IntrinsicModel(config.model.args) if model is None else model
        self.model_type = config.model.type
        self.score_function_ = ScoreFunction(self.model_type, self.diffusion_process)
        self.velocity_function_ = VelocityFunction(
            self.model_type, self.diffusion_process
        )
        self.noise_function_ = NoiseFunction(self.model_type, self.diffusion_process)
        self.data_prediction_function_ = DataPredictionFunction(
            self.model_type, self.diffusion_process
        )

        self.gaussian_process = get_gaussian_process(
            config.gaussian_process.type, **config.gaussian_process.args
        )
        self.stochastic_process = StochasticProcess(self.path, self.gaussian_process)

        self.energy_guidance = (
            EnergyGuidance(self.config.energy_guidance)
            if hasattr(config, "energy_guidance")
            else None
        )
        self.alpha = 1

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

    def get_type(self):
        return "FunctionalDiffusion"

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
            Sample from the functional diffusion model.
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
            - samples (tensor): samples from the functional diffusion model; tensor (T, B, N, D)
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

    def sample_process(
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
            Sample from the functional diffusion model.
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
            - samples (tensor): samples from the functional diffusion model; tensor (T, B, N, D)
        """
        if t_span is not None:
            t_span = t_span.to(self.device)

        if batch_size is None:
            extra_batch_size = torch.tensor((1,), device=self.device)
        elif isinstance(batch_size, int):
            extra_batch_size = torch.tensor((batch_size,), device=self.device)
        else:
            if (
                isinstance(batch_size, torch.Size)
                or isinstance(batch_size, Tuple)
                or isinstance(batch_size, List)
            ):
                extra_batch_size = torch.tensor(batch_size, device=self.device)
            else:
                assert False, "Invalid batch size"

        if x_0 is not None and condition is not None:
            assert (
                x_0.shape[0] == condition.shape[0]
            ), "The batch size of x_0 and condition must be the same"
            data_batch_size = x_0.shape[0]
        elif x_0 is not None:
            data_batch_size = x_0.shape[0]
        elif condition is not None:
            data_batch_size = condition.shape[0]
        else:
            data_batch_size = 1

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            assert hasattr(
                self, "solver"
            ), "solver must be specified in config or solver_config"
            solver = self.solver

        if x_0 is None:
            x = self.gaussian_process.sample_from_prior(
                dims=n_dims,
                n_samples=torch.prod(extra_batch_size) * data_batch_size,
                n_channels=n_channels,
            )
        else:
            x = x_0
            # x.shape = (B*N, D)

        if isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
            def drift(t, x):
                return self.diffusion_process.reverse_ode(
                    function=self.model,
                    function_type=self.model_type,
                    condition=condition,
                ).drift(t, x)

            if solver.library == "torchdiffeq_adjoint":
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self.model),
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                            adjoint_params=find_parameters(self.model),
                        )
            else:
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                        )
        else:
            raise NotImplementedError("Not implemented")

        if len(extra_batch_size.shape) == 0:
            data = data.reshape(
                -1, extra_batch_size, data_batch_size, n_channels, *n_dims
            )
        else:
            data = data.reshape(
                -1, *extra_batch_size, data_batch_size, n_channels, *n_dims
            )
        # data.shape = (T, B, N, D)

        if batch_size is None:
            if x_0 is None and condition is None:
                data = data.squeeze(1).squeeze(1)
                # data.shape = (T, D)
            else:
                data = data.squeeze(1)
                # data.shape = (T, N, D)
        else:
            if x_0 is None and condition is None:
                data = data.squeeze(1 + len(extra_batch_size))
                # data.shape = (T, B, D)
            else:
                # data.shape = (T, B, N, D)
                pass

        return data

    def sample_with_log_prob(
        self,
        n_dims: List[int] = None,
        n_channels: int = None,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: torch.Tensor = None,
        log_prob_x_0: torch.Tensor = None,
        function_log_prob_x_0: Union[callable, nn.Module] = None,
        condition: torch.Tensor = None,
        with_grad: bool = True,
        solver_config: EasyDict = None,
        using_Hutchinson_trace_estimator: bool = True,
    ) -> Tuple[torch.Tensor]:
        """
        Overview:
            Sample from the model with log probability.
        Arguments:
            - t_span (torch.Tensor): time span to sample over
            - x_0 (torch.Tensor): initial condition
            - log_prob_x_0 (torch.Tensor): log probability of the initial condition
            - function_log_prob_x_0 (Union[callable, nn.Module]): function to compute the log probability of the initial condition
            - condition (torch.Tensor): condition
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
            - using_Hutchinson_trace_estimator (bool): whether to use Hutchinson trace estimator
        Returns:
            - x1 (torch.Tensor): sampled data
        """

        x1, log_likelihood, logp_x1_minus_logp_x0 = self.sample_process_with_log_prob(
            n_dims=n_dims,
            n_channels=n_channels,
            t_span=t_span,
            batch_size=batch_size,
            x_0=x_0,
            log_prob_x_0=log_prob_x_0,
            function_log_prob_x_0=function_log_prob_x_0,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
            using_Hutchinson_trace_estimator=using_Hutchinson_trace_estimator,
        )

        return x1[-1], log_likelihood[-1], logp_x1_minus_logp_x0[-1]

    def sample_process_with_log_prob(
        self,
        n_dims: List[int] = None,
        n_channels: int = None,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: torch.Tensor = None,
        log_prob_x_0: torch.Tensor = None,
        function_log_prob_x_0: Union[callable, nn.Module] = None,
        condition: torch.Tensor = None,
        with_grad: bool = True,
        solver_config: EasyDict = None,
        using_Hutchinson_trace_estimator: bool = True,
    ):
        """
        Overview:
            Sample from the model with log probability.
        Arguments:
            - t_span (torch.Tensor): time span to sample over
            - x_0 (torch.Tensor): initial condition
            - log_prob_x_0 (torch.Tensor): log probability of the initial condition
            - function_log_prob_x_0 (Union[callable, nn.Module]): function to compute the log probability of the initial condition
            - condition (torch.Tensor): condition
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
            - using_Hutchinson_trace_estimator (bool): whether to use Hutchinson trace estimator
        Returns:
            - x1 (torch.Tensor): sampled data
        """

        model_drift = lambda t, x: self.diffusion_process.reverse_ode(
            function=self.model,
            function_type=self.model_type,
            condition=condition,
        ).drift(t, x)
        model_params = find_parameters(self)

        def compute_trace_of_jacobian_general(dx, x):
            # if x is complex, change to real
            if x.dtype == torch.complex64 or x.dtype == torch.complex128:
                x = x.real
            # Assuming x has shape (B, D1, ..., Dn)
            shape = x.shape[1:]  # get the shape of a single element in the batch
            outputs = torch.zeros(
                x.shape[0], device=x.device, dtype=x.dtype
            )  # trace for each batch
            # Iterate through each index in the product of dimensions
            for index in torch.cartesian_prod(*(torch.arange(s) for s in shape)):
                if len(index.shape) > 0:
                    index = tuple(index)
                else:
                    index = (index,)
                grad_outputs = torch.zeros_like(x)
                grad_outputs[(slice(None), *index)] = (
                    1  # set one at the specific index across all batches
                )
                grads = torch.autograd.grad(
                    outputs=dx, inputs=x, grad_outputs=grad_outputs, retain_graph=True
                )[0]
                outputs += grads[(slice(None), *index)]
            return outputs

        def compute_trace_of_jacobian_by_Hutchinson_Skilling(dx, x, eps):
            """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

            fn_eps = torch.sum(dx * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x, create_graph=True)[0]
            outputs = torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
            return outputs

        def composite_drift(t, x):
            # where x is actually x0_and_diff_logp, (x0, diff_logp), which is a tuple containing x and logp_xt_minus_logp_x0
            with torch.set_grad_enabled(True):
                t = t.detach()
                x_t = x[0].detach()
                logp_xt_minus_logp_x0 = x[1]

                x_t.requires_grad = True
                t.requires_grad = True

                dx = model_drift(t, x_t)
                if using_Hutchinson_trace_estimator:
                    noise = torch.randn_like(x_t, device=x_t.device)
                    logp_drift = -compute_trace_of_jacobian_by_Hutchinson_Skilling(
                        dx, x_t, noise
                    )
                else:
                    logp_drift = -compute_trace_of_jacobian_general(dx, x_t)

                return dx, logp_drift

        if batch_size is None:
            extra_batch_size = torch.tensor((1,), device=self.device)
        elif isinstance(batch_size, int):
            extra_batch_size = torch.tensor((batch_size,), device=self.device)
        else:
            if (
                isinstance(batch_size, torch.Size)
                or isinstance(batch_size, Tuple)
                or isinstance(batch_size, List)
            ):
                extra_batch_size = torch.tensor(batch_size, device=self.device)
            else:
                assert False, "Invalid batch size"

        if x_0 is not None and condition is not None:
            assert (
                x_0.shape[0] == condition.shape[0]
            ), "The batch size of x_0 and condition must be the same"
            data_batch_size = x_0.shape[0]
        elif x_0 is not None:
            data_batch_size = x_0.shape[0]
        elif condition is not None:
            data_batch_size = condition.shape[0]
        else:
            data_batch_size = 1

        if x_0 is None:
            x_0 = self.gaussian_process.sample_from_prior(
                dims=n_dims,
                n_samples=torch.prod(extra_batch_size) * data_batch_size,
                n_channels=n_channels,
            )

        x0_and_diff_logp = (x_0, torch.zeros(x_0.shape[0], device=x_0.device))

        if t_span is None:
            t_span = torch.linspace(0.0, 1.0, 1000).to(x.device)
        else:
            t_span = t_span.to(x_0.device)

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            # solver = ODESolver(library="torchdiffeq_adjoint")
            solver = ODESolver(library="torchdiffeq")

        if with_grad:
            x1_and_logpx1 = solver.integrate(
                drift=composite_drift,
                x0=x0_and_diff_logp,
                t_span=t_span,
                # adjoint_params=model_params,
            )
        else:
            # TODO: check if it is correct
            with torch.no_grad():
                x1_and_logpx1 = solver.integrate(
                    drift=composite_drift,
                    x0=x0_and_diff_logp,
                    t_span=t_span,
                    # adjoint_params=model_params,
                )

        logp_x1_minus_logp_x0 = x1_and_logpx1[1]
        x1 = x1_and_logpx1[0]

        if log_prob_x_0 is not None:
            log_likelihood = log_prob_x_0 + logp_x1_minus_logp_x0
        elif function_log_prob_x_0 is not None:
            log_prob_x_0 = function_log_prob_x_0(x0)
            log_likelihood = log_prob_x_0 + logp_x1_minus_logp_x0
        elif x_0 is not None:
            # TODO: check if it is correct

            log_prob_x_0 = self.gaussian_process.prior_likelihood(x_0)
            log_likelihood = log_prob_x_0 + logp_x1_minus_logp_x0
        else:
            log_likelihood = torch.zeros_like(
                logp_x1_minus_logp_x0, device=logp_x1_minus_logp_x0.device
            )

        return x1, log_likelihood, logp_x1_minus_logp_x0

    def inverse_sample(
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
            Inverse sample from the model.
        Arguments:
            - n_dims (List[int]): list of dimensions of inputs
            - n_channels (int): number of independent channels to draw samples for
            - t_span (torch.Tensor): time span to sample over
            - batch_size (Union[torch.Size, int, Tuple[int], List[int]]): batch size for sampling
            - x_0 (torch.Tensor): initial condition
            - condition (torch.Tensor): condition
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
        Returns:
            - data (torch.Tensor): sampled data
        """
        return self.inverse_sample_process(
            n_dims=n_dims,
            n_channels=n_channels,
            t_span=t_span,
            batch_size=batch_size,
            x_0=x_0,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
        )[-1]

    def inverse_sample_process(
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
            Inverse sample from the model.
        Arguments:
            - n_dims (List[int]): list of dimensions of inputs
            - n_channels (int): number of independent channels to draw samples for
            - t_span (torch.Tensor): time span to sample over
            - batch_size (Union[torch.Size, int, Tuple[int], List[int]]): batch size for sampling
            - x_0 (torch.Tensor): initial condition
            - condition (torch.Tensor): condition
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
        Returns:
            - data (torch.Tensor): sampled data
        """
        if t_span is not None:
            t_span = t_span.to(self.device)

        if batch_size is None:
            extra_batch_size = torch.tensor((1,), device=self.device)
        elif isinstance(batch_size, int):
            extra_batch_size = torch.tensor((batch_size,), device=self.device)
        else:
            if (
                isinstance(batch_size, torch.Size)
                or isinstance(batch_size, Tuple)
                or isinstance(batch_size, List)
            ):
                extra_batch_size = torch.tensor(batch_size, device=self.device)
            else:
                assert False, "Invalid batch size"

        if x_0 is not None and condition is not None:
            assert (
                x_0.shape[0] == condition.shape[0]
            ), "The batch size of x_0 and condition must be the same"
            data_batch_size = x_0.shape[0]
        elif x_0 is not None:
            data_batch_size = x_0.shape[0]
        elif condition is not None:
            data_batch_size = condition.shape[0]
        else:
            data_batch_size = 1

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            assert hasattr(
                self, "solver"
            ), "solver must be specified in config or solver_config"
            solver = self.solver

        if x_0 is None:
            x = self.gaussian_process.sample_from_prior(
                dims=n_dims,
                n_samples=torch.prod(extra_batch_size) * data_batch_size,
                n_channels=n_channels,
            )
        else:
            x = x_0
            # x.shape = (B*N, D)

        if isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
            def reverse_drift(t, x):
                return self.diffusion_process.forward_ode(
                    function=self.model,
                    function_type=self.model_type,
                    condition=condition,
                ).drift(t, x)

            if solver.library == "torchdiffeq_adjoint":
                if with_grad:
                    data = solver.integrate(
                        drift=reverse_drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self),
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=reverse_drift,
                            x0=x,
                            t_span=t_span,
                            adjoint_params=find_parameters(self),
                        )
            else:
                if with_grad:
                    data = solver.integrate(
                        drift=reverse_drift,
                        x0=x,
                        t_span=t_span,
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=reverse_drift,
                            x0=x,
                            t_span=t_span,
                        )
        else:
            raise NotImplementedError("Not implemented")

        if len(extra_batch_size.shape) == 0:
            data = data.reshape(
                -1, extra_batch_size, data_batch_size, n_channels, *n_dims
            )
        else:
            data = data.reshape(
                -1, *extra_batch_size, data_batch_size, n_channels, *n_dims
            )
        # data.shape = (T, B, N, D)

        if batch_size is None:
            if x_0 is None and condition is None:
                data = data.squeeze(1).squeeze(1)
                # data.shape = (T, D)
            else:
                data = data.squeeze(1)
                # data.shape = (T, N, D)
        else:
            if x_0 is None and condition is None:
                data = data.squeeze(1 + len(extra_batch_size))
                # data.shape = (T, B, D)
            else:
                # data.shape = (T, B, N, D)
                pass

        return data

    def inverse_sample_with_log_prob(
        self,
        t_span: torch.Tensor = None,
        x_0: torch.Tensor = None,
        log_prob_x_0: torch.Tensor = None,
        function_log_prob_x_0: Union[callable, nn.Module] = None,
        condition: torch.Tensor = None,
        with_grad: bool = True,
        solver_config: EasyDict = None,
        using_Hutchinson_trace_estimator: bool = True,
    ) -> Tuple[torch.Tensor]:
        """
        Overview:
            Inverse sample from the model with log probability.
        Arguments:
            - t_span (torch.Tensor): time span to sample over
            - x_0 (torch.Tensor): initial condition
            - log_prob_x_0 (torch.Tensor): log probability of the initial condition
            - function_log_prob_x_0 (Union[callable, nn.Module]): function to compute the log probability of the initial condition
            - condition (torch.Tensor): condition
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
            - using_Hutchinson_trace_estimator (bool): whether to use Hutchinson trace estimator
        Returns:
            - x1 (torch.Tensor): sampled data
            - log_likelihood (torch.Tensor): log likelihood of the sampled data
            - logp_x1_minus_logp_x0 (torch.Tensor): log probability difference between the sampled data and the initial condition
        """

        (
            x1,
            log_likelihood,
            logp_x1_minus_logp_x0,
        ) = self.inverse_sample_process_with_log_prob(
            t_span=t_span,
            x_0=x_0,
            log_prob_x_0=log_prob_x_0,
            function_log_prob_x_0=function_log_prob_x_0,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
            using_Hutchinson_trace_estimator=using_Hutchinson_trace_estimator,
        )

        return x1[-1], log_likelihood[-1], logp_x1_minus_logp_x0[-1]

    def inverse_sample_process_with_log_prob(
        self,
        t_span: torch.Tensor = None,
        x_0: torch.Tensor = None,
        log_prob_x_0: torch.Tensor = None,
        function_log_prob_x_0: Union[callable, nn.Module] = None,
        condition: torch.Tensor = None,
        with_grad: bool = True,
        solver_config: EasyDict = None,
        using_Hutchinson_trace_estimator: bool = True,
    ):
        """
        Overview:
            Inverse sample from the model with log probability.
        Arguments:
            - t_span (torch.Tensor): time span to sample over
            - x_0 (torch.Tensor): initial condition
            - log_prob_x_0 (torch.Tensor): log probability of the initial condition
            - function_log_prob_x_0 (Union[callable, nn.Module]): function to compute the log probability of the initial condition
            - condition (torch.Tensor): condition
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
            - using_Hutchinson_trace_estimator (bool): whether to use Hutchinson trace estimator
        Returns:
            - x1 (torch.Tensor): sampled data
        """

        def reverse_drift(t, x):
            return self.diffusion_process.forward_ode(
                function=self.model,
                function_type=self.model_type,
                condition=condition,
            ).drift(t, x)

        model_drift = lambda t, x: reverse_drift(t, x)
        model_params = find_parameters(self)

        def compute_trace_of_jacobian_general(dx, x):
            # if x is complex, change to real
            if x.dtype == torch.complex64 or x.dtype == torch.complex128:
                x = x.real
            # Assuming x has shape (B, D1, ..., Dn)
            shape = x.shape[1:]  # get the shape of a single element in the batch
            outputs = torch.zeros(
                x.shape[0], device=x.device, dtype=x.dtype
            )  # trace for each batch
            # Iterate through each index in the product of dimensions
            for index in torch.cartesian_prod(*(torch.arange(s) for s in shape)):
                if len(index.shape) > 0:
                    index = tuple(index)
                else:
                    index = (index,)
                grad_outputs = torch.zeros_like(x)
                grad_outputs[(slice(None), *index)] = (
                    1  # set one at the specific index across all batches
                )
                grads = torch.autograd.grad(
                    outputs=dx, inputs=x, grad_outputs=grad_outputs, retain_graph=True
                )[0]
                outputs += grads[(slice(None), *index)]
            return outputs

        def compute_trace_of_jacobian_by_Hutchinson_Skilling(dx, x, eps):
            """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

            fn_eps = torch.sum(dx * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x, create_graph=True)[0]
            outputs = torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
            return outputs

        def composite_drift(t, x):
            # where x is actually x0_and_diff_logp, (x0, diff_logp), which is a tuple containing x and logp_xt_minus_logp_x0
            with torch.set_grad_enabled(True):
                t = t.detach()
                x_t = x[0].detach()
                logp_xt_minus_logp_x0 = x[1]

                x_t.requires_grad = True
                t.requires_grad = True

                dx = model_drift(t, x_t)
                if using_Hutchinson_trace_estimator:
                    noise = torch.randn_like(x_t, device=x_t.device)
                    logp_drift = -compute_trace_of_jacobian_by_Hutchinson_Skilling(
                        dx, x_t, noise
                    )
                    # logp_drift = - divergence_approx(dx, x_t, noise)
                else:
                    logp_drift = -compute_trace_of_jacobian_general(dx, x_t)

                return dx, logp_drift

        x0_and_diff_logp = (x_0, torch.zeros(x_0.shape[0], device=x_0.device))

        if t_span is None:
            t_span = torch.linspace(0.0, 1.0, 1000).to(x.device)
        else:
            t_span = t_span.to(x_0.device)

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            # solver = ODESolver(library="torchdiffeq_adjoint")
            solver = ODESolver(library="torchdiffeq")

        if with_grad:
            x1_and_logpx1 = solver.integrate(
                drift=composite_drift,
                x0=x0_and_diff_logp,
                t_span=t_span,
                # adjoint_params=model_params,
            )
        else:
            # TODO: check if it is correct
            with torch.no_grad():
                x1_and_logpx1 = solver.integrate(
                    drift=composite_drift,
                    x0=x0_and_diff_logp,
                    t_span=t_span,
                    # adjoint_params=model_params,
                )

        logp_x1_minus_logp_x0 = x1_and_logpx1[1]
        x1 = x1_and_logpx1[0]

        if log_prob_x_0 is not None:
            log_likelihood = log_prob_x_0 + logp_x1_minus_logp_x0
        elif function_log_prob_x_0 is not None:
            log_prob_x_0 = function_log_prob_x_0(x0)
            log_likelihood = log_prob_x_0 + logp_x1_minus_logp_x0
        elif x_0 is not None:
            # TODO: check if it is correct

            log_prob_x_0 = self.gaussian_process.prior_likelihood(x_0)
            # x0_1d = x_0.reshape(x_0.shape[0], -1)
            # log_prob_x_0 = Independent(
            #     Normal(
            #         loc=torch.zeros_like(x0_1d, device=x0_1d.device),
            #         scale=torch.ones_like(x0_1d, device=x0_1d.device),
            #     ),
            #     1,
            # ).log_prob(x0_1d)
            log_likelihood = log_prob_x_0 + logp_x1_minus_logp_x0
        else:
            log_likelihood = torch.zeros_like(
                logp_x1_minus_logp_x0, device=logp_x1_minus_logp_x0.device
            )

        return x1, log_likelihood, logp_x1_minus_logp_x0

    def functional_flow_matching_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor = None,
        condition: torch.Tensor = None,
        average: bool = True,
        sum_all_elements: bool = True,
    ):
        """
        Overview:
            Compute the functional flow matching loss.
        Arguments:
            - x0 (torch.Tensor): initial condition
            - x1 (torch.Tensor): final condition
            - condition (torch.Tensor): condition
            - average (bool): whether to average the loss
            - sum_all_elements (bool): whether to sum all elements
        Returns:
            - loss (torch.Tensor): functional flow matching loss
        """

        def get_batch_size_and_device(x):
            if isinstance(x, torch.Tensor):
                return x.shape[0], x.device
            elif isinstance(x, TensorDict):
                return x.shape, x.device
            elif isinstance(x, treetensor.torch.Tensor):
                return list(x.values())[0].shape[0], list(x.values())[0].device
            else:
                raise NotImplementedError("Unknown type of x {}".format(type))

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

        if self.model_type == "noise_function":
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x0)
            t_random = (
                torch.rand(batch_size, device=device)
                * (self.diffusion_process.t_max - eps)
                + eps
            )
            if x1 is None:
                noise = self.gaussian_process.sample_from_prior(
                    dims=x0.shape[2:],
                    n_samples=x0.shape[0],
                    n_channels=x0.shape[1],
                )
            else:
                noise = x1
            std = self.diffusion_process.std(t_random, x0)
            x_t = self.diffusion_process.scale(t_random, x0) * x0 + std * noise
            velocity_value = (
                self.diffusion_process.drift(t_random, x_t)
                + 0.5
                * self.diffusion_process.diffusion_squared(t_random, x_t)
                * self.model(t_random, x_t, condition=condition)
                / std
            )
            velocity = self.diffusion_process.velocity(t_random, x0, noise=noise)
            loss = get_loss(velocity_value, velocity)
            return loss
        elif self.model_type == "score_function":
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x0)
            t_random = (
                torch.rand(batch_size, device=device)
                * (self.diffusion_process.t_max - eps)
                + eps
            )
            if x1 is None:
                noise = self.gaussian_process.sample_from_prior(
                    dims=x0.shape[2:],
                    n_samples=x0.shape[0],
                    n_channels=x0.shape[1],
                )
            else:
                noise = x1
            std = self.diffusion_process.std(t_random, x0)
            x_t = self.diffusion_process.scale(t_random, x0) * x0 + std * noise
            velocity_value = self.diffusion_process.drift(
                t_random, x_t
            ) - 0.5 * self.diffusion_process.diffusion_squared(
                t_random, x_t
            ) * self.model(
                t_random, x_t, condition=condition
            )
            velocity = self.diffusion_process.velocity(t_random, x0, noise=noise)
            loss = get_loss(velocity_value, velocity)
            return loss
        elif self.model_type == "velocity_function":
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x0)
            t_random = (
                torch.rand(batch_size, device=device)
                * (self.diffusion_process.t_max - eps)
                + eps
            )
            if x1 is None:
                noise = self.gaussian_process.sample_from_prior(
                    dims=x0.shape[2:],
                    n_samples=x0.shape[0],
                    n_channels=x0.shape[1],
                )
            else:
                noise = x1
            std = self.diffusion_process.std(t_random, x0)
            x_t = self.diffusion_process.scale(t_random, x0) * x0 + std * noise
            velocity_value = self.model(t_random, x_t, condition=condition)
            velocity = self.diffusion_process.velocity(t_random, x0, noise=noise)
            loss = get_loss(velocity_value, velocity)
            return loss
        elif self.model_type == "data_prediction_function":
            # TODO: check if this is correct
            eps = 1e-5
            batch_size, device = get_batch_size_and_device(x0)
            t_random = (
                torch.rand(batch_size, device=device)
                * (self.diffusion_process.t_max - eps)
                + eps
            )
            if x1 is None:
                noise = self.gaussian_process.sample_from_prior(
                    dims=x0.shape[2:],
                    n_samples=x0.shape[0],
                    n_channels=x0.shape[1],
                )
            else:
                noise = x1
            std = self.diffusion_process.std(t_random, x0)
            x_t = self.diffusion_process.scale(t_random, x0) * x0 + std * noise
            D = (
                0.5
                * self.diffusion_process.diffusion_squared(t_random, x0)
                / self.diffusion_process.covariance(t_random, x0)
            )
            velocity_value = (
                self.diffusion_process.drift_coefficient(t_random, x_t) + D
            ) * x_t - D * self.diffusion_process.scale(t_random, x_t) * self.model(
                t_random, x_t, condition=condition
            )
            velocity = self.diffusion_process.velocity(t_random, x0, noise=noise)
            loss = get_loss(velocity_value, velocity)
            return loss
        else:
            raise NotImplementedError(
                "Unknown type of velocity function {}".format(type)
            )

    def energy_guidance_loss(
        self,
        energy_model: Union[torch.nn.Module, Callable],
        x: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor],
        condition: Union[torch.Tensor, TensorDict, treetensor.torch.Tensor] = None,
    ):
        """
        Overview:
            The loss function for training Energy Guidance, CEP guidance method, as proposed in the paper \
            "Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning"

        Arguments:
            energy_model (:obj:`Union[torch.nn.Module, Callable]`): The energy model to compute the energy of the input.
            x (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input.
            condition (:obj:`Union[torch.Tensor, TensorDict, treetensor.torch.Tensor]`): The input condition.        
        """
        eps = 1e-4
        t_random = torch.rand((x.shape[0],), device=self.device) * (1.0 - eps) + eps
        t_random = torch.stack([t_random] * x.shape[1], dim=1)
        if condition is not None:
            condition_repeat = torch.stack([condition] * x.shape[1], axis=1)
            condition_repeat_reshape = condition_repeat.reshape(
                condition_repeat.shape[0] * condition_repeat.shape[1],
                *condition_repeat.shape[2:]
            )
            x_reshape = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
            energy = energy_model(x_reshape, condition_repeat_reshape).detach()
            energy = energy.reshape(x.shape[0], x.shape[1]).squeeze(dim=-1)
        else:
            x_reshape = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
            energy = energy_model(x_reshape).detach()
            energy = energy.reshape(x.shape[0], x.shape[1]).squeeze(dim=-1)
        x_t = self.diffusion_process.direct_sample(t_random, x, condition)
        if condition is not None:
            condition_repeat = torch.stack([condition] * x_t.shape[1], axis=1)
            condition_repeat_reshape_new = condition_repeat.reshape(
                condition_repeat.shape[0] * condition_repeat.shape[1],
                *condition_repeat.shape[2:]
            )
            x_t_reshape = x_t.reshape(x_t.shape[0] * x_t.shape[1], *x_t.shape[2:])
            t_random_reshape = t_random.reshape(t_random.shape[0] * t_random.shape[1])
            xt_energy_guidance = self.energy_guidance(
                t_random_reshape, x_t_reshape, condition_repeat_reshape_new
            )
            xt_energy_guidance = xt_energy_guidance.reshape(
                x_t.shape[0], x_t.shape[1]
            ).squeeze(dim=-1)
        else:
            # xt_energy_guidance = self.energy_guidance(t_random, x_t).squeeze(dim=-1)
            x_t_reshape = x_t.reshape(x_t.shape[0] * x_t.shape[1], *x_t.shape[2:])
            t_random_reshape = t_random.reshape(t_random.shape[0] * t_random.shape[1])
            xt_energy_guidance = self.energy_guidance(t_random_reshape, x_t_reshape)
            xt_energy_guidance = xt_energy_guidance.reshape(
                x_t.shape[0], x_t.shape[1]
            ).squeeze(dim=-1)
        log_xt_relative_energy = nn.LogSoftmax(dim=1)(xt_energy_guidance)
        x0_relative_energy = nn.Softmax(dim=1)(energy * self.alpha)
        loss = -torch.mean(
            torch.sum(x0_relative_energy * log_xt_relative_energy, axis=-1)
        )
        return loss

    def sample_with_energy_guidance(
        self,
        n_dims: List[int],
        n_channels: int,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: torch.Tensor = None,
        condition: torch.Tensor = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
        guidance_scale: float = 1.0,
    ):
        """
        Overview:
            Sample from the functional diffusion model.
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
            - samples (tensor): samples from the functional diffusion model; tensor (T, B, N, D)
        """
        return self.sample_process_with_energy_guidance(
            n_dims=n_dims,
            n_channels=n_channels,
            t_span=t_span,
            batch_size=batch_size,
            x_0=x_0,
            condition=condition,
            with_grad=with_grad,
            solver_config=solver_config,
            guidance_scale=guidance_scale,
        )[-1]

    def sample_process_with_energy_guidance(
        self,
        n_dims: List[int],
        n_channels: int,
        t_span: torch.Tensor = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: torch.Tensor = None,
        condition: torch.Tensor = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
        guidance_scale: float = 1.0,
    ):
        """
        Overview:
            Sample from the functional diffusion model.
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
            - samples (tensor): samples from the functional diffusion model; tensor (T, B, N, D)
        """
        if t_span is not None:
            t_span = t_span.to(self.device)

        if batch_size is None:
            extra_batch_size = torch.tensor((1,), device=self.device)
        elif isinstance(batch_size, int):
            extra_batch_size = torch.tensor((batch_size,), device=self.device)
        else:
            if (
                isinstance(batch_size, torch.Size)
                or isinstance(batch_size, Tuple)
                or isinstance(batch_size, List)
            ):
                extra_batch_size = torch.tensor(batch_size, device=self.device)
            else:
                assert False, "Invalid batch size"

        if x_0 is not None and condition is not None:
            assert (
                x_0.shape[0] == condition.shape[0]
            ), "The batch size of x_0 and condition must be the same"
            data_batch_size = x_0.shape[0]
        elif x_0 is not None:
            data_batch_size = x_0.shape[0]
        elif condition is not None:
            data_batch_size = condition.shape[0]
        else:
            data_batch_size = 1

        if solver_config is not None:
            solver = get_solver(solver_config.type)(**solver_config.args)
        else:
            assert hasattr(
                self, "solver"
            ), "solver must be specified in config or solver_config"
            solver = self.solver

        if x_0 is None:
            x = self.gaussian_process.sample_from_prior(
                dims=n_dims,
                n_samples=torch.prod(extra_batch_size) * data_batch_size,
                n_channels=n_channels,
            )
        else:
            x = x_0
            # x.shape = (B*N, D)

        if isinstance(solver, ODESolver):
            # TODO: make it compatible with TensorDict
            def drift(t, x):
                return self.diffusion_process.reverse_ode_with_energy_guidance(
                    function=self.model,
                    function_type=self.model_type,
                    energy_guidance_function=lambda t, x: self.energy_guidance.calculate_energy_guidance(
                        t, x, condition=condition, guidance_scale=guidance_scale
                    ),
                    # condition=condition,
                ).drift(t, x)

            if solver.library == "torchdiffeq_adjoint":
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self.model),
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                            adjoint_params=find_parameters(self.model),
                        )
            else:
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                        )
        else:
            raise NotImplementedError("Not implemented")

        if len(extra_batch_size.shape) == 0:
            data = data.reshape(
                -1, extra_batch_size, data_batch_size, n_channels, *n_dims
            )
        else:
            data = data.reshape(
                -1, *extra_batch_size, data_batch_size, n_channels, *n_dims
            )
        # data.shape = (T, B, N, D)

        if batch_size is None:
            if x_0 is None and condition is None:
                data = data.squeeze(1).squeeze(1)
                # data.shape = (T, D)
            else:
                data = data.squeeze(1)
                # data.shape = (T, N, D)
        else:
            if x_0 is None and condition is None:
                data = data.squeeze(1 + len(extra_batch_size))
                # data.shape = (T, B, D)
            else:
                # data.shape = (T, B, N, D)
                pass

        return data
