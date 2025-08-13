from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import ot
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal
from easydict import EasyDict

from airfoil_generation.model.probability_path import ConditionalProbabilityPath
from airfoil_generation.model.intrinsic_model import IntrinsicModel
from airfoil_generation.model.stochastic_process import StochasticProcess
from airfoil_generation.dataset.toy_dataset import get_gaussian_process
from airfoil_generation.numerical_solvers import ODESolver
from airfoil_generation.numerical_solvers import get_solver
from airfoil_generation.utils import find_parameters


class AsyncTemporalScheduler:
    """
    Overview:
        Asynchronous temporal scheduler for the model. The world-clock time is from [0, 1].
        This class will be used to schedule the sampling of the model. For each point, there is a local time in [0, 1].
    """

    def __init__(
        self,
        t_span: Tuple[float, float] = (0.0, 1.0),
        device: Union[str, torch.device] = "cpu",
        mode: str = "linear",
    ):
        """
        Overview:
            Initialize the scheduler.
        Arguments:
            - t_span (Tuple[float, float]): time span to sample over
            - device (Union[str, torch.device]): device to use
            - mode (str): mode of the scheduler, currently only "linear" is supported
        """
        self.t_span = t_span
        self.device = device
        self.mode = mode
        self.t0 = None
        self.t1 = None


    def set_initial_time(self, t: torch.Tensor):
        """
        Overview:
            Set the initial time for the scheduler for every point.
        Arguments:
            - t (torch.Tensor): initial time
        """
        if isinstance(t, torch.Tensor):
            self.t0 = t.to(self.device)
        else:
            self.t0 = torch.tensor(t, device=self.device)
        
        # Set t1 to be the end of the time span for each point
        if self.t0.dim() == 0:
            # Scalar case
            self.t1 = torch.tensor(self.t_span[1], device=self.device)
        else:
            # Tensor case - each point gets the same end time
            self.t1 = torch.full_like(self.t0, self.t_span[1], device=self.device)

    def get_local_time(self, t: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Get the local time for each point. The local time will be  from self.t0 to self.t1 according to the world-clock time `t` and the scheduler mode.
        Arguments:
            - t (torch.Tensor): world-clock time
        Returns:
            - local_time (torch.Tensor): local time for each point
        """
        if self.t0 is None or self.t1 is None:
            raise ValueError("Initial time must be set before getting local time. Call set_initial_time() first.")
        
        # Ensure t is on the correct device
        if isinstance(t, torch.Tensor):
            t = t.to(self.device)
        else:
            t = torch.tensor(t, device=self.device)
        
        if self.mode == "linear":
            # Linear interpolation from t0 to t1 based on world-clock time t
            # t should be in [0, 1] (world-clock time)
            # local_time = t0 + t * (t1 - t0)
            
            # Handle broadcasting for different tensor shapes
            if self.t0.dim() == 0 and t.dim() == 0:
                # Both scalars
                local_time = self.t0 + t * (self.t1 - self.t0)
            elif self.t0.dim() > 0 and t.dim() == 0:
                # t0 is tensor, t is scalar
                local_time = self.t0 + t * (self.t1 - self.t0)
            elif self.t0.dim() == 0 and t.dim() > 0:
                # t0 is scalar, t is tensor
                local_time = self.t0 + t * (self.t1 - self.t0)
            else:
                # Both are tensors - broadcast appropriately
                # t is of shape [B]
                # self.t0 and self.t1 are tensors of shape [B, D1, ..., Dn]
                local_time = self.t0 + torch.einsum(
                    'b,b...->b...', t, (self.t1 - self.t0)
                )
            
            return local_time
        else:
            raise NotImplementedError(f"Scheduler mode '{self.mode}' is not implemented. Only 'linear' mode is supported.")

    def get_world_time(self, local_time: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Get the world-clock time from local time. Inverse operation of get_local_time.
        Arguments:
            - local_time (torch.Tensor): local time for each point
        Returns:
            - world_time (torch.Tensor): world-clock time
        """
        if self.t0 is None or self.t1 is None:
            raise ValueError("Initial time must be set before getting world time. Call set_initial_time() first.")
        
        # Ensure local_time is on the correct device
        if isinstance(local_time, torch.Tensor):
            local_time = local_time.to(self.device)
        else:
            local_time = torch.tensor(local_time, device=self.device)
        
        if self.mode == "linear":
            # Inverse of linear interpolation: t = (local_time - t0) / (t1 - t0)
            world_time = (local_time - self.t0) / (self.t1 - self.t0)
            return world_time
        else:
            raise NotImplementedError(f"Scheduler mode '{self.mode}' is not implemented. Only 'linear' mode is supported.")

    def reset(self):
        """
        Overview:
            Reset the scheduler by clearing the initial and final times.
        """
        self.t0 = None
        self.t1 = None

    def is_initialized(self) -> bool:
        """
        Overview:
            Check if the scheduler has been initialized with initial times.
        Returns:
            - bool: True if initialized, False otherwise
        """
        return self.t0 is not None and self.t1 is not None

    def get_time_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Overview:
            Get the current time range (t0, t1) for each point.
        Returns:
            - Tuple[torch.Tensor, torch.Tensor]: (t0, t1) tensors
        """
        if not self.is_initialized():
            raise ValueError("Scheduler is not initialized. Call set_initial_time() first.")
        return self.t0, self.t1




class AsyncTemporalOptimalTransportFunctionalFlow(nn.Module):
    """
    Overview:
        Async Temporal Optimal transport functional flow model.
    Interfaces:
        "__init__", "forward", "sample", "sample_process", "inverse_sample", "inverse_sample_process",
        "inverse_sample_with_log_prob", "inverse_sample_process_with_log_prob", "functional_flow_matching_loss",
        "optimal_transport_functional_flow_matching_loss"
    """

    def __init__(
        self,
        config: EasyDict,
        model: nn.Module = None,
    ):
        """
        Overview:
            Initialize the model.
        Arguments:
            - config (EasyDict): configuration for the model
            - model (nn.Module): intrinsic model
        """
        super().__init__()

        self.config = config
        self.device = config.device
        self.path = ConditionalProbabilityPath(config.path)
        self.model = IntrinsicModel(config.model.args) if model is None else model

        self.async_temporal_scheduler = AsyncTemporalScheduler(device=self.device)

        self.gaussian_process = get_gaussian_process(
            config.gaussian_process.type, **config.gaussian_process.args
        )

        self.temporal_gaussian_process = get_gaussian_process(
            config.temporal_gaussian_process.type, **config.temporal_gaussian_process.args
        )

        self.stochastic_process = StochasticProcess(self.path, self.gaussian_process)

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

    def get_type(self):
        return "AsyncTemporalOptimalTransportFunctionalFlow"

    def forward(
        self,
    ):
        pass

    def sample(
        self,
        n_dims: List[int],
        n_channels: int,
        t_span: torch.Tensor = None,
        async_temporal_scheduler: AsyncTemporalScheduler = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: torch.Tensor = None,
        condition: torch.Tensor = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the model.
        Arguments:
            - n_dims (List[int]): list of dimensions of inputs
            - n_channels (int): number of independent channels to draw samples for
            - t_span (torch.Tensor): time span to sample over
            - async_temporal_scheduler (AsyncTemporalScheduler): scheduler for asynchronous temporal sampling
            - batch_size (Union[torch.Size, int, Tuple[int], List[int]]): batch size for sampling
            - x_0 (torch.Tensor): initial condition
            - condition (torch.Tensor): condition
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
        Returns:
            - data (torch.Tensor): sampled data
        """
        return self.sample_process(
            n_dims=n_dims,
            n_channels=n_channels,
            t_span=t_span,
            async_temporal_scheduler=async_temporal_scheduler,
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
        async_temporal_scheduler: AsyncTemporalScheduler = None,
        batch_size: Union[torch.Size, int, Tuple[int], List[int]] = None,
        x_0: torch.Tensor = None,
        condition: torch.Tensor = None,
        with_grad: bool = False,
        solver_config: EasyDict = None,
    ):
        """
        Overview:
            Sample from the model.
        Arguments:
            - n_dims (List[int]): list of dimensions of inputs
            - n_channels (int): number of independent channels to draw samples for
            - t_span (torch.Tensor): time span to sample over
            - async_temporal_scheduler (AsyncTemporalScheduler): scheduler for asynchronous temporal sampling
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

        if async_temporal_scheduler is None:
            async_temporal_scheduler = self.async_temporal_scheduler


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
                t = async_temporal_scheduler.get_local_time(t)
                velocity = self.model(t=t, x=x, condition=condition)
                relative_velocity = async_temporal_scheduler.t1 - async_temporal_scheduler.t0
                return velocity * relative_velocity

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

        model_drift = lambda t, x: self.model(t, x, condition)
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
                reverse_t = t_span.max() - t + t_span.min()
                return -self.model(t=reverse_t, x=x, condition=condition)

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
            reverse_t = t_span.max() - t + t_span.min()
            return -self.model(t=reverse_t, x=x, condition=condition)

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
        x1: torch.Tensor,
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

        t_random = self.temporal_gaussian_process.sample_from_prior(
            dims=[x0.shape[2]],
            n_samples=batch_size,
            n_channels=1,
        ) * 0.5 + 0.5
        t_random_origin = t_random.clone()

        # clip t_random to [0, 1]
        t_random = torch.clamp(t_random, 0.0, 1.0).squeeze(-1).to(self.device) * self.stochastic_process.t_max

        # generate a random ratio from 0 to 1
        shrink_ratio_random = torch.rand(batch_size, 1, 1, device=self.device)

        t_random = 1 - (1 - shrink_ratio_random) * (1 - t_random)

        # plot t_random and t_random_origin, which is of shape (batch_size, 1, 257), plot the first item, which is of shape (257,)
        if True:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            for i in range(batch_size):
                plt.plot(t_random_origin[i,0].cpu().numpy(), label="t_random_origin")
                plt.plot(t_random[i,0].cpu().numpy(), label="t_random")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.title("t_random and t_random_origin")
            plt.legend()
            plt.grid()
            plt.show()

        x_t = self.stochastic_process.direct_sample(t_random, x0, x1)

        velocity_value = self.model(t_random, x_t, condition=condition)
        velocity = self.stochastic_process.velocity(t_random, x0, x1)
        loss = get_loss(velocity_value, velocity)
        return loss

    def optimal_transport_functional_flow_matching_loss(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        condition: torch.Tensor = None,
        average: bool = True,
        sum_all_elements: bool = True,
    ):
        """
        Overview:
            Compute the optimal transport functional flow matching loss.
        Arguments:
            - x0 (torch.Tensor): initial condition
            - x1 (torch.Tensor): final condition
            - condition (torch.Tensor): condition
            - average (bool): whether to average the loss
            - sum_all_elements (bool): whether to sum all elements
        Returns:
            - loss (torch.Tensor): optimal transport functional flow matching loss
        """

        a = ot.unif(x0.shape[0])
        b = ot.unif(x1.shape[0])
        # TODO: make it compatible with TensorDict and treetensor.torch.Tensor
        if x0.dim() > 2:
            x0_ = x0.reshape(x0.shape[0], -1)
        else:
            x0_ = x0
        if x1.dim() > 2:
            x1_ = x1.reshape(x1.shape[0], -1)
        else:
            x1_ = x1

        M = torch.cdist(x0_, x1_) ** 2
        p = ot.emd(a, b, M.detach().cpu().numpy())
        assert np.all(np.isfinite(p)), "p is not finite"

        p_flatten = p.flatten()
        p_flatten = p_flatten / p_flatten.sum()

        choices = np.random.choice(
            p.shape[0] * p.shape[1], p=p_flatten, size=x0.shape[0], replace=True
        )

        i, j = np.divmod(choices, p.shape[1])
        x0_ot = x0[i]
        x1_ot = x1[j]
        if condition is not None:
            # condition_ot = condition0_ot = condition1_ot = condition[j]
            condition_ot = condition[j]
        else:
            condition_ot = None

        return self.functional_flow_matching_loss(
            x0=x0_ot,
            x1=x1_ot,
            condition=condition_ot,
            average=average,
            sum_all_elements=sum_all_elements,
        )








