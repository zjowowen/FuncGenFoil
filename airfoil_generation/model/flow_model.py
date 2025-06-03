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


class FunctionalFlow(nn.Module):
    """
    Overview:
        A functional flow model.
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
            Initialize the functional flow model.
        Arguments:
            - config (EasyDict): configuration for the model
            - model (nn.Module): intrinsic model
        """
        super().__init__()

        self.config = config
        self.device = config.device
        self.path = ConditionalProbabilityPath(config.path)
        self.model = IntrinsicModel(config.model.args) if model is None else model

        self.gaussian_process = get_gaussian_process(
            config.gaussian_process.type, **config.gaussian_process.args
        )
        self.stochastic_process = StochasticProcess(self.path, self.gaussian_process)

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

    def get_type(self):
        return "FunctionalFlow"

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
                return self.model(t=t, x=x, condition=condition)

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
                data = data.squeeze(1 + len(extra_batch_size.shape))
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
            Inverse sample from the functional flow model.
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
            Inverse sample from the functional flow model.
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
                data = data.squeeze(1 + len(extra_batch_size.shape))
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
            Inverse sample from the functional flow model with log probability.
        Arguments:
            - t_span (tensor): time span to sample over
            - x_0 (tensor): initial condition for sampling
            - log_prob_x_0 (tensor): log probability of the initial condition
            - function_log_prob_x_0 (callable, nn.Module): function to compute the log probability of the initial condition
            - condition (tensor): condition for sampling
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
            - using_Hutchinson_trace_estimator (bool): whether to use Hutchinson trace estimator
        Returns:
            - x1 (tensor): samples from the functional flow model; tensor (B, D)
            - log_likelihood (tensor): log likelihood of the samples; tensor (B)
            - logp_x1_minus_logp_x0 (tensor): log probability difference; tensor (B)
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
            Inverse sample from the functional flow model with log probability.
        Arguments:
            - t_span (tensor): time span to sample over
            - x_0 (tensor): initial condition for sampling
            - log_prob_x_0 (tensor): log probability of the initial condition
            - function_log_prob_x_0 (callable, nn.Module): function to compute the log probability of the initial condition
            - condition (tensor): condition for sampling
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
            - using_Hutchinson_trace_estimator (bool): whether to use Hutchinson trace estimator
        Returns:
            - x1 (tensor): samples from the functional flow model; tensor (T, B, N, D)
            - log_likelihood (tensor): log likelihood of the samples; tensor (T, B)
            - logp_x1_minus_logp_x0 (tensor): log probability difference; tensor (T, B)
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


class FunctionalFlowForRegression(nn.Module):
    """
    Overview:
        A functional flow model for regression.
    Interface:
        "__init__", "get_type", "forward", "sample", "sample_process"
    """

    def __init__(
        self,
        config: EasyDict,
        model: nn.Module = None,
        prior: torch.Tensor = None,
    ):
        """
        Overview:
            Initialize the functional flow model for regression.
        Arguments:
            - config (EasyDict): configuration for the model
            - model (nn.Module): intrinsic model
            - prior (tensor): prior
        """
        super().__init__()

        self.config = config
        self.device = config.device
        self.path = ConditionalProbabilityPath(config.path)
        self.model = IntrinsicModel(config.model.args) if model is None else model

        self.gaussian_process = get_gaussian_process(
            config.gaussian_process.type, **config.gaussian_process.args
        )
        self.stochastic_process = StochasticProcess(self.path, self.gaussian_process)

        if hasattr(config, "solver"):
            self.solver = get_solver(config.solver.type)(**config.solver.args)

        self.prior = nn.Parameter(prior)

    def get_type(self):
        return "FunctionalFlowForRegression"

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
            Sample from the functional flow model for regression.
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
            Sample from the functional flow model for regression.
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
                return self.model(t=t, x=x, condition=condition)

            if solver.library == "torchdiffeq_adjoint":
                if with_grad:
                    data = solver.integrate(
                        drift=drift,
                        x0=x,
                        t_span=t_span,
                        adjoint_params=find_parameters(self),
                    )
                else:
                    with torch.no_grad():
                        data = solver.integrate(
                            drift=drift,
                            x0=x,
                            t_span=t_span,
                            adjoint_params=find_parameters(self),
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
                data = data.squeeze(1 + len(extra_batch_size.shape))
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
            Inverse sample from the functional flow model for regression.
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
            - samples (tensor): samples from the functional flow model; tensor (B, D)
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
            Inverse sample from the functional flow model for regression.
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
                data = data.squeeze(1 + len(extra_batch_size.shape))
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
            Inverse sample from the functional flow model with log probability.
        Arguments:
            - t_span (tensor): time span to sample over
            - x_0 (tensor): initial condition for sampling
            - log_prob_x_0 (tensor): log probability of the initial condition
            - function_log_prob_x_0 (callable, nn.Module): function to compute the log probability of the initial condition
            - condition (tensor): condition for sampling
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
            - using_Hutchinson_trace_estimator (bool): whether to use Hutchinson trace estimator
        Returns:
            - x1 (tensor): samples from the functional flow model; tensor (B, D)
            - log_likelihood (tensor): log likelihood of the samples; tensor (B)
            - logp_x1_minus_logp_x0 (tensor): log probability difference; tensor (B)
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
            Inverse sample from the functional flow model with log probability.
        Arguments:
            - t_span (tensor): time span to sample over
            - x_0 (tensor): initial condition for sampling
            - log_prob_x_0 (tensor): log probability of the initial condition
            - function_log_prob_x_0 (callable, nn.Module): function to compute the log probability of the initial condition
            - condition (tensor): condition for sampling
            - with_grad (bool): whether to compute gradients
            - solver_config (EasyDict): configuration for the solver
            - using_Hutchinson_trace_estimator (bool): whether to use Hutchinson trace estimator
        Returns:
            - x1 (tensor): samples from the functional flow model; tensor (T, B, N, D)
            - log_likelihood (tensor): log likelihood of the samples; tensor (T, B)
            - logp_x1_minus_logp_x0 (tensor): log probability difference; tensor (T, B)
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
            - x0 (tensor): initial condition for sampling
            - x1 (tensor): target condition for sampling
            - condition (tensor): condition for sampling
            - average (bool): whether to average the loss
            - sum_all_elements (bool): whether to sum all elements of the loss
        Returns:
            - loss (tensor): functional flow matching loss
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
