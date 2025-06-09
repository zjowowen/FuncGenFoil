from typing import List
import torch
import torch.nn as nn
from neuralop.models import FNO
import gpytorch


def t_allhot(t, shape: List):
    """
    Overview:
        Converts a scalar time to a one-hot time tensor.
    Arguments:
        - t (torch.Tensor): scalar time tensor
        - shape (List): shape of the output tensor
    Returns:
        - t (torch.Tensor): one-hot time tensor
    """
    batch_size = shape[0]
    n_channels = shape[1]
    dim = shape[2:]
    n_dim = len(dim)

    t = t.view(batch_size, *[1] * (1 + n_dim))
    t = t * torch.ones(batch_size, 1, *dim, device=t.device)
    return t


def make_posn_embed(batch_size: int, dims: List[int]):
    """
    Overview:
        Create spatial embeddings for the input grid.
    Arguments:
        - batch_size (int): batch size
        - dims (List): dimensions of the input grid
    Returns:
        - emb (torch.Tensor): spatial embeddings
    """
    if len(dims) == 1:
        # Single channel of spatial embeddings
        emb = torch.linspace(0, 1, dims[0])
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1)
    elif len(dims) == 2:
        # 2 Channels of spatial embeddings
        x1 = torch.linspace(0, 1, dims[1]).repeat(dims[0], 1).unsqueeze(0)
        x2 = torch.linspace(0, 1, dims[0]).repeat(dims[1], 1).T.unsqueeze(0)
        emb = torch.cat((x1, x2), dim=0)  # (2, dims[0], dims[1])

        # Repeat along new batch channel
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (batch_size, 2, *dims)

    ## new
    elif len(dims) == 3:
        x1 = (
            torch.linspace(0, 1, dims[0])
            .reshape(1, dims[0], 1, 1)
            .repeat(1, 1, dims[1], dims[2])
        )
        x2 = (
            torch.linspace(0, 1, dims[1])
            .reshape(1, 1, dims[1], 1)
            .repeat(1, dims[0], 1, dims[2])
        )
        x3 = (
            torch.linspace(0, 1, dims[2])
            .reshape(1, 1, 1, dims[2])
            .repeat(1, dims[0], dims[1], 1)
        )
        emb = torch.cat((x1, x2, x3), dim=0)

        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # (batch_size, 3, *dims)

    else:
        raise NotImplementedError

    return emb


class FourierNeuralOperator(nn.Module):
    """
    Overview:
        Fourier Neural Operator model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        modes: int,
        vis_channels: int,
        hidden_channels: int,
        proj_channels: int,
        x_dim: int = 1,
        t_scaling: float = 1,
        n_layers: int = 4,
        n_conditions: int = 0,
    ):
        """
        Overview:
            Initialize the model.
        Arguments:
            - modes (int): number of Fourier modes
            - vis_channels (int): number of visual channels
            - hidden_channels (int): number of hidden channels
            - proj_channels (int): number of projection channels
            - x_dim (int): number of dimensions of the input grid
            - t_scaling (float): scaling factor for the time
        """
        super().__init__()

        self.t_scaling = t_scaling
        n_modes = (modes,) * x_dim  # Same number of modes in each x dimension
        in_channels = (
            vis_channels + x_dim + 1 + n_conditions
        )  # visual channels + spatial embedding + time embedding

        self.model = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channels=proj_channels,
            in_channels=in_channels,
            out_channels=vis_channels,
            n_layers=n_layers,
        )

    def forward(self, t, x, condition=None):
        """
        Overview:
            Forward pass of the model.
        Arguments:
            - t (torch.Tensor): time tensor
            - x (torch.Tensor): input tensor
            - condition (torch.Tensor): condition tensor
        Returns:
            - out (torch.Tensor): output tensor
        """
        u = x
        # u: (batch_size, channels, h, w)
        # t: either scalar or (batch_size,)

        t = t / self.t_scaling
        # print("t in the model:{}".format(t))
        batch_size = u.shape[0]
        dims = u.shape[2:]

        if t.dim() == 0 or t.numel() == 1:
            t = torch.ones(u.shape[0], device=t.device) * t

        assert t.dim() == 1
        assert t.shape[0] == u.shape[0]

        # Concatenate time as a new channel
        t = t_allhot(t, u.shape)
        # print('t max:{}, t min:{}'.format(t.max(), t.min()))
        # Concatenate position as new channel(s)
        posn_emb = make_posn_embed(batch_size, dims).to(u.device)
        if condition is not None:
            condition = condition.unsqueeze(2).expand(
                -1, -1, u.shape[2]
            )  # 扩展成 (B, 11, 257)
            u = torch.cat((u, posn_emb, t, condition), dim=1).float()
        else:
            u = torch.cat((u, posn_emb, t), dim=1).float()  # todo fix precision

        out = self.model(u)

        return out


# Assume LearnableMeanFieldFromGP class is defined above or imported
# (Using the version from the previous corrected response)
class LearnableMeanFieldFromGP(gpytorch.models.ApproximateGP):
    """
    Overview:
        A gpytorch model that defines a learnable mean field. This field is obtained
        from the mean parameter of a MeanFieldVariationalDistribution.
        It's designed to be used as a learnable set of weights for an inner product.
    """

    def __init__(self, num_channels: int, spatial_shape: tuple, x_dim: int):
        """
        Arguments:
            - num_channels (int): Number of independent mean fields (e.g., vis_channels).
            - spatial_shape (tuple): Spatial dimensions of the field (e.g., (H, W)).
            - x_dim (int): Number of spatial dimensions (e.g., 2 for (H,W)).
        """
        # Input validation
        if not isinstance(spatial_shape, tuple):
            raise TypeError("spatial_shape must be a tuple.")
        if not all(
            isinstance(dim, int) and dim > 0 for dim in spatial_shape
        ):  # Ensure positive dims
            raise TypeError("All elements of spatial_shape must be positive integers.")
        if len(spatial_shape) != x_dim:
            raise ValueError(
                f"Length of spatial_shape ({len(spatial_shape)}) "
                f"must match x_dim ({x_dim})."
            )

        self.num_channels = num_channels
        self.spatial_shape = spatial_shape
        self.x_dim = x_dim
        # Ensure spatial_shape elements are used as Python ints for prod
        _num_spatial_points_val = 1
        for dim_size in spatial_shape:
            _num_spatial_points_val *= dim_size
        self.num_spatial_points = _num_spatial_points_val

        # Create inducing points on a grid. These define the locations where the mean field is parameterized.
        # They are fixed (learn_inducing_locations=False).
        grid_coords = [torch.linspace(0, 1, s) for s in spatial_shape]
        mesh = torch.meshgrid(*grid_coords, indexing="ij")
        # inducing_points shape: (num_spatial_points, x_dim)
        inducing_points = torch.stack([m.flatten() for m in mesh], dim=-1)

        # Define the variational distribution.
        # We use MeanField, implying a diagonal covariance in the variational posterior.
        # batch_shape=[num_channels] means we have 'num_channels' independent distributions.
        variational_distribution = (
            gpytorch.variational.MeanFieldVariationalDistribution(
                num_inducing_points=self.num_spatial_points,
                batch_shape=torch.Size([num_channels]),
            )
        )

        # Define the variational strategy.
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )

        # Initialize the ApproximateGP model.
        super().__init__(variational_strategy)

        # Define mean and kernel modules (required by gpytorch.models.ApproximateGP).
        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([num_channels])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([num_channels]),
                ard_num_dims=x_dim if x_dim > 0 else None,
            ),  # ard_num_dims can be None if x_dim is 0
            batch_shape=torch.Size([num_channels]),
        )

    def forward(self, x_grid_points):
        """
        Standard forward method for a GP model.
        It computes the mean and covariance at given points 'x_grid_points'.
        """
        mean_x = self.mean_module(x_grid_points)
        covar_x = self.covar_module(x_grid_points)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_learnable_mean_field(self):
        """
        Returns the learnable mean field, reshaped to (num_channels, *spatial_shape).
        This field is directly based on the 'variational_mean' parameter of the variational distribution.
        """
        # model.variational_strategy.variational_distribution is a GPyTorch distribution (e.g., MultivariateNormal)
        # its .mean property is what we want, which is derived from the underlying _variational_distribution's parameters.
        mean_field_flat = self.variational_strategy.variational_distribution.mean
        return mean_field_flat.view(self.num_channels, *self.spatial_shape)


class FourierNeuralOperatorBasedValueFunction(nn.Module):
    """
    Overview:
        Fourier Neural Operator model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        modes: int,
        vis_channels: int,
        hidden_channels: int,
        proj_channels: int,
        spatial_shape: tuple,  # Spatial dimensions of input x, e.g., (H, W)
        x_dim: int = 1,
        t_scaling: float = 1,
        n_layers: int = 4,
        n_conditions: int = 0,
    ):
        """
        Overview:
            Initialize the model.
        Arguments:
            - modes (int): number of Fourier modes
            - vis_channels (int): number of visual channels
            - hidden_channels (int): number of hidden channels
            - proj_channels (int): number of projection channels
            - x_dim (int): number of dimensions of the input grid
            - t_scaling (float): scaling factor for the time
        """
        super().__init__()

        self.model = FourierNeuralOperator(
            modes=modes,
            vis_channels=vis_channels,
            hidden_channels=hidden_channels,
            proj_channels=proj_channels,
            x_dim=x_dim,
            t_scaling=t_scaling,
            n_layers=n_layers,
            n_conditions=n_conditions,
        )

        # Initialize the gpytorch-based learnable field module.
        # This 'gp' module holds the learnable weights for the inner product.
        self.gp = LearnableMeanFieldFromGP(
            num_channels=vis_channels,  # Matches output channels of self.model
            spatial_shape=spatial_shape,
            x_dim=x_dim,
        )

    def forward(self, t, x, condition=None):
        """
        Overview:
            Forward pass of the model.
        Arguments:
            - t (torch.Tensor): time tensor
            - x (torch.Tensor): input tensor
            - condition (torch.Tensor): condition tensor
        Returns:
            - out (torch.Tensor): output tensor
        """

        # features shape: (batch_size, vis_channels, *spatial_shape)
        features = self.model(t, x, condition)

        # Get the learnable mean field (weights for inner product)
        # gp_kernel shape: (vis_channels, *spatial_shape)
        gp_kernel = self.gp.get_learnable_mean_field()

        # Ensure gp_kernel is on the same device as features.
        # This is important if the model is moved to a GPU.
        gp_kernel = gp_kernel.to(features.device)

        # Sanity check for spatial dimension consistency (optional, good for debugging)
        if gp_kernel.shape[1:] != features.shape[2:]:
            raise ValueError(
                f"GP kernel spatial shape {gp_kernel.shape[1:]} "
                f"does not match feature spatial shape {features.shape[2:]}. "
                "Ensure 'spatial_shape' argument during ValueFunction init matches data's spatial resolution."
            )

        # Compute the inner product.
        # Sums over channel ('c') and all spatial dimensions ('...').
        # Result is per batch item ('b').
        value = torch.einsum("c...,bc...->b", gp_kernel, features)

        return value


class FourierNeuralOperatorDeterministic(nn.Module):
    """
    Overview:
        Fourier Neural Operator model.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        modes: int,
        vis_channels: int,
        hidden_channels: int,
        proj_channels: int,
        x_dim: int = 1,
        n_layers: int = 4,
        n_conditions: int = 0,
    ):
        """
        Overview:
            Initialize the model.
        Arguments:
            - modes (int): number of Fourier modes
            - vis_channels (int): number of visual channels
            - hidden_channels (int): number of hidden channels
            - proj_channels (int): number of projection channels
            - x_dim (int): number of dimensions of the input grid
        """
        super().__init__()

        n_modes = (modes,) * x_dim  # Same number of modes in each x dimension
        in_channels = (
            vis_channels + x_dim + 1 + n_conditions
        )  # visual channels + spatial embedding + time embedding

        self.model = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            projection_channels=proj_channels,
            in_channels=in_channels,
            out_channels=vis_channels,
            n_layers=n_layers,
        )

    def forward(self, x, condition=None):
        """
        Overview:
            Forward pass of the model.
        Arguments:
            - x (torch.Tensor): input tensor
            - condition (torch.Tensor): condition tensor
        Returns:
            - out (torch.Tensor): output tensor
        """
        u = torch.zeros_like(x)
        # u: (batch_size, channels, h, w)

        batch_size = u.shape[0]
        dims = u.shape[2:]

        posn_emb = make_posn_embed(batch_size, dims).to(u.device)
        if condition is not None:
            condition = condition.unsqueeze(2).expand(
                -1, -1, u.shape[2]
            )  # 扩展成 (B, 11, 257)
            u = torch.cat((u, posn_emb, condition), dim=1).float()
        else:
            u = torch.cat((u, posn_emb), dim=1).float()  # todo fix precision

        out = self.model(u)

        return out
