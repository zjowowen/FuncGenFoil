import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.gaussian_process.kernels import Matern,RBF,WhiteKernel,ConstantKernel,RationalQuadratic


def make_2d_grid(dims, x_min=0, x_max=1):
    """
    Overview:
        Create a 2D grid of points for some interval [x_min, x_max] and dimensions dims.
    Arguments:
        - dims (list): list of dimensions for the grid
        - x_min (float): minimum value of the grid
        - x_max (float): maximum value of the grid
    Returns:
        - x1 (tensor): grid of x values for the first dimension
        - x2 (tensor): grid of x values for the second dimension
        - grid (tensor): flattened grid of points
    """
    # Makes a 2D grid in the format of (n_grid, 2)
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x1, x2 = torch.meshgrid(x1, x2, indexing="ij")
    grid = torch.cat(
        (x1.contiguous().view(x1.numel(), 1), x2.contiguous().view(x2.numel(), 1)),
        dim=1,
    )
    return x1, x2, grid


def make_3d_grid(dims, x_min=0, x_max=1):
    """
    Overview:
        Create a 3D grid of points for some interval [x_min, x_max] and dimensions dims.
    Arguments:
        - dims (list): list of dimensions for the grid
        - x_min (float): minimum value of the grid
        - x_max (float): maximum value of the grid
    Returns:
        - x1 (tensor): grid of x values for the first dimension
        - x2 (tensor): grid of x values for the second dimension
        - x3 (tensor): grid of x values for the third dimension
        - grid (tensor): flattened grid of points
    """
    x1 = torch.linspace(x_min, x_max, dims[0])
    x2 = torch.linspace(x_min, x_max, dims[1])
    x3 = torch.linspace(x_min, x_max, dims[2])
    x1, x2, x3 = torch.meshgrid(x1, x2, x3, indexing="ij")
    grid = torch.cat(
        (
            x1.contiguous().view(x1.numel(), 1),
            x2.contiguous().view(x2.numel(), 1),
            x3.contiguous().view(x3.numel(), 1),
        ),
        dim=1,
    )
    return x1, x2, x3, grid


def make_grid(dims, x_min=0, x_max=1):
    """
    Overview:
        Create a grid of points for some interval [x_min, x_max] and dimensions dims.
    Arguments:
        - dims (list): list of dimensions for the grid
        - x_min (float): minimum value of the grid
        - x_max (float): maximum value of the grid
    Returns:
        - grid (tensor): flattened grid of points
    """
    if len(dims) == 1:
        grid = torch.linspace(x_min, x_max, dims[0])
        grid = grid.unsqueeze(-1)
    elif len(dims) == 2:
        _, _, grid = make_2d_grid(dims)
    elif len(dims) == 3:  # Shouldn't try it, too large
        _, _, _, grid = make_3d_grid(dims)

    return grid


def matern_kernel_cov(grids, length_scale, nu):
    """
    Overview:
        Create a Matern kernel covariance matrix for a given grid of points.
        
    Arguments:
        - grids (tensor): grid of points, for example, [n_points, 1 or 2]
        - length_scale (float): length scale of the kernel
        - nu (float): smoothness parameter of the kernel
    Returns:
        - kernel (tensor): Matern kernel covariance matrix
    """
    
    kernel = 1.0 * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=nu)
    return kernel(grids)


def rbf_kernel_cov(grids, length_scale):
    """
    Overview:
        Create a RBF kernel covariance matrix for a given grid of points.
        math:
        k(x_i, x_j) = \\exp\\left(- \\frac{d(x_i, x_j)^2}{2l^2} \\right)
    Arguments:
        - grids (tensor): grid of points, for example, [n_points, 1 or 2]
        - length_scale (float): length scale of the kernel
    Returns:
        - kernel (tensor): RBF kernel covariance matrix
    """
    kernel = 1.0 * RBF(length_scale=length_scale, length_scale_bounds="fixed")
    return kernel(grids)

    
def white_kernel_cov(grids, noise_level):
    """
    Overview:
        Create a WhiteKernel covariance matrix for a given grid of points.
        .. math::
        k(x_1, x_2) = noise\\_level \\text{ if } x_i == x_j \\text{ else } 0
    Arguments:
        - grids (tensor): grid of points, for example, [n_points, 1 or 2]
        - noise_level (float): noise level of the kernel
    Returns:
        - kernel (tensor): WhiteKernel covariance matrix
    """
    # Create the WhiteKernel with the specified noise level

    kernel = 1.0 * WhiteKernel(noise_level=noise_level, noise_level_bounds="fixed")
    # Compute the covariance matrix for the given grid
    return kernel(grids)
    

def get_gaussian_process(kernel_type, **kwargs):
    if kernel_type == "matern":
        return MaternGaussianProcess(**kwargs)
    elif kernel_type == "rbf":
        return RBFGaussianProcess(**kwargs)
    elif kernel_type=="white":
        return WhiteGaussianProcess(**kwargs)
    else:
        raise ValueError(f"Unsupported kernel_type '{kernel_type}'. Supported types are: 'matern', 'rbf', 'white'.")


class MaternGaussianProcess(torch.distributions.distribution.Distribution):
    """
    Overview:
        A Gaussian process with a Matern kernel.
    Interface:
        "__init__", "new_dist", "sample", "sample_from_prior", "sample_train_data", "prior_likelihood"
    """

    def __init__(
        self,
        length_scale: float,
        nu: float,
        device: torch.device,
        dims: list,
    ):
        """
        Overview:
            Initialize the Gaussian process.
        Arguments:
            - length_scale (float): length scale of the kernel
            - nu (float): smoothness parameter of the kernel
            - device (torch.device): device to run the GP on
            - dims (list): list of dimensions for the grid
        """

        jitter = 1e-6
        n_points = np.prod(dims)  
        grids = make_grid(dims)
        matern_ker = matern_kernel_cov(grids, length_scale, nu)

        self.length_scale = length_scale
        self.nu = nu
        self.dims = dims

        base_mu = torch.zeros(n_points).float().to(device)
        # add jitter
        base_cov = torch.tensor(matern_ker).float() + jitter * torch.eye(
            matern_ker.shape[0]
        )
        base_cov = base_cov.to(torch.float64).to(
            device
        )  # can help improve numerical stability

        self.base_dist = torch.distributions.MultivariateNormal(
            base_mu,
            scale_tril=torch.linalg.cholesky_ex(base_cov)[0].to(torch.float32),
        )

        self.device = device

    def new_dist(self, dims: list):
        """
        Overview:
            Creates a Normal distribution at the points in x.
        Arguments:
            - dims (list): list of dimensions for the grid
        Returns:
            - base_dist (torch.distributions.MultivariateNormal): a Gaussian at x
        """
        jitter = 1e-6
        n_points = np.prod(dims)
        grids = make_grid(dims)
        matern_ker = matern_kernel_cov(grids, self.length_scale, self.nu)

        base_mu = torch.zeros(n_points).float().to(self.device)
        base_cov = torch.tensor(matern_ker).float() + jitter * torch.eye(
            matern_ker.shape[0]
        )
        base_cov = base_cov.to(torch.float64).to(self.device)

        base_dist = torch.distributions.MultivariateNormal(
            base_mu,
            scale_tril=torch.linalg.cholesky_ex(base_cov)[0].to(torch.float32),
        )

        return base_dist

    def sample(self, dims: list, n_samples: int = 1, n_channels: int = 1):
        """
        Overview:
            Draws samples from the GP prior.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
        Returns:
            - samples (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """

        if dims == self.dims:
            distr = self.base_dist
        else:
            distr = self.new_dist(dims)
        samples = distr.sample(
            sample_shape=torch.Size(
                [
                    n_samples * n_channels,
                ]
            )
        )
        samples = samples.reshape(n_samples, n_channels, *dims)

        return samples

    def sample_from_prior(self, dims: list, n_samples: int = 1, n_channels: int = 1):
        """
        Overview:
            Draws samples from the GP prior.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
        Returns:
            - samples (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        samples = self.base_dist.sample(
            sample_shape=torch.Size(
                [
                    n_samples * n_channels,
                ]
            )
        )
        samples = samples.reshape(n_samples, n_channels, *dims)

        return samples

    def sample_train_data(
        self, dims: list, n_samples: int = 1, n_channels: int = 1, nbatch: int = 200000
    ):
        """
        Overview:
            Draws samples from the GP prior.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
            - nbatch (int): batch size for sampling
        Returns:
            - samples_all (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        samples_all = []

        sampled_num = 0
        nbatch = np.min([n_samples, nbatch])

        while sampled_num < n_samples:
            temp_sample = self.sample_from_prior(dims, nbatch, n_channels)
            sampled_num += len(temp_sample)
            samples_all.append(temp_sample)

        samples_all = torch.vstack(samples_all)[:n_samples]
        return samples_all

    def prior_likelihood(self, x: torch.Tensor):
        """
        Overview:
            Calculate the likelihood of the input.
        Arguments:
            - dims (list): list of dimensions for the grid
            - x (tensor): input to calculate the likelihood for, shape (n_batch, n_channels, ...)
        Returns:
            - logp (tensor): log likelihood of the input
        """

        x = torch.flatten(x, start_dim=1)

        if self.base_dist.loc.shape == x.shape[1:]:
            distr = self.base_dist
        else:
            distr = self.new_dist(x.shape[1:])

        logp = distr.log_prob(x)
        return logp


class RBFGaussianProcess(torch.distributions.distribution.Distribution):
    """
    Overview:
        A Gaussian process with a RBF kernel.
    Interface:
        "__init__", "new_dist", "sample", "sample_from_prior", "sample_train_data", "prior_likelihood"
    """

    def __init__(
        self,
        length_scale: float,
        device: torch.device,
        dims: list,
    ):
        """
        Overview:
            Initialize the Gaussian process.
        Arguments:
            - length_scale (float): length scale of the kernel
            - device (torch.device): device to run the GP on
            - dims (list): list of dimensions for the grid
        """

        jitter = 1e-6
        n_points = np.prod(dims)  #计算 dims 列表中所有元素的乘积，即网格中点的总数。
        grids = make_grid(dims)
        rbf_ker = rbf_kernel_cov(grids, length_scale)

        self.length_scale = length_scale
        self.dims = dims

        base_mu = torch.zeros(n_points).float().to(device)
        # add jitter
        base_cov = torch.tensor(rbf_ker).float() + jitter * torch.eye(
            rbf_ker.shape[0]
        )
        base_cov = base_cov.to(torch.float64).to(
            device
        )  # can help improve numerical stability

        self.base_dist = torch.distributions.MultivariateNormal(
            base_mu,
            scale_tril=torch.linalg.cholesky_ex(base_cov)[0].to(torch.float32),
        )

        self.device = device

    def new_dist(self, dims: list):
        """
        Overview:
            Creates a new Gaussian distribution for a given grid.
        Arguments:
            - dims (list): list of dimensions for the grid
        Returns:
            - base_dist (torch.distributions.MultivariateNormal): a Gaussian at x
        """
        jitter = 1e-6
        n_points = np.prod(dims)
        grids = make_grid(dims)
        rbf_ker = rbf_kernel_cov(grids, self.length_scale)

        base_mu = torch.zeros(n_points).float().to(self.device)
        base_cov = torch.tensor(rbf_ker).float() + jitter * torch.eye(
            rbf_ker.shape[0]
        )
        base_cov = base_cov.to(torch.float64).to(self.device)

        base_dist = torch.distributions.MultivariateNormal(
            base_mu,
            scale_tril=torch.linalg.cholesky_ex(base_cov)[0].to(torch.float32),
        )

        return base_dist

    def sample(self, dims: list, n_samples: int = 1, n_channels: int = 1):
        """
        Overview:
            Draws samples from the GP prior.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
        Returns:
            - samples (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """

        if dims == self.dims:
            distr = self.base_dist
        else:
            distr = self.new_dist(dims)
        samples = distr.sample(
            sample_shape=torch.Size(
                [
                    n_samples * n_channels,
                ]
            )
        )
        samples = samples.reshape(n_samples, n_channels, *dims)

        return samples

    def sample_from_prior(self, dims: list, n_samples: int = 1, n_channels: int = 1):
        """
        Overview:
            Draws samples from the GP prior using the base distribution.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
        Returns:
            - samples (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        samples = self.base_dist.sample(
            sample_shape=torch.Size(
                [
                    n_samples * n_channels,
                ]
            )
        )
        samples = samples.reshape(n_samples, n_channels, *dims)

        return samples

    def sample_train_data(
        self, dims: list, n_samples: int = 1, n_channels: int = 1, nbatch: int = 200000
    ):
        """
        Overview:
            Draws samples from the GP prior in batches.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
            - nbatch (int): batch size for sampling
        Returns:
            - samples_all (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        samples_all = []

        sampled_num = 0
        nbatch = np.min([n_samples, nbatch])

        while sampled_num < n_samples:
            temp_sample = self.sample_from_prior(dims, nbatch, n_channels)
            sampled_num += len(temp_sample)
            samples_all.append(temp_sample)

        samples_all = torch.vstack(samples_all)[:n_samples]
        return samples_all

    def prior_likelihood(self, x: torch.Tensor):
        """
        Overview:
            Calculate the likelihood of the input under the GP prior.
        Arguments:
            - dims (list): list of dimensions for the grid
            - x (tensor): input to calculate the likelihood for, shape (n_batch, n_channels, ...)
        Returns:
            - logp (tensor): log likelihood of the input
        """

        x = torch.flatten(x, start_dim=1)

        if self.base_dist.loc.shape == x.shape[1:]:
            distr = self.base_dist
        else:
            distr = self.new_dist(x.shape[1:])

        logp = distr.log_prob(x)
        return logp

class WhiteGaussianProcess(torch.distributions.distribution.Distribution):
    """
    Overview:
        A Gaussian process with a Whitekernel.
    Interface:
        "__init__", "new_dist", "sample", "sample_from_prior", "sample_train_data", "prior_likelihood"
    """

    def __init__(
        self,
        noise_level: float,
        device: torch.device,
        dims: list,
    ):
        """
        Overview:
            Initialize the Gaussian process with a WhiteKernel.
        Arguments:
            - noise_level (float): noise level of the kernel
            - device (torch.device): device to run the GP on
            - dims (list): list of dimensions for the grid
        """

        jitter = 1e-6
        n_points = np.prod(dims)  #计算 dims 列表中所有元素的乘积，即网格中点的总数。
        grids = make_grid(dims)
        white_kernel = white_kernel_cov(grids, noise_level)
        self.noise_level = noise_level
 
        self.dims = dims

        base_mu = torch.zeros(n_points).float().to(device)
        # add jitter
        base_cov = torch.tensor(white_kernel).float() + jitter * torch.eye(
            white_kernel.shape[0]
        )
        base_cov = base_cov.to(torch.float64).to(
            device
        )  # can help improve numerical stability

        self.base_dist = torch.distributions.MultivariateNormal(
            base_mu,
            scale_tril=torch.linalg.cholesky_ex(base_cov)[0].to(torch.float32),
        )

        self.device = device

    def new_dist(self, dims: list):
        """
        Overview:
            Creates a new Gaussian distribution for a given grid.
        Arguments:
            - dims (list): list of dimensions for the grid
        Returns:
            - base_dist (torch.distributions.MultivariateNormal): a Gaussian at x
        """
        jitter = 1e-6
        n_points = np.prod(dims)
        grids = make_grid(dims)
        white_ker = white_kernel_cov(grids, self.noise_level)

        base_mu = torch.zeros(n_points).float().to(self.device)
        base_cov = torch.tensor(white_ker).float() + jitter * torch.eye(
            white_ker.shape[0]
        )
        base_cov = base_cov.to(torch.float64).to(self.device)

        base_dist = torch.distributions.MultivariateNormal(
            base_mu,
            scale_tril=torch.linalg.cholesky_ex(base_cov)[0].to(torch.float32),
        )

        return base_dist

    def sample(self, dims: list, n_samples: int = 1, n_channels: int = 1):
        """
        Overview:
            Draws samples from the GP prior.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
        Returns:
            - samples (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """

        if dims == self.dims:
            distr = self.base_dist
        else:
            distr = self.new_dist(dims)
        samples = distr.sample(
            sample_shape=torch.Size(
                [
                    n_samples * n_channels,
                ]
            )
        )
        samples = samples.reshape(n_samples, n_channels, *dims)

        return samples

    def sample_from_prior(self, dims: list, n_samples: int = 1, n_channels: int = 1):
        """
        Overview:
            Draws samples from the GP prior.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
        Returns:
            - samples (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        samples = self.base_dist.sample(
            sample_shape=torch.Size(
                [
                    n_samples * n_channels,
                ]
            )
        )
        samples = samples.reshape(n_samples, n_channels, *dims)

        return samples

    def sample_train_data(
        self, dims: list, n_samples: int = 1, n_channels: int = 1, nbatch: int = 200000
    ):
        """
        Overview:
            Draws samples from the GP prior.
        Arguments:
            - dims (list): list of dimensions of inputs; e.g. for a 64x64 input grid, dims=[64, 64]
            - n_samples (int): number of samples to draw
            - n_channels (int): number of independent channels to draw samples for
            - nbatch (int): batch size for sampling
        Returns:
            - samples_all (tensor): samples from the GP; tensor (n_samples, n_channels, dims[0], dims[1], ...)
        """
        samples_all = []

        sampled_num = 0
        nbatch = np.min([n_samples, nbatch])

        while sampled_num < n_samples:
            temp_sample = self.sample_from_prior(dims, nbatch, n_channels)
            sampled_num += len(temp_sample)
            samples_all.append(temp_sample)

        samples_all = torch.vstack(samples_all)[:n_samples]
        return samples_all

    def prior_likelihood(self, x: torch.Tensor):
        """
        Overview:
            Calculate the likelihood of the input.
        Arguments:
            - dims (list): list of dimensions for the grid
            - x (tensor): input to calculate the likelihood for, shape (n_batch, n_channels, ...)
        Returns:
            - logp (tensor): log likelihood of the input
        """

        x = torch.flatten(x, start_dim=1)

        if self.base_dist.loc.shape == x.shape[1:]:
            distr = self.base_dist
        else:
            distr = self.new_dist(x.shape[1:])

        logp = distr.log_prob(x)
        return logp
