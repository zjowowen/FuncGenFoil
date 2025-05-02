from typing import List
import torch
import torch.nn as nn
from neuralop.models import FNO


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
