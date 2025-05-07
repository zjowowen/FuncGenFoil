import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from airfoil_generation.utils.training import LpLoss
from airfoil_generation.utils.normalizer import UnitGaussianNormalizer

from tqdm import tqdm

from airfoil_generation.neural_networks.point_cloud_data_process import (
    compute_triangle_area_,
    compute_tetrahedron_volume_,
    compute_measure_per_elem_,
    compute_node_measures,
    convert_structured_data,
)


def pinv(a, rrank, rcond=1e-3):
    """
    Compute the (Moore-Penrose) pseudo-inverse of a matrix.

    Calculate the generalized inverse of a matrix using its
    singular-value decomposition (SVD) and including all
    *large* singular values.

        Parameters:
            a : float[M, N]
                Matrix to be pseudo-inverted.
            rrank : int
                Maximum rank
            rcond : float, optional
                Cutoff for small singular values.
                Singular values less than or equal to
                ``rcond * largest_singular_value`` are set to zero.
                Default: ``1e-3``.

        Returns:
            B : float[N, M]
                The pseudo-inverse of `a`.

    """
    u, s, vt = np.linalg.svd(a, full_matrices=False)

    # discard small singular values
    cutoff = rcond * s[0]
    large = s > cutoff
    large[rrank:] = False
    s = np.divide(1, s, where=large, out=s)
    s[~large] = 0

    res = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))
    return res


def compute_edge_gradient_weights_helper(nodes, node_dims, adj_list, rcond=1e-3):
    """
    Compute weights for gradient computation
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)     =          :
       :                                :
    xj - x                        f(xj) - f(x)

    in matrix form   dx  nabla f(x)   = df.

    The pseudo-inverse of dx is pinvdx.
    Then gradient f(x) for any function f, is pinvdx * df
    We store directed edges (x, x1), (x, x2), ..., (x, xj)
    And its associated weight pinvdx[:,1], pinvdx[:,2], ..., pinvdx[:,j]
    Then the gradient can be efficiently computed with scatter_add

    When these points are on a degerated plane or surface, the gradient towards the
    normal direction is 0.


        Parameters:
            nodes : float[nnodes, ndims]
            node_dims : int[nnodes], the intrisic dimensionality of the node
                        if the node is on a volume, it is 3
                        if the node is on a surface, it is 2
                        if the node is on a line, it is 1
                        if it is on different type of elements, take the maximum


            adj_list : list of set, saving neighbors for each nodes
            rcond : float, truncate the singular values in numpy.linalg.pinv at rcond*largest_singular_value


        Return :

            directed_edges : int[nedges,2]
            edge_gradient_weights   : float[nedges, ndims]

            * the directed_edges (and adjacent list) include all node pairs that share the element

    """

    nnodes, ndims = nodes.shape
    directed_edges = []
    edge_gradient_weights = []
    for a in range(nnodes):
        dx = np.zeros((len(adj_list[a]), ndims))
        for i, b in enumerate(adj_list[a]):
            dx[i, :] = nodes[b, :] - nodes[a, :]
            directed_edges.append([a, b])
        edge_gradient_weights.append(pinv(dx, rrank=node_dims[a], rcond=rcond).T)

    directed_edges = np.array(directed_edges, dtype=int)
    edge_gradient_weights = np.concatenate(edge_gradient_weights, axis=0)
    return directed_edges, edge_gradient_weights, adj_list


def compute_edge_gradient_weights(nodes, elems, rcond=1e-3):
    """
    Compute weights for gradient computation
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)     =          :
       :                                :
    xj - x                        f(xj) - f(x)

    in matrix form   dx  nable f(x)   = df.

    The pseudo-inverse of dx is pinvdx.
    Then gradient f(x) for any function f, is pinvdx * df
    We store directed edges (x, x1), (x, x2), ..., (x, xj)
    And its associated weight pinvdx[:,1], pinvdx[:,2], ..., pinvdx[:,j]
    Then the gradient can be efficiently computed with scatter_add

    When these points are on a degerated plane or surface, the gradient towards the
    normal direction is 0.


        Parameters:
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1].
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            rcond : float, truncate the singular values in numpy.linalg.pinv at rcond*largest_singular_value


        Return :

            directed_edges : int[nedges,2]
            edge_gradient_weights   : float[nedges, ndims]

            * the directed_edges (and adjacent list) include all node pairs that share the element

    """

    nnodes, ndims = nodes.shape
    nelems, _ = elems.shape
    # Initialize adjacency list as a list of sets
    # Use a set to store unique directed edges
    adj_list = [set() for _ in range(nnodes)]

    # Initialize node_dims to store the maximum dimensionality at that node
    node_dims = np.zeros(nnodes, dtype=int)
    # Loop through each element and create directed edges
    for elem in elems:
        elem_dim, e = elem[0], elem[1:]
        node_dims[e] = np.maximum(node_dims[e], elem_dim)
        e = e[e >= 0]
        nnodes_per_elem = len(e)
        for i in range(nnodes_per_elem):
            # Add each node's neighbors to its set
            adj_list[e[i]].update([e[j] for j in range(nnodes_per_elem) if j != i])

    return compute_edge_gradient_weights_helper(nodes, node_dims, adj_list, rcond=rcond)


def preprocess_data(nodes_list, elems_list, features_list):
    """
    Compute node measures (length, area or volume for each node),
    for each element, compute its length, area or volume,
    equally assign it to its nodes.

        Parameters:
            nodes_list :  list of float[nnodes, ndims]
            elems_list :  list of float[nelems, max_num_of_nodes_per_elem+1].
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            features_list  : list of float[nnodes, nfeatures]


        Return :
            nnodes         :  int
            node_mask      :  int[ndata, max_nnodes, 1]               (1 for node, 0 for padding)
            nodes          :  float[ndata, max_nnodes, ndims]      (padding 0)
            node_measures  :  float[ndata, max_nnodes, 1]               (padding NaN)
            features       :  float[ndata, max_nnodes, nfeatures]  (padding 0)
            directed_edges :  float[ndata, max_nedges, 2]          (padding 0)
            edge_gradient_weights   :  float[ndata, max_nedges, ndims]      (padding 0)
    """
    ndata = len(nodes_list)
    ndims, nfeatures = nodes_list[0].shape[1], features_list[0].shape[1]
    nnodes = np.array([nodes.shape[0] for nodes in nodes_list], dtype=int)
    max_nnodes = max(nnodes)

    print("Preprocessing data : computing node_mask")
    node_mask = np.zeros((ndata, max_nnodes, 1), dtype=int)
    for i in range(ndata):
        node_mask[i, : nnodes[i], :] = 1

    print("Preprocessing data : computing nodes")
    nodes = np.zeros((ndata, max_nnodes, ndims))
    for i in range(ndata):
        nodes[i, : nnodes[i], :] = nodes_list[i]

    print("Preprocessing data : computing node_measures")
    # The mesh might have elements with different dimensionalities (e.g., 1D edges, 2D faces, 3D volumes).
    # If any mesh includes both 1D and 2D elements, it is assumed that all meshes in the dataset will also include both types of elements.
    # This ensures uniformity in processing and avoids inconsistencies in element handling.
    node_measures = np.full((ndata, max_nnodes, ndims), np.nan)
    nmeasures = 0
    for i in tqdm(range(ndata)):
        measures = compute_node_measures(nodes_list[i], elems_list[i])
        if i == 0:
            nmeasures = measures.shape[1]
        node_measures[i, : nnodes[i], :nmeasures] = measures
    node_measures = node_measures[..., :nmeasures]

    print("Preprocessing data : computing features")
    features = np.zeros((ndata, max_nnodes, nfeatures))
    for i in range(ndata):
        features[i, : nnodes[i], :] = features_list[i]

    print("Preprocessing data : computing directed_edges and edge_gradient_weights")
    directed_edges_list, edge_gradient_weights_list = [], []
    for i in tqdm(range(ndata)):
        directed_edges, edge_gradient_weights, edge_adj_list = (
            compute_edge_gradient_weights(nodes_list[i], elems_list[i])
        )
        directed_edges_list.append(directed_edges)
        edge_gradient_weights_list.append(edge_gradient_weights)
    nedges = np.array(
        [directed_edges.shape[0] for directed_edges in directed_edges_list]
    )
    max_nedges = max(nedges)
    directed_edges, edge_gradient_weights = np.zeros(
        (ndata, max_nedges, 2), dtype=int
    ), np.zeros((ndata, max_nedges, ndims))
    for i in range(ndata):
        directed_edges[i, : nedges[i], :] = directed_edges_list[i]
        edge_gradient_weights[i, : nedges[i], :] = edge_gradient_weights_list[i]

    return (
        nnodes,
        node_mask,
        nodes,
        node_measures,
        features,
        directed_edges,
        edge_gradient_weights,
    )


def compute_node_weights(nnodes, node_measures, equal_measure=False):
    """
    Compute node weights based on node measures (length, area, volume,......).

    This function calculates weights for each node using its corresponding measures.
    If `equal_measure` is set to True, the node measures are recomputed as equal, |Omega|/n,
    where `|Omega|` is the total measure and `n` is the number of nodes with nonzero measures.
    Node weights are computed such that their sum equals 1, using the formula:
        node_weight = node_measure / sum(node_measures)

    If there are several types of measures, compute weights for each type of measures, and normalize it by nmeasures
    node_weight = 1/nmeasures * node_measure / sum(node_measures)

    Parameters:
        nnodes int[ndata]:
            Number of nodes for each data instance.

        node_measures float[ndata, max_nnodes, nmeasures]:
            Each value corresponds to the measure of a node.
            Padding with NaN is used for indices greater than or equal to the number of nodes (`nnodes`), or nodes do not have measure

        equal_measure (bool, optional):
            If True, node measures are uniformly distributed as |Omega|/n. Default is False.

    Returns:
        node_measures float[ndata, max_nnodes, nmeasures]:
            Updated array of node measures with shape, maintaining the same padding structure (But with padding 0).
            If equal_measure is False, the measures remains unchanged
        node_weights float[ndata, max_nnodes, nmeasures]:
            Array of computed node weights, maintaining the same padding structure.
    """

    ndata, max_nnodes, nmeasures = node_measures.shape
    node_measures_new = np.zeros((ndata, max_nnodes, nmeasures))
    node_weights = np.zeros((ndata, max_nnodes, nmeasures))
    if equal_measure:
        for i in range(ndata):
            for j in range(nmeasures):
                # take average for nonzero measure nodes
                indices = np.isfinite(node_measures[i, :, j])
                node_measures_new[i, indices, j] = sum(
                    node_measures[i, indices, j]
                ) / sum(indices)
    else:
        # replace all NaN value to 0
        node_measures_new = np.nan_to_num(node_measures, nan=0.0)

    # node weight is the normalization of node measure
    for i in range(ndata):
        for j in range(nmeasures):
            node_weights[i, : nnodes[i], j] = (
                1.0
                / nmeasures
                * node_measures_new[i, : nnodes[i], j]
                / sum(node_measures_new[i, : nnodes[i], j])
            )

    return node_measures_new, node_weights


def _get_act(act):
    if act == "tanh":
        func = F.tanh
    elif act == "gelu":
        func = F.gelu
    elif act == "relu":
        func = F.relu_
    elif act == "elu":
        func = F.elu_
    elif act == "leaky_relu":
        func = F.leaky_relu_
    elif act == "none":
        func = None
    else:
        raise ValueError(f"{act} is not supported")
    return func


def compute_Fourier_modes_helper(ndims, nks, Ls):
    """
    Compute Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k

        Parameters:
            ndims : int
            nks   : int[ndims]
            Ls    : float[ndims]

        Return :
            k_pairs : float[nmodes, ndims]
    """
    assert len(nks) == len(Ls) == ndims
    if ndims == 1:
        nk, Lx = nks[0], Ls[0]
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(1, nk + 1):
            k_pairs[i, :] = 2 * np.pi / Lx * kx
            k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
            i += 1

    elif ndims == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2 * nx * ny + nx + ny
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if ky == 0 and kx <= 0:
                    continue

                k_pairs[i, :] = 2 * np.pi / Lx * kx, 2 * np.pi / Ly * ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1

    elif ndims == 3:
        nx, ny, nz = nks
        Lx, Ly, Lz = Ls
        nk = 4 * nx * ny * nz + 2 * (nx * ny + nx * nz + ny * nz) + nx + ny + nz
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(-ny, ny + 1):
                for kz in range(0, nz + 1):
                    if kz == 0 and (ky < 0 or (ky == 0 and kx <= 0)):
                        continue

                    k_pairs[i, :] = (
                        2 * np.pi / Lx * kx,
                        2 * np.pi / Ly * ky,
                        2 * np.pi / Lz * kz,
                    )
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")

    k_pairs = k_pairs[np.argsort(k_pair_mag), :]
    return k_pairs


def compute_Fourier_modes(ndims, nks, Ls):
    """
    Compute `nmeasures` sets of Fourier modes number k
    Fourier bases are cos(kx), sin(kx), 1
    * We cannot have both k and -k

        Parameters:
            ndims : int
            nks   : int[ndims * nmeasures]
            Ls    : float[ndims * nmeasures]

        Return :
            k_pairs : float[nmodes, ndims, nmeasures]
    """
    assert len(nks) == len(Ls)
    nmeasures = len(nks) // ndims
    k_pairs = np.stack(
        [
            compute_Fourier_modes_helper(
                ndims, nks[i * ndims : (i + 1) * ndims], Ls[i * ndims : (i + 1) * ndims]
            )
            for i in range(nmeasures)
        ],
        axis=-1,
    )

    return k_pairs


def compute_Fourier_bases(nodes, modes):
    """
    Compute Fourier bases for the whole space
    Fourier bases are cos(kx), sin(kx), 1

        Parameters:
            nodes        : float[batch_size, nnodes, ndims]
            modes        : float[nmodes, ndims, nmeasures]

        Return :
            bases_c, bases_s : float[batch_size, nnodes, nmodes, nmeasures]
            bases_0 : float[batch_size, nnodes, 1, nmeasures]
    """
    # temp : float[batch_size, nnodes, nmodes, nmeasures]
    temp = torch.einsum("bxd,kdw->bxkw", nodes, modes)

    bases_c = torch.cos(temp)
    bases_s = torch.sin(temp)
    batch_size, nnodes, _, nmeasures = temp.shape
    bases_0 = torch.zeros(
        batch_size, nnodes, 1, nmeasures, dtype=temp.dtype, device=temp.device
    )
    return bases_c, bases_s, bases_0


################################################################
# Fourier layer
################################################################
class SpectralConv(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        nmodes, ndims, nmeasures = modes.shape
        self.modes = modes
        self.nmeasures = nmeasures
        self.scale = 1 / (in_channels * out_channels)

        self.weights_c = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_s = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, nmodes, nmeasures, dtype=torch.float
            )
        )
        self.weights_0 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, 1, nmeasures, dtype=torch.float)
        )

    def forward(self, x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0):
        """
        Compute Fourier neural layer
            Parameters:
                x                   : float[batch_size, in_channels, nnodes]
                bases_c, bases_s    : float[batch_size, nnodes, nmodes, nmeasures]
                bases_0             : float[batch_size, nnodes, 1, nmeasures]
                wbases_c, wbases_s  : float[batch_size, nnodes, nmodes, nmeasures]
                wbases_0            : float[batch_size, nnodes, 1, nmeasures]

            Return :
                x                   : float[batch_size, out_channels, nnodes]
        """
        x_c_hat = torch.einsum("bix,bxkw->bikw", x, wbases_c)
        x_s_hat = -torch.einsum("bix,bxkw->bikw", x, wbases_s)
        x_0_hat = torch.einsum("bix,bxkw->bikw", x, wbases_0)

        weights_c, weights_s, weights_0 = self.weights_c, self.weights_s, self.weights_0

        f_c_hat = torch.einsum("bikw,iokw->bokw", x_c_hat, weights_c) - torch.einsum(
            "bikw,iokw->bokw", x_s_hat, weights_s
        )
        f_s_hat = torch.einsum("bikw,iokw->bokw", x_s_hat, weights_c) + torch.einsum(
            "bikw,iokw->bokw", x_c_hat, weights_s
        )
        f_0_hat = torch.einsum("bikw,iokw->bokw", x_0_hat, weights_0)

        x = (
            torch.einsum("bokw,bxkw->box", f_0_hat, bases_0)
            + 2 * torch.einsum("bokw,bxkw->box", f_c_hat, bases_c)
            - 2 * torch.einsum("bokw,bxkw->box", f_s_hat, bases_s)
        )

        return x


def compute_gradient(f, directed_edges, edge_gradient_weights):
    """
    Compute gradient of field f at each node
    The gradient is computed by least square.
    Node x has neighbors x1, x2, ..., xj

    x1 - x                        f(x1) - f(x)
    x2 - x                        f(x2) - f(x)
       :      gradient f(x)   =         :
       :                                :
    xj - x                        f(xj) - f(x)

    in matrix form   dx  nable f(x)   = df.

    The pseudo-inverse of dx is pinvdx.
    Then gradient f(x) for any function f, is pinvdx * df
    directed_edges stores directed edges (x, x1), (x, x2), ..., (x, xj)
    edge_gradient_weights stores its associated weight pinvdx[:,1], pinvdx[:,2], ..., pinvdx[:,j]

    Then the gradient can be computed
    gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)]
    with scatter_add for each edge


        Parameters:
            f : float[batch_size, in_channels, nnodes]
            directed_edges : int[batch_size, max_nedges, 2]
            edge_gradient_weights : float[batch_size, max_nedges, ndims]

        Returns:
            x_gradients : float Tensor[batch_size, in_channels*ndims, max_nnodes]
            * in_channels*ndims dimension is gradient[x_1], gradient[x_2], gradient[x_3]......
    """

    f = f.permute(0, 2, 1)
    batch_size, max_nnodes, in_channels = f.shape
    _, max_nedges, ndims = edge_gradient_weights.shape
    # Message passing : compute message = edge_gradient_weights * (f_source - f_target) for each edge
    # target\source : int Tensor[batch_size, max_nedges]
    # message : float Tensor[batch_size, max_nedges, in_channels*ndims]

    target, source = (
        directed_edges[..., 0],
        directed_edges[..., 1],
    )  # source and target nodes of edges
    message = torch.einsum(
        "bed,bec->becd",
        edge_gradient_weights,
        f[torch.arange(batch_size).unsqueeze(1), source]
        - f[torch.arange(batch_size).unsqueeze(1), target],
    ).reshape(batch_size, max_nedges, in_channels * ndims)

    # f_gradients : float Tensor[batch_size, max_nnodes, in_channels*ndims]
    f_gradients = torch.zeros(
        batch_size,
        max_nnodes,
        in_channels * ndims,
        dtype=message.dtype,
        device=message.device,
    )
    f_gradients.scatter_add_(
        dim=1, src=message, index=target.unsqueeze(2).repeat(1, 1, in_channels * ndims)
    )

    return f_gradients.permute(0, 2, 1)


class PCNO(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        train_sp_L="independently",
        act="gelu",
    ):
        super(PCNO, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
            
            Parameters: 
                ndims : int 
                    Dimensionality of the problem
                modes : float[nmodes, ndims, nmeasures]
                    It contains nmodes modes k, and Fourier bases include : cos(k x), sin(k x), 1  
                    * We cannot have both k and -k
                nmeasures : int
                    Number of measures
                    There might be different integrals with different measures
                train_sp_L: bool or str
                    The way to train sp_L.
                    False: means we dont train sp_L
                    'together': means sp_L will be trained with other params together.
                    'independently' : means sp_L will be trained in another optimizer independently.
                layers : list of int
                    number of channels of each layer
                    The lifting layer first lifts to layers[0]
                    The first Fourier layer maps between layers[0] and layers[1]
                    ...
                    The nlayers Fourier layer maps between layers[nlayers-1] and layers[nlayers]
                    The number of Fourier layers is len(layers) - 1
                fc_dim : int 
                    hidden layers for the projection layer, when fc_dim > 0, otherwise there is no hidden layer
                in_dim : int 
                    The number of channels for the input function
                    For example, when the coefficient function and locations (a(x, y), x, y) are inputs, in_dim = 3
                out_dim : int 
                    The number of channels for the output function
                act : string (default gelu)
                    The activation function

            
            Returns:
                Point cloud neural operator

        """
        self.modes = modes
        self.nmeasures = nmeasures

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.train_sp_L = train_sp_L
        self.sp_L = nn.Parameter(
            torch.ones(ndims, nmeasures), requires_grad=bool(train_sp_L)
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims * in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)
        self.softsign = F.softsign

        self.normal_params = []  #  group of params which will be trained normally
        self.sp_L_params = []  #  group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.sp_L:
                self.normal_params.append(param)
            else:
                if self.train_sp_L == "together":
                    self.normal_params.append(param)
                elif self.train_sp_L == "independently":
                    self.sp_L_params.append(param)
                elif self.train_sp_L == False:
                    continue
                else:
                    raise ValueError(f"{self.train_sp_L} is not supported")

    def forward(self, x, aux):
        """
        Forward evaluation.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws;
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            Parameters:
                x : Tensor float[batch_size, max_nnomdes, in_dim]
                    Input data
                aux : list of Tensor, containing
                    node_mask : Tensor int[batch_size, max_nnomdes, 1]
                                1: node; otherwise 0

                    nodes : Tensor float[batch_size, max_nnomdes, ndim]
                            nodal coordinate; padding with 0

                    node_weights  : Tensor float[batch_size, max_nnomdes, nmeasures]
                                    rho(x)dx used for nmeasures integrations; padding with 0

                    directed_edges : Tensor int[batch_size, max_nedges, 2]
                                     direted edge pairs; padding with 0
                                     gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)]

                    edge_gradient_weights      : Tensor float[batch_size, max_nedges, ndim]
                                                 pinvdx on each directed edge
                                                 gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)]


            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim]
                       Input data

        """
        length = len(self.ws)

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = aux
        # bases: float[batch_size, nnodes, nmodes]
        bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, self.modes * self.sp_L)
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(
                self.softsign(
                    compute_gradient(x, directed_edges, edge_gradient_weights)
                )
            )
            x = x1 + x2 + x3
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        return x


class PointCloudNeuralOperator(nn.Module):
    def __init__(
        self,
        ndims,
        modes,
        nmeasures,
        layers,
        fc_dim=128,
        in_dim=3,
        out_dim=1,
        train_sp_L="independently",
        act="gelu",
    ):
        super(PointCloudNeuralOperator, self).__init__()

        """
        The overall network. 
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws; 
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
            
            Parameters: 
                ndims : int 
                    Dimensionality of the problem
                modes : float[nmodes, ndims, nmeasures]
                    It contains nmodes modes k, and Fourier bases include : cos(k x), sin(k x), 1  
                    * We cannot have both k and -k
                nmeasures : int
                    Number of measures
                    There might be different integrals with different measures
                train_sp_L: bool or str
                    The way to train sp_L.
                    False: means we dont train sp_L
                    'together': means sp_L will be trained with other params together.
                    'independently' : means sp_L will be trained in another optimizer independently.
                layers : list of int
                    number of channels of each layer
                    The lifting layer first lifts to layers[0]
                    The first Fourier layer maps between layers[0] and layers[1]
                    ...
                    The nlayers Fourier layer maps between layers[nlayers-1] and layers[nlayers]
                    The number of Fourier layers is len(layers) - 1
                fc_dim : int 
                    hidden layers for the projection layer, when fc_dim > 0, otherwise there is no hidden layer
                in_dim : int 
                    The number of channels for the input function
                    For example, when the coefficient function and locations (a(x, y), x, y) are inputs, in_dim = 3
                out_dim : int 
                    The number of channels for the output function
                act : string (default gelu)
                    The activation function

            
            Returns:
                Point cloud neural operator

        """
        self.modes = nn.Parameter(modes, requires_grad=False)
        self.nmeasures = nmeasures

        self.layers = layers
        self.fc_dim = fc_dim

        self.ndims = ndims
        self.in_dim = in_dim

        self.fc0 = nn.Linear(in_dim, layers[0])

        self.sp_convs = nn.ModuleList(
            [
                SpectralConv(in_size, out_size, modes)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )
        self.train_sp_L = train_sp_L
        self.sp_L = nn.Parameter(
            torch.ones(ndims, nmeasures), requires_grad=bool(train_sp_L)
        )

        self.ws = nn.ModuleList(
            [
                nn.Conv1d(in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        self.gws = nn.ModuleList(
            [
                nn.Conv1d(ndims * in_size, out_size, 1)
                for in_size, out_size in zip(self.layers, self.layers[1:])
            ]
        )

        if fc_dim > 0:
            self.fc1 = nn.Linear(layers[-1], fc_dim)
            self.fc2 = nn.Linear(fc_dim, out_dim)
        else:
            self.fc2 = nn.Linear(layers[-1], out_dim)

        self.act = _get_act(act)
        self.softsign = F.softsign

        self.normal_params = []  #  group of params which will be trained normally
        self.sp_L_params = []  #  group of params which may be trained specially
        for _, param in self.named_parameters():
            if param is not self.sp_L:
                self.normal_params.append(param)
            else:
                if self.train_sp_L == "together":
                    self.normal_params.append(param)
                elif self.train_sp_L == "independently":
                    self.sp_L_params.append(param)
                elif self.train_sp_L == False:
                    continue
                else:
                    raise ValueError(f"{self.train_sp_L} is not supported")

    def forward(self, t, x, condition=None):
        """
        Forward evaluation.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. len(layers)-1 layers of the point cloud neural layers u' = (W + K + D)(u).
           linear functions  W: parameterized by self.ws;
           integral operator K: parameterized by self.sp_convs with nmeasures different integrals
           differential operator D: parameterized by self.gws
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

            Parameters:
                x : Tensor float[batch_size, max_nnomdes, in_dim]
                    Input data
                condition : list of Tensor, containing
                    node_mask : Tensor int[batch_size, max_nnomdes, 1]
                                1: node; otherwise 0

                    nodes : Tensor float[batch_size, max_nnomdes, ndim]
                            nodal coordinate; padding with 0

                    node_weights  : Tensor float[batch_size, max_nnomdes, nmeasures]
                                    rho(x)dx used for nmeasures integrations; padding with 0

                    directed_edges : Tensor int[batch_size, max_nedges, 2]
                                     direted edge pairs; padding with 0
                                     gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)]

                    edge_gradient_weights      : Tensor float[batch_size, max_nedges, ndim]
                                                 pinvdx on each directed edge
                                                 gradient f(x) = sum_i pinvdx[:,i] * [f(xi) - f(x)]


            Returns:
                G(x) : Tensor float[batch_size, max_nnomdes, out_dim]
                       Input data

        """
        x = x.permute(0, 2, 1)
        length = len(self.ws)

        # nodes: float[batch_size, nnodes, ndims]
        node_mask, nodes, node_weights, directed_edges, edge_gradient_weights = (
            condition["node_mask"],
            condition["nodes"],
            condition["node_weights"],
            condition["directed_edges"],
            condition["edge_gradient_weights"],
        )
        # bases: float[batch_size, nnodes, nmodes]
        bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, self.modes * self.sp_L)
        # node_weights: float[batch_size, nnodes, nmeasures]
        # wbases: float[batch_size, nnodes, nmodes, nmeasures]
        # set nodes with zero measure to 0
        wbases_c = torch.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = torch.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = torch.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, x.shape[1], 1)
        params = condition["params"].unsqueeze(-2).repeat(1, x.shape[1], 1)
        x = torch.concatenate([t, x, condition["x"], params], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)
            x2 = w(x)
            x3 = gw(
                self.softsign(
                    compute_gradient(x, directed_edges, edge_gradient_weights)
                )
            )
            x = x1 + x2 + x3
            if self.act is not None and i != length - 1:
                x = self.act(x)

        x = x.permute(0, 2, 1)

        if self.fc_dim > 0:
            x = self.fc1(x)
            if self.act is not None:
                x = self.act(x)

        x = self.fc2(x)

        x = x.permute(0, 2, 1)

        return x
