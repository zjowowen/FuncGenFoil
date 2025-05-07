import torch
import numpy as np

####################
# pytorch version
####################


def compute_triangle_area_torch(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the triangle area from 3 points in 2D or 3D using PyTorch.
    points shape: (3, ndims)  where ndims = 2 or 3.
    """
    # If points are 2D, pad them to 3D for cross product
    if points.size(1) == 2:
        # Pad z=0 to each point
        zeros = torch.zeros(points.size(0), 1, device=points.device)
        points_3d = torch.cat([points, zeros], dim=1)
    else:
        points_3d = points

    ab = points_3d[1] - points_3d[0]
    ac = points_3d[2] - points_3d[0]
    cross_product = torch.cross(ab, ac, dim=0)
    return 0.5 * torch.norm(cross_product, p=2)


def compute_tetrahedron_volume_torch(points: torch.Tensor) -> torch.Tensor:
    """
    Compute the tetrahedron volume from 4 points in 3D using PyTorch.
    points shape: (4, 3).
    """
    ab = points[1] - points[0]
    ac = points[2] - points[0]
    ad = points[3] - points[0]
    # scalar triple product
    cross_ab_ac = torch.cross(ab, ac, dim=0)
    scalar_triple = torch.dot(cross_ab_ac, ad)
    volume = torch.abs(scalar_triple) / 6.0
    return volume


def compute_measure_per_elem_torch(points: torch.Tensor, elem_dim: int) -> torch.Tensor:
    """
    Compute the measure (length, area, or volume) of an element given its points.
      - 2 points -> length
      - 3 points -> triangle area
      - 4 points -> either polygon area (split into 2 triangles if elem_dim=2)
                    or tetrahedron volume if elem_dim=3
    """
    npoints, ndims = points.size()
    if npoints == 2:
        # length
        s = torch.norm(points[0] - points[1], p=2)
    elif npoints == 3:
        # triangle area
        s = compute_triangle_area_torch(points)
    elif npoints == 4:
        if elem_dim == 2:
            # treat first 3 points as one triangle, last 3 points as another
            # e.g. points[:3,:] and points[1:,:]
            s = compute_triangle_area_torch(
                points[:3, :]
            ) + compute_triangle_area_torch(points[1:, :])
        elif elem_dim == 3:
            # tetrahedron volume
            s = compute_tetrahedron_volume_torch(points)
        else:
            raise ValueError(f"elem_dim {elem_dim} is not recognized")
    else:
        raise ValueError(f"npoints {npoints} is not recognized")
    return s


def compute_node_measures_torch(
    nodes: torch.Tensor, elems: torch.Tensor
) -> torch.Tensor:
    """
    Compute node measures (length, area, volume, depending on the element dimension)
    and distribute them equally among the connected nodes.

    Parameters:
      nodes: Float tensor with shape (nnodes, ndims).
      elems: Long tensor with shape (nelems, max_num_of_nodes_per_elem+1).
             The first entry in each row is elem_dim,
             the rest are the node indices (e.g. shape = [elem_dim, n0, n1, ..., -1, ...]).

    Returns:
      measures: Float tensor with shape (nnodes, M)
                where M <= ndims is the number of distinct measure types encountered.
                E.g., if there are line (1D) and triangle (2D) elements, M could be 2.
    """
    nnodes, ndims = nodes.size()
    # We will store measures in shape (nnodes, ndims),
    # but only some columns may actually be used (depending on which dimensions appear).
    measures = torch.full((nnodes, ndims), float("nan"), device=nodes.device)
    measure_types = [False] * ndims  # track which dimensions appear

    for elem in elems:
        elem_dim = elem[0].item()
        # gather valid node indices (filter out any negative padding)
        valid_node_indices = elem[1:]
        valid_node_indices = valid_node_indices[valid_node_indices >= 0]

        # compute measure for this element
        points_elem = nodes[valid_node_indices, :]
        s = compute_measure_per_elem_torch(points_elem, elem_dim)

        ne = valid_node_indices.size(0)

        # If there's nothing stored yet in measures[:, elem_dim-1], put 0 in place of NaN
        measures[valid_node_indices, elem_dim - 1] = torch.nan_to_num(
            measures[valid_node_indices, elem_dim - 1], nan=0.0
        )
        # distribute the measure equally to each connected node
        measures[valid_node_indices, elem_dim - 1] += s / ne

        measure_types[elem_dim - 1] = True

    # Return only the columns that were used
    measure_types_t = torch.tensor(measure_types, device=nodes.device)
    # measure_types_t is e.g. [False, True, True], indexing we do by nonzero
    used_indices = measure_types_t.nonzero().squeeze(-1)
    if used_indices.numel() == 0:
        # No measures found
        return torch.full((nnodes, 0), float("nan"), device=nodes.device)
    else:
        return measures[:, used_indices]


def convert_structured_data_torch(
    coords_list, features, nnodes_per_elem=3, feature_include_coords=True, device="cpu"
):
    """
    Convert structured 2D data to unstructured data.

    Parameters:
        coords_list : list of length 2
            [coordx, coordy] where
            coordx, coordy: torch.FloatTensor of shape (ndata, nx, ny)

        features : torch.FloatTensor
            shape (ndata, nx, ny, nfeatures)

        nnodes_per_elem : int
            Number of nodes per element (3 for triangle, 4 for quad)

        feature_include_coords : bool
            Whether the x,y coordinates should be appended to the feature channels

        device : str
            "cuda" or "cpu". The returned data will be on this device.

    Returns:
        nodes_list : list of length ndata
            Each entry is a (nnodes, 2) FloatTensor of node coordinates.

        elems_list : list of length ndata
            Each entry is a (nelems, nnodes_per_elem+1) LongTensor of element connectivity.
            The first column is elem_dim = 2 (for 2D).
            The rest are node indices. Negative entries are padding if needed.

        features_list : list of length ndata
            Each entry is a (nnodes, nfeatures_new) FloatTensor of node features,
            where nfeatures_new = (nfeatures + 2) if feature_include_coords else nfeatures.
    """
    print("convert_structured_data so far only supports 2D problems")
    ndims = len(coords_list)
    assert ndims == 2, "This function only supports 2D coords_list."

    # coords_list = [coordx, coordy]
    coordx, coordy = coords_list
    # Move coords & features to the specified device
    coordx = coordx.to(device)
    coordy = coordy.to(device)
    features = features.to(device)

    # (ndata, nx, ny)
    ndata, nx, ny = coordx.shape
    nnodes = nx * ny

    # For triangular elements: nelems = 2 * (nx-1)*(ny-1)
    # For quadrilateral elements: nelems = (nx-1)*(ny-1)
    # In your code, there's a formula: (nx-1)*(ny-1)*(5 - nnodes_per_elem)
    # - If nnodes_per_elem=3, that factor = 2
    # - If nnodes_per_elem=4, that factor = 1
    nelems = (nx - 1) * (ny - 1) * (5 - nnodes_per_elem)

    # Reshape coords into (ndata, nnodes). Then stack along dim=2 → shape (ndata, nnodes, 2).
    coordx_flat = coordx.reshape(ndata, nnodes)
    coordy_flat = coordy.reshape(ndata, nnodes)
    nodes = torch.stack([coordx_flat, coordy_flat], dim=2)  # (ndata, nnodes, 2)

    # Reshape features to (ndata, nnodes, nfeatures)
    # Optionally append the x,y coords as additional features
    nfeatures = features.shape[-1]
    features_2d = features.reshape(ndata, nnodes, nfeatures)
    if feature_include_coords:
        # Append the coords (which has shape (ndata, nnodes, 2)) to the features
        features_2d = torch.cat([features_2d, nodes], dim=-1)
        nfeatures += 2

    # Build the local connectivity for a single 2D slice
    # shape → (nelems, nnodes_per_elem + 1)
    # We'll store elem_dim=2 in the first column
    elems_2d = torch.zeros(
        (nelems, nnodes_per_elem + 1), dtype=torch.long, device=device
    )
    # Fill in the connectivity
    idx = 0
    for i in range(nx - 1):
        for j in range(ny - 1):
            if nnodes_per_elem == 4:
                # The single quad element
                # Format: (elem_dim, n0, n1, n2, n3)
                # n0 = i*ny + j
                # n1 = i*ny + (j+1)
                # n2 = (i+1)*ny + (j+1)
                # n3 = (i+1)*ny + j
                elems_2d[idx, :] = torch.tensor(
                    [
                        2,
                        i * ny + j,
                        i * ny + j + 1,
                        (i + 1) * ny + j + 1,
                        (i + 1) * ny + j,
                    ],
                    device=device,
                )
                idx += 1
            else:
                # Triangle mesh
                # We produce 2 triangles per cell
                #  First triangle: (2, n0, n1, n2)
                #  Second triangle: (2, n0, n2, n3)
                n0 = i * ny + j
                n1 = i * ny + j + 1
                n2 = (i + 1) * ny + j + 1
                n3 = (i + 1) * ny + j

                elems_2d[idx, :] = torch.tensor([2, n0, n1, n2], device=device)
                elems_2d[idx + 1, :] = torch.tensor([2, n0, n2, n3], device=device)
                idx += 2

    # Now replicate this connectivity for each of the ndata slices
    # shape → (ndata, nelems, nnodes_per_elem + 1)
    # We can do something like unsqueeze the first dimension and repeat:
    elems_2d_expanded = elems_2d.unsqueeze(0).repeat(ndata, 1, 1)

    # Finally, split into lists
    # nodes_list[i]   : shape (nnodes, 2)
    # elems_list[i]   : shape (nelems, nnodes_per_elem+1)
    # features_list[i]: shape (nnodes, nfeatures)
    nodes_list = [nodes[i] for i in range(ndata)]
    elems_list = [elems_2d_expanded[i] for i in range(ndata)]
    features_list = [features_2d[i] for i in range(ndata)]

    return nodes_list, elems_list, features_list


####################
# numpy version
####################


def compute_triangle_area_(points):
    ab = points[1, :] - points[0, :]
    ac = points[2, :] - points[0, :]
    cross_product = np.cross(ab, ac)
    return 0.5 * np.linalg.norm(cross_product)


def compute_tetrahedron_volume_(points):
    ab = points[1, :] - points[0, :]
    ac = points[2, :] - points[0, :]
    ad = points[3, :] - points[0, :]
    # Calculate the scalar triple product
    volume = abs(np.dot(np.cross(ab, ac), ad)) / 6
    return volume


def compute_measure_per_elem_(points, elem_dim):
    """
    Compute element measure (length, area or volume)
    for 2-point  element, compute its length
    for 3-point  element, compute its area
    for 4-point  element, compute its area if elem_dim=2; compute its volume if elem_dim=3
    equally assign it to its nodes

        Parameters:
            points : float[npoints, ndims]
            elem_dim : int

        Returns:
            s : float
    """

    npoints, ndims = points.shape
    if npoints == 2:
        s = np.linalg.norm(points[0, :] - points[1, :])
    elif npoints == 3:
        s = compute_triangle_area_(points)
    elif npoints == 4:
        assert npoints == 3 or npoints == 4
        if elem_dim == 2:
            s = compute_triangle_area_(points[:3, :]) + compute_triangle_area_(
                points[1:, :]
            )
        elif elem_dim == 3:
            s = compute_tetrahedron_volume_(points)
        else:
            raise ValueError("elem dim ", elem_dim, "is not recognized")
    else:
        raise ValueError("npoints ", npoints, "is not recognized")
    return s


def compute_node_measures(nodes, elems):
    """
    Compute node measures  (separate length, area and volume ... for each node),
    For each element, compute its length, area or volume s,
    equally assign it to its ne nodes (measures[:] += s/ne).

        Parameters:
            nodes : float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1].
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.

        Return :
            measures : float[nnodes, nmeasures]
                       padding NaN for nodes that do not have measures
                       nmeasures >= 1: number of measures with different dimensionalities
                       For example, if there are both lines and triangles, nmeasures = 2

    """
    nnodes, ndims = nodes.shape
    measures = np.full((nnodes, ndims), np.nan)
    measure_types = [False] * ndims
    for elem in elems:
        elem_dim, e = elem[0], elem[1:]
        e = e[e >= 0]
        ne = len(e)
        # compute measure based on elem_dim
        s = compute_measure_per_elem_(nodes[e, :], elem_dim)
        # assign it to cooresponding measures
        measures[e, elem_dim - 1] = np.nan_to_num(measures[e, elem_dim - 1], nan=0.0)
        measures[e, elem_dim - 1] += s / ne
        measure_types[elem_dim - 1] = True

    # return only nonzero measures
    return measures[:, measure_types]


def convert_structured_data(
    coords_list, features, nnodes_per_elem=3, feature_include_coords=True
):
    """
    Convert structured data, to unstructured data
                    ny-1                                                                  ny-1   2ny-1
                    ny-2                                                                  ny-2    .
                    .                                                                       .     .
    y direction     .          nodes are ordered from left to right/bottom to top           .     .
                    .                                                                       .     .
                    1                                                                       1     ny+1
                    0                                                                       0     ny
                        0 - 1 - 2 - ... - nx-1   (x direction)

        Parameters:
            coords_list            :  list of ndims float[nnodes, nx, ny], for each dimension
            features               :  float[nelems, nx, ny, nfeatures]
            nnodes_per_elem        :  int, nnodes_per_elem = 3: triangle mesh; nnodes_per_elem = 4: quad mesh
            feature_include_coords :  boolean, whether treating coordinates as features, if coordinates
                                      are treated as features, they are concatenated at the end

        Return :
            nodes_list :     list of float[nnodes, ndims]
            elems : int[nelems, max_num_of_nodes_per_elem+1].
                    The first entry is elem_dim, the dimensionality of the element.
                    The elems array can have some padding numbers, for example, when
                    we have both line segments and triangles, the padding values are
                    -1 or any negative integers.
            features_list  : list of float[nnodes, nfeatures]
    """
    print("convert_structured_data so far only supports 2d problems")
    ndims = len(coords_list)
    assert ndims == 2
    coordx, coordy = coords_list
    ndata, nx, ny = coords_list[0].shape
    nnodes, nelems = nx * ny, (nx - 1) * (ny - 1) * (5 - nnodes_per_elem)
    nodes = np.stack(
        (coordx.reshape((ndata, nnodes)), coordy.reshape((ndata, nnodes))), axis=2
    )
    if feature_include_coords:
        nfeatures = features.shape[-1] + ndims
        features = np.concatenate(
            (features.reshape((ndata, nnodes, -1)), nodes), axis=-1
        )
    else:
        nfeatures = features.shape[-1]
        features = features.reshape((ndata, nnodes, -1))

    elems = np.zeros((nelems, nnodes_per_elem + 1), dtype=int)
    for i in range(nx - 1):
        for j in range(ny - 1):
            ie = i * (ny - 1) + j
            if nnodes_per_elem == 4:
                elems[ie, :] = (
                    2,
                    i * ny + j,
                    i * ny + j + 1,
                    (i + 1) * ny + j + 1,
                    (i + 1) * ny + j,
                )
            else:
                elems[2 * ie, :] = 2, i * ny + j, i * ny + j + 1, (i + 1) * ny + j + 1
                elems[2 * ie + 1, :] = (
                    2,
                    i * ny + j,
                    (i + 1) * ny + j + 1,
                    (i + 1) * ny + j,
                )

    elems = np.tile(elems, (ndata, 1, 1))

    nodes_list = [nodes[i, ...] for i in range(ndata)]
    elems_list = [elems[i, ...] for i in range(ndata)]
    features_list = [features[i, ...] for i in range(ndata)]

    return nodes_list, elems_list, features_list
