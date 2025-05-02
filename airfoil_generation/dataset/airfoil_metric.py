import numpy as np
import torch
import torch.nn as nn


def torch_factorial(x):
    # x should be a tensor (can be int or float)
    return torch.exp(torch.lgamma(x + 1))


def calculate_airfoil_metric_n15(x, y, n_inner_steps=2000):
    """
    Overview:
        Get parsec features from airfoil data.
    Arguments:
        x: airfoil data x(npoints,)
        y: airfoil data y(npoints,)

    Returns:
        parsec_features: parsec features (nfeatures,), which include:
            - rf: Leading Edge Radius,
            - t4u: 4% Chord Upper Surface Thickness,
            - t4l: 4% Chord Lower Surface Thickness,
            - xumax: Upper Surface Maximum Thickness X Coordinate,
            - yumax: Upper Surface Maximum Thickness Y Coordinate,
            - xlmax: Lower Surface Maximum Thickness X Coordinate,
            - ylmax: Lower Surface Maximum Thickness Y Coordinate,
            - t25u: 25% Chord Upper Surface Thickness,
            - t25l: 25% Chord Lower Surface Thickness,
            - angle: Upper Surface Trailing Edge Angle,
            - te1: Trailing Edge Thickness,
            - xr: Trailing Edge Load X Coordinate,
            - yr: Trailing Edge Load Y Coordinate,
            - t60u: 60% Chord Upper Surface Thickness,
            - t60l: 60% Chord Lower Surface Thickness
    """

    a = (y[0] - y[9]) / (x[0] - x[9])
    theta_radians = torch.atan(a)
    theta_degrees = torch.rad2deg(theta_radians)
    angle = theta_degrees

    def fit_CST_parallel(x_coords, y_coords, n_cst=12, n1=0.5, n2=1.0):
        n_x = len(x_coords)
        n = n_cst

        # ------------------------------------------------------------------
        # 1.  Inputs
        # ------------------------------------------------------------------
        # n, n1, n2 : integers
        # x_coords  : 1-D tensor of shape (Nx,) already on the desired device
        # ------------------------------------------------------------------

        device = x_coords.device  # keep whatever device you are using
        dtype = x_coords.dtype  # keep your precision (float32/float64)

        # ------------------------------------------------------------------
        # 2.  Prepare r = 0 … n as one tensor
        # ------------------------------------------------------------------
        r = torch.arange(0, n + 1, device=device, dtype=dtype)  # (n+1,)

        # ------------------------------------------------------------------
        # 3.  Binomial coefficients k[r]  ( = n! / (r!(n−r)!) )
        #     lgamma is numerically stable and already runs on the GPU
        # ------------------------------------------------------------------
        lgamma_n_plus1 = torch.lgamma(torch.tensor(n + 1, device=device, dtype=dtype))
        k = torch.exp(
            lgamma_n_plus1
            - torch.lgamma(r + 1)  # r!
            - torch.lgamma(n - r + 1)  # (n-r)!
        )  # shape: (n+1,)

        # ------------------------------------------------------------------
        # 4.  Broadcast powers and coefficients
        # ------------------------------------------------------------------
        # Bring r to shape (n+1, 1) so it broadcasts against x_coords (1, Nx)
        r_col = r.unsqueeze(1)  # (n+1, 1)

        pow1 = n1 + r_col  # exponent for x_coords
        pow2 = (n + n2) - r_col  # exponent for (1 - x_coords)

        # k needs the same leading dimension, then broadcasts automatically
        k_col = k.unsqueeze(1)  # (n+1, 1)

        # ------------------------------------------------------------------
        # 5.  Final matrix A0
        # ------------------------------------------------------------------
        A0 = k_col * x_coords.pow(pow1) * (1 - x_coords).pow(pow2)  # (n+1, Nx)

        A0 = A0.T
        yu = y_coords[:n_x].flip(0)
        yl = y_coords[n_x - 1 :]
        te = (yu[-1] - yl[-1]) / 2
        au = torch.linalg.lstsq(A0, yu - x_coords * yu[-1], rcond=None)[0]
        al = torch.linalg.lstsq(A0, yl - x_coords * yl[-1], rcond=None)[0]
        return au, al, te

    au, al, te = fit_CST_parallel(x_coords=x[:129].flip(0), y_coords=y, n_cst=12)

    x2 = torch.arange(0, 1.001, 0.0001).to(x)

    def CST_base(x_coords, n_cst=12, n_x=129, n1=0.5, n2=1.0):

        n_x = len(x_coords)
        n = n_cst

        # ------------------------------------------------------------------
        # 1.  Inputs
        # ------------------------------------------------------------------
        # n, n1, n2 : integers
        # x_coords  : 1-D tensor of shape (Nx,) already on the desired device
        # ------------------------------------------------------------------

        device = x_coords.device  # keep whatever device you are using
        dtype = x_coords.dtype  # keep your precision (float32/float64)

        # ------------------------------------------------------------------
        # 2.  Prepare r = 0 … n as one tensor
        # ------------------------------------------------------------------
        r = torch.arange(0, n + 1, device=device, dtype=dtype)  # (n+1,)

        # ------------------------------------------------------------------
        # 3.  Binomial coefficients k[r]  ( = n! / (r!(n−r)!) )
        #     lgamma is numerically stable and already runs on the GPU
        # ------------------------------------------------------------------
        lgamma_n_plus1 = torch.lgamma(torch.tensor(n + 1, device=device, dtype=dtype))
        k = torch.exp(
            lgamma_n_plus1
            - torch.lgamma(r + 1)  # r!
            - torch.lgamma(n - r + 1)  # (n-r)!
        )  # shape: (n+1,)

        # ------------------------------------------------------------------
        # 4.  Broadcast powers and coefficients
        # ------------------------------------------------------------------
        # Bring r to shape (n+1, 1) so it broadcasts against x_coords (1, Nx)
        r_col = r.unsqueeze(1)  # (n+1, 1)

        pow1 = n1 + r_col  # exponent for x_coords
        pow2 = (n + n2) - r_col  # exponent for (1 - x_coords)

        # k needs the same leading dimension, then broadcasts automatically
        k_col = k.unsqueeze(1)  # (n+1, 1)

        # ------------------------------------------------------------------
        # 5.  Final matrix A0
        # ------------------------------------------------------------------
        A0 = k_col * x_coords.pow(pow1) * (1 - x_coords).pow(pow2)  # (n+1, Nx)

        return A0.T

    A0_fine = CST_base(x_coords=x2, n_cst=12)
    yu = A0_fine @ au + x2 * te
    yl = A0_fine @ al - x2 * te
    t4u = yu[400]
    t25u = yu[2500]
    t60u = yu[6000]
    t4l = yl[400]
    t25l = yl[2500]
    t60l = yl[6000]
    te1 = te

    yumax = yu.max()
    ylmax = yl.min()

    def soft_argmax(x, beta=100.0):  # beta = 1/temperature
        """
        Returns a pseudo-index in [0, len(x)-1] with gradients.
        For very large beta it approaches argmax.
        """
        *_, n = x.shape if x.ndim > 1 else (None, x.shape[0])
        w = torch.softmax(beta * x, dim=-1)  # shape (..., n)
        indices = torch.arange(n, device=x.device, dtype=x.dtype)
        return (w * indices).sum(dim=-1)  # pseudo-index

    # pseudo-index
    xumax_soft = soft_argmax(yu, beta=10000.0) / 10010.0  # differentiable
    xlmax_soft = soft_argmax(-yl, beta=10000.0) / 10010.0  # differentiable
    xr_soft = soft_argmax(yl, beta=10000.0) / 10010.0  # differentiable

    # xumax = x2[torch.argmax(yu)]
    xumax = xumax_soft
    # xlmax = x2[torch.argmin(yl)]
    xlmax = xlmax_soft
    yr = yl.max()
    # xr = x2[torch.argmax(yl)]
    xr = xr_soft

    xdata = x[126:131]
    ydata = y[126:131]
    points = torch.stack([xdata, ydata], dim=1)

    def minimize(x, y):
        xc = torch.ones(1, requires_grad=True).to(x) * 0.0025
        yc = torch.ones(1, requires_grad=True).to(x) * 0.0
        r = torch.ones(1, requires_grad=True).to(x) * torch.std(points).detach()
        inner_learning_rate = 0.05

        for _ in range(n_inner_steps):
            pred_r = torch.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            loss = torch.sum((pred_r - r) ** 2)

            if _ == n_inner_steps - 1:
                print(f"loss1: {loss.item()}")
            grad = torch.autograd.grad(loss, [xc, yc, r], create_graph=True)
            xc = xc - inner_learning_rate * grad[0]
            yc = yc - inner_learning_rate * grad[1]
            r = r - inner_learning_rate * grad[2]

        r = r[0]

        return xc, yc, r

    # enable gradient tracking
    with torch.enable_grad():

        xc_1, yc_1, rf = minimize(xdata, ydata)

    return (
        rf,
        t4u,
        t4l,
        xumax,
        yumax,
        xlmax,
        ylmax,
        t25u,
        t25l,
        angle,
        te1,
        xr,
        yr,
        t60u,
        t60l,
    )
