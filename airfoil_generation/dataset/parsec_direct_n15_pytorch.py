from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor
from torch import nn

import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
from scipy.special import factorial


# ------------------------------------------------------------------
#                         helpers
# ------------------------------------------------------------------
def _to_tensor(x, *, dtype=torch.float64, device=None) -> Tensor:
    """
    Take `Tensor | ndarray | list`  ➜  torch.Tensor(dtype,float64)

    The layer works in float64 to match NumPy’s default
    precision and to make the comparison tests easier.
    """
    if isinstance(x, Tensor):
        out = x.clone().detach()
    else:
        out = torch.as_tensor(x)
    return out.to(device=device, dtype=dtype)


def _binom_coeff(n: int, r: Tensor) -> Tensor:
    """
    Compute `n choose r` in a numerically stable way by using
    log-gamma to avoid over/underflow for large *n*.
    """
    n = torch.as_tensor(float(n), dtype=r.dtype, device=r.device)
    return torch.exp(
        torch.lgamma(n + 1) - torch.lgamma(r + 1) - torch.lgamma(n - r + 1)
    )


# ------------------------------------------------------------------
#                       CSTLayer (Torch)
# ------------------------------------------------------------------
class CSTLayerTorch(nn.Module):
    """
    Torch version of the original *CSTLayer*.

    • Every constant is stored as a *buffer* so it is moved
      together with `.to(device)` or `.cuda()`.
    • Methods returning matrices now give back **torch.Tensor**.
    """

    def __init__(
        self,
        x_coords: np.ndarray | list | Tensor | None = None,
        n_cst: int = 12,
        n_x: int = 129,
        n1: float = 0.5,
        n2: float = 1.0,
        dtype=torch.float64,
        device=None,
    ):
        super().__init__()

        if x_coords is None:
            theta = torch.linspace(
                math.pi, 2 * math.pi, n_x, dtype=dtype, device=device
            )
            x_coords = (torch.cos(theta) + 1.0) / 2.0
        else:
            x_coords = _to_tensor(x_coords, dtype=dtype, device=device)
            n_x = int(x_coords.numel())

        # immutable hyper-parameters
        self.n_cst = n_cst
        self.n1 = float(n1)
        self.n2 = float(n2)
        self.n_x = n_x

        self.register_buffer("x_coords", x_coords)

        # pre-compute A0 because it never changes
        self.register_buffer("A0_const", self._build_A0())

    # ------------------------------------------------------------------
    #                       matrix builders
    # ------------------------------------------------------------------
    def _build_A0(self) -> Tensor:
        """Return the *A0* matrix  (n_x × (n+1))."""
        n, n1, n2, x = self.n_cst, self.n1, self.n2, self.x_coords
        r = torch.arange(0, n + 1, dtype=x.dtype, device=x.device)
        k = _binom_coeff(n, r)  # shape: (n+1,)

        A0 = (
            k  # broadcasted (n+1,)
            * x.unsqueeze(1) ** (n1 + r)  # (n_x, n+1)
            * (1 - x).unsqueeze(1) ** (n + n2 - r)
        )
        return A0  # shape (n_x, n+1)

    def _build_derivative(self) -> Tuple[Tensor, Tensor]:
        """
        See original implementation.
        Returns:
            A1, A2  (with the leading/trailing x-points removed)
        """
        n, n1, n2 = self.n_cst, self.n1, self.n2
        x = self.x_coords[1:-1]  # (n_x-2,)
        r = torch.arange(0, n + 1, dtype=x.dtype, device=x.device)
        k = _binom_coeff(n, r)  # (n+1,)

        # broadcasting helpers
        x1 = x.unsqueeze(1)  # (n_x-2,1)
        x2 = (1 - x).unsqueeze(1)  # (n_x-2,1)

        # same closed-form expressions as the NumPy version
        A1 = k * (
            -(x1 ** (n1 + r - 1))
            * x2 ** (n + n2 - r - 1)
            * (x1 * (n + n2 - r) + (n1 + r) * (x1 - 1))
        )
        A2 = (
            k
            * x1 ** (n1 + r - 2)
            * x2 ** (n + n2 - r - 2)
            * (
                x1**2 * (-n + n2**2 + 2 * n2 * (n - r) - n2 + r + (n - r) ** 2)
                + 2 * x1 * (x1 - 1) * (n1 * n2 + n1 * (n - r) + n2 * r + r * (n - r))
                + (x1 - 1) ** 2 * (n1**2 + 2 * n1 * r - n1 + r**2 - r)
            )
        )
        return A1, A2  # each (n_x-2, n+1)

    # ------------------------------------------------------------------
    #                       public  API
    # ------------------------------------------------------------------
    def A0_matrix(self) -> Tensor:
        return self.A0_const.clone()

    def derivative_matrix(self) -> Tuple[Tensor, Tensor]:
        return tuple(t.clone() for t in self._build_derivative())

    # ------------------------------ fitting ---------------------------
    def fit_CST(
        self,
        y_coords: Tensor | np.ndarray,
        n_x: int = 129,
    ) -> Tuple[Tensor, Tensor, float]:
        """
        Least-square fit of upper / lower coefficients and trailing edge.
        Fully differentiable **except** for the explicit `lstsq` call that
        is run under `torch.no_grad()` (same as NumPy version).
        """
        y_coords = _to_tensor(y_coords, device=self.x_coords.device)

        # split upper & lower side
        yu = y_coords[:n_x].flip(0)  # (n_x,)
        yl = y_coords[n_x - 1 :]
        te = (yu[-1] - yl[-1]) / 2.0

        # build target rhs
        rhs_u = yu - self.x_coords * yu[-1]
        rhs_l = yl - self.x_coords * yl[-1]

        # PyTorch ≥ 1.12: torch.linalg.lstsq  ➜  very close to np.linalg.lstsq
        au = torch.linalg.lstsq(self.A0_const, rhs_u).solution
        al = torch.linalg.lstsq(self.A0_const, rhs_l).solution
        return au, al, float(te)


# ------------------------------------------------------------------
#                Fit_airfoil   (Torch rewrite)
# ------------------------------------------------------------------
class FitAirfoilTorch:
    """
    A literal translation of the NumPy/SciPy *Fit_airfoil* class:
    * The internal heavy lifting (CST) is performed in Torch.
    * Optimisation (`scipy.optimize.minimize`) is kept unchanged
      for exact reproducibility; the objective uses NumPy.
    """

    def __init__(self, data: np.ndarray | Tensor):
        # keep the raw data in NumPy so that SciPy works
        if isinstance(data, Tensor):
            data = data.cpu().numpy()
        self.data = np.asarray(data, dtype=np.float64)
        self.parsec_features = self.get_parsec_n15()

    # --------------------------------------------------------------
    def get_parsec_n15(self) -> np.ndarray:
        data = self.data
        x, y = data[:, 0], data[:, 1]

        a = (data[0, -1] - data[9, -1]) / (data[0, 0] - data[9, 0])
        theta_degrees = math.degrees(math.atan(a))
        angle = theta_degrees

        # ---------------- CST fit (Torch) --------------------------
        cst = CSTLayerTorch(n_cst=12, x_coords=x[:129][::-1])
        au, al, te = cst.fit_CST(torch.from_numpy(y), n_x=129)

        # Sample the fitted curves on a fine grid
        x2 = np.arange(0.0, 1.0001, 0.0001, dtype=np.float64)
        cst_fine = CSTLayerTorch(n_cst=12, x_coords=x2)
        A0_fine = cst_fine.A0_matrix().cpu().numpy()

        yu = (A0_fine @ au.cpu().numpy()) + cst_fine.x_coords.cpu().numpy() * te
        yl = (A0_fine @ al.cpu().numpy()) - cst_fine.x_coords.cpu().numpy() * te

        # thickness values
        t4u, t25u, t60u = yu[400], yu[2500], yu[6000]
        t4l, t25l, t60l = yl[400], yl[2500], yl[6000]
        te1 = te

        # extreme points & reflex
        yumax, ylmax = yu.max(), yl.min()
        xumax, xlmax = x2[yu.argmax()], x2[yl.argmin()]
        yr = yl.max()
        xr = x2[yl.argmax()]

        # leading-edge radius via circle fit (SciPy)
        pts = data[126:131]  # (5,2)
        xdata, ydata = pts[:, 0], pts[:, 1]
        initial_guess = (0.0025, 0.0, np.std([xdata, ydata]))
        res = minimize(
            FitAirfoilTorch._objective,
            initial_guess,
            args=(xdata, ydata),
            method="Nelder-Mead",
        )
        rf, *_ = res.x  # leading-edge radius

        return np.asarray(
            [
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
            ],
            dtype=np.float64,
        )

    # static --------------------------------------------------------
    @staticmethod
    def _objective(params, x, y):
        xc, yc, r = params
        return np.sum((np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r) ** 2)
