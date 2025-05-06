import numpy as np
from scipy.interpolate import splev, splprep
from scipy import optimize
from scipy.stats import linregress


import torch


class Fit_airfoil_11:
    """
    Fit airfoil by 3 order Bspline and extract Parsec features.
    airfoil (npoints,2)
    """

    def __init__(self, iLE=128):
        self.iLE = iLE

    def compute_parsec_features(self, airfoil):

        self.tck, self.u = splprep(airfoil.T, s=0)
        rle = self.get_rle()
        xup, yup, yxxup = self.get_up()
        xlo, ylo, yxxlo = self.get_lo()
        yteup = airfoil[0, 1]
        ytelo = airfoil[-1, 1]
        alphate, betate = self.get_te_angle(airfoil)

        self.parsec_features = np.array(
            [
                rle,
                xup,
                yup,
                yxxup,
                xlo,
                ylo,
                yxxlo,
                (yteup + ytelo) / 2,
                yteup - ytelo,
                alphate,
                betate,
            ]
        )

        return self.parsec_features

    def get_rle(self):
        uLE = self.u[self.iLE]
        xu, yu = splev(uLE, self.tck, der=1)  # dx/du
        xuu, yuu = splev(uLE, self.tck, der=2)  # ddx/du^2
        K = abs(xu * yuu - xuu * yu) / (xu**2 + yu**2) ** 1.5  # curvature
        return 1 / K

    def get_up(self):
        def f(u_tmp):
            x_tmp, y_tmp = splev(u_tmp, self.tck)
            return -y_tmp

        res = optimize.minimize_scalar(
            f, bounds=(0, self.u[self.iLE]), method="bounded"
        )
        uup = res.x
        xup, yup = splev(uup, self.tck)

        xu, yu = splev(uup, self.tck, der=1)  # dx/du
        xuu, yuu = splev(uup, self.tck, der=2)  # ddx/du^2
        # yx = yu/xu
        yxx = (yuu * xu - xuu * yu) / xu**3
        return xup, yup, yxx

    def get_lo(self):
        def f(u_tmp):
            x_tmp, y_tmp = splev(u_tmp, self.tck)
            return y_tmp

        res = optimize.minimize_scalar(
            f, bounds=(self.u[self.iLE], 1), method="bounded"
        )
        ulo = res.x
        xlo, ylo = splev(ulo, self.tck)

        xu, yu = splev(ulo, self.tck, der=1)  # dx/du
        xuu, yuu = splev(ulo, self.tck, der=2)  # ddx/du^2
        # yx = yu/xu
        yxx = (yuu * xu - xuu * yu) / xu**3
        return xlo, ylo, yxx

    def get_te_angle(self, airfoil):

        n = int(0.02 * airfoil.shape[0])
        k1 = linregress(airfoil[:n, 0], airfoil[:n, 1])[0]
        k2 = linregress(airfoil[-n:, 0], airfoil[-n:, 1])[0]
        alphate = np.arctan(k1)
        betate = np.arctan(k2)
        return alphate, betate


class Fit_airfoil_11_Torch:
    """
    Fit airfoil by 3rd‐order B‐spline (SciPy) and extract PARSEC features
    using PyTorch for all subsequent algebra and “minimize_scalar” steps.
    Input airfoil should be a torch.Tensor of shape (n_points, 2).
    """

    def __init__(
        self,
        iLE: int = 128,
        inner_steps: int = 1000,
        lr: float = 0.01,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float64,
    ):
        self.iLE = iLE
        self.inner_steps = inner_steps
        self.lr = lr
        self.device = device
        self.dtype = dtype

    def compute_parsec_features(self, airfoil: torch.Tensor) -> torch.Tensor:
        # 1) fit B‐spline once with SciPy
        af_np = airfoil.detach().cpu().numpy()
        # ensure shape (2, N)
        tck, u = splprep(af_np.T, s=0, k=3)
        self.tck = tck
        self.u = u

        # 2) compute all features
        rle = self.get_rle()
        xup, yup, yxxup = self.get_up()
        xlo, ylo, yxxlo = self.get_lo()
        # trailing‐edge y coords:
        yte_up = af_np[0, 1]
        yte_lo = af_np[-1, 1]
        alphate, betate = self.get_te_angles(airfoil)

        feats = torch.stack(
            [
                rle,
                xup,
                yup,
                yxxup,
                xlo,
                ylo,
                yxxlo,
                torch.tensor(
                    (yte_up + yte_lo) / 2.0, device=self.device, dtype=self.dtype
                ),
                torch.tensor(yte_up - yte_lo, device=self.device, dtype=self.dtype),
                alphate,
                betate,
            ]
        )
        return feats

    def get_rle(self) -> torch.Tensor:
        # radius of leading edge = 1/curvature at uLE
        uLE = float(self.u[self.iLE])
        dx1, dy1 = splev(uLE, self.tck, der=1)
        dx2, dy2 = splev(uLE, self.tck, der=2)
        dx1_t = torch.tensor(dx1, device=self.device, dtype=self.dtype)
        dy1_t = torch.tensor(dy1, device=self.device, dtype=self.dtype)
        dx2_t = torch.tensor(dx2, device=self.device, dtype=self.dtype)
        dy2_t = torch.tensor(dy2, device=self.device, dtype=self.dtype)
        K = torch.abs(dx1_t * dy2_t - dx2_t * dy1_t) / (
            dx1_t.pow(2) + dy1_t.pow(2)
        ).pow(1.5)
        return 1.0 / K

    def get_up(self):
        # minimize f(u) = - y(u) over [0, uLE]
        uLE = float(self.u[self.iLE])
        # start halfway
        u_t = torch.tensor(
            uLE * 0.5, device=self.device, dtype=self.dtype, requires_grad=False
        )
        for _ in range(self.inner_steps):
            # evaluate y and y'
            y = float(splev(u_t.item(), self.tck)[1])
            dy = float(splev(u_t.item(), self.tck, der=1)[1])
            # f' = - dy/du
            grad = -dy
            u_t = u_t - self.lr * grad
            # clamp to domain [0, uLE]
            u_t = torch.clamp(u_t, 0.0, uLE)

        u_opt = u_t.item()
        xup, yup = splev(u_opt, self.tck)
        dx1, dy1 = splev(u_opt, self.tck, der=1)
        dx2, dy2 = splev(u_opt, self.tck, der=2)
        # y'' wrt x: (d²y/du² dx/du - d²x/du² dy/du) / (dx/du)^3
        yxx = (dy2 * dx1 - dx2 * dy1) / (dx1**3)

        return (
            torch.tensor(xup, device=self.device, dtype=self.dtype),
            torch.tensor(yup, device=self.device, dtype=self.dtype),
            torch.tensor(yxx, device=self.device, dtype=self.dtype),
        )

    def get_lo(self):
        # minimize f(u) = + y(u) over [uLE, 1]
        uLE = float(self.u[self.iLE])
        u_t = torch.tensor(
            (1.0 + uLE) * 0.5, device=self.device, dtype=self.dtype, requires_grad=False
        )
        for _ in range(self.inner_steps):
            y = float(splev(u_t.item(), self.tck)[1])
            dy = float(splev(u_t.item(), self.tck, der=1)[1])
            # f' = + dy/du
            grad = +dy
            u_t = u_t - self.lr * grad
            u_t = torch.clamp(u_t, uLE, 1.0)

        u_opt = u_t.item()
        xlo, ylo = splev(u_opt, self.tck)
        dx1, dy1 = splev(u_opt, self.tck, der=1)
        dx2, dy2 = splev(u_opt, self.tck, der=2)
        yxx = (dy2 * dx1 - dx2 * dy1) / (dx1**3)

        return (
            torch.tensor(xlo, device=self.device, dtype=self.dtype),
            torch.tensor(ylo, device=self.device, dtype=self.dtype),
            torch.tensor(yxx, device=self.device, dtype=self.dtype),
        )

    def get_te_angles(self, airfoil: torch.Tensor):
        # linear‐fit slope on first/last 2% of the points
        N = airfoil.shape[0]
        n = max(2, int(0.02 * N))
        x1 = airfoil[:n, 0]
        y1 = airfoil[:n, 1]
        x2 = airfoil[-n:, 0]
        y2 = airfoil[-n:, 1]

        # slope = cov(x,y)/var(x)
        def slope(x, y):
            xm = x.mean()
            ym = y.mean()
            cov = ((x - xm) * (y - ym)).sum()
            var = ((x - xm) ** 2).sum()
            return cov / var

        k1 = slope(x1, y1)
        k2 = slope(x2, y2)
        alpha = torch.atan(k1)
        beta = torch.atan(k2)
        return alpha.to(self.device), beta.to(self.device)


# --- example usage ---
if __name__ == "__main__":
    # suppose we have some airfoil (n,2) as a torch.Tensor
    import matplotlib.pyplot as plt

    # build a dummy airfoil: a NACA‐like shape
    x = np.linspace(0, 1, 200)
    yt = 0.1 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )
    xp = np.concatenate([x, x[::-1]])
    yp = np.concatenate([yt, -yt[::-1]])
    foil = torch.from_numpy(np.stack([xp, yp], axis=1)).float()

    fitter = Fit_airfoil_11_Torch(
        iLE=100,
        inner_steps=2000,
        lr=0.005,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    pf = fitter.compute_parsec_features(foil)
    print("PARSEC features:", pf)
