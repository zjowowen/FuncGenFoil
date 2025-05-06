import math
import numpy as np
from numpy.linalg import lstsq
from scipy.interpolate import splev, splprep
from scipy import optimize
from scipy.special import factorial
from scipy.stats import linregress


class Fit_airfoil_11:
    """
    Fit airfoil by 3 order Bspline and extract Parsec features.
    airfoil (npoints,2)
    """

    def __init__(self, airfoil, iLE=128):
        airfoil = airfoil[:: ((airfoil.shape[0] - 1) // 256)]
        self.iLE = int((airfoil.shape[0] - 1) / 2)
        self.tck, self.u = splprep(airfoil.T, s=0)

        # parsec features
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
        # xu,yu = splev(0, self.tck, der=1)
        # yx = yu/xu
        # alphate = np.arctan(yx)

        # xu,yu = splev(1, self.tck, der=1)
        # yx = yu/xu
        # betate = np.arctan(yx)
        # alphate = np.arctan((airfoil[0,1]-airfoil[1,1])/(airfoil[0,0]-airfoil[1,0]))
        # betate = np.arctan((airfoil[-1,1]-airfoil[-2,1])/(airfoil[-1,0]-airfoil[-2,0]))

        n = int(0.02 * airfoil.shape[0])
        k1 = linregress(airfoil[:n, 0], airfoil[:n, 1])[0]
        k2 = linregress(airfoil[-n:, 0], airfoil[-n:, 1])[0]
        alphate = np.arctan(k1)
        betate = np.arctan(k2)
        return alphate, betate
