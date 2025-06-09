import math
import numpy as np
from numpy.linalg import lstsq
from scipy.optimize import minimize
from scipy.special import factorial


class CSTLayer:
    def __init__(self, x_coords=None, n_cst=12, n_x=129, n1=0.5, n2=1.0):
        if x_coords is None:  # use n_x to generate x_coords
            """
            only work for same x coordinates for both side of airfoil
            airfoil points from upper TE ---> LE ---> lower TE
            """
            self.n_x = n_x
            theta = np.linspace(np.pi, 2 * np.pi, n_x)
            self.x_coords = (np.cos(theta) + 1.0) / 2
        else:
            self.n_x = len(x_coords)
            self.x_coords = x_coords

        self.n1 = n1
        self.n2 = n2
        self.n_cst = n_cst
        self.A0 = self.A0_matrix()

    def A0_matrix(self):
        """
        y = A0.T.dot(au) + 0.5 * te * x
        """
        n = self.n_cst
        n1 = self.n1
        n2 = self.n2
        n_x = self.n_x
        x = self.x_coords
        k = np.zeros(n + 1)
        A0 = np.zeros([n + 1, n_x])

        for r in range(n + 1):
            k[r] = factorial(n) / factorial(r) / factorial(n - r)
            A0[r, :] = k[r] * x ** (n1 + r) * (1 - x) ** (n + n2 - r)
        return A0.T

    def derivative_matrix(self):
        """
        y1 = A1.T.dot(au) + 0.5 * te
        y2 = A2.T.dot(au)
        K = (1+y1**2)**(3/2)/y2
        remove 0 and 1, derivates can be nan, use x_coords[1:-1] instead
        """
        n = self.n_cst
        n1 = self.n1
        n2 = self.n2
        n_x = self.n_x - 2
        x = self.x_coords[1:-1]
        k = np.zeros(n + 1)
        A1 = np.zeros([n + 1, n_x])
        A2 = np.zeros([n + 1, n_x])

        for r in range(n + 1):
            k[r] = factorial(n) / factorial(r) / factorial(n - r)
            A1[r, :] = k[r] * (
                -(x ** (n1 + r - 1))
                * (1 - x) ** (n + n2 - r - 1)
                * (x * (n + n2 - r) + (n1 + r) * (x - 1))
            )
            A2[r, :] = k[r] * (
                x ** (n1 + r - 2)
                * (1 - x) ** (n + n2 - r - 2)
                * (
                    x**2 * (-n + n2**2 + 2 * n2 * (n - r) - n2 + r + (n - r) ** 2)
                    + 2 * x * (x - 1) * (n1 * n2 + n1 * (n - r) + n2 * r + r * (n - r))
                    + (x - 1) ** 2 * (n1**2 + 2 * n1 * r - n1 + r**2 - r)
                )
            )
        return A1.T, A2.T

    def fit_CST(self, y_coords, n_x=129):
        A0 = self.A0_matrix()
        yu = y_coords[:n_x][::-1]
        yl = y_coords[n_x - 1 :]
        te = (yu[-1] - yl[-1]) / 2
        au = lstsq(A0, yu - self.x_coords * yu[-1], rcond=None)[0]
        al = lstsq(A0, yl - self.x_coords * yl[-1], rcond=None)[0]
        return au, al, te

    def fit_CST_up(self, y_coords, n_x=129):
        A0 = self.A0_matrix()
        yu = y_coords[:n_x][::-1]
        yl = y_coords[n_x - 1 :]
        te = (yu[-1] - yl[-1]) / 2
        au = lstsq(A0, yu - self.x_coords * yu[-1], rcond=None)[0]
        # al = lstsq(A0,yl-self.x_coords*yl[-1],rcond=None)[0]
        return au, te

    def fit_CST_low(self, y_coords, n_x=129):
        A0 = self.A0_matrix()
        yu = y_coords[:n_x][::-1]
        yl = y_coords[n_x - 1 :]
        te = (yu[-1] - yl[-1]) / 2
        # au = lstsq(A0,yu-self.x_coords*yu[-1],rcond=None)[0]
        al = lstsq(A0, yl - self.x_coords * yl[-1], rcond=None)[0]
        return al, te


class Fit_airfoil_15:
    """
    Overview:
        Fit airfoil by 3 order Bspline and extract Parsec features.
        airfoil (npoints,2)
    Interface:
        __init__, get_parsec_n15, objective
    """

    def __init__(self, data):
        self.data = data
        self.parsec_features = self.get_parsec_n15()

    def get_parsec_n15(self):
        """
        Overview:
            Get parsec features from airfoil data.
        Arguments:
            data: airfoil data (npoints,2)
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
        data = self.data
        x = data[:, 0]
        y = data[:, 1]

        a = (data[0, -1] - data[9, -1]) / (data[0, 0] - data[9, 0])
        theta_radians = math.atan(a)
        theta_degrees = math.degrees(theta_radians)
        angle = theta_degrees

        cst = CSTLayer(n_cst=12, x_coords=x[:129][::-1])
        au, al, te = cst.fit_CST(y, n_x=129)
        x2 = np.arange(0, 1.001, 0.0001)
        cst2 = CSTLayer(n_cst=12, x_coords=x2)
        yu = cst2.A0.dot(au) + cst2.x_coords * te
        yl = cst2.A0.dot(al) - cst2.x_coords * te
        t4u = yu[400]
        t25u = yu[2500]
        t60u = yu[6000]
        t4l = yl[400]
        t25l = yl[2500]
        t60l = yl[6000]
        te1 = te

        yumax = yu.max()
        ylmax = yl.min()
        xumax = x2[np.argmax(yu)]
        xlmax = x2[np.argmin(yl)]
        yr = yl.max()
        xr = x2[np.argmax(yl)]

        points = data[126:131, :]
        xdata = points[:, 0]
        ydata = points[:, 1]
        initial_guess = (0.0025, 0, np.std([xdata, ydata]))
        result = minimize(
            __class__.objective,
            initial_guess,
            args=(xdata, ydata),
            method="Nelder-Mead",
        )

        xc, yc, r = result.x

        # print(np.sum((np.sqrt((xdata - xc) ** 2 + (ydata - yc) ** 2) - r) ** 2))

        rf = r
        # breakpoint()
        return np.array(
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
            ]
        )

    @staticmethod
    def objective(params, x, y):
        xc, yc, r = params
        return np.sum((np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - r) ** 2)


def process_file(args):
    key, value, parsec_params_path = args
    feature = Fit_airfoil_15(value)
    with open(parsec_params_path, "a") as f:
        f.write(key)
        f.write(",")
        f.write(",".join(map(str, feature.parsec_features)))
        f.write("\n")


if __name__ == "__main__":

    import os
    import sys
    from multiprocessing import Pool
    import h5py
    from tqdm import tqdm
    import argparse

    # parse data folder from command line
    parser = argparse.ArgumentParser(description="Parsec Direct N15")
    parser.add_argument(
        "--data_folder", type=str, required=True, help="Path to the data folder"
    )
    args = parser.parse_args()
    data_folder = args.data_folder

    def fitparsec(dataset_name):
        parsec_params_path = (
            f"{args.data_folder}/{dataset_name}/{dataset_name}_parsec_params_15.txt"
        )
        root_path = f"{args.data_folder}/{dataset_name}/{dataset_name}_airfoils.h5"

        if os.path.exists(parsec_params_path):
            print(f"file {parsec_params_path} already exists, exiting program")
            sys.exit(1)

        with h5py.File(root_path, "r") as f:
            minlen = min(map(len, f.keys()))
            g = (
                (key, f[key][:], parsec_params_path)
                for key in f.keys()
                if len(key) == minlen
            )

            # Using multiprocessing to speed up the process
            # with Pool(processes=16) as pool:
            #     pool.map(
            #         process_file,
            #         g
            #     )

            # do not use multiprocessing:
            for item in tqdm(g):
                process_file(item)

    fitparsec("airfoils_test")
    fitparsec("data_4000")
    fitparsec("r05")
    fitparsec("r06")
    fitparsec("supercritical_airfoil")
