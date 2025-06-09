import numpy as np
from scipy.interpolate import splev, splprep
from scipy import optimize
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


def process_file(args):
    key, value, parsec_params_path = args
    feature = Fit_airfoil_11(value)
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
    parser = argparse.ArgumentParser(description="Parsec Direct N11")
    parser.add_argument(
        "--data_folder", type=str, required=True, help="Path to the data folder"
    )
    args = parser.parse_args()
    data_folder = args.data_folder

    def fitparsec(dataset_name):
        parsec_params_path = (
            f"{args.data_folder}/{dataset_name}/{dataset_name}_parsec_params_11.txt"
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
