import numpy as np
import torch
from airfoil_generation.dataset.airfoil_metric_batchwise import (
    calculate_airfoil_metric_n15_batch,
)


class Fit_airfoil_15_v2:
    """
    Overview:
        Fit airfoil by 3 order Bspline and extract Parsec features.
        airfoil (npoints,2)
    Interface:
        __init__, get_parsec_n15
    """

    def __init__(self, data, device=None):
        self.data = data
        self.device = (
            device
            if device is not None and torch.cuda.is_available()
            else torch.device("cpu")
        )
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

        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(0)

        (
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
        ) = calculate_airfoil_metric_n15_batch(x_tensor, y_tensor)

        rf = rf.item()
        t4u = t4u.item()
        t4l = t4l.item()
        xumax = xumax.item()
        yumax = yumax.item()
        xlmax = xlmax.item()
        ylmax = ylmax.item()
        t25u = t25u.item()
        t25l = t25l.item()
        angle = angle.item()
        te1 = te1.item()
        xr = xr.item()
        yr = yr.item()
        t60u = t60u.item()
        t60l = t60l.item()

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


def process_file(args):
    key, value, parsec_params_path = args
    feature = Fit_airfoil_15_v2(value)
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
    parser = argparse.ArgumentParser(description="Parsec Direct N15 V2")
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
