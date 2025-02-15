import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import lstsq
from scipy.spatial.distance import pdist, squareform
from scipy.special import factorial

minimax = {}

N_X = 257
xs = (np.cos(np.linspace(0, 2 * np.pi, N_X)) + 1) / 2


def norm(
    data, min_max_xy, trained_on=["supercritical_airfoil", "data_4000", "r05", "r06"]
):
    """
    rescale to [-1,1]
    """
    global minimax
    trained_on = frozenset(trained_on)
    try:
        mini, maxi = minimax[trained_on]
    except:
        minimax[trained_on] = min_max_xy
        mini, maxi = minimax[trained_on]
    mini, maxi = mini.numpy(), maxi.numpy()
    return (data - mini) / (maxi - mini) * 2 - 1


def de_norm(
    data, min_max_xy, trained_on=["supercritical_airfoil", "data_4000", "r05", "r06"]
):
    """
    rescale back to [mini,maxi]
    """
    global minimax
    trained_on = frozenset(trained_on)
    try:
        mini, maxi = minimax[trained_on]
    except:
        minimax[trained_on] = min_max_xy
        mini, maxi = minimax[trained_on]
    mini, maxi = mini.numpy(), maxi.numpy()
    return (data + 1) / 2 * (maxi - mini) + mini


def vis_airfoil(data, idx, dir_name="output_airfoil"):
    os.makedirs(dir_name, exist_ok=True)
    # 将data可视化，画出散点图
    plt.scatter(data[:, 0], data[:, 1])

    file_path = f"{dir_name}/{idx}.png"
    plt.savefig(file_path, dpi=100, bbox_inches="tight", pad_inches=0.0)
    # Clear the plot cache
    plt.clf()


def vis_airfoil2(source, target, idx, dir_name="output_airfoil", sample_type="ddpm"):
    # breakpoint()
    os.makedirs(dir_name, exist_ok=True)

    ## 将source和target放到一张图
    plt.scatter(
        source[:, 0], source[:, 1], c="red", label="source"
    )  # plot source points in red
    plt.scatter(
        target[:, 0], target[:, 1], c="blue", label="target"
    )  # plot target points in blue
    plt.legend()  # show legend

    file_path = f"{dir_name}/{sample_type}_{idx}.png"
    plt.savefig(file_path, dpi=100, bbox_inches="tight", pad_inches=0.0)
    # Clear the plot cache
    plt.clf()


def calculate_smoothness(airfoil):
    smoothness = 0.0
    num_points = airfoil.shape[0]

    for i in range(num_points):
        p_idx = (i - 1) % num_points
        q_idx = (i + 1) % num_points

        p = airfoil[p_idx]
        q = airfoil[q_idx]

        if p[0] == q[0]:  # 处理垂直于x轴的线段
            distance = abs(airfoil[i, 0] - p[0])
        else:
            m = (q[1] - p[1]) / (q[0] - p[0])
            b = p[1] - m * p[0]

            distance = abs(m * airfoil[i, 0] - airfoil[i, 1] + b) / np.sqrt(m**2 + 1)

        smoothness += distance

    return smoothness


def get_loss_smooth(airfoils):
    batch_size, num_points, _ = airfoils.shape
    loss_smooth = torch.tensor(0.0, device=airfoils.device)  # 确保与输入在同一设备上

    # 获取前一和后一点的索引
    p_idx = (torch.arange(num_points) - 1) % num_points
    q_idx = (torch.arange(num_points) + 1) % num_points

    # 获取 p, q 和当前点的坐标
    p = airfoils[:, p_idx, :]  # 前一点
    q = airfoils[:, q_idx, :]  # 后一点
    current_points = airfoils  # 当前点

    # 计算斜率和截距
    slopes = (q[:, :, 1] - p[:, :, 1]) / (q[:, :, 0] - p[:, :, 0] + 1e-6)  # 避免除零
    intercepts = p[:, :, 1] - slopes * p[:, :, 0]

    # 处理距离
    vertical_mask = p[:, :, 0] == q[:, :, 0]

    distance = torch.zeros(batch_size, num_points, device=airfoils.device)
    distance[vertical_mask] = torch.abs(
        current_points[..., 0][vertical_mask] - p[..., 0][vertical_mask]
    )

    # 非垂直情况
    non_vertical_mask = ~vertical_mask
    distance[non_vertical_mask] = torch.abs(
        slopes[non_vertical_mask] * current_points[..., 0][non_vertical_mask]
        - current_points[..., 1][non_vertical_mask]
        + intercepts[non_vertical_mask]
    ) / torch.sqrt(slopes[non_vertical_mask] ** 2 + 1)

    loss_smooth += distance.sum(dim=1).mean()  # 对每个 batch 和每个 point 求和

    return loss_smooth


def cal_diversity_score(data, subset_size=10, sample_times=1000):
    # Average log determinant
    N = data.shape[0]
    data = data.reshape(N, -1)
    mean_logdet = 0
    for i in range(sample_times):
        ind = np.random.choice(N, size=subset_size, replace=False)
        subset = data[ind]
        D = squareform(pdist(subset, "euclidean"))
        S = np.exp(-0.5 * np.square(D))
        (sign, logdet) = np.linalg.slogdet(S)
        mean_logdet += logdet
    return mean_logdet / sample_times


def cst_fit(source):
    """
    source: [257,2]
    """
    # x = source[:,0]
    y = source[:, 1]
    cst = CSTLayer()
    au, al, te = cst.fit_CST(y, n_x=129)  # 拟合中的x坐标和数量需要与原始翼型一致
    yu = cst.A0.dot(au) + cst.x_coords * te
    yl = cst.A0.dot(al) - cst.x_coords * te
    new_y = np.concatenate([yu[::-1], yl[1:]])
    source[:, 1] = new_y
    return source


def perturb_cst(source, std):
    """
    source: [257,2]
    """
    source = source.numpy()
    y = source[:, 1]
    cst = CSTLayer()
    au, al, te = cst.fit_CST(y, n_x=129)  # 拟合中的x坐标和数量需要与原始翼型一致
    noise_au = np.random.randn(*au.shape) * std + 1
    au *= noise_au
    noise_al = np.random.randn(*al.shape) * std + 1
    al *= noise_al
    noise_te = np.random.randn() * std + 1
    te *= noise_te
    yu = cst.A0.dot(au) + cst.x_coords * te
    yl = cst.A0.dot(al) - cst.x_coords * te
    new_y = np.concatenate([yu[::-1], yl[1:]])
    source[:, 1] = new_y
    return torch.from_numpy(source)


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


def plot_airfoils(
    airfoil_list,
    fix_indices=None,
    min_error_idx=-1,
    save_path=f"airfoils_plot_diff.png",
    equel_axis=False,
):
    plt.clf()
    plt.figure(figsize=(10, 6))
    for i, airfoil in enumerate(airfoil_list, 1):
        x, y = airfoil[:, 0], airfoil[:, 1]
        if i == len(airfoil_list):
            plt.plot(
                x,
                y,
                label="Gt",
                linestyle="--",
                color="orange",
                linewidth=0.7,
                alpha=0.7,
            )
            if fix_indices is not None:
                plt.scatter(
                    x[fix_indices],
                    y[fix_indices],
                    color="red",
                    s=3,
                    label="Fixed Points",
                )
        elif i == min_error_idx:
            plt.plot(
                x,
                y,
                linestyle="-.",
                linewidth=0.7,
                alpha=0.7,
                label=f"Min error {min_error_idx}",
            )
        else:
            plt.plot(x, y, linewidth=0.7, alpha=0.4)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Airfoil Shapes")
    plt.grid(True)
    if equel_axis:
        plt.axis("equal")
    plt.legend()
    plt.savefig(save_path, dpi=500)


def find_parameters(module):
    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


if __name__ == "__main__":
    breakpoint()
