import torch

from typing import Literal, Optional, Tuple


def _d_along_length(f: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """1-D central differences along axis 1 (batchwise)."""
    front = (f[:, 1] - f[:, 0]) / (t[:, 1] - t[:, 0])
    back = (f[:, -1] - f[:, -2]) / (t[:, -1] - t[:, -2])
    centre = (f[:, 2:] - f[:, :-2]) / (t[:, 2:] - t[:, :-2])
    return torch.cat((front.unsqueeze(1), centre, back.unsqueeze(1)), 1)


def curvature_radius_center_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    t: Optional[torch.Tensor] = None,
    parameter: Literal["index", "arclength"] = "index",
    eps: float = 1.0e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched signed curvature, radius and centre of curvature.

    Returns
    --------
    kappa   : (B, L)  – signed curvature
    radius  : (B, L)  – signed radius (= 1/κ)
    center  : (B, L, 2) – coordinates (Cx, Cy)
              NaN where derivatives are (nearly) zero.
    """
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be 2-D tensors of identical shape (B,L).")
    B, L = x.shape
    device, dtype = x.device, x.dtype

    # ---------------------------------------------------- #
    # 1. parameter grid t                                  #
    # ---------------------------------------------------- #
    if t is None:
        if parameter == "index":
            t = torch.arange(L, dtype=dtype, device=device).expand(B, L)
        elif parameter == "arclength":
            ds = torch.sqrt((x[:, 1:] - x[:, :-1]) ** 2 + (y[:, 1:] - y[:, :-1]) ** 2)
            t = torch.cat(
                (torch.zeros(B, 1, dtype=dtype, device=device), torch.cumsum(ds, 1)), 1
            )
        else:
            raise ValueError("parameter must be 'index' or 'arclength'.")
    else:  # user-supplied grid
        if t.ndim == 1:
            if t.numel() != L:
                raise ValueError("1-D t must have length L.")
            t = t.to(device=device, dtype=dtype).expand(B, L)
        elif t.ndim == 2 and t.shape == (B, L):
            t = t.to(device=device, dtype=dtype)
        else:
            raise ValueError("t must be (L,) or (B,L).")

    # ---------------------------------------------------- #
    # 2. derivatives                                       #
    # ---------------------------------------------------- #
    dx = _d_along_length(x, t)
    dy = _d_along_length(y, t)
    d2x = _d_along_length(dx, t)
    d2y = _d_along_length(dy, t)

    speed2 = dx**2 + dy**2
    speed = torch.sqrt(speed2).clamp_min(eps)

    # ---------------------------------------------------- #
    # 3. signed curvature and radius                       #
    # ---------------------------------------------------- #
    num = dx * d2y - dy * d2x  # keep the sign!
    denom = speed2.pow(1.5).clamp_min(eps)
    kappa = num / denom  # (B,L)
    radius = 1.0 / kappa  # signed radius

    # ---------------------------------------------------- #
    # 4. centre of curvature                               #
    # ---------------------------------------------------- #
    nx = -dy / speed  # unit normal components
    ny = dx / speed
    Cx = x + radius * nx  # (B,L)
    Cy = y + radius * ny
    center = torch.stack((Cx, Cy), dim=-1)  # (B,L,2)

    # guard against ill-defined points (speed≈0)
    mask_zero = speed2 < eps
    kappa = kappa.masked_fill(mask_zero, float("nan"))
    radius = radius.masked_fill(mask_zero, float("nan"))
    center = center.masked_fill(mask_zero.unsqueeze(-1), float("nan"))
    return kappa, radius, center


def torch_factorial(x):
    return torch.exp(torch.lgamma(x + 1))


def calculate_airfoil_metric_n15_batch(x, y, stacked: bool = False):
    """
    Batch version of calculate_airfoil_metric_n15.

    Args:
        x: Tensor [B, 257]
        y: Tensor [B, 257]

    Returns:
        parsec_features: tuple of tensors each of shape [B]
    """
    B = x.shape[0]

    a = (y[:, 0] - y[:, 9]) / (x[:, 0] - x[:, 9])
    theta_radians = torch.atan(a)
    angle = torch.rad2deg(theta_radians)

    def fit_CST_parallel_batch(x_coords, y_coords, n_cst=12, n1=0.5, n2=1.0):
        B, n_x = x_coords.shape
        r = torch.arange(0, n_cst + 1, device=x_coords.device, dtype=x_coords.dtype)

        lgamma_n_plus1 = torch.lgamma(
            torch.tensor(n_cst + 1, device=x_coords.device, dtype=x_coords.dtype)
        )
        k = torch.exp(
            lgamma_n_plus1 - torch.lgamma(r + 1) - torch.lgamma(n_cst - r + 1)
        )

        r_col = r.unsqueeze(1)
        pow1 = n1 + r_col
        pow2 = (n_cst + n2) - r_col
        k_col = k.unsqueeze(1)

        A0 = (
            k_col
            * x_coords.unsqueeze(1).pow(pow1.unsqueeze(0))
            * (1 - x_coords.unsqueeze(1)).pow(pow2.unsqueeze(0))
        )
        A0 = A0.permute(0, 2, 1)

        yu = y_coords[:, :n_x].flip(1)
        yl = y_coords[:, n_x - 1 :]

        te = (yu[:, -1] - yl[:, -1]) / 2

        yu_end = yu[:, -1].unsqueeze(1)
        yl_end = yl[:, -1].unsqueeze(1)

        au = torch.linalg.lstsq(A0, yu - x_coords * yu_end, rcond=None)[0]
        al = torch.linalg.lstsq(A0, yl - x_coords * yl_end, rcond=None)[0]

        return au, al, te

    au, al, te = fit_CST_parallel_batch(x[:, :129].flip(1), y, n_cst=12)

    x2 = torch.arange(0, 1.001, 0.0001, device=x.device, dtype=x.dtype)
    x2 = x2.unsqueeze(0).repeat(B, 1)

    def CST_base_batch(x_coords, n_cst=12, n1=0.5, n2=1.0):
        B, n_x = x_coords.shape
        r = torch.arange(0, n_cst + 1, device=x_coords.device, dtype=x_coords.dtype)

        lgamma_n_plus1 = torch.lgamma(
            torch.tensor(n_cst + 1, device=x_coords.device, dtype=x_coords.dtype)
        )
        k = torch.exp(
            lgamma_n_plus1 - torch.lgamma(r + 1) - torch.lgamma(n_cst - r + 1)
        )

        r_col = r.unsqueeze(1)
        pow1 = n1 + r_col
        pow2 = (n_cst + n2) - r_col
        k_col = k.unsqueeze(1)

        A0 = (
            k_col
            * x_coords.unsqueeze(1).pow(pow1.unsqueeze(0))
            * (1 - x_coords.unsqueeze(1)).pow(pow2.unsqueeze(0))
        )
        return A0.permute(0, 2, 1)

    A0_fine = CST_base_batch(x2, n_cst=12)
    yu = (A0_fine @ au.unsqueeze(-1)).squeeze(-1) + x2 * te.unsqueeze(1)
    yl = (A0_fine @ al.unsqueeze(-1)).squeeze(-1) - x2 * te.unsqueeze(1)

    t4u = yu[:, 400]
    t25u = yu[:, 2500]
    t60u = yu[:, 6000]
    t4l = yl[:, 400]
    t25l = yl[:, 2500]
    t60l = yl[:, 6000]

    yumax, _ = yu.max(dim=1)
    ylmax, _ = yl.min(dim=1)

    def soft_argmax(x, beta=10000.0):
        indices = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0)
        weights = torch.softmax(beta * x, dim=1)
        return (weights * indices).sum(dim=1)

    xumax = soft_argmax(yu) / 10010.0
    xlmax = soft_argmax(-yl) / 10010.0
    xr = soft_argmax(yl) / 10010.0
    yr, _ = yl.max(dim=1)

    te1 = te

    kappa, radius, center = curvature_radius_center_batch(
        x=x,
        y=y,
        parameter="index",
    )  # of shape (b,257)

    rf = radius[:, 128]

    if stacked:
        return torch.stack(
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
            dim=1,
        )
    else:
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


if __name__ == "__main__":

    B = 32
    x_batch = torch.rand(B, 257)
    y_batch = torch.rand(B, 257)

    features = calculate_airfoil_metric_n15_batch(x_batch, y_batch, n_inner_steps=20000)

    from airfoil_generation.dataset.airfoil_metric import calculate_airfoil_metric_n15

    for feature in features:
        print(feature.shape)  # Should print torch.Size([32])

    for i in range(B):
        single_feature = calculate_airfoil_metric_n15(
            x_batch[i], y_batch[i], n_inner_steps=20000
        )
        for j, feature in enumerate(single_feature):
            assert torch.allclose(
                feature, features[j][i], atol=1e-6
            ), f"Mismatch in feature {j} for batch item {i}"
    print("All features match for batch and single item calculations.")
    print("Batchwise airfoil metric calculation is correct.")
