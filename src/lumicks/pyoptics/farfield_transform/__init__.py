import numpy as np

from .czt import near_field_to_far_field_czt

__all__ = ["ff_to_bfp", "ff_to_bfp_angle", "near_field_to_far_field_czt"]


def ff_to_bfp(
    Exff: np.ndarray,
    Eyff: np.ndarray,
    Ezff: np.ndarray,
    Sx: np.ndarray,
    Sy: np.ndarray,
    Sz: np.ndarray,
    n_medium: float,
    n_bfp: float,
):
    Sp = np.hypot(Sx, Sy)
    cosP = np.ones(Sp.shape)
    sinP = np.zeros(Sp.shape)
    region = Sp > 0
    cosP[region] = Sx[region] / Sp[region]
    sinP[region] = Sy[region] / Sp[region]
    sinP[Sp == 0] = 0
    Et = np.zeros(Exff.shape, dtype="complex128")
    Ep = Et.copy()
    Ex_bfp = Et.copy()
    Ey_bfp = Et.copy()

    Et[Sz > 0] = (
        (
            (Exff[Sz > 0] * cosP[Sz > 0] + Eyff[Sz > 0] * sinP[Sz > 0]) * Sz[Sz > 0]
            - Ezff[Sz > 0] * Sp[Sz > 0]
        )
        * (n_medium / n_bfp) ** 0.5
        / (Sz[Sz > 0]) ** 0.5
    )
    Ep[Sz > 0] = (
        (Eyff[Sz > 0] * cosP[Sz > 0] - Exff[Sz > 0] * sinP[Sz > 0])
        * (n_medium / n_bfp) ** 0.5
        / (Sz[Sz > 0]) ** 0.5
    )

    Ex_bfp[Sz > 0] = Et[Sz > 0] * cosP[Sz > 0] - Ep[Sz > 0] * sinP[Sz > 0]
    Ey_bfp[Sz > 0] = Ep[Sz > 0] * cosP[Sz > 0] + Et[Sz > 0] * sinP[Sz > 0]

    return Ex_bfp, Ey_bfp


def ff_to_bfp_angle(
    Exff: np.ndarray,
    Eyff: np.ndarray,
    Ezff: np.ndarray,
    cosPhi: np.ndarray,
    sinPhi: np.ndarray,
    cosTheta: np.ndarray,
    n_medium: float,
    n_bfp: float,
):
    Et = np.zeros(Exff.shape, dtype="complex128")
    Ep = Et.copy()
    Ex_bfp = Et.copy()
    Ey_bfp = Et.copy()

    sinTheta = (1 - cosTheta**2) ** 0.5

    roc = cosTheta > 0  # roc == Region of convergence, avoid division by zero

    Et[roc] = (
        (
            (Exff[roc] * cosPhi[roc] + Eyff[roc] * sinPhi[roc]) * cosTheta[roc]
            - Ezff[roc] * sinTheta[roc]
        )
        * (n_medium / n_bfp) ** 0.5
        / (cosTheta[roc]) ** 0.5
    )

    Ep[roc] = (
        (Eyff[roc] * cosPhi[roc] - Exff[roc] * sinPhi[roc])
        * (n_medium / n_bfp) ** 0.5
        / (cosTheta[roc]) ** 0.5
    )

    Ex_bfp[roc] = Et[roc] * cosPhi[roc] - Ep[roc] * sinPhi[roc]
    Ey_bfp[roc] = Ep[roc] * cosPhi[roc] + Et[roc] * sinPhi[roc]

    return Ex_bfp, Ey_bfp
