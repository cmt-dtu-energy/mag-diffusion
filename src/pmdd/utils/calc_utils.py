"""
Functions for validation of physicality of magnetic fields
As we define magnetic fields to have no magnetic sources inside the measurement area,
we know that the divergence and curl of true magnetic fields is 0. This can be checked
with the following functions.
However, calculating these values for a true 3-D magnetic field.
"""

import numpy as np


def curl_3d(field) -> np.ndarray:
    Fx_y = np.gradient(field[0], axis=1)
    Fy_x = np.gradient(field[1], axis=0)
    Fx_z = np.gradient(field[0], axis=2)
    Fy_z = np.gradient(field[1], axis=2)
    Fz_x = np.gradient(field[2], axis=0)
    Fz_y = np.gradient(field[2], axis=1)
    # Taking gradients of center layer only
    return np.stack([Fz_y - Fy_z, Fx_z - Fz_x, Fy_x - Fx_y], axis=0)[:, :, :, 1]


def curl_2d(field) -> np.ndarray:
    Fx_y = np.gradient(field[0], axis=1)
    Fy_x = np.gradient(field[1], axis=0)
    return Fy_x - Fx_y


def div_3d(field) -> np.ndarray:
    Fx_x = np.gradient(field[0], axis=0)
    Fy_y = np.gradient(field[1], axis=1)
    Fz_z = np.gradient(field[2], axis=2)
    # Taking gradients of center layer only
    div = np.stack([Fx_x, Fy_y, Fz_z], axis=0)[:, :, :, 1]

    return div.sum(axis=0)


def div_2d(field) -> np.ndarray:
    Fx_x = np.gradient(field[0], axis=0)
    Fy_y = np.gradient(field[1], axis=1)
    div = np.stack([Fx_x, Fy_y], axis=0)

    return div.sum(axis=0)
