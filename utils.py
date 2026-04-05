"""Utility functions"""

import numpy as np
from typing import Tuple


def diff_mean_layers_values(vp1, vs1, rho1,
                            vp2, vs2, rho2):
    # TODO: не указаны типы и размерности и комментарии по действиям
    # TODO: чем отличается от diff_mean_values
    # REPLY: т.к. ф-ии, вызывающую эту и функцию ниже, не вызываются, у этих функций по цепочкке не фиксированы размерности. Можно удалить.
    # REPLY: отличается отсутствием зависимости от углов
    """Расчёт вспомогательных параметров (разницы, средние значения)
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
    Return:
        vp_diff, vs_diff, rho_diff - разницы
        vp_mean, vs_mean, rho_mean - средние значения
    """
    vp_mean = (vp1 + vp2) / 2.0
    vs_mean = (vs1 + vs2) / 2.0
    rho_mean = (rho1 + rho2) / 2.0

    vp_diff = vp2 - vp1
    vs_diff = vs2 - vs1
    rho_diff = rho2 - rho1

    return vp_diff, vs_diff, rho_diff, vp_mean, vs_mean, rho_mean


def diff_mean_values(vp1, vs1, rho1,
                     vp2, vs2, rho2,
                     angle):
    """Расчёт вспомогательных параметров (разницы, средние значения, углы)
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        angle_rad - угол наклона P-волны в слое 1, радианы
        angle_rad_2 - угол наклона P-волны в слое 2, радианы
        vp_diff, vs_diff, rho_diff - разности
        vp_mean, vs_mean, rho_mean - средние значения
    """

    vp_diff, vs_diff, rho_diff, vp_mean, vs_mean, rho_mean = diff_mean_layers_values(
        vp1, vs1, rho1, vp2, vs2, rho2
    )
    angle_rad = complex(angle / 180.0 * np.pi, 0)
    angle_rad_2 = np.arcsin(np.sin(angle_rad) / vp1 * vp2)
    angle_rad_mean = (angle_rad_2 + angle_rad) / 2.0

    return (angle_rad, angle_rad_2, angle_rad_mean,
            vp_diff, vs_diff, rho_diff,
            vp_mean, vs_mean, rho_mean)
