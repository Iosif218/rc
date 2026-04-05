"""Reflection coefficient using various equations"""
import numpy as np

from utils import diff_mean_values


def aki_rpp(vp1, vs1, rho1, vp2, vs2, rho2, angle):
    # TODO: не указаны типы и размерности и комментарии по действиям
    # По размерностям: эти функции сейчас не используются, размерности могут быть разные. Можно удалить.
    # По комментариям: расчёты проводятся в лоб, комментировать нечего.
    # Для лучшего понимания надо смотреть, как Аки, Ричардс, Шуэ это выводили.
    """Расчёт к. отражения с помощью аппроксимации Аки-Ричардса
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        Rpp - комплексный к. отражения PP-волн
    """

    (angle_rad, angle_rad_2, angle_rad_mean,
     vp_diff, vs_diff, dn_diff,
     vp_mean, vs_mean, dn_mean) = diff_mean_values(vp1, vs1, rho1, vp2, vs2, rho2, angle)

    common_part_1 = vs_mean / vp_mean * np.sin(angle_rad_mean) ** 2
    common_part_2 = np.tan(angle_rad_mean) ** 2

    rpp = (0.5 * (1.0 - 4.0 * common_part_1) * dn_diff / dn_mean
           + 0.5 * (1.0 + common_part_2) * vp_diff / vp_mean
           - 4.0 * common_part_1 * vs_diff / vs_mean)
    return rpp


def shuye_rpp(vp1, vs1, rho1, vp2, vs2, rho2, angle):
    """Расчёт к. отражения с помощью аппроксимации Шуэ
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        Rpp - комплексный к. отражения PP-волн
    """
    (angle_rad, angle_rad_2, angle_rad_mean,
     vp_diff, vs_diff, rho_diff,
     vp_mean, vs_mean, dn_mean) = diff_mean_values(vp1, vs1, rho1, vp2, vs2, rho2, angle)

    vp_diff_vp_mean = vp_diff / vp_mean
    rho_diff_dn_mean = rho_diff / dn_mean
    vs_mean_vp_mean = vs_mean / vp_mean
    sin2 = np.sin(angle_rad_mean) ** 2

    rpp = (
            0.5 * (rho_diff_dn_mean + vp_diff_vp_mean)
            + (0.5 * vp_diff_vp_mean
               - 4.0 * vs_mean_vp_mean * vs_mean_vp_mean
               * (0.5 * rho_diff_dn_mean + vs_diff / vs_mean)) * sin2
            + 0.5 * vp_diff_vp_mean * sin2 * sin2 / (1.0 - sin2)
    )
    return rpp


def liquids_rpp(vp1, rho1, vp2, rho2, angle):
    """Расчёт к. отражения на границе жидких сред
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        rpp - комплексный к. отражения PP-волн
    """
    z1 = vp1 * rho1
    z2 = vp2 * rho2
    angle_rad = angle * np.pi / 180.0
    angle_1_cos = np.cos(angle_rad)
    angle_2_cos = np.cos(np.arcsin(np.sin(angle_rad) * vp2 / vp1))

    common_param_1 = z1 * angle_2_cos
    common_param_2 = z2 * angle_1_cos

    rpp = (common_param_2 - common_param_1) / (common_param_2 + common_param_1)
    return rpp


def rsh(vs1, rho1, vs2, rho2, angle):
    """Расчёт к. отражения SH волны на границе жидких сред
    Input parameters:
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        rsh - комплексный к. отражения SH-волн
    """

    angle_rad = angle * np.pi / 180.0
    vs1_vs2 = vs1 * vs2
    val1 = rho1 * vs1 * vs1_vs2 * np.cos(angle_rad)
    val2 = rho2 * vs2 * vs1_vs2 * np.cos(np.arcsin(np.sin(angle_rad) * vs2 / vs1))

    rsh = (val1 - val2) / (val1 + val2)
    return rsh
