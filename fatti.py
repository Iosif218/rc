"""Fatti approximation"""
import numpy as np
from numpy import typing as npt


def fatti_coeffs(angle: float, vsvp2: float):
    # TODO: не указаны типы и размерности и комментарии по действиям
    # REPLY: Здесь и ниже расчёты делаются в лоб.
    # Чтобы понять их смысл, надо открыть формулы расчётов к. Фатти и прочитать их
    # комментарии здесь излишни и вредны
    """Расчёт к. Фатти
    Input parameters:
        angle - угол наклона P-волны, градусы
        vsvp2 - отношение (vs / vp)**2
    Return:
        a, b, c - промежуточные коэффициенты
    """
    a = 0.5 * (1 + np.square(np.tan(angle)))
    b = -4 * vsvp2 * np.square(np.sin(angle))
    c = 0.5 - a - 0.5 * b
    return a, b, c


def fatti(angle: float, vsvp2: float, k: float, m: float):
    """Расчёт к. отражения с использованием аппроксимации Фатти
    Input parameters:
        angle - угол наклона P-волны, градусы
        vsvp2 - отношение (vs / vp)**2
        k, m - корреляционные параметры
    Return:
        a_p, a_s, a_d - параметры Фатти
    """
    a, a_s, a_d = fatti_coeffs(angle, vsvp2)
    a_p = a + a_s * k + a_d * m
    return a_p, a_s, a_d


def calc_fatti_rpp(ai: npt.NDArray, si: npt.NDArray, rho: npt.NDArray, angle_deg: npt.NDArray, constant_vsvp: float = None):
    """ Расчёт к. отражения PP-волн с использованием аппроксимации Фатти (Аки-Ричардса)
    Input parameters:
        ai - трасса АИ
        si - трасса СИ
        rho - трасса плотностей
        angle_deg - угол наклона P-волны, градусы
        constant_vsvp -  отношение (vs / vp)**2, рассчитывается автоматически, если None
    Return:
        Rpp - к. отражения PP-волны
    """

    rpp = np.zeros(len(ai))

    # Заполнение vsvp
    if constant_vsvp is None:
        vsvp2 = (si/ai)**2
    else:
        vsvp2 = np.ones(len(ai)) * (constant_vsvp**2)

    # Расчет коэффициента отражения
    for i in range(1, len(ai)):
        coeffs = fatti_coeffs(angle_deg / 180 * np.pi, vsvp2[i])
        rpp[i] = (
                2 * (coeffs[0] * (ai[i] - ai[i-1])/ (ai[i] + ai[i-1])) +
                2 * (coeffs[1] * (si[i] - si[i-1])/ (si[i] + si[i-1])) +
                2 * (coeffs[2] * (rho[i] - rho[i - 1]) / (rho[i] + rho[i - 1]))
                 )
    return rpp
