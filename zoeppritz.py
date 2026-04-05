"""Уравнения Цёппритца"""

import numpy as np
from numpy import typing as npt


def calc_angles(vp1: float, vs1: float, vp2: float, vs2: float, angle: float):
    # TODO: не указаны типы и размерности и комментарии по действиям
    # REPLY: типы указал, размерности функции не важны (если только пользователь не собирается подавать разные...)
    # Комментарии по действиям у расчётов уравнений Цеппритца абсолютно бессмысленны: надо смотреть оригинал уравнений в учебниках для лучшего понимания.
    # Перечисление входных-выходных параметров тоже чрезмерно, т.к. они очевидны, если посмотреть сами уравнения.
    # Многие функции не используются, можно снести.
    """Расчёт вспомогательных параметров для аппроксимации Цёппритца
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        p - параметр луча
        p2 - квадрат параметра луча
        sin_p1 - синус наклона P-волны в верхнем слое
        sin_p2 - синус наклона P-волны в нижнем слое
        sin_s1 - синус наклона S-волны в верхнем слое
        sin_s2 - синус наклона S-волны в нижнем слое
        cos_p1 - косинус наклона P-волны в верхнем слое
        cos_p2 - косинус наклона P-волны в нижнем слое
        cos_s1 - косинус наклона S-волны в верхнем слое
        cos_s2 - косинус наклона S-волны в нижнем слое
        cos_p1_vp1 - косинус наклона P-волны в верхнем слое, разделенный на vp1
        cos_p2_vp2 - косинус наклона P-волны в нижнем слое, разделенный на vp2
        cos_s1_vs1 - косинус наклона S-волны в верхнем слое, разделенный на vs1
        cos_s2_vs2 - косинус наклона S-волны в нижнем слое, разделенный на vs2
    """

    sin_p1 = np.sin(complex(angle / 180.0 * np.pi, 0))

    p = sin_p1 / vp1
    p2 = p * p

    sin_p2 = p * vp2
    sin_s1 = p * vs1
    sin_s2 = p * vs2

    cos_p1 = np.sqrt(1.0 - sin_p1 * sin_p1)
    cos_p2 = np.sqrt(1.0 - sin_p2 * sin_p2)
    cos_s1 = np.sqrt(1.0 - sin_s1 * sin_s1)
    cos_s2 = np.sqrt(1.0 - sin_s2 * sin_s2)

    cos_p1_vp1 = cos_p1 / vp1
    cos_p2_vp2 = cos_p2 / vp2
    cos_s1_vs1 = cos_s1 / vs1
    cos_s2_vs2 = cos_s2 / vs2

    return (p, p2,
            sin_p1, sin_p2, sin_s1, sin_s2,
            cos_p1, cos_p2, cos_s1, cos_s2,
            cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2)


def calc_coeffs(rho1: float, vs1: float, rho2: float, vs2: float, p2: float,
                cos_p1_vp1: float, cos_p2_vp2: float,
                cos_s1_vs1: float, cos_s2_vs2: float):
    """Расчёт промежуточных знаечений для уравнений Цёппритца
    Input parameters:
        rho1 - плотность в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho2 - плотность в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        p2 - квадрат параметра луча
        cos_p1_vp1 - косинус P-волны в верхнем слое, разделенный на vp1
        cos_p2_vp2 - косинус P-волны в нижнем слое, разделенный на vp2
        cos_s1_vs1 - косинус S-волны в верхнем слое, разделенный на vs1
        cos_s2_vs2 - косинус S-волны в нижнем слое, разделенный на vs2
    Return:
        a, b, c, d, e, f, g, h, dd - промежуточные константы
    """

    twice_vs22_dn2 = 2.0 * vs2 * vs2 * rho2
    twice_vs12_dn1 = 2.0 * vs1 * vs1 * rho1
    twice_vs22_dn2_p2 = twice_vs22_dn2 * p2
    twice_vs12_dn1_p2 = twice_vs12_dn1 * p2

    a = rho2 - rho1 - twice_vs22_dn2_p2 + twice_vs12_dn1_p2
    b = rho2 - twice_vs22_dn2_p2 + twice_vs12_dn1_p2
    c = rho1 - twice_vs12_dn1_p2 + twice_vs22_dn2_p2
    d = twice_vs22_dn2 - twice_vs12_dn1

    e = b * cos_p1_vp1 + c * cos_p2_vp2
    f = b * cos_s1_vs1 + c * cos_s2_vs2
    g = a - d * cos_p1_vp1 * cos_s2_vs2
    h = a - d * cos_p2_vp2 * cos_s1_vs1

    dd = e * f + g * h * p2

    return a, b, c, d, e, f, g, h, dd


def calc_rpp(vp1: float, vs1: float, rho1: float, vp2: float, vs2: float, rho2: float, angle: float):
    """ Расчёт к. отражения PP-волн через уравнения Цёппритца
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        rpp - комплексные к. отражения PP-волны
    """

    (p, p2,
     sin_p1, sin_p2, sin_s1, sin_s2,
     cos_p1, cos_p2, cos_s1, cos_s2,
     cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2) = calc_angles(vp1, vs1, vp2, vs2, angle)

    a, b, c, d, e, f, g, h, dd = calc_coeffs(rho1, vs1, rho2, vs2, p2,
                                             cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2)

    rpp = ((b * cos_p1_vp1 - c * cos_p2_vp2) * f - (a + d * cos_p1_vp1 * cos_s2_vs2) * h * p2) / dd
    return rpp


def calc_rps(vp1: float, vs1: float, rho1: float, vp2: float, vs2: float, rho2: float, angle: float):
    """ Расчёт к. отражения PS-волн через уравнения Цёппритца
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        rps - комплексные к. отражения PS-волны
    """

    (p, p2,
     sin_p1, sin_p2, sin_s1, sin_s2,
     cos_p1, cos_p2, cos_s1, cos_s2,
     cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2) = calc_angles(vp1, vs1, vp2, vs2, angle)

    a, b, c, d, e, f, g, h, dd = calc_coeffs(rho1, vs1, rho2, vs2, p2,
                                             cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2)

    rps = (-2.0 * cos_p1_vp1
           * (a * b + c * d * cos_p2_vp2 * cos_s2_vs2)
           * p * vp1 / vs1 / dd)
    return rps


def calc_rsp(vp1: float, vs1: float, rho1: float, vp2: float, vs2: float, rho2: float, angle: float):
    """ Расчёт к. отражения SP-волн через уравнения Цёппритца
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        rsp - комплексные к. отражения SP-волны
    """

    (p, p2,
     sin_p1, sin_p2, sin_s1, sin_s2,
     cos_p1, cos_p2, cos_s1, cos_s2,
     cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2) = calc_angles(vp1, vs1, vp2, vs2, angle)

    a, b, c, d, e, f, g, h, dd = calc_coeffs(rho1, vs1, rho2, vs2, p2,
                                             cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2)

    rsp = (-2.0 * cos_s1_vs1
           * (a * b + c * d * cos_p2_vp2 * cos_s2_vs2)
           * p * vs1 / vp1 / dd)
    return rsp


def calc_rss(vp1: float, vs1: float, rho1: float, vp2: float, vs2: float, rho2: float, angle: float):
    """ Расчёт к. отражения SS-волн через уравнения Цёппритца
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        rss - комплексные к. отражения SS-волны
    """

    (p, p2,
     sin_p1, sin_p2, sin_s1, sin_s2,
     cos_p1, cos_p2, cos_s1, cos_s2,
     cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2) = calc_angles(vp1, vs1, vp2, vs2, angle)

    a, b, c, d, e, f, g, h, dd = calc_coeffs(rho1, vs1, rho2, vs2, p2,
                                             cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2)

    rss = -((b * cos_s1_vs1 - c * cos_s2_vs2) * e
            - (a + d * cos_p2_vp2 * cos_s1_vs1) * g * p2) / dd
    return rss


def calc_rc(vp1: float, vs1: float, dn1: float, vp2: float, vs2: float, dn2: float, angle: float):
    """ Расчёт к. отражения через уравнения Цёппритца
    Input parameters:
        vp1 - скорость P-волн в верхнем слое
        vs1 - скорость S-волн в верхнем слое
        rho1 - плотность в верхнем слое
        vp2 - скорость P-волн в нижнем слое
        vs2 - скорость S-волн в нижнем слое
        rho2 - плотность в нижнем слое
        angle - угол наклона P-волны, градусы
    Return:
        rpp - комплексные к. отражения PP-волны
        rps - комплексные к. отражения PS-волны
        rsp - комплексные к. отражения SP-волны
        rss - комплексные к. отражения SS-волны
    """

    (p, p2,
     sin_p1, sin_p2, sin_s1, sin_s2,
     cos_p1, cos_p2, cos_s1, cos_s2,
     cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2) = calc_angles(vp1, vs1, vp2, vs2, angle)

    a, b, c, d, e, f, g, h, dd = calc_coeffs(dn1, vs1, dn2, vs2, p2,
                                             cos_p1_vp1, cos_p2_vp2, cos_s1_vs1, cos_s2_vs2)

    rpp = ((b * cos_p1_vp1 - c * cos_p2_vp2) * f
           - (a + d * cos_p1_vp1 * cos_s2_vs2) * h * p2) / dd
    rss = ((c * cos_s2_vs2 - b * cos_s1_vs1) * e
           + (a + d * cos_p2_vp2 * cos_s1_vs1) * g * p2) / dd
    rps = (-2.0 * cos_p1_vp1
           * (a * b + c * d * cos_p2_vp2 * cos_s2_vs2)
           * p * vp1 / vs1 / dd)
    rsp = (-2.0 * (cos_s1_vs1
                   * (a * b + c * d * cos_p2_vp2 * cos_s2_vs2))
           * p * vs1 / vp1 / dd)
    return rpp, rps, rsp, rss


def calc_rpp_mat(vp: npt.NDArray, vs: npt.NDArray, rho: npt.NDArray, angles: npt.NDArray):
    """ Расчёт к. отражения PP-волн для массивов
    Input parameters:
        vp - трасса vp
        vs - трасса vs
        rho - трасса плотностей
        angles - трасса углов наклона P-волны, градусы
    Return:
        Rpp - к. отражения PP-волн, 2d массив комплексных чисел
    """
    rpp = np.zeros((len(angles), len(vp)), dtype=complex)
    for i in range(rpp.shape[0]):
        for j in range(1, rpp.shape[1]):
            j1 = j - 1
            rpp[i, j] = calc_rpp(vp[j1], vs[j1], rho[j1], vp[j], vs[j], rho[j], angles[i])
    return rpp


def calc_rpp_fixed_angle(vp: npt.NDArray, vs: npt.NDArray, dn: npt.NDArray, angle: float):
    """ Расчёт к. отражения PP-волн для массивов для одного угла
    Input parameters:
        vp - трасса vp
        vs - трасса vs
        rho - трасса плотностей
        angle - угол наклона P-волны, градусы
    Return:
        Rpp - к. отражения PP-волн, 1d массив комплексных чисел
    """
    rpp = np.zeros(len(vp), dtype=complex)
    for i in range(1, len(vp)):
        i1 = i - 1
        rpp[i] = calc_rpp(vp[i1], vs[i1], dn[i1], vp[i], vs[i], dn[i], angle)
    return rpp


if __name__ == '__main__':
    # TODO: зачем здесь main
    # RELPY: в качестве проверки и для тестовых расчётов, если потребуются.
    # Можно удалить, но штука полезная
    from matplotlib import pyplot as plt

    # input params
    vp1, vp2 = 1500, 2000
    vs1, vs2 = 300, 800
    dn1, dn2 = 1.5, 2.5
    angles = np.arange(90)

    rpp = np.zeros(len(angles), dtype=complex)
    for i in range(len(angles)):
        rpp[i] = calc_rpp(vp1, vs1, dn1, vp2, vs2, dn2, angles[i])

    plt.plot(np.abs(rpp), color='black')
    plt.plot(np.real(rpp), color='red')
    plt.plot(np.imag(rpp), color='blue')
    plt.grid()
    plt.show()
