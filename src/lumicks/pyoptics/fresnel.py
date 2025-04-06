# import numpy as np


def rs(kz1, kz2):
    return (kz1 - kz2) / (kz1 + kz2)


def rp(n1, kz1, n2, kz2):
    return (n2**2 * kz1 - n1**2 * kz2) / (n2**2 * kz1 + n1**2 * kz2)


def ts(kz1, kz2):
    return 1 + rs(kz1, kz2)


def tp(n1, kz1, n2, kz2):
    return (1 + rp(n1, kz1, n2, kz2)) * n1 / n2


# def ref_fresnel(n0, n1, n2, d, kp, lambdas):
#     kz0 = ((2 * np.pi * n0 / lambdas)**2 - kp**2 + 0j)**0.5
#     kz1 = ((2 * np.pi * n1 / lambdas)**2 - kp**2 + 0j)**0.5
#     kz2 = ((2 * np.pi * n2 / lambdas)**2 - kp**2 + 0j)**0.5

#     rs01 = rs(kz0, kz1)
#     rs12 = rs(kz1, kz2)

#     rp01 = rp(n0, kz0, n1, kz1)
#     rp12 = rp(n1, kz1, n2, kz2)

#     ts01 = 1 + rs01
#     tp01 = (1 + rp01) * n0 / n1
#     ts12 = 1 + rs12
#     tp12 = (1 + rp12) * n1 / n2

#     # Fresnel, source:
#     # Regenerating evanescent waves from a silver superlens
#     # Nicholas Fang, Zhaowei Liu, Ta-Jen Yen, and Xiang Zhang
#     # Optics Express, 2003, 11, 682-687
#     rs = (rs01 + rs12 * np.exp(2j * kz1 * d)) / (1 + rs01 * rs12 * np.exp(2j * kz1 * d))
#     rp = (rp01 + rp12 * np.exp(2j * kz1 * d)) / (1 + rp01 * rp12 * np.exp(2j * kz1 * d))
#     ts = ts01 * ts12 * np.exp(1j * kz1 * d) / (1 + rs01 * rs12 * np.exp(2j * kz1 * d))
#     tp = tp01 * tp12 * np.exp(1j * kz1 * d) / (1 + rp01 * rp12 * np.exp(2j * kz1 * d))

#     return rs, rp, tp, ts
