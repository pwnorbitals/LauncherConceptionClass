import numpy as np
import matplotlib.pyplot as plt

# Constantes
g0 = 9.80665  # m/s2
Re = 6378.137  # km
Omega_e = 7.292e-5  # rad/s
mu = 398600.5  # km3/s2

# Data
Zp = 200  # km
Rp = Re + Zp
Za = 35786  # km
Ra = Re + Za
inc_deg = 5.2  # deg
inc_rad = np.deg2rad(inc_deg)
M_payload = 3800  # kg
lat_deg = 5.2  # deg
lat_rad = np.deg2rad(lat_deg)


# Injection requirement
azimuth_rad = np.arcsin(np.cos(inc_rad)/np.cos(lat_rad))
azimuth_deg = np.rad2deg(azimuth_rad)

a = (Ra + Rp) / 2
Vp = np.sqrt(mu * (2/Rp - 1/a)) * 1000

Vi = Omega_e * Re * np.cos(lat_rad) * np.cos(azimuth_rad)
losses = 2.452e-3 * Zp**2 + 1.051 * Zp + 1387.5

deltaV_prop = Vp - Vi + losses

# Optimization
# Parameters


def TwoStages(deltaV, Isp1, Isp2, k1, k2, b2=3, M_payload=3800):
    g0 = 9.80665
    Omega1 = k1 / (1 + k1)
    Omega2 = k2 / (1 + k2)
    deltaV_temp = 0

    while abs(deltaV - deltaV_temp) > 10:
        b1 = (1 / Omega1) * (1 - (Isp2 / Isp1) * (1 - Omega2 * b2))
        deltaV_1 = Isp1 * g0 * np.log(b1)
        deltaV_2 = Isp2 * g0 * np.log(b2)
        deltaV_temp = deltaV_1 + deltaV_2
        b2 += 0.05

    a1 = (1 + k1) / b1 - k1
    a2 = (1 + k2) / b2 - k2

    Mi2 = M_payload / a2
    Mi1 = Mi2 / a1

    Me2 = (1 + a2) / (1 + k2) * Mi2
    Me1 = (1 + a1) / (1 + k1) * Mi1

    Ms2 = k2 * Me2
    Ms1 = k1 * Me1

    return deltaV_temp, Mi1, Mi2, Me1, Me2, Ms1, Ms2


def ThreeStages(deltaV, Isp1, Isp2, Isp3, k1, k2, k3, b3=1, M_payload=3800):
    g0 = 9.80665
    Omega1 = k1 / (1 + k1)
    Omega2 = k2 / (1 + k2)
    Omega3 = k3 / (1 + k3)
    deltaV_temp = 0

    while abs(deltaV - deltaV_temp) > 10:
        b2 = (1 / Omega2) * (1 - (Isp3 / Isp2) * (1 - Omega3 * b3))
        b1 = (1 / Omega1) * (1 - (Isp2 / Isp1) * (1 - Omega2 * b2))
        deltaV_1 = Isp1 * g0 * np.log(b1)
        deltaV_2 = Isp2 * g0 * np.log(b2)
        deltaV_3 = Isp3 * g0 * np.log(b3)
        deltaV_temp = deltaV_1 + deltaV_2 + deltaV_3
        b3 += 0.05

    a1 = (1 + k1) / b1 - k1
    a2 = (1 + k2) / b2 - k2
    a3 = (1 + k3) / b3 - k3

    Mi3 = M_payload / a3
    Mi2 = Mi3 / a2
    Mi1 = Mi2 / a1

    Me3 = (1 + a3) / (1 + k3) * Mi3
    Me2 = (1 + a2) / (1 + k2) * Mi2
    Me1 = (1 + a1) / (1 + k1) * Mi1

    Ms3 = k3 * Me3
    Ms2 = k2 * Me2
    Ms1 = k1 * Me1

    return deltaV_temp, Mi1, Mi2, Mi3, Me1, Me2, Me3, Ms1, Ms2, Ms3

