import numpy as np
import matplotlib.pyplot as plt
import math as m

# Constantes
g0 = 9.80665  # m/s2 (2.pi/86164)
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

Vi = (Omega_e * Re * np.cos(lat_rad) * np.sin(azimuth_rad)) * 1000
losses = 2.452e-3 * Zp**2 + 1.051 * Zp + 1387.5

deltaV_prop = Vp - Vi + losses

print("Vp = ", Vp, "\nVi = ", Vi, "\nLosses = ", losses, "\ndeltaV_prop = ", deltaV_prop)

# Optimization
# Parameters

# Propellants
class Propellant:
    def __init__(self, name, code, possible_stages, Ispv, Isp_mean, k):
        self.name = name
        self.code = code
        self.possible_stages = possible_stages
        self.Ispv = Ispv
        self.Isp_mean = Isp_mean
        self.k = k

Ref_pet = Propellant('Refined Petroleum 1', 'LOX-RP1', [1, 2, 3], 330, 287, 0.150)
Liq_OH = Propellant('Liquide oxygen - liquid hydrogen', 'LOX-LK2', [2,3], 440, 0, 0.22)
Sol_liq = Propellant('Solid-liquid', 'Solid', [1], 300, 260, 0.1)

# Different possible scenarii

#With two stages
two_stages_1 = [Ref_pet, Ref_pet]
two_stages_2 = [Ref_pet, Liq_OH]
two_stages_3 = [Sol_liq, Ref_pet]
two_stages_4 = [Sol_liq, Liq_OH]

#With three stages
three_stages_1 = [Ref_pet, Ref_pet, Ref_pet]
three_stages_2 = [Ref_pet, Ref_pet, Liq_OH]
three_stages_3 = [Ref_pet, Liq_OH, Ref_pet]
three_stages_4 = [Ref_pet, Liq_OH, Liq_OH]
three_stages_5 = [Sol_liq, Ref_pet, Ref_pet]
three_stages_6 = [Sol_liq, Ref_pet, Liq_OH]
three_stages_7 = [Sol_liq, Liq_OH, Liq_OH]
three_stages_8 = [Sol_liq, Liq_OH, Ref_pet]

scenarii = [two_stages_1, two_stages_2, two_stages_3, two_stages_4, three_stages_1, three_stages_2, three_stages_3, three_stages_4, three_stages_5, three_stages_6, three_stages_7, three_stages_8]        


def TwoStages(deltaV, Isp1, Isp2, k1, k2, b2=3, M_payload=3800):
    g0 = 9.80665
    Omega1 = k1 / (1 + k1)
    Omega2 = k2 / (1 + k2)
    deltaV_temp = 0
    it = 0 # Counts Lagrange Multiplier methods iterations

    while abs(deltaV - deltaV_temp) > 20:
        b1 = (1 / Omega1) * (1 - (Isp2 / Isp1) * (1 - Omega2 * b2))
        deltaV_1 = Isp1 * g0 * np.log(b1)
        deltaV_2 = Isp2 * g0 * np.log(b2)
        deltaV_temp = deltaV_1 + deltaV_2
        it += 1
        #print("abs(deltaV - deltaV_temp) : ", abs(deltaV - deltaV_temp))

        if deltaV_temp < deltaV:
            b2 += 0.05
        else:
            b2 -= 0.05

    a1 = (1 + k1) / b1 - k1
    a2 = (1 + k2) / b2 - k2

    Mi2 = M_payload / a2
    Mi1 = Mi2 / a1

    Me2 = (1 + a2) / (1 + k2) * Mi2
    Me1 = (1 + a1) / (1 + k1) * Mi1

    Ms2 = k2 * Me2
    Ms1 = k1 * Me1

    Mtot = Mi1 + Mi2 + M_payload

    return deltaV_temp, Mi1, Mi2, Me1, Me2, Ms1, Ms2, Mtot, it


def ThreeStages(deltaV, Isp1, Isp2, Isp3, k1, k2, k3, b3=1, M_payload=3800):
    g0 = 9.80665
    Omega1 = k1 / (1 + k1)
    Omega2 = k2 / (1 + k2)
    Omega3 = k3 / (1 + k3)
    deltaV_temp = 0
    it = 0 # Counts Lagrange multiplier methods iterations

    while abs(deltaV - deltaV_temp) > 60:
        b2 = (1 / Omega2) * (1 - (Isp3 / Isp2) * (1 - Omega3 * b3))
        b1 = (1 / Omega1) * (1 - (Isp2 / Isp1) * (1 - Omega2 * b2))
        deltaV_1 = Isp1 * g0 * np.log(b1)
        deltaV_2 = Isp2 * g0 * np.log(b2)
        deltaV_3 = Isp3 * g0 * np.log(b3)
        deltaV_temp = deltaV_1 + deltaV_2 + deltaV_3
        it += 1
        #print("abs(deltaV - deltaV_temp) : ", abs(deltaV - deltaV_temp))

        if deltaV_temp < deltaV:
            b3 += 0.05
        else:
            b3 -= 0.05

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

    Mtot = Mi1 + Mi2 + Mi3 + M_payload

    return deltaV_temp, Mi1, Mi2, Mi3, Me1, Me2, Me3, Ms1, Ms2, Ms3, Mtot, it

def Checks(Ms, Mi, M_payload):
    """
    Small functions that checks if the mass specifications are respected
    """
    if 500 < Ms[0] < 100000: # First stage mass check
        print("\nStage 1 min and max structural mass : CHECK")
    else:
        print("\nStage 1 min and max structural mass : NOT RESPECTED\nStage 1 structural mass : ", Ms[0], "\nSpecifications : 500 < Ms1 < 100 000 kg.")
    if 200 < Ms[1] < 80000: # Second stage mass check
        print("\nStage 2 min and max structural mass : CHECK")
    else:
        print("\nStage 2 min and max structural mass : NOT RESPECTED\nStage 2 structural mass : ", Ms[1], "\nSpecifications : 200 < Ms2 < 80 000 kg.")
    if len(Ms) > 2: # Third stage mass check
        if 200 < Ms[2] < 50000:
            print("\nStage 3 min and max structural mass : CHECK")
        else:
            print("\nStage 3 min and max structural mass : NOT RESPECTED\nStage 3 structural mass : ", Ms[2], "\nSpecifications : 200 < Ms3 < 50 000 kg.")
    
    if len(Mi) > 2: # First stage mass equilibrium check
        if (Mi[0] > Mi[1] + Mi[2] + M_payload):
            print("\nFirst stage mass bigger than the mass of the rest of the launcher : CHECK")
        else:
            print("\nFirst stage mass bigger than the mass of the rest of the launcher : NOT RESPECTED")
        if (Mi[1] > Mi[2] + M_payload):
            print("\nMass of stage 2 bigger than the mass of the rest of the launcher : CHECK")
        else:
            print("\nMass of stage 2 bigger than the mass of the rest of the launcher : NOT RESPECTED")
    else:
        if (Mi[0] > Mi[1] + M_payload):
            print("\nFirst stage mass bigger than the mass of the rest of the launcher : CHECK")


def Test(scenarii):
    for i, scenario in enumerate(scenarii):
        if len(scenario) == 2: # Two stages scenario
            Isp1 = scenario[0].Isp_mean
            Isp2 = scenario[1].Ispv
            k1 = scenario[0].k
            k2 = scenario[1].k
            
            deltaV_true, Mi1, Mi2, Me1, Me2, Ms1, Ms2, Mtot, it = TwoStages(deltaV_prop, Isp1, Isp2, k1, k2)
            
            Mi = [Mi1, Mi2] # Mass lists for specifications tests.
            Ms = [Ms1, Ms2]

            print("\n#-----------\nScenario n°", i, " :\nNumber of iterations : ", it)
            print("\nTotal mass : Mtot = ", Mtot)
            print("Total stage masses : Mi1 = ", Mi1, "; Mi2 = ", Mi2)
            print("Structural masses : Ms1 = ", Ms1 ,", Ms2 = ", Ms2)
            print("Propellant masses : Me1 = ", Me1, "; Me2 = ", Me2)
            
        else:   # Three stages scenario
            Isp1 = scenario[0].Isp_mean
            Isp2 = scenario[1].Ispv
            Isp3 = scenario[2].Ispv
            k1 = scenario[0].k
            k2 = scenario[1].k
            k3 = scenario[2].k
            
            deltaV_temp, Mi1, Mi2, Mi3, Me1, Me2, Me3, Ms1, Ms2, Ms3, Mtot, it = ThreeStages(deltaV_prop, Isp1, Isp2, Isp3, k1, k2, k3)
            
            Mi = [Mi1, Mi2, Mi3] # Mass lists for specifications tests.
            Ms = [Ms1, Ms2, Ms3]

            print("\n#-----------\nScenario n°", i, " :\nNumber of iterations : ", it)
            print("\nTotal mass : Mtot = ", Mtot)
            print("Total stage masses : Mi1 = ", Mi1, "; Mi2 = ", Mi2)
            print("Structural masses : Ms1 = ", Ms1 ,", Ms2 = ", Ms2)
            print("Propellant masses : Me1 = ", Me1, "; Me2 = ", Me2)

        Checks(Ms, Mi, M_payload)

        i += 1


# Test if the specifications are respected for each mass propellant scenario
#Test(scenarii)