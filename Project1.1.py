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
propellants = [Ref_pet, Liq_OH, Sol_liq]

# Different possible scenarii
two_stages = [(prop1, prop2) if prop1 != Liq_OH and prop2 != Sol_liq  else () 
                    for prop1 in propellants for prop2 in propellants]
three_stages = [(prop1, prop2, prop3) if prop1 != Liq_OH and prop2 != Sol_liq and prop3 != Sol_liq else () 
                    for prop1 in propellants for prop2 in propellants for prop3 in propellants]
scenarii =  list(filter(None, two_stages + three_stages))


def NStages(n, deltaV, Isp, k, bn=10, M_payload=3800, threshold = 0.1, step = 0.00001, itmax = 1000000):

    assert(len(Isp) == n)
    assert(len(k) == n)

    g0 = 9.80665
    Omega = k / (1 + k)
    deltaV_current = 0
    it = 0 # Counts Lagrange multiplier methods iterations

    while abs(deltaV_prop - deltaV_current) > threshold:
        

        b = np.zeros((n,)); b[n-1] = bn
        for j in range(n-2, -1, -1):
            b[j] =  (1 / Omega[j]) * (1 - ((Isp[j+1] / Isp[j]) * (1 - (Omega[j] * b[j+1]))))
            if b[j] < 1:
                raise RuntimeError("b < 1")
 
        deltaVs = Isp * g0 * np.log(b)
        deltaV_current = np.sum(deltaVs)
        it += 1
        #print("abs(deltaV - deltaV_temp) : ", abs(deltaV - deltaV_temp))

        bn += step if deltaV_current < deltaV_prop else -step
        if it > itmax :
            raise RuntimeError("iterations limit")


    a = ((1 + k) / b) - k

    Mi = np.zeros((n+1,)); Mi[n] = M_payload
    for j in range(n-1, -1, -1):
        Mi[j] = Mi[j+1] / a[j]

    Me = ((1 - a) / (1 + k)) * Mi[:n]
    Ms = k * Me
    Mtot = np.sum(Mi)

    return deltaV_current, Mi, Me, Ms, Mtot, it, b, a, Omega


    

def Checks(Ms, Mi, M_payload):
    """
    Small functions that checks if the mass specifications are respected
    """
    if 500 < Ms[0] < 100000: # First stage mass check
        print("\nStage 1 min and max structural mass : CHECK")
    else:
        print("\nStage 1 min and max structural mass : NOT RESPECTED\nStage 1 structural mass : ", Ms[0], "\nSpecifications : 500 < Ms1 < 100 000 kg.")
        return False
    if 200 < Ms[1] < 80000: # Second stage mass check
        print("\nStage 2 min and max structural mass : CHECK")
    else:
        print("\nStage 2 min and max structural mass : NOT RESPECTED\nStage 2 structural mass : ", Ms[1], "\nSpecifications : 200 < Ms2 < 80 000 kg.")
        return False
    if len(Ms) > 2: # Third stage mass check
        if 200 < Ms[2] < 50000:
            print("Stage 3 min and max structural mass : CHECK")
        else:
            print("Stage 3 min and max structural mass : NOT RESPECTED\nStage 3 structural mass : ", Ms[2], "\nSpecifications : 200 < Ms3 < 50 000 kg.")
            return False
    if len(Mi) > 2: # First stage mass equilibrium check
        if (Mi[0] > Mi[1] + Mi[2] + M_payload):
            print("First stage mass bigger than the mass of the rest of the launcher : CHECK")
        else:
            print("First stage mass bigger than the mass of the rest of the launcher : NOT RESPECTED")
            return False
        if (Mi[1] > Mi[2] + M_payload):
            print("Mass of stage 2 bigger than the mass of the rest of the launcher : CHECK")
        else:
            print("Mass of stage 2 bigger than the mass of the rest of the launcher : NOT RESPECTED")
            return False
    else:
        if (Mi[0] > Mi[1] + M_payload):
            print("First stage mass bigger than the mass of the rest of the launcher : CHECK")

    return True


def Test(scenarii):
    for i, scenario in enumerate(scenarii):

        Isp = np.array([prop.Isp_mean if i == 0 else prop.Ispv for i,prop in enumerate(scenario)])
        k = np.array([prop.k for prop in scenario])

        try:
            deltaV, Mi, Me, Ms, Mtot, it, b, a, Omega = NStages(len(scenario), deltaV_prop, Isp, k)
        except RuntimeError as e :
            print("\n#-----------\nScenario n°", i, " : ",[prop.code for prop in scenario],"\nFailed : ", str(e))
            continue
                

        print("\n#-----------\nScenario n°", i, " : ",[prop.code for prop in scenario],"\nNumber of iterations : ", it)
        print("Exact dV : deltaV = ", deltaV)
        print("B coeffs : b = ", b)
        print("A coeffs : a = ", a)
        print("Omega coeffs : Omega = ", Omega)
        print("Total mass : Mtot = ", Mtot)
        print("Total stage masses : Mi = ", Mi)
        print("Structural masses : Ms = ", Ms)
        print("Propellant masses : Me = ", Me)

        Checks(Ms, Mi, M_payload)

        i += 1


# Test if the specifications are respected for each mass propellant scenario
Test(scenarii)