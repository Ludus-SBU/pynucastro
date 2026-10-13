# An eigenvalue solver that solves for the laminar flame speed
# following the prescription of Chamulak et al 2007 and
# Khoklov et al 1995.
# Author - Josh Martin 2026

# Loading dependencies
print("Importing numpy...")
import numpy as np
print("Importing solve_ivp...")
from scipy.integrate import solve_ivp
print("Importing brentq...")
from scipy.optimize import brentq
print("Importing pynucastro...")
import pynucastro as pyna
print("Importing screening...")
from pynucastro.screening import chugunov_2007 # Importing our specific screening rate
print("Importing conductivity...")
from pynucastro.conductivity import conductivity

# Load your EOS
print("Loading EOS")
eos = pyna.StellarEOS()
print("EOS Loaded")

# Loading in a Network
print("Loading in Network")
net = pyna.common_networks.aprox13()
print("Network Loaded")

def negative_F_check(t, y, S_lam, p_fixed, rho_fuel):
    return y[0]

def divergent_F_check(t, y, S_lam, p_fixed, rho_fuel):
    return (1e34 - y[0])

negative_F_check.terminal = True
negative_F_check.direction = -1

divergent_F_check.terminal = True
divergent_F_check.direction = -1


def invert_eos_for_rho(T, p_fixed, Y):
    # This function finds rho via another shooting method
    def shooter(rho, T, Y):
        state = eos.pe_state(rho, T, Y)
        residual = state.p - p_fixed
        return residual

    rho_lower_bound = 1e6 # may need to change this
    rho_upper_bound = 1e10 # may need to change this

    rho_from_inverted_eos = brentq(
        f=shooter,
        a=rho_lower_bound,
        b=rho_upper_bound,
        args=(T, Y)
    )

    state_final = eos.pe_state(rho_from_inverted_eos, T, Y)

    return state_final


def RHS(t, y, S_lam, p_fixed, rho_fuel):
    # 0) Unpack variables from y
    F, T, *Y = y
    Y = np.array(Y)
    # 1) get rho, cp from EOS
    state = invert_eos_for_rho(T, p_fixed, Y)
    rho = state.rho
    cp  = state.cp
    # 2) Create RHS of dYdt diff eqs
    dYdt = net.rhs(t, Y, rho, T, screen_func=chugunov_2007)
    # 3) Compute epsilon via dYdt equations
    epsilon = net.energy_release(dYdt)
    # 4) Conductivity -> K & opacities
    K = conductivity.get_K(state, net)
    # 5) Compute u
    u = rho_fuel * S_lam / rho
    # 6) dTdt and dFdt
    dTdt = -u * F / K
    dFdt = (u * rho * epsilon) - u * S_lam * rho_fuel * cp * F / K

    return [dFdt, dTdt, *dYdt]


def integrator(S_lam, y0, p_fixed, tmax, rho_fuel):
    sol = solve_ivp(
        fun=RHS,
        t_span=[0, tmax], 
        y0=y0,
        args=(S_lam, p_fixed, rho_fuel),
        events=(negative_F_check, divergent_F_check), 
        method='BDF', 
        rtol=1e-5, 
        atol=1e-8
        )

    # Extracting the solution
    F_final = sol.y[0, -1] # indexed as [state variable, time point]

    print(f"Tried S_lam = {S_lam}, returned F = {F_final}")

    # Returning the function we are trying to get to zero
    return F_final


# --- Setting Inital Values ---
T_fuel = 1e9 # Need to set
rho_fuel = 2e9 # Need to set

F0 = 2e-8 # arbitrarily small positive value
tmax = 1e-8 # will need to change this

# Setting initial mass fractions
species = net.unique_nuclei

X0 = np.zeros(len(species))
ids = {} # creating a dictionary of ids so that we can change the network at will

for id, nuc in enumerate(species):
    ids[nuc] = id

print(ids)

X0[ids[pyna.Nucleus("he4")]] = 0
X0[ids[pyna.Nucleus("c12")]] = 0.5
X0[ids[pyna.Nucleus("o16")]] = 0.5
X0[ids[pyna.Nucleus("ne20")]] = 0
print("Initial mass fractions set")

A  = np.array([nuc.A for nuc in net.unique_nuclei])
Y0 = X0/A

# Initial state vector
y0 = [F0, T_fuel, *Y0]

# Finding the Fixed Pressure
state = eos.pe_state(rho_fuel, T_fuel, Y0)
p_fixed = state.p
print(f"Fixed pressure p = {p_fixed}")


## Using brentq to Find The Root
S_lam_lower_bound = 1e4 #cm/s
S_lam_upper_bound = 1e8 #cm/s

print("Beginning Shooter")
S_lam_solution, details = brentq(
    f=integrator,
    a=S_lam_lower_bound,
    b=S_lam_upper_bound,
    args=(y0, p_fixed, tmax, rho_fuel),
    full_output=True
)

print(f"S_lam: {S_lam_solution:.6f}")
print(f"Converged in {details.iterations} iterations.")

# With optimized S_lam, we need to find delta_lam