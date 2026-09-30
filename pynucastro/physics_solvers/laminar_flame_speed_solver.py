# An eigenvalue solver that solves for the laminar flame speed
# following the prescription of Chamulak et al 2007 and
# Khoklov et al 1995.
# Author - Josh Martin 2026

# Loading dependencies
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import pynucastro as pyna
from pynucastro.screening import screen5 # Importing our specific screening rate

# Load your EOS
eos = pyna.StellarEOS()

# Loading in a Network
net = pyna.common_networks.approx13()

def get_init_comps(X4, X12, X16, X20):
    # NEEDS WORK
    return Y0_comp

def RHS(t, y, S_lam, p_fixed):
    # 0) Unpack variables from y
    F, T, *Y = y
    Y = np.array(Y)
    # 1) get rho, cp from EOS
    rho, cp = eos(T, p_fixed, Y) '''NEEDS WORK'''
    # 2) Integrate screened reaction network -> dY_i/dt, epsilon
    dYdt = net.rhs(T, rho, Y) '''NEEDS WORK'''
    # 3) Compute epsilon via dY_i/dt
    epsilon = net.energy_release(dYdt, Y) ''' NEEDS WORK '''
    # 4) Conductivity -> K & opacities
    K = conductivity(rho, T, Y) ''' NEEDS WORK '''
    # 5) dTdt and dFdt
    dTdt = -S_lam * F / K
    dFdt = (S_lam * rho * epsilon) - S_lam**2 * rho * cp * F / K

    return [dFdt, dTdt, *dYdt]


def integrator(S_lam, y0, p_fixed, tmax):
    sol = solve_ivp(
        fun=RHS,
        t_span=[0, tmax], 
        y0=y0,
        args=(S_lam, p_fixed), 
        method='BDF', 
        rtol=1e-5, 
        atol=1e-8
        )

    # Extracting the solution
    F_final = sol.y[0, -1] # indexed as [state variable, time point]

    # Returning the function we are trying to get to zero
    return F_final


# --- Setting Inital Values ---
T_fuel = 1e8
F0 = 1e-10 # arbitrarily small positive value
p_fixed = # Need to set (if needed)
tmax = 1e-6 # will need to change this

# Defining initial mass fractions
X4 = 0
X12 = 0.5
X16 = 0.5
X20 = 0

Y0_comp = get_init_comps(X4, X12, X16, X20)

# Initial state vector
y0 = [F0, T_fuel, *Y0_comp]


## Using brentq to Find The Root
S_lam_lower_bound = 1e4 #cm/s
S_lam_upper_bound = 1e8 #cm/s

S_lam_solution, details = brentq(
    f=integrator,
    a=S_lam_lower_bound,
    b=S_lam_upper_bound,
    args=(y0, p_fixed, tmax),
    full_output=True
)

print(f"S_lam: {S_lam_solution:.6f}")
print(f"Converged in {details.iterations} iterations.")

# With optimized S_lam, we need to find delta_lam