# An eigenvalue solver that solves for the laminar flame speed
# following the prescription of Chamulak et al 2007 and
# Khoklov et al 1995.
# Author - Josh Martin 2026

# Loading dependencies
import matplotlib.pyplot as plt
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
from pynucastro.conductivity.conductivity import Conductivity
print("Importing composition...")
from pynucastro.nucdata import Composition
print("Importing Electron EOS")
from pynucastro.eos.electron_eos import ElectronEOS

rl = pyna.ReacLibLibrary()

all_nuclei = ["p", "he4",
              "c12", "n13",
              "o16", "f17", 
              "ne20", "na23",
              "mg24", "al27", "si28",
              "p31", "s32", "cl35",
              "ar36", "k39", "ca40",
              "sc43", "ti44", "v47",
              "cr48", "mn51",
              "fe52","co55","ni56"]

lib = rl.linking_nuclei(all_nuclei)

# Load your EOS
print("Loading EOS")
eos = pyna.StellarEOS()
print("EOS Loaded")

# Loading in Electron EOS
ele_eos = ElectronEOS(include_positrons=True)
print("Electron EOS Loaded")

# Loading in a Network
print("Loading in Network")
net = pyna.PythonNetwork(libraries=[lib])
print("Network Loaded")

def plot_F(t, F, S_lam):
    fig = plt.figure()
    ax = fig.add_subplot(111) #(111) meaning (nrows, ncols, index) 
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Heat Flux")

    ax.plot(t[1:], np.abs(F[1:]), "-r", lw = 3)

    plt.tick_params(axis='both')  # Make tick labels larger

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1e-13, 1e-6)
    #ax.set_ylim(1e18, 1e23)

    plt.savefig(f"/gpfs/projects/CalderGroup/JoshSpace/flame_speed_plots/HeatFlux_S_lam_{S_lam}.png")
    plt.close(fig)

def plot_T(t, T, S_lam):
    fig = plt.figure()
    ax = fig.add_subplot(111) #(111) meaning (nrows, ncols, index) 
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temp (K)")

    ax.plot(t[1:], T[1:], "-r", lw = 3)

    plt.tick_params(axis='both')  # Make tick labels larger

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(1e-13, 1e-6)

    plt.savefig(f"/gpfs/projects/CalderGroup/JoshSpace/flame_speed_plots/Temp_S_lam_{S_lam}.png")
    plt.close(fig)

def plot_X(t, Y, S_lam, all_nuclei, A):
    fig = plt.figure()
    ax = fig.add_subplot(111) #(111) meaning (nrows, ncols, index) 
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mass Fraction")

    # X = Y * A
    ax.plot(t[1:], A[2] * Y[2, 1:], "-r", label=all_nuclei[2], lw = 2)
    ax.plot(t[1:], A[4] * Y[4, 1:], "-b", label=all_nuclei[4], lw = 2)
    ax.plot(t[1:], A[6] * Y[6, 1:], "-g", label=all_nuclei[6], lw = 2)
    ax.plot(t[1:], A[8] * Y[8, 1:], color="gold", label=all_nuclei[8], lw = 2)
    ax.plot(t[1:], A[10]* Y[10, 1:], color="purple", label=all_nuclei[10], lw = 2)
    ax.plot(t[1:], A[24]* Y[24, 1:], color="pink", label=all_nuclei[24], lw = 2)

    plt.tick_params(axis='both')  # Make tick labels larger

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(5e-11, 1e-6)
    ax.set_ylim(1e-6, 1)
    ax.legend()

    plt.savefig(f"/gpfs/projects/CalderGroup/JoshSpace/flame_speed_plots/Comp_S_lam_{S_lam}.png")
    plt.close(fig)

def positive_F_check(t, y, S_lam, p_fixed, rho_fuel):
    return y[0]

#def divergent_F_check(t, y, S_lam, p_fixed, rho_fuel):
#    return (1e36 + y[0])

def divergent_T_check(t, y, S_lam, p_fixed, rho_fuel):
    return (1e10 - y[1])

positive_F_check.terminal = True
positive_F_check.direction = 1

#divergent_F_check.terminal = True
#divergent_F_check.direction = -1

divergent_T_check.terminal = True
divergent_T_check.direction = -1


def invert_eos_for_rho(T, p_fixed, comp):
    # This function finds rho via another shooting method
    def shooter(rho, T, comp):
        state = eos.pe_state(rho, T, comp)
        residual = state.p - p_fixed
        return residual

    rho_lower_bound = 1e6 # may need to change this
    rho_upper_bound = 1e10 # may need to change this

    rho_from_inverted_eos = brentq(
        f=shooter,
        a=rho_lower_bound,
        b=rho_upper_bound,
        args=(T, comp)
    )

    state_final = eos.pe_state(rho_from_inverted_eos, T, comp)

    return state_final, rho_from_inverted_eos


def RHS(t, y, S_lam, p_fixed, rho_fuel):
    # 0) Unpack variables from y
    F, T, *Y = y
    Y = np.array(Y)

    comp = Composition(net.unique_nuclei)
    comp.set_molar_array(Y)

    # 1) get rho, cp from EOS
    state, rho = invert_eos_for_rho(T, p_fixed, comp)
    cp  = state.c_p
    print(f"Values for t = {t:.2e}")
    print(f"c_p = {cp:.2e}")
    print(f"rho = {rho:.2e}")

    thermo = pyna.ThermoState(rho=rho, T=T, comp=comp)

    # 2) Create RHS of dYdt diff eqs
    ydots = net.evaluate_ydots(thermo, screen_func=chugunov_2007)
    dYdt = np.array([ydots[nuc] for nuc in net.unique_nuclei])
    # 3) Compute epsilon via dYdt equations
    epsilon = net.evaluate_energy_generation(thermo, screen_func=chugunov_2007)
    print(f"epsilon = {epsilon:.2e}")
    
    ele_state, pos_state = ele_eos.pe_state(rho, T, comp)
    p_ele = ele_state.p + pos_state.p

    # 4) Conductivity -> K & opacities
    K = Conductivity.get_K(state, comp, rho, T, p_ele, net)
    print(f"K = {K:.2e}")
    # 5) Compute u
    u = rho_fuel * S_lam / rho
    print(f"u = {u:.2e}")
    # 6) dTdt and dFdt
    dTdt = -u * F / K
    dFdt = (u * rho * epsilon) + (u * S_lam * rho_fuel * cp * F / K)

    return [dFdt, dTdt, *dYdt]


def integrator(S_lam, y0, p_fixed, tmax, rho_fuel, all_nuclei, A):
    sol = solve_ivp(
        fun=RHS,
        t_span=[0, tmax], 
        y0=y0,
        args=(S_lam, p_fixed, rho_fuel),
        events=(positive_F_check, divergent_T_check), 
        method='BDF', 
        rtol=1e-5, 
        atol=1e-8
        )

    # Extracting the solution
    F_final = sol.y[0, -1] # indexed as [state variable, time point]

    # Extracting full solutions for plots
    t = sol.t
    F = sol.y[0, :]
    T = sol.y[1, :]
    Y = sol.y[2:, :]

    plot_F(t, F, S_lam)
    plot_T(t, T, S_lam)
    plot_X(t, Y, S_lam, all_nuclei, A)

    print(f"Tried S_lam = {S_lam}, returned F = {F_final:.2e}")

    if sol.t_events[0].size > 0:
        print("Integrator stopped due to positive F")
        return 1.0
    elif sol.t_events[1].size > 0:
        print("Integrator stopped because F diverged")
        return -1.0
    else:
        print("Integrator stopped because tmax was reached")
        F_peak = np.max(np.abs(F))
        F_norm = F_final / F_peak
        print(f"F_norm = {F_norm}")
        return -1 * F_norm


# --- Setting Inital Values ---
T_fuel = 1e8 # Need to set
rho_fuel = 2e9 # Need to set

F0 = -1 # arbitrarily small negative value
tmax = 1e-6 # will need to change this

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

comp = pyna.Composition(net.unique_nuclei)

for nuc in net.unique_nuclei:
    comp.set_nuc(nuc, X0[ids[nuc]])

# Finding the Fixed Pressure
state = eos.pe_state(rho_fuel, T_fuel, comp)
p_fixed = state.p
print(f"Fixed pressure p = {p_fixed}")


## Using brentq to Find The Root
S_lam_lower_bound = 4.375e6 #cm/s
S_lam_upper_bound = 4.400e6 #cm/s


print("Beginning Shooter")
S_lam_solution, details = brentq(
    f=integrator,
    a=S_lam_lower_bound,
    b=S_lam_upper_bound,
    args=(y0, p_fixed, tmax, rho_fuel, all_nuclei, A),
    full_output=True
)

'''
print("Beginning Test S_lams")
S_lams = np.linspace(2e5, 1e6, 20)
for S_lam in S_lams:
    print(f"Testing S_lam = {S_lam:.2e}")
    integrator(S_lam, y0, p_fixed, tmax, rho_fuel, all_nuclei, A)
'''
print(f"S_lam: {S_lam_solution:.6f}")
print(f"Converged in {details.iterations} iterations.")

# With optimized S_lam, we need to find delta_lam