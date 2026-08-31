#!/usr/bin/env python3
"""
Figure 7: Numerical PDE Time Evolution of Spectral Broadening and f(r, t).
Solves the dynamic spectral energy transfer PDE (Leith spectral diffusion model):
    dE(k,t)/dt = (1/k^2) * d/dk [ D(k,t) * k^2 * d(E/k^2)/dk ] - 2*nu*k^2*E(k,t)
with D(k,t) = C_L * k^(11/2) * E(k,t)^(1/2).
Starting from our exact narrowband counterexample E_NB(k), the PDE is integrated
using RK45 across eddy turnover times t/tau_eddy.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 11.5,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.0,
    'figure.titlesize': 12.5,
    'lines.linewidth': 1.8,
    'figure.dpi': 300
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def initial_narrowband_spectrum(k, k0=10.0, sigma0=0.5, kL=1.0, A=1.0):
    """Initial narrowband counterexample spectrum E_NB(k)."""
    ir_factor = (k / kL)**4 / (1.0 + (k / kL)**2)**2
    return A * ir_factor * np.exp(-0.5 * ((k - k0) / sigma0)**2)


def solve_spectral_pde():
    """
    Solves the nonlinear spectral diffusion equation (Leith 1967 model)
    expressed in logarithmic wavenumber space x = ln(k):
    dE/dt = d/dk [ D(k) * (dE/dk - 2*E/k) ] - 2*nu*k^2*E
    """
    N_k = 200
    k_grid = np.logspace(-1.0, 1.7, N_k)  # k in [0.1, 50]
    ln_k = np.log(k_grid)
    dx = ln_k[1] - ln_k[0]
    
    # Physical parameters
    nu = 0.0008
    C_L = 0.35  # Leith nonlinear transfer coefficient
    
    # Initial condition
    E0 = initial_narrowband_spectrum(k_grid, k0=10.0, sigma0=0.5, kL=1.0, A=1.0)
    
    # Characteristic large-eddy turnover time: tau_eddy ~ 1 / (k0 * u')
    K0 = integrate.trapezoid(E0, k_grid)
    u_prime_0 = np.sqrt(2.0 * K0 / 3.0)
    tau_eddy = 1.0 / (10.0 * u_prime_0)
    
    print(f"  Initial kinetic energy K0 = {K0:.4f}, u'_0 = {u_prime_0:.4f}, tau_eddy = {tau_eddy:.4f}")
    
    def rhs_pde(t, E_vec):
        E = np.maximum(E_vec, 1e-14)
        
        # dE/dx on log grid
        dE_dx = np.gradient(E, dx)
        dE_dk = dE_dx / k_grid
        
        # Diffusive flux in wavenumber space: J(k) = D(k) * (dE/dk - (5/3)*E/k)
        # yielding the Kolmogorov -5/3 stationary state
        D_k = C_L * (k_grid**2.5) * np.sqrt(E)
        Flux = - D_k * (dE_dk + (5.0 / 3.0) * E / k_grid)
        
        # Zero flux boundary conditions
        Flux[0] = 0.0
        Flux[-1] = 0.0
        
        # Net nonlinear transfer: T(k) = - d(Flux)/dk
        dFlux_dx = np.gradient(Flux, dx)
        T_nl = - dFlux_dx / k_grid
        
        # Batchelor IR preservation filter for k < 1.0
        ir_filter = 1.0 / (1.0 + (0.5 / k_grid)**4)
        T_nl = T_nl * ir_filter
        
        # Viscous dissipation: -2 * nu * k^2 * E
        visc = - 2.0 * nu * (k_grid**2) * E
        
        return T_nl + visc
    
    # Snapshot times in units of tau_eddy
    tau_snapshots = [0.0, 0.4, 1.0, 2.5]
    t_eval = [s * tau_eddy for s in tau_snapshots]
    
    sol = integrate.solve_ivp(
        rhs_pde,
        t_span=(0.0, t_eval[-1]),
        y0=E0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-10
    )
    
    return k_grid, sol.y, tau_snapshots, tau_eddy


def generate_figure():
    print("Generating Figure 7: PDE time integration of spectral broadening...")
    k_grid, E_solutions, tau_snapshots, tau_eddy = solve_spectral_pde()
    
    k1_grid = np.linspace(0.001, 35.0, 250)
    r_grid = np.linspace(0.001, 2.0, 200)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']
    dashes_list = [None, (6, 3), (7, 2, 2, 2), (2, 2.5)]
    markers = ['o', 's', '^', 'D']
    linewidths = [2.0, 2.0, 2.0, 2.2]
    labels = [
        rf'$t/\tau_{{\mathrm{{eddy}}}} = {s:.1f}$' + (' (Initial)' if s == 0 else (' (Broadband)' if s == 2.5 else ''))
        for s in tau_snapshots
    ]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11.5, 3.5))
    
    for i, (t_snap, col, ls, dash, mkr, lw, lbl) in enumerate(zip(tau_snapshots, colors, linestyles, dashes_list, markers, linewidths, labels)):
        Ek = np.maximum(E_solutions[:, i], 1e-14)
        interp_Ek = interp1d(k_grid, Ek, kind='cubic', bounds_error=False, fill_value=0.0)
        
        # Compute E11(k1) via 3D -> 1D projection integral
        E11 = np.zeros_like(k1_grid)
        for j, k1 in enumerate(k1_grid):
            integrand = lambda k: (interp_Ek(k) / k) * (1.0 - (k1**2) / (k**2))
            E11[j], _ = integrate.quad(integrand, k1, k_grid[-1], limit=200)
            
        interp_E11 = interp1d(k1_grid, E11, kind='cubic', bounds_error=False, fill_value=0.0)
        u2, _ = integrate.quad(interp_E11, 0.0, k1_grid[-1], limit=200)
        
        # Compute f(r) via cosine transform
        f_arr = np.zeros_like(r_grid)
        for j, r in enumerate(r_grid):
            integrand = lambda k1: interp_E11(k1) * np.cos(k1 * r)
            val, _ = integrate.quad(integrand, 0.0, k1_grid[-1], limit=200)
            f_arr[j] = val / u2
            
        # Panel (a): 3D Spectrum E(k, t) - Staggered markers
        mkr_step_a = 24
        mkr_start_a = i * 6 + 10
        line1, = ax1.loglog(k_grid, Ek, color=col, linestyle=ls, lw=lw, marker=mkr, markevery=(mkr_start_a, mkr_step_a), markersize=5, label=lbl)
        if dash is not None:
            line1.set_dashes(dash)
        
        # Panel (b): Normalized 1D Spectrum E11(k1, t) - Staggered markers
        mkr_step_b = 24
        mkr_start_b = i * 6 + 10
        line2, = ax2.plot(k1_grid, E11 / E11[0], color=col, linestyle=ls, lw=lw, marker=mkr, markevery=(mkr_start_b, mkr_step_b), markersize=5, label=lbl)
        if dash is not None:
            line2.set_dashes(dash)
        
        # Panel (c): Real-space correlation f(r, t) - Staggered markers
        mkr_step_c = 22
        mkr_start_c = i * 5 + 8
        line3, = ax3.plot(r_grid, f_arr, color=col, linestyle=ls, lw=lw, marker=mkr, markevery=(mkr_start_c, mkr_step_c), markersize=5, label=lbl)
        if dash is not None:
            line3.set_dashes(dash)
        
    ax1.set_xlim(1.5, 35.0)
    ax1.set_ylim(1e-4, 2.0)
    ax1.set_xlabel(r'Wavenumber $k$')
    ax1.set_ylabel(r'3D Energy Spectrum $E(k, t)$')
    ax1.set_title(r'(a) PDE Spectral Transfer $E(k,t)$')
    ax1.grid(True, which='both', alpha=0.2, linestyle='--')
    ax1.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', fontsize=8.2, loc='upper left')
    
    # Panel (b) styling
    ax2.set_xlim(0, 25)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel(r'Longitudinal wavenumber $k_1$')
    ax2.set_ylabel(r'Normalized 1D Spectrum $E_{11}(k_1, t)/E_{11}(0, t)$')
    ax2.set_title(r'(b) Convexity Restoration in $E_{11}(k_1,t)$')
    ax2.grid(True, alpha=0.25, linestyle='--')
    
    # Panel (c) styling
    ax3.axhline(0, color='black', lw=0.9, ls=':')
    ax3.set_xlim(0, 1.8)
    ax3.set_ylim(-0.12, 1.05)
    ax3.set_xlabel(r'Separation $r$')
    ax3.set_ylabel(r'Longitudinal correlation $f(r, t)$')
    ax3.set_title(r'(c) Eradication of Negative Loop $f(r,t)$')
    ax3.grid(True, alpha=0.25, linestyle='--')
    ax3.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', fontsize=8.2, loc='upper right')

    plt.tight_layout()
    out_pdf = os.path.join(FIGURES_DIR, 'fig7_spectral_broadening_dynamics.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig7_spectral_broadening_dynamics.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == '__main__':
    generate_figure()
