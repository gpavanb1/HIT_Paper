#!/usr/bin/env python3
"""
Figure 4: Narrowband Batchelor-Consistent Counterexample.
Demonstrates that an admissible, smooth, divergence-free narrowband spectrum with
Batchelor infrared behavior E(k) ~ k^4 produces an unconditionally monotonic E11(k1)
and a pronounced genuine negative loop in the longitudinal autocorrelation f(r).
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
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10.5,
    'figure.titlesize': 13,
    'lines.linewidth': 1.8,
    'figure.dpi': 300
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def narrowband_batchelor(k, k0=10.0, sigma=0.5, A=1.0, kL=1.0):
    ir_factor = (k/kL)**4 / (1.0 + (k/kL)**2)**2
    return A * ir_factor * np.exp(-0.5 * ((k - k0)/sigma)**2)


def generate_figure():
    print("Generating Figure 4: Narrowband counterexample...")
    k_grid_nb = np.linspace(0.01, 20.0, 300)
    E_nb = narrowband_batchelor(k_grid_nb)

    k1_grid_nb = np.linspace(0.001, 25.0, 250)
    E11_nb = np.zeros_like(k1_grid_nb)
    for i, k1 in enumerate(k1_grid_nb):
        integrand = lambda k: (narrowband_batchelor(k)/k) * (1.0 - (k1**2)/(k**2))
        E11_nb[i], _ = integrate.quad(integrand, k1, 30.0, limit=300)

    E11_interp = interp1d(k1_grid_nb, E11_nb, kind='cubic', bounds_error=False, fill_value=0.0)
    norm_nb, _ = integrate.quad(E11_interp, 0.0, 25.0, limit=300)

    r_grid_nb = np.linspace(0.001, 2.5, 250)
    f_nb = np.zeros_like(r_grid_nb)
    for i, r in enumerate(r_grid_nb):
        integrand = lambda k1: E11_interp(k1) * np.cos(k1 * r)
        val, _ = integrate.quad(integrand, 0.0, 25.0, limit=300)
        f_nb[i] = val / norm_nb

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.0, 3.2))

    ax1.plot(k_grid_nb, E_nb, 'C0-', lw=1.8)
    ax1.set_xlabel(r'$k$')
    ax1.set_ylabel(r'$E(k)$')
    ax1.set_title(r'(a) 3D Spectrum $E(k)$')
    ax1.grid(True, alpha=0.25, linestyle='--')

    ax2.plot(k1_grid_nb, E11_nb, 'C2-', lw=1.8)
    ax2.set_xlabel(r'$k_1$')
    ax2.set_ylabel(r'$E_{11}(k_1)$')
    ax2.set_title(r'(b) 1D Spectrum $E_{11}(k_1)$')
    ax2.grid(True, alpha=0.25, linestyle='--')

    ax3.plot(r_grid_nb, f_nb, 'C3-', lw=1.8)
    ax3.axhline(0, color='black', lw=0.8, ls=':')
    ax3.fill_between(r_grid_nb, f_nb, 0, where=(f_nb < 0), color='C3', alpha=0.25, label=f'Min $f(r) = {f_nb.min():.3f}$')
    ax3.set_xlabel(r'Separation $r$')
    ax3.set_ylabel(r'$f(r)$')
    ax3.set_title(r'(c) Negative Loop in $f(r)$')
    ax3.grid(True, alpha=0.25, linestyle='--')
    ax3.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', loc='upper right')

    plt.tight_layout()
    out_pdf = os.path.join(FIGURES_DIR, 'fig4_narrowband_counterexample.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig4_narrowband_counterexample.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == '__main__':
    generate_figure()
