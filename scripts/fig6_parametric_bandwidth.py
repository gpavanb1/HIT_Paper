#!/usr/bin/env python3
"""
Figure 6: Parametric Transition of f(r) with Spectral Bandwidth.
Demonstrates the suppression of the negative loop in f(r) as spectral bandwidth sigma
broadens relative to peak wavenumber k0.
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
    print("Generating Figure 6: Parametric transition vs bandwidth...")
    k1_grid_nb = np.linspace(0.001, 25.0, 250)
    sigmas = [0.3, 0.6, 1.2, 2.5]
    r_scan = np.linspace(0.001, 2.5, 200)
    min_f_vals = []
    sigma_dense = np.linspace(0.25, 4.0, 16)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    for s in sigmas:
        E11_arr = np.zeros_like(k1_grid_nb)
        for i, k1 in enumerate(k1_grid_nb):
            integrand = lambda k: (narrowband_batchelor(k, sigma=s)/k) * (1.0 - (k1**2)/(k**2))
            E11_arr[i], _ = integrate.quad(integrand, k1, 30.0, limit=200)
        
        interp_s = interp1d(k1_grid_nb, E11_arr, kind='cubic', bounds_error=False, fill_value=0.0)
        norm_s, _ = integrate.quad(interp_s, 0.0, 25.0, limit=200)
        
        f_arr = np.zeros_like(r_scan)
        for i, r in enumerate(r_scan):
            integrand = lambda k1: interp_s(k1) * np.cos(k1 * r)
            val, _ = integrate.quad(integrand, 0.0, 25.0, limit=200)
            f_arr[i] = val / norm_s
            
        ax1.plot(r_scan, f_arr, lw=1.8, label=rf'$\sigma = {s}$')

    ax1.axhline(0, color='black', lw=0.8, ls=':')
    ax1.set_xlabel(r'Separation $r$')
    ax1.set_ylabel(r'$f(r)$')
    ax1.set_title(r'(a) Longitudinal correlation $f(r)$ vs $\sigma$')
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', fontsize=9.5)

    for s in sigma_dense:
        E11_arr = np.zeros_like(k1_grid_nb)
        for i, k1 in enumerate(k1_grid_nb):
            integrand = lambda k: (narrowband_batchelor(k, sigma=s)/k) * (1.0 - (k1**2)/(k**2))
            E11_arr[i], _ = integrate.quad(integrand, k1, 30.0, limit=200)
        interp_s = interp1d(k1_grid_nb, E11_arr, kind='cubic', bounds_error=False, fill_value=0.0)
        norm_s, _ = integrate.quad(interp_s, 0.0, 25.0, limit=200)
        
        f_arr = np.zeros_like(r_scan)
        for i, r in enumerate(r_scan):
            integrand = lambda k1: interp_s(k1) * np.cos(k1 * r)
            val, _ = integrate.quad(integrand, 0.0, 25.0, limit=200)
            f_arr[i] = val / norm_s
        min_f_vals.append(np.min(f_arr))

    ax2.plot(sigma_dense / 10.0, min_f_vals, 'C0o-', lw=1.8, markersize=5)
    ax2.axhline(0, color='black', lw=0.8, ls=':')
    ax2.set_xlabel(r'Relative bandwidth $\sigma / k_0$')
    ax2.set_ylabel(r'$\min_r f(r)$')
    ax2.set_title(r'(b) Minimum of $f(r)$ vs Bandwidth')
    ax2.grid(True, alpha=0.25, linestyle='--')

    plt.tight_layout()
    out_pdf = os.path.join(FIGURES_DIR, 'fig6_parametric_bandwidth.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig6_parametric_bandwidth.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == '__main__':
    generate_figure()
