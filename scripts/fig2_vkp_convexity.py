#!/usr/bin/env python3
"""
Figure 2: von Karman-Pao Spectrum, 1D Longitudinal Spectrum E11(k1), and Convexity d2E11/dk1^2.
Demonstrates the finite near-origin concavity region (d2E11/dk1^2 < 0) resulting from
Batchelor infrared scaling and the convex cascade region.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
from scipy import integrate
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


def von_karman_pao_spectrum(k, L=1.0, u_rms=1.0, nu=1e-5):
    C_k = 1.5
    epsilon = u_rms**3 / L
    eta = (nu**3 / epsilon)**0.25
    kL = k * L
    E = 1.5 * u_rms**2 * L * (kL)**4 / (1.0 + kL**2)**(17.0 / 6.0)
    E *= np.exp(-1.5 * C_k * (k * eta)**(4.0 / 3.0))
    return E


def E11_vkp_single(k1):
    if k1 <= 1e-10:
        val, _ = integrate.quad(lambda k: von_karman_pao_spectrum(k)/k, 1e-10, 1000.0, limit=200)
    else:
        val, _ = integrate.quad(lambda k: (von_karman_pao_spectrum(k)/k)*(1.0 - (k1**2)/(k**2)), k1, 1000.0, limit=200)
    return val


def d2E11_vkp_single(k1):
    term1 = 2.0 * von_karman_pao_spectrum(k1) / (k1**2)
    term2, _ = integrate.quad(lambda k: 2.0 * von_karman_pao_spectrum(k) / (k**3), k1, 1000.0, limit=200)
    return term1 - term2


def generate_figure():
    print("Generating Figure 2: VKP spectrum and derivatives...")
    k_vals = np.logspace(-2, 2, 200)
    E_vals = [von_karman_pao_spectrum(k) for k in k_vals]

    k1_vals = np.linspace(0.005, 4.0, 100)
    E11_vals = np.array([E11_vkp_single(k) for k in k1_vals])
    d2E11_vals = np.array([d2E11_vkp_single(k) for k in k1_vals])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.4))

    ax1.loglog(k_vals, E_vals, color='#1f77b4', linestyle='-', lw=2.0, label=r'$E(k)$ (von Kármán–Pao)')
    ax1.plot(k1_vals, E11_vals, color='#2ca02c', linestyle='--', lw=2.0, label=r'$E_{11}(k_1)$ (1D longitudinal)')
    ax1.set_xlabel(r'Wavenumber $k, k_1$')
    ax1.set_ylabel(r'Energy spectrum')
    ax1.set_title(r'(a) 3D and 1D Energy Spectra')
    ax1.grid(True, which='both', alpha=0.25, linestyle='--')
    ax1.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8')

    ax2.plot(k1_vals, d2E11_vals, color='#d62728', linestyle='-', lw=1.8)
    ax2.axhline(0, color='black', lw=0.8, ls=':')
    ax2.fill_between(k1_vals, d2E11_vals, 0, where=(d2E11_vals < 0), color='#d62728', alpha=0.2, label=r'Concave ($d^2E_{11}/dk_1^2 < 0$)')
    ax2.fill_between(k1_vals, d2E11_vals, 0, where=(d2E11_vals >= 0), color='#2ca02c', alpha=0.15, label=r'Convex ($d^2E_{11}/dk_1^2 \geq 0$)')
    ax2.set_xlabel(r'Wavenumber $k_1$')
    ax2.set_ylabel(r'$d^2E_{11}/dk_1^2$')
    ax2.set_title(r'(b) Convexity of $E_{11}(k_1)$')
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8')

    plt.tight_layout()
    out_pdf = os.path.join(FIGURES_DIR, 'fig2_vkp_convexity.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig2_vkp_convexity.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == '__main__':
    generate_figure()
