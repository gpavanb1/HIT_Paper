#!/usr/bin/env python3
"""
Figure 1: Schoenberg Kernel and Longitudinal Projection Kernel.
Plots the isotropic Schoenberg projection kernel Omega_3(x) = sin(x)/x and the
longitudinal projection kernel K_11(x) = 3(sin(x) - x*cos(x))/x^3 in R^3,
illustrating their oscillatory, sign-changing behavior.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
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


def generate_figure():
    print("Generating Figure 1: Schoenberg Kernel...")
    x = np.linspace(0.01, 16, 1000)
    omega3 = np.sin(x) / x
    kernel_11 = 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(x, omega3, 'C0-', label=r'Schoenberg isotropic kernel $\Omega_3(x) = \frac{\sin x}{x}$')
    ax.plot(x, kernel_11, 'C1--', label=r'Longitudinal kernel $K_{11}(x) = 3\left(\frac{\sin x - x\cos x}{x^3}\right)$')
    ax.axhline(0, color='black', lw=0.8, ls=':')
    ax.fill_between(x, omega3, 0, where=(omega3 < 0), color='C0', alpha=0.15)
    ax.fill_between(x, kernel_11, 0, where=(kernel_11 < 0), color='C1', alpha=0.15)
    ax.set_xlabel(r'Dimensionless separation $x = kr$')
    ax.set_ylabel(r'Kernel value')
    ax.set_title(r'Sign-changing projection kernels in $\mathbb{R}^3$')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8')

    out_pdf = os.path.join(FIGURES_DIR, 'fig1_schoenberg_kernel.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig1_schoenberg_kernel.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == '__main__':
    generate_figure()
