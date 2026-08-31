#!/usr/bin/env python3
"""
Figure 0: Sedov Self-Similar Closed-Form Solution.
Plots the exact Kummer hypergeometric solution f(xi) = 1F1(0.8, 2.5, -xi^2/8)
demonstrating unconditional non-negativity for self-preserving decay.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
from scipy.special import hyp1f1
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
    print("Generating Figure 0: Sedov self-similar solution...")
    xi = np.linspace(0, 15, 500)
    # Sedov's analytical solution for self-similar decay: f(xi) = 1F1(a, b, -xi^2/8) with a=0.8, b=2.5
    a_sedov, b_sedov = 0.8, 2.5
    f_sedov = hyp1f1(a_sedov, b_sedov, -xi**2 / 8.0)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(xi, f_sedov, 'C0-', lw=2.0, label=r'Sedov solution $f(\xi) = {}_1F_1\left(0.8;\, 2.5;\, -\frac{\xi^2}{8}\right)$')
    ax.axhline(0, color='black', lw=0.8, ls=':')
    ax.set_xlabel(r'Similarity variable $\xi = r / \ell(t)$')
    ax.set_ylabel(r'Longitudinal correlation $f(\xi)$')
    ax.set_title(r'Sedov exact self-similar solution: $f(\xi) \geq 0$ for all $\xi$')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', loc='upper right')

    out_pdf = os.path.join(FIGURES_DIR, 'fig0_sedov_solution.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig0_sedov_solution.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == '__main__':
    generate_figure()
