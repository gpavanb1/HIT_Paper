#!/usr/bin/env python3
"""
Figure 3: Longitudinal Autocorrelation f(r) for von Karman and von Karman-Pao Spectra.
Demonstrates:
  (a) Near range r/L in [0, 8]: Exact continuous quadrature vs coarse discrete FFT aliasing artifact.
  (b) Asymptotic tails r/L in [0, 30]: Exact analytical Macdonald K_{1/3} Bessel decay for pure von Karman
      alongside high-precision evaluation for the full von Karman-Pao (VKP) spectrum with viscous cutoff,
      confirming smooth, strictly positive monotonic exponential decay across >13 orders of magnitude.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
import mpmath as mp
from scipy import integrate
from scipy.special import kv, gamma
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


def f_vk_analytical(r, L=1.0):
    xi = np.maximum(r / L, 1e-12)
    coeff = (2.0**(2.0/3.0)) / gamma(1.0/3.0)
    return coeff * (xi**(1.0/3.0)) * kv(1.0/3.0, xi)


def compute_vkp_tail_mpmath(r_vals, L=1.0, u_rms=1.0, nu=1e-5):
    """High-precision numerical integration of f(r) for VKP using mpmath."""
    mp.mp.dps = 25
    epsilon = u_rms**3 / L
    eta = (nu**3 / epsilon)**0.25

    def E_vkp(k):
        kL = k * L
        return 1.5 * u_rms**2 * L * (kL)**4 / (1.0 + kL**2)**(mp.mpf('17.0') / 6) * mp.exp(-2.25 * (k * eta)**(mp.mpf('4.0') / 3))

    u2 = (mp.mpf(2) / 3) * mp.quad(E_vkp, [0, 10, mp.inf])

    f_vals = []
    for r in r_vals:
        r_mp = mp.mpf(r)
        def integrand(k):
            kr = k * r_mp
            if kr < 1e-4:
                k11 = mp.mpf(2)/3 - (mp.mpf(2)/15)*(kr**2)
            else:
                k11 = 2 * (mp.sin(kr) - kr * mp.cos(kr)) / (kr)**3
            return E_vkp(k) * k11

        if r < 1.0:
            val = mp.quad(integrand, [0, 10, mp.inf])
        else:
            val = mp.quadosc(integrand, [0, mp.inf], zeros=lambda n: n * mp.pi / r_mp)
        f_vals.append(float(val / u2))

    return np.array(f_vals)


def generate_figure():
    print("Generating Figure 3: f(r) for VK and VKP spectra...")
    k1_dense = np.linspace(0.0001, 100.0, 1000)
    E11_dense = np.array([E11_vkp_single(k) for k in k1_dense])
    E11_spline = interp1d(k1_dense, E11_dense, kind='cubic', bounds_error=False, fill_value=0.0)

    norm_vkp, _ = integrate.quad(E11_spline, 0, 80.0, limit=500)

    r_fine = np.linspace(0.01, 8.0, 120)
    f_fine = np.zeros_like(r_fine)
    for i, r in enumerate(r_fine):
        val, _ = integrate.quad(lambda k: E11_spline(k)*np.cos(k*r), 0, 80.0, limit=500)
        f_fine[i] = val / norm_vkp

    # Panel (b): Asymptotic decay lines out to r/L = 30
    r_dense = np.linspace(0.05, 30.0, 500)
    # Pure von Karman exact analytical Bessel function
    f_vk_dense = f_vk_analytical(r_dense)

    # Full von Karman-Pao with viscous cutoff computed via mpmath
    print("Computing VKP high-precision curve via mpmath...")
    r_vkp = np.linspace(0.05, 30.0, 40)
    f_vkp = compute_vkp_tail_mpmath(r_vkp)

    # Aliased FFT demonstration
    k_fft = np.linspace(0, 30, 128)
    E11_fft = np.array([E11_spline(k) for k in k_fft])
    dr_fft = np.pi / 30.0
    r_fft_grid = np.arange(len(k_fft)) * dr_fft
    f_fft = np.fft.irfft(E11_fft, n=len(k_fft)*2)
    f_fft = f_fft[:len(k_fft)]
    f_fft /= f_fft[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.6))

    # Panel (a): Near range comparison
    ax1.plot(r_fine, f_fine, color='#1f77b4', linestyle='-', lw=2.0, label=r'Continuous Quad ($f \geq 0$)')
    ax1.plot(r_fft_grid[r_fft_grid <= 8.0], f_fft[r_fft_grid <= 8.0], color='#d62728', linestyle='--', lw=1.8, label='Coarse FFT (aliasing dip)')
    ax1.axhline(0, color='black', lw=0.8, ls=':')
    ax1.set_xlabel(r'Separation $r/L$')
    ax1.set_ylabel(r'$f(r)$')
    ax1.set_title(r'(a) Near range $r/L \in [0, 8]$')
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', loc='upper right')

    # Panel (b): Asymptotic Tails (Pure von Karman vs Full VKP via mpmath)
    ax2.semilogy(r_dense, f_vk_dense, color='#1f77b4', linestyle='-', lw=2.0, label=r'Pure vK (Exact $K_{1/3}$ Bessel)')
    ax2.semilogy(r_vkp, f_vkp, color='#ff7f0e', linestyle='--', lw=2.0, label=r'Full VKP ($\mathtt{mpmath}$ quad)')
    ax2.set_xlabel(r'Separation $r/L$')
    ax2.set_ylabel(r'$f(r)$ (log scale)')
    ax2.set_title(r'(b) Asymptotic Tail $r/L \in [0, 30]$')
    ax2.set_ylim(1e-15, 1.5)
    ax2.grid(True, which='both', alpha=0.25, linestyle='--')
    ax2.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', loc='upper right')
    ax2.set_xlabel(r'Separation $r/L$')
    ax2.set_ylabel(r'$f(r)$ (log scale)')
    ax2.set_title(r'(b) Asymptotic Tail $r/L \in [0, 30]$')
    ax2.set_ylim(1e-15, 1.5)
    ax2.grid(True, which='both', alpha=0.25, linestyle='--')
    ax2.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', loc='upper right')

    plt.tight_layout()
    out_pdf = os.path.join(FIGURES_DIR, 'fig3_vkp_correlation.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig3_vkp_correlation.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == '__main__':
    generate_figure()
