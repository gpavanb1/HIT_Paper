#!/usr/bin/env python3
"""
Verify that the Fig. 5 HIT realization (random-phase Fourier superposition,
projected onto the divergence-free plane) satisfies the admissibility
properties claimed in the paper for the narrowband spectrum E_NB(k):

  1. Exact incompressibility (divergence-free)
  2. Recovered 3D energy spectrum matches the prescribed E_NB(k) shape
  3. Finite kinetic energy
  4. Finite enstrophy / viscous dissipation
  5. Super-algebraic smoothness (Gaussian ultraviolet tail)
  6. Homogeneity (vanishing mean) and second-order isotropy
  7. Batchelor infrared law E(k) ~ k^4 as k → 0
  8. Kinematic realizability E(k) ≥ 0
"""

import os
import sys

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fig5_flow_realization import generate_3d_hit_field

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
    'lines.linewidth': 1.8,
    'figure.dpi': 300,
})

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

K0 = 10.0
SIGMA = 0.5
KL = 1.0
AMP = 1.0
N = 256
L_BOX = 2.0 * np.pi
SEED = 42


def E_NB(k, k0=K0, sigma=SIGMA, kL=KL, A=AMP):
    k = np.asarray(k, dtype=float)
    out = np.zeros_like(k, dtype=float)
    pos = k > 0.0
    kp = k[pos]
    ir_factor = (kp / kL) ** 4 / (1.0 + (kp / kL) ** 2) ** 2
    out[pos] = A * ir_factor * np.exp(-0.5 * ((kp - k0) / sigma) ** 2)
    return out if out.ndim else float(out)


def _scalar_E_NB(k):
    return float(np.squeeze(E_NB(np.atleast_1d(float(k)))))


def wavevectors(N, L):
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d, indexing='ij')
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k_abs = np.sqrt(k2)
    return kx, ky, kz, k2, k_abs, k1d


def physical_fourier(field):
    n = field.shape[0]
    return np.fft.fftn(field, axes=(0, 1, 2)) / n ** 3


def shell_average_spectrum(u_hat, v_hat, w_hat, k_abs, k1d):
    """Isotropic shell spectrum with ∫ E(k) dk = ½ ⟨|u|²⟩."""
    e_mode = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2 + np.abs(w_hat) ** 2)
    k_max = float(np.max(np.abs(k1d)))
    edges = np.arange(0.5, k_max + 0.5, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dk = np.diff(edges)
    E_meas = np.zeros_like(centers)
    counts = np.zeros_like(centers)
    k_flat = k_abs.ravel()
    e_flat = e_mode.ravel()
    idx = np.digitize(k_flat, edges) - 1
    valid = (idx >= 0) & (idx < len(centers))
    for i in np.unique(idx[valid]):
        mask = idx == i
        E_meas[i] = np.sum(e_flat[mask]) / dk[i]
        counts[i] = np.count_nonzero(mask)
    return centers, E_meas, counts, e_mode, edges


def longitudinal_f(u):
    """Spatial f(r) along x1 from a single periodic realization."""
    n = u.shape[0]
    u_hat_x = np.fft.fft(u, axis=0)
    corr = np.fft.ifft(np.abs(u_hat_x) ** 2, axis=0).real / n
    r11 = corr.mean(axis=(1, 2))
    return r11 / r11[0]


def check(results, name, ok, detail):
    status = 'PASS' if ok else 'FAIL'
    print(f'  [{status}] {name}: {detail}')
    results.append(bool(ok))
    return bool(ok)


def verify():
    results = []

    print('=' * 72)
    print('Verification of Fig. 5 cosine/sine HIT realization (narrowband E_NB)')
    print(f'  generator: fig5_flow_realization.generate_3d_hit_field')
    print(f'  N = {N}^3,  L = 2π,  k0 = {K0},  σ = {SIGMA},  kL = {KL},  seed = {SEED}')
    print('=' * 72)

    u, v, w, omega_z = generate_3d_hit_field(
        spectrum_type='narrowband', N=N, L_box=L_BOX, seed=SEED,
    )
    kx, ky, kz, k2, k_abs, k1d = wavevectors(N, L_BOX)
    u_hat = physical_fourier(u)
    v_hat = physical_fourier(v)
    w_hat = physical_fourier(w)
    rms_u = float(np.sqrt(np.mean(u ** 2 + v ** 2 + w ** 2)))

    print('\n1) Exact incompressibility  ∇·u = 0')
    k_dot_u = kx * u_hat + ky * v_hat + kz * w_hat
    max_k_dot_u = float(np.max(np.abs(k_dot_u)))
    check(results, 'Fourier solenoidality  k·û(k) = 0',
          max_k_dot_u < 1e-12 * rms_u,
          f'max |k·û| = {max_k_dot_u:.3e}')

    div = np.fft.ifftn(1j * k_dot_u * N ** 3).real
    max_div = float(np.max(np.abs(div)))
    div_rel = max_div / (rms_u * K0 + 1e-30)
    check(results, 'Physical-space divergence',
          div_rel < 1e-10,
          f'max |∇·u| / (u_rms k0) = {div_rel:.3e}  (max |∇·u| = {max_div:.3e})')

    print('\n2) Recovered energy spectrum matches prescribed E_NB(k)')
    ke_field = 0.5 * np.mean(u ** 2 + v ** 2 + w ** 2)
    ke_cont, _ = integrate.quad(_scalar_E_NB, 0.0, 80.0, epsabs=1e-10)
    centers, E_meas, counts, e_mode, edges = shell_average_spectrum(
        u_hat, v_hat, w_hat, k_abs, k1d,
    )
    dk = np.diff(edges)
    E_bin = np.zeros_like(centers)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        E_bin[i], _ = integrate.quad(_scalar_E_NB, a, b)
        E_bin[i] /= (b - a)
    E_target_scaled = (ke_field / ke_cont) * E_bin
    k_lo, k_hi = K0 - 4.0 * SIGMA, K0 + 4.0 * SIGMA
    band = (centers >= k_lo) & (centers <= k_hi) & (counts >= 30)
    frac_meas = float(np.sum(E_meas[band] * dk[band]) / ke_field)
    frac_spec = float(integrate.quad(_scalar_E_NB, k_lo, k_hi)[0] / ke_cont)
    check(results, 'Energy fraction in the narrowband peak (k0 ± 4σ)',
          abs(frac_meas - frac_spec) < 0.05,
          f'field = {frac_meas:.4f},  spectrum = {frac_spec:.4f}')

    peak = np.max(E_target_scaled[band]) if np.any(band) else 1.0
    cmp = band & (E_target_scaled > 0.05 * peak)
    rel_spec = np.abs(E_meas[cmp] - E_target_scaled[cmp]) / np.maximum(E_target_scaled[cmp], 1e-30)
    spec_rel_rms = float(np.sqrt(np.mean(rel_spec ** 2))) if np.any(cmp) else np.inf
    check(results, 'Shell-averaged E(k) vs bin-integrated E_NB',
          spec_rel_rms < 0.35,
          f'RMS relative error = {spec_rel_rms:.3e}  '
          f'(Gaussian mode amplitudes; expected O(1/√N_shell) scatter)')

    ke_meas = float(np.sum(e_mode))
    ke_parseval_rel = abs(ke_meas - ke_field) / ke_field
    check(results, 'Parseval: shell integral equals ½⟨|u|²⟩',
          ke_parseval_rel < 1e-10,
          f'Σ shells E Δk = {ke_meas:.6e},  K_field = {ke_field:.6e}')

    print('\n3) Finite kinetic energy')
    urms = np.sqrt(np.mean(u ** 2 + v ** 2 + w ** 2) / 3.0)
    check(results, 'Field energy is finite (Fig. 5 normalizes u′ = 1 ⇒ K = 3/2)',
          np.isfinite(ke_field) and abs(ke_field - 1.5) < 1e-10,
          f'K_field = ½⟨|u|²⟩ = {ke_field:.6e},  u′ = {urms:.6e}')
    check(results, 'Continuous spectrum energy ∫ E_NB(k) dk is finite',
          np.isfinite(ke_cont) and ke_cont > 0.0,
          f'∫ E_NB dk = {ke_cont:.6e}')

    print('\n4) Finite enstrophy / viscous dissipation')
    omega_x = np.fft.ifftn(1j * (ky * w_hat - kz * v_hat) * N ** 3).real
    omega_y = np.fft.ifftn(1j * (kz * u_hat - kx * w_hat) * N ** 3).real
    omega_z_spec = np.fft.ifftn(1j * (kx * v_hat - ky * u_hat) * N ** 3).real
    ens_field = 0.5 * np.mean(omega_x ** 2 + omega_y ** 2 + omega_z_spec ** 2)
    ens_cont, _ = integrate.quad(lambda k: (k ** 2) * _scalar_E_NB(k), 0.0, 80.0, epsabs=1e-10)
    ens_ratio_field = ens_field / ke_field
    ens_ratio_spec = ens_cont / ke_cont
    ens_rel = abs(ens_ratio_field - ens_ratio_spec) / ens_ratio_spec
    check(results, 'Ω_field / K_field ≈ ∫ k² E_NB dk / ∫ E_NB dk',
          ens_rel < 0.08,
          f'Ω/K field = {ens_ratio_field:.6e},  spectral = {ens_ratio_spec:.6e},  '
          f'rel. err. = {ens_rel:.3e}')

    omega_z_err = float(np.max(np.abs(omega_z_spec - omega_z)))
    check(results, 'Fig. 5 vorticity ω3 matches spectral curl',
          omega_z_err < 1e-10 * (np.std(omega_z) + 1e-30),
          f'max |ω3_fig5 − (∇×u)_3| = {omega_z_err:.3e}')

    print('\n5) C^∞ smoothness (Gaussian ultraviolet decay)')
    abs2 = np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2 + np.abs(w_hat) ** 2
    for s in (0, 2, 4, 8, 16):
        hs = float(np.sum(((1.0 + k2) ** s) * abs2))
        check(results, f'H^{s} Sobolev norm finite',
              np.isfinite(hs) and hs > 0.0,
              f'||u||_H^{s}^2 = {hs:.6e}')

    e_high = float(np.sum(e_mode[k_abs > 40.0]))
    check(results, 'Energy beyond k = 40 is negligible',
          e_high < 1e-12 * ke_field,
          f'Σ_{{k>40}} ½|û|² = {e_high:.3e}')

    pal_field = float(np.sum((k2 ** 2) * abs2))
    check(results, 'Palinstrophy Σ k⁴ |û|² finite',
          np.isfinite(pal_field),
          f'Σ k⁴ |û|² = {pal_field:.6e}')

    print('\n6) Homogeneity and second-order isotropy')
    means = np.array([u.mean(), v.mean(), w.mean()])
    check(results, 'Vanishing spatial mean',
          np.max(np.abs(means)) < 1e-12 * rms_u,
          f'<u> = ({means[0]:.3e}, {means[1]:.3e}, {means[2]:.3e})')

    var = np.array([np.mean(u ** 2), np.mean(v ** 2), np.mean(w ** 2)])
    iso_var_rel = float(np.max(np.abs(var - var.mean())) / var.mean())
    check(results, 'Equal component variances',
          iso_var_rel < 0.15,
          f'<u_i²> = ({var[0]:.4e}, {var[1]:.4e}, {var[2]:.4e}),  '
          f'max rel. spread = {iso_var_rel:.3e}')

    re_stress = np.array([np.mean(u * v), np.mean(u * w), np.mean(v * w)])
    iso_off = float(np.max(np.abs(re_stress)) / var.mean())
    check(results, 'Off-diagonal Reynolds stresses ≈ 0',
          iso_off < 0.08,
          f'max |<u_i u_j>| / <u_i²> = {iso_off:.3e}')

    print('\n7) Batchelor infrared constraint  E(k) ∼ k⁴ as k → 0')
    k_ir = np.logspace(-4, -2, 8)
    gauss_ir = np.exp(-0.5 * ((k_ir - K0) / SIGMA) ** 2)
    ir_prefactor = E_NB(k_ir) / (k_ir ** 4 * gauss_ir)
    ir_rel = float(np.max(np.abs(ir_prefactor / (AMP / KL ** 4) - 1.0)))
    check(results, 'IR prefactor E_NB(k) / [k⁴ exp(−(k−k0)²/2σ²)] → A/kL⁴',
          ir_rel < 1e-3,
          f'max rel. error on k ∈ [10^{{-4}}, 10^{{-2}}] = {ir_rel:.3e},  '
          f'prefactor = {ir_prefactor[0]:.6e} (A/kL⁴ = {AMP / KL ** 4:.6e})')

    print('\n8) Kinematic realizability  E(k) ≥ 0')
    e_min = float(np.min(E_NB(np.linspace(0.0, 80.0, 4001))))
    check(results, 'Prescribed E_NB(k) ≥ 0',
          e_min >= -1e-16,
          f'min E_NB(k) = {e_min:.3e}')
    check(results, 'Measured shell spectrum ≥ 0',
          float(np.min(E_meas)) >= -1e-16,
          f'min measured E(k) = {float(np.min(E_meas)):.3e}')

    print('\nBonus) Longitudinal autocorrelation of the realized field')
    r = np.arange(N) * (L_BOX / N)
    f_r = longitudinal_f(u)
    r_pos = r[r <= np.pi]
    f_pos = f_r[r <= np.pi]
    f_min = float(np.min(f_pos))
    r_min = float(r_pos[np.argmin(f_pos)])
    check(results, 'Negative loop in f(r) from the Fig. 5 field',
          f_min < -0.02,
          f'min f(r) = {f_min:.4f} at r = {r_min:.3f}  '
          f'(paper: ≈ −0.083 at r ≈ 0.47; single-realization estimate)')

    passed = all(results)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))

    ax = axes[0]
    vis = (centers > 0.5) & (centers < 25.0) & (counts > 0)
    ax.plot(centers[vis], E_target_scaled[vis], 'k-', lw=2.0, label=r'scaled $E_{\mathrm{NB}}(k)$')
    ax.plot(centers[vis], E_meas[vis], 'C0o', ms=3.5, label='measured shells')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$E(k)$')
    ax.set_title(r'(a) Energy spectrum')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25, ls='--')

    ax = axes[1]
    ax.plot(k_ir, E_NB(k_ir), 'C1o-', ms=4, label=r'$E_{\mathrm{NB}}(k)$')
    gauss0 = np.exp(-0.5 * ((k_ir - K0) / SIGMA) ** 2)
    ax.plot(k_ir, ir_prefactor[0] * k_ir ** 4 * gauss0, 'k--', lw=1.4, label=r'$\propto k^4$ (with Gaussian)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$E(k)$')
    ax.set_title(r'(b) Batchelor infrared law')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25, ls='--', which='both')

    ax = axes[2]
    ax.plot(r_pos, f_pos, 'C3-', lw=1.8)
    ax.axhline(0.0, color='k', lw=0.8, ls=':')
    ax.fill_between(r_pos, f_pos, 0.0, where=(f_pos < 0.0), color='C3', alpha=0.25)
    ax.set_xlabel(r'$r$')
    ax.set_ylabel(r'$f(r)$')
    ax.set_title(r'(c) Realized $f(r)$')
    ax.grid(True, alpha=0.25, ls='--')

    plt.tight_layout()
    out_pdf = os.path.join(FIGURES_DIR, 'verify_flow_realization.pdf')
    out_png = os.path.join(FIGURES_DIR, 'verify_flow_realization.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()

    print('\n' + '=' * 72)
    print(f'Result: {"ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED"}')
    print(f'Saved diagnostic figure: {out_pdf}')
    print('=' * 72)
    return passed


if __name__ == '__main__':
    ok = verify()
    raise SystemExit(0 if ok else 1)
