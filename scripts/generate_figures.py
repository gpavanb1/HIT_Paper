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
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 11,
    'lines.linewidth': 1.5,
    'figure.dpi': 300
})

# Create figures directory relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Figure 1: Schoenberg Kernel & Positive Definiteness
# -------------------------------------------------------------
print("Generating Figure 1: Schoenberg Kernel...")
x = np.linspace(0.01, 16, 1000)
omega3 = np.sin(x) / x
kernel_11 = 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3)

fig, ax = plt.subplots(figsize=(5.5, 3.2))
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
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_schoenberg_kernel.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig1_schoenberg_kernel.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 2. Figure 2: Von Karman-Pao spectrum, E11(k1), and d2E11/dk1^2
# -------------------------------------------------------------
print("Generating Figure 2: VKP spectrum and derivatives...")

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

k_vals = np.logspace(-2, 2, 200)
E_vals = [von_karman_pao_spectrum(k) for k in k_vals]

k1_vals = np.linspace(0.005, 4.0, 100)
E11_vals = np.array([E11_vkp_single(k) for k in k1_vals])
d2E11_vals = np.array([d2E11_vkp_single(k) for k in k1_vals])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))

ax1.loglog(k_vals, E_vals, 'C0-', label=r'$E(k)$ (von Kármán–Pao)')
ax1.plot(k1_vals, E11_vals, 'C2--', label=r'$E_{11}(k_1)$ (1D longitudinal)')
ax1.set_xlabel(r'Wavenumber $k, k_1$')
ax1.set_ylabel(r'Energy spectrum')
ax1.set_title(r'(a) 3D and 1D Energy Spectra')
ax1.grid(True, which='both', alpha=0.25, linestyle='--')
ax1.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8')

ax2.plot(k1_vals, d2E11_vals, 'C3-', lw=1.6)
ax2.axhline(0, color='black', lw=0.8, ls=':')
ax2.fill_between(k1_vals, d2E11_vals, 0, where=(d2E11_vals < 0), color='C3', alpha=0.2, label=r'Concave ($d^2E_{11}/dk_1^2 < 0$)')
ax2.fill_between(k1_vals, d2E11_vals, 0, where=(d2E11_vals >= 0), color='C2', alpha=0.15, label=r'Convex ($d^2E_{11}/dk_1^2 \geq 0$)')
ax2.set_xlabel(r'Wavenumber $k_1$')
ax2.set_ylabel(r'$d^2E_{11}/dk_1^2$')
ax2.set_title(r'(b) Convexity of $E_{11}(k_1)$')
ax2.grid(True, alpha=0.25, linestyle='--')
ax2.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8')

plt.savefig(os.path.join(FIGURES_DIR, 'fig2_vkp_convexity.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig2_vkp_convexity.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 3. Figure 3: f(r) for von Karman-Pao
# -------------------------------------------------------------
print("Generating Figure 3: f(r) for VKP spectrum...")

k1_dense = np.linspace(0.0001, 100.0, 1000)
E11_dense = np.array([E11_vkp_single(k) for k in k1_dense])
E11_spline = interp1d(k1_dense, E11_dense, kind='cubic', bounds_error=False, fill_value=0.0)

norm_vkp, _ = integrate.quad(E11_spline, 0, 80.0, limit=500)

r_fine = np.linspace(0.01, 8.0, 120)
f_fine = np.zeros_like(r_fine)
for i, r in enumerate(r_fine):
    val, _ = integrate.quad(lambda k: E11_spline(k)*np.cos(k*r), 0, 80.0, limit=500)
    f_fine[i] = val / norm_vkp

r_tail = np.linspace(8.0, 30.0, 50)
f_tail = np.zeros_like(r_tail)
for i, r in enumerate(r_tail):
    val, _ = integrate.quad(lambda k: E11_spline(k)*np.cos(k*r), 0, 80.0, limit=500)
    f_tail[i] = max(val / norm_vkp, 1e-15)

# Aliased FFT demonstration
k_fft = np.linspace(0, 30, 128)
E11_fft = np.array([E11_spline(k) for k in k_fft])
dr_fft = np.pi / 30.0
r_fft_grid = np.arange(len(k_fft)) * dr_fft
f_fft = np.fft.irfft(E11_fft, n=len(k_fft)*2)
f_fft = f_fft[:len(k_fft)]
f_fft /= f_fft[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))

ax1.plot(r_fine, f_fine, 'C0-', label=r'Exact Quad: $f(r) \geq 0$')
ax1.plot(r_fft_grid[r_fft_grid <= 8.0], f_fft[r_fft_grid <= 8.0], 'C3--', lw=1.2, label='Coarse FFT (aliasing artifact)')
ax1.axhline(0, color='black', lw=0.8, ls=':')
ax1.set_xlabel(r'Separation $r/L$')
ax1.set_ylabel(r'$f(r)$')
ax1.set_title(r'(a) Near range $r/L \in [0, 8]$')
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8')

ax2.semilogy(r_fine, f_fine, 'C0-')
ax2.semilogy(r_tail, f_tail, 'C0-', label=r'Smooth positive decay')
ax2.set_xlabel(r'Separation $r/L$')
ax2.set_ylabel(r'$f(r)$ (log scale)')
ax2.set_title(r'(b) Asymptotic Tail $r/L \in [0, 30]$')
ax2.grid(True, which='both', alpha=0.25, linestyle='--')
ax2.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8')

plt.savefig(os.path.join(FIGURES_DIR, 'fig3_vkp_correlation.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig3_vkp_correlation.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 4. Figure 4: Counterexample - Narrowband Batchelor Spectrum
# -------------------------------------------------------------
print("Generating Figure 4: Narrowband counterexample...")

def narrowband_batchelor(k, k0=10.0, sigma=0.5, A=1.0, kL=1.0):
    ir_factor = (k/kL)**4 / (1.0 + (k/kL)**2)**2
    return A * ir_factor * np.exp(-0.5 * ((k - k0)/sigma)**2)

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

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.6))

ax1.plot(k_grid_nb, E_nb, 'C0-')
ax1.set_xlabel(r'$k$')
ax1.set_ylabel(r'$E(k)$')
ax1.set_title(r'(a) Batchelor Narrowband $E(k)$')
ax1.grid(True, alpha=0.25, linestyle='--')

ax2.plot(k1_grid_nb, E11_nb, 'C2-')
ax2.set_xlabel(r'$k_1$')
ax2.set_ylabel(r'$E_{11}(k_1)$')
ax2.set_title(r'(b) 1D Spectrum $E_{11}(k_1)$')
ax2.grid(True, alpha=0.25, linestyle='--')

ax3.plot(r_grid_nb, f_nb, 'C3-', lw=1.6)
ax3.axhline(0, color='black', lw=0.8, ls=':')
ax3.fill_between(r_grid_nb, f_nb, 0, where=(f_nb < 0), color='C3', alpha=0.25, label=f'Min $f(r) = {f_nb.min():.3f}$')
ax3.set_xlabel(r'Separation $r$')
ax3.set_ylabel(r'$f(r)$')
ax3.set_title(r'(c) Negative Loop in $f(r)$')
ax3.grid(True, alpha=0.25, linestyle='--')
ax3.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', loc='upper right')

plt.savefig(os.path.join(FIGURES_DIR, 'fig4_narrowband_counterexample.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig4_narrowband_counterexample.png'), dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 5. Figure 5: Parametric transition - Negative loop vs bandwidth
# -------------------------------------------------------------
print("Generating Figure 5: Parametric transition vs bandwidth...")

sigmas = [0.3, 0.6, 1.2, 2.5]
r_scan = np.linspace(0.001, 2.5, 200)
min_f_vals = []
sigma_dense = np.linspace(0.25, 4.0, 16)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9))

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
        
    ax1.plot(r_scan, f_arr, label=rf'$\sigma = {s}$')

ax1.axhline(0, color='black', lw=0.8, ls=':')
ax1.set_xlabel(r'Separation $r$')
ax1.set_ylabel(r'$f(r)$')
ax1.set_title(r'(a) Longitudinal correlation $f(r)$ vs $\sigma$')
ax1.grid(True, alpha=0.25, linestyle='--')
ax1.legend(frameon=True, edgecolor='none', facecolor='#f8f8f8', fontsize=8)

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

ax2.plot(sigma_dense / 10.0, min_f_vals, 'C0o-', markersize=4)
ax2.axhline(0, color='black', lw=0.8, ls=':')
ax2.set_xlabel(r'Relative bandwidth $\sigma / k_0$')
ax2.set_ylabel(r'$\min_r f(r)$')
ax2.set_title(r'(b) Depth of Negative Loop vs Spectral Breadth')
ax2.grid(True, alpha=0.25, linestyle='--')

plt.savefig(os.path.join(FIGURES_DIR, 'fig5_parametric_bandwidth.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(FIGURES_DIR, 'fig5_parametric_bandwidth.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All figures successfully generated in figures/ directory!")
