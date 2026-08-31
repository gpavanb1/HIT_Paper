#!/usr/bin/env python3
"""
Figure 5: 2D Cross-Sectional Flow Field Realizations (Eq. 110 / Eq. 93).
Visualizes random-phase Fourier superpositions of 3D isotropic, divergence-free
velocity and vorticity fields:
(a) Narrowband u1 field with streamlines showing alternating lanes (negative correlation),
(b) Narrowband out-of-plane vorticity omega3 showing cyclonic/anticyclonic vortex cores,
(c) Broadband von Karman-Pao u1 field showing multi-scale turbulent cascade.
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


def generate_3d_hit_field(spectrum_type='narrowband', N=256, L_box=2.0*np.pi, seed=42):
    np.random.seed(seed)
    
    kx = np.fft.fftfreq(N, d=L_box/(2.0*np.pi*N))
    ky = np.fft.fftfreq(N, d=L_box/(2.0*np.pi*N))
    kz = np.fft.rfftfreq(N, d=L_box/(2.0*np.pi*N))
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2_safe = np.copy(K2)
    K2_safe[0, 0, 0] = 1.0
    K_safe = np.sqrt(K2_safe)
    
    if spectrum_type == 'narrowband':
        k0 = 10.0
        sigma = 0.5
        kL = 1.0
        ir_factor = (K_safe/kL)**4 / (1.0 + (K_safe/kL)**2)**2
        E = ir_factor * np.exp(-0.5 * ((K_safe - k0)/sigma)**2)
    elif spectrum_type == 'broadband':
        L = 1.0
        kL = K_safe * L
        C_k = 1.5
        nu = 5e-4
        epsilon = 1.0
        eta = (nu**3 / epsilon)**0.25
        E = 1.5 * L * (kL)**4 / (1.0 + kL**2)**(17.0 / 6.0)
        E *= np.exp(-1.5 * C_k * (K_safe * eta)**(4.0 / 3.0))
    
    E[0, 0, 0] = 0.0
    
    v_hat = (np.random.randn(3, N, N, N//2 + 1) + 1j * np.random.randn(3, N, N, N//2 + 1))
    
    k_dot_v = KX * v_hat[0] + KY * v_hat[1] + KZ * v_hat[2]
    u_hat = np.zeros_like(v_hat)
    u_hat[0] = v_hat[0] - (k_dot_v / K2_safe) * KX
    u_hat[1] = v_hat[1] - (k_dot_v / K2_safe) * KY
    u_hat[2] = v_hat[2] - (k_dot_v / K2_safe) * KZ
    
    amp = np.sqrt(np.maximum(0.0, E) / (4.0 * np.pi * K2_safe))
    amp[0, 0, 0] = 0.0
    u_hat *= amp
    
    u = np.fft.irfftn(u_hat[0], s=(N, N, N), axes=(0, 1, 2))
    v = np.fft.irfftn(u_hat[1], s=(N, N, N), axes=(0, 1, 2))
    w = np.fft.irfftn(u_hat[2], s=(N, N, N), axes=(0, 1, 2))
    
    omega_z_hat = 1j * (KX * u_hat[1] - KY * u_hat[0])
    omega_z = np.fft.irfftn(omega_z_hat, s=(N, N, N), axes=(0, 1, 2))
    
    urms = np.sqrt(np.mean(u**2 + v**2 + w**2) / 3.0)
    u /= urms
    v /= urms
    w /= urms
    omega_z /= urms
    
    return u, v, w, omega_z


def generate_figure():
    print("Generating Figure 5: 2D Flow field realization...")
    u_nb, v_nb, w_nb, wz_nb = generate_3d_hit_field('narrowband', N=256, seed=42)
    u_bb, v_bb, w_bb, wz_bb = generate_3d_hit_field('broadband', N=256, seed=42)

    x_coords = np.linspace(0, 2.0*np.pi, 256)
    y_coords = np.linspace(0, 2.0*np.pi, 256)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8), sharex=True, sharey=True)

    # (a) Narrowband u1
    ax = axes[0]
    im1 = ax.imshow(u_nb[:, :, 0].T, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='coolwarm', vmin=-2.5, vmax=2.5)
    skip = 4
    ax.streamplot(x_coords[::skip], y_coords[::skip], u_nb[::skip, ::skip, 0].T, v_nb[::skip, ::skip, 0].T, 
                  color='black', density=1.1, linewidth=0.6, arrowsize=0.7, arrowstyle='->')

    # Sample anti-correlated pairs separated by r = pi / k0 ≈ 0.31
    # Pair 1: (x1=1.24, y=3.62) has u1 < 0 (blue) and (x2=1.56, y=3.62) has u1 > 0 (red)
    # Pair 2: (x1=5.35, y=0.55) has u1 > 0 (red) and (x2=5.66, y=0.55) has u1 < 0 (blue)
    pairs = [
        {'x1': 1.24, 'x2': 1.24 + np.pi/10.0, 'y': 3.65, 'txt_offset': (0, 0.42), 'label': r'$\mathbf{x}_1, \mathbf{x}_1+r\hat{\mathbf{e}}_1$'},
        {'x1': 5.35, 'x2': 5.35 + np.pi/10.0, 'y': 0.55, 'txt_offset': (0, 0.42), 'label': r'$\mathbf{x}_2, \mathbf{x}_2+r\hat{\mathbf{e}}_1$'}
    ]
    for i, p in enumerate(pairs):
        x1, x2, y = p['x1'], p['x2'], p['y']
        # Connecting line
        ax.plot([x1, x2], [y, y], color='lime', lw=2.0, zorder=6, solid_capstyle='round')
        # Dots for the two sample points
        ax.plot(x1, y, 'o', color='lime', markeredgecolor='black', markeredgewidth=1.2, markersize=6.0, zorder=7)
        ax.plot(x2, y, 'o', color='lime', markeredgecolor='black', markeredgewidth=1.2, markersize=6.0, zorder=7)
        # Annotation text
        ox, oy = p['txt_offset']
        ax.annotate(r'Pair: $u_1 u_1 < 0$' + '\n' + r'($r \approx \pi/k_0$)', 
                    xy=((x1 + x2)/2, y), xytext=((x1 + x2)/2 + ox, y + oy),
                    ha='center', va='bottom', fontsize=7.8,
                    color='black',
                    bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='lime', alpha=0.95, lw=1.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8, shrinkA=1, shrinkB=3),
                    zorder=8)

    ax.set_title(r'(a) Narrowband $u_1(\mathbf{x})$ ($k_0=10$)')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
    ax.set_yticks([0, np.pi, 2*np.pi])
    ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

    # (b) Narrowband vorticity
    ax = axes[1]
    w_lim = np.percentile(np.abs(wz_nb[:, :, 0]), 99)
    im2 = ax.imshow(wz_nb[:, :, 0].T, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='PuOr', vmin=-w_lim, vmax=w_lim)
    ax.set_title(r'(b) Narrowband Vorticity $\omega_3(\mathbf{x})$')
    ax.set_xlabel(r'$x_1$')
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

    # (c) Broadband u1
    ax = axes[2]
    im3 = ax.imshow(u_bb[:, :, 0].T, origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], cmap='coolwarm', vmin=-2.5, vmax=2.5)
    ax.streamplot(x_coords[::skip], y_coords[::skip], u_bb[::skip, ::skip, 0].T, v_bb[::skip, ::skip, 0].T, 
                  color='black', density=1.1, linewidth=0.6, arrowsize=0.7, arrowstyle='->')
    ax.set_title(r'(c) Broadband $u_1(\mathbf{x})$ (von Kármán–Pao)')
    ax.set_xlabel(r'$x_1$')
    ax.set_xticks([0, np.pi, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])

    # Colorbars
    cbar1 = fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.18, shrink=0.85)
    cbar1.set_label(r'$u_1 / u^\prime$', labelpad=3)
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation='horizontal', pad=0.18, shrink=0.85)
    cbar2.set_label(r'$\omega_3 / (u^\prime / L)$', labelpad=3)
    cbar3 = fig.colorbar(im3, ax=axes[2], orientation='horizontal', pad=0.18, shrink=0.85)
    cbar3.set_label(r'$u_1 / u^\prime$', labelpad=3)

    plt.tight_layout()
    out_pdf = os.path.join(FIGURES_DIR, 'fig5_flow_realization.pdf')
    out_png = os.path.join(FIGURES_DIR, 'fig5_flow_realization.png')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_pdf} and {out_png}")


if __name__ == '__main__':
    generate_figure()
