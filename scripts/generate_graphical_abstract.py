#!/usr/bin/env python3
"""
Graphical Abstract for JFM:
Single-panel high-resolution visualization with aspect ratio 1.2 : 1 (e.g. 6cm x 5cm / 6in x 5in).
Visualizes the 2D cross-section of 3D isotropic divergence-free turbulent velocity field u1 
with streamlines and two-point correlation sampling pairs (dots), without text/captions.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
os.makedirs('/tmp/matplotlib_cache', exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_3d_hit_field(spectrum_type='narrowband', N=384, L_box=2.0*np.pi, seed=42):
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
    
    urms = np.sqrt(np.mean(u**2 + v**2 + w**2) / 3.0)
    u /= urms
    v /= urms
    w /= urms
    
    return u, v, w


def generate_graphical_abstract():
    print("Generating Graphical Abstract (1.2:1 aspect ratio, high resolution)...")
    N = 384
    u_nb, v_nb, _ = generate_3d_hit_field('narrowband', N=N, seed=42)

    # 1.2 : 1 aspect ratio: x in [0, 2*pi], y in [0, 2*pi / 1.2]
    y_max = 2.0 * np.pi / 1.2
    ny = int(N / 1.2)
    
    x_coords = np.linspace(0, 2.0*np.pi, N)
    y_coords = np.linspace(0, y_max, ny)
    
    u_slice = u_nb[:, :ny, 0].T
    v_slice = v_nb[:, :ny, 0].T

    # Find candidate pairs with strong anti-correlation across r = pi/10
    r_sep = np.pi / 10.0
    
    # 4 prominent, well-distributed pairs across the domain with strong anti-correlation:
    # Each pair has one dot firmly in a red zone (u1 > 0) and one dot firmly in a blue zone (u1 < 0).
    # 1. Top-left: x1=0.48, y=4.88 -> u(x1)=+2.31 (red), u(x2)=-2.31 (blue)
    # 2. Top-right: x1=4.78, y=4.76 -> u(x1)=-1.78 (blue), u(x2)=+1.92 (red)
    # 3. Center (shifted top-right): x1=3.24, y=2.86 -> u(x1)=+1.65 (red), u(x2)=-1.05 (blue)
    # 4. Bottom-right: x1=4.88, y=1.22 -> u(x1)=-1.50 (blue), u(x2)=+1.20 (red)
    pairs = [
        (0.48, 4.88),
        (4.78, 4.76),
        (3.24, 2.86),
        (4.88, 1.22),
    ]

    fig = plt.figure(figsize=(6.0, 5.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # Smooth colormap of u1 velocity field
    ax.imshow(u_slice, origin='lower', extent=[0, 2*np.pi, 0, y_max], 
              cmap='coolwarm', vmin=-2.6, vmax=2.6, interpolation='bicubic')

    # Streamlines showing flow structure
    skip = 2
    ax.streamplot(x_coords[::skip], y_coords[::skip], 
                  u_slice[::skip, ::skip], v_slice[::skip, ::skip], 
                  color='#111111', density=1.25, linewidth=0.75, arrowsize=0.85, arrowstyle='->')

    # Anti-correlated two-point sampling pairs (dots)
    for x1, y in pairs:
        x2 = x1 + r_sep
        # Outer dark outline for high contrast against both red and blue
        ax.plot([x1, x2], [y, y], color='black', lw=4.5, zorder=12, solid_capstyle='round')
        # Bright high-visibility core
        ax.plot([x1, x2], [y, y], color='#00FF55', lw=2.8, zorder=13, solid_capstyle='round')
        # Dots
        ax.plot(x1, y, 'o', color='#00FF55', markeredgecolor='black', markeredgewidth=1.6, markersize=8.5, zorder=15)
        ax.plot(x2, y, 'o', color='#00FF55', markeredgecolor='black', markeredgewidth=1.6, markersize=8.5, zorder=15)

    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, y_max)

    out_jpg = os.path.join(FIGURES_DIR, 'graphical_abstract.jpg')
    out_png = os.path.join(FIGURES_DIR, 'graphical_abstract.png')
    out_pdf = os.path.join(FIGURES_DIR, 'graphical_abstract.pdf')
    out_jpg_600 = os.path.join(FIGURES_DIR, 'graphical_abstract_600dpi.jpg')
    
    # Save required formats
    plt.savefig(out_jpg, dpi=300, format='jpeg', pil_kwargs={'quality': 98})
    plt.savefig(out_png, dpi=300, format='png')
    plt.savefig(out_pdf, format='pdf')
    plt.close()

    # 600 DPI version for high-definition / cover
    fig = plt.figure(figsize=(6.0, 5.0), dpi=600)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(u_slice, origin='lower', extent=[0, 2*np.pi, 0, y_max], 
              cmap='coolwarm', vmin=-2.6, vmax=2.6, interpolation='bicubic')
    ax.streamplot(x_coords[::skip], y_coords[::skip], 
                  u_slice[::skip, ::skip], v_slice[::skip, ::skip], 
                  color='#111111', density=1.25, linewidth=0.75, arrowsize=0.85, arrowstyle='->')
    for x1, y in pairs:
        x2 = x1 + r_sep
        ax.plot([x1, x2], [y, y], color='black', lw=4.5, zorder=12, solid_capstyle='round')
        ax.plot([x1, x2], [y, y], color='#00FF55', lw=2.8, zorder=13, solid_capstyle='round')
        ax.plot(x1, y, 'o', color='#00FF55', markeredgecolor='black', markeredgewidth=1.6, markersize=8.5, zorder=15)
        ax.plot(x2, y, 'o', color='#00FF55', markeredgecolor='black', markeredgewidth=1.6, markersize=8.5, zorder=15)
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, y_max)
    plt.savefig(out_jpg_600, dpi=600, format='jpeg', pil_kwargs={'quality': 98})
    plt.close()

    print("Generated Graphical Abstract files:")
    print(f" - {out_jpg} (300 DPI, Aspect Ratio 1.2:1)")
    print(f" - {out_jpg_600} (600 DPI, Aspect Ratio 1.2:1)")
    print(f" - {out_png}")
    print(f" - {out_pdf}")


if __name__ == '__main__':
    generate_graphical_abstract()
