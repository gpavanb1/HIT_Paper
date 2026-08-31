#!/usr/bin/env python3
"""
Master runner script to generate all figures for the paper.
Calls the individual figure generation scripts in sequence.
"""

import os
import sys

# Ensure scripts directory is in python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import fig0_sedov_solution
import fig1_schoenberg_kernel
import fig2_vkp_convexity
import fig3_vkp_correlation
import fig4_narrowband_counterexample
import fig5_flow_realization
import fig6_parametric_bandwidth
import fig7_spectral_broadening_dynamics

MODULES = [
    ("Figure 0: Sedov Solution", fig0_sedov_solution),
    ("Figure 1: Schoenberg Kernel", fig1_schoenberg_kernel),
    ("Figure 2: VKP Convexity", fig2_vkp_convexity),
    ("Figure 3: VKP Correlation", fig3_vkp_correlation),
    ("Figure 4: Narrowband Counterexample", fig4_narrowband_counterexample),
    ("Figure 5: 2D Flow Realization", fig5_flow_realization),
    ("Figure 6: Parametric Bandwidth", fig6_parametric_bandwidth),
    ("Figure 7: Spectral Broadening Dynamics", fig7_spectral_broadening_dynamics),
]


def main():
    print("=" * 60)
    print("Generating all figures for HIT Paper...")
    print("=" * 60)
    for name, module in MODULES:
        print(f"\n--- Running: {name} ---")
        module.generate_figure()
    print("\n" + "=" * 60)
    print("All figures successfully generated and saved to figures/")
    print("=" * 60)


if __name__ == '__main__':
    main()
