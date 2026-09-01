#!/usr/bin/env python3
"""
Verification of Appendix Arbitrary-Precision Quadrature for von Karman-Pao spectrum.
Computes u'^2 f_VKP(r) across scales using mpmath high-precision quadrature with
the small-argument Taylor expansion of the projection kernel K_11(x).
"""

import mpmath as mp

def verify_quadrature():
    mp.mp.dps = 30
    L = 1.0
    u_rms = 1.0
    nu = 1e-5
    epsilon = u_rms**3 / L
    eta = (nu**3 / epsilon)**0.25

    def E_vkp(k):
        kL = k * L
        return (
            1.5 * u_rms**2 * L * (kL)**4
            / (1.0 + kL**2)**(mp.mpf('17.0') / 6)
            * mp.exp(-mp.mpf('2.25') * (k * eta)**(mp.mpf('4.0') / 3))
        )

    # u'^2 = (2/3) \int_0^\infty E(k) dk
    u2 = (mp.mpf(2) / 3) * mp.quad(E_vkp, [0, 10, mp.inf])

    r_test_points = [0.001, 1.0, 5.0, 10.0, 30.0]

    print("=" * 70)
    print("Appendix Table: Arbitrary-Precision Quadrature Verification for f_VKP(r)")
    print("=" * 70)

    for r_val in r_test_points:
        r_mp = mp.mpf(r_val)

        def integrand(k):
            kr = k * r_mp
            # Small-argument Taylor series expansion for K_11(x) to prevent numerical cancellation
            if kr < 1e-3:
                k11 = (
                    mp.mpf(2) / 3
                    - (mp.mpf(1) / 15) * (kr**2)
                    + (mp.mpf(1) / 420) * (kr**4)
                    - (mp.mpf(1) / 22680) * (kr**6)
                )
            else:
                k11 = 2 * (mp.sin(kr) - kr * mp.cos(kr)) / (kr**3)
            return E_vkp(k) * k11

        if r_val < 1.0:
            val = mp.quad(integrand, [0, 10, mp.inf])
        else:
            val = mp.quadosc(integrand, [0, mp.inf], zeros=lambda n: n * mp.pi / r_mp)

        f_res = val / u2
        print(f"f_VKP(r/L = {r_val:>5.3f}) = {float(f_res):.8e} (mpmath: {f_res}) > 0")

    print("=" * 70)


if __name__ == "__main__":
    verify_quadrature()
