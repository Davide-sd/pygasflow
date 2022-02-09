import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.testing.compare import compare_images
from pygasflow.solvers import (
    isentropic_solver, fanno_solver, rayleigh_solver, shockwave_solver
)
from pygasflow.shockwave import (
    theta_from_mach_beta, beta_from_mach_max_theta,
    beta_theta_max_for_unit_mach_downstream,
    max_theta_c_from_mach, mach_cone_angle_from_shock_angle,
    load_data
)

# NOTE: as I'm too lazy to go and test every single function of the module,
# I went to test the most common ones. I did that by comparing the generated
# pictures with the original ones, which were vetted during the development
# of the module with the reference book.

def test_isentropic_solver():
    M = np.linspace(1e-05, 4, 100)
    r = isentropic_solver("m", M)

    plt.figure()
    plt.plot(M, r[4], label=r"$p / p^{*}$")
    plt.plot(M, r[5], label=r"$\rho / \rho^{*}$")
    plt.plot(M, r[6], label=r"$T / T^{*}$")
    plt.plot(M, r[8], label=r"$A / A^{*}$")
    plt.plot(M, r[1], label=r"$P / P_{0}$")
    plt.plot(M, r[2], label=r"$\rho / \rho_{0}$")
    plt.plot(M, r[3], label=r"$T / T_{0}$")

    plt.xlim(0, max(M))
    plt.ylim(0, 3)
    plt.xlabel("M")
    plt.ylabel("ratios")
    plt.grid(which='major', linestyle='-', alpha=0.7)
    plt.grid(which='minor', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title("Isentropic Flow")
    plt.savefig("test_isentropic.png")

    rc = compare_images("test_isentropic.png", "imgs/isentropic.png", 0.001)
    assert rc is None

def test_fanno_solver():
    M = np.linspace(1e-05, 5, 100)
    r = fanno_solver("m", M)

    plt.figure()
    plt.plot(M, r[1], label=r"$p / p^{*}$")
    plt.plot(M, r[2], label=r"$\rho / \rho^{*}$")
    plt.plot(M, r[3], label=r"$T / T^{*}$")
    plt.plot(M, r[4], label=r"$P0 / P_{0}^{*}$")
    plt.plot(M, r[5], label=r"$U / U^{*}$")
    plt.plot(M, r[6], label=r"$4f L^{*} / D$")
    plt.plot(M, r[7], label=r"$(s^{*}-s)/R$")

    plt.xlim(0, max(M))
    plt.ylim(0, 3)
    plt.xlabel("M")
    plt.ylabel("ratios")
    plt.grid(which='major', linestyle='-', alpha=0.7)
    plt.grid(which='minor', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title("Fanno Flow")
    plt.savefig("test_fanno.png")

    rc = compare_images("test_fanno.png", "imgs/fanno.png", 0.001)
    assert rc is None


def test_rayleigh_solver():
    M = np.linspace(1e-05, 5, 100)
    r = rayleigh_solver("m", M)

    plt.figure()
    plt.plot(M, r[1], label=r"$p / p^{*}$")
    plt.plot(M, r[2], label=r"$\rho / \rho^{*}$")
    plt.plot(M, r[3], label=r"$T / T^{*}$")
    plt.plot(M, r[4], label=r"$P0 / P_{0}^{*}$")
    plt.plot(M, r[5], label=r"$T0 / T_{0}^{*}$")
    plt.plot(M, r[6], label=r"$U / U^{*}$")
    plt.plot(M, r[7], label=r"$(s^{*}-s)/R$")

    plt.xlim(0, max(M))
    plt.ylim(0, 3)
    plt.xlabel("M")
    plt.ylabel("ratios")
    plt.grid(which='major', linestyle='-', alpha=0.7)
    plt.grid(which='minor', linestyle=':', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title("Rayleigh Flow")
    plt.savefig("test_rayleigh.png")

    rc = compare_images("test_rayleigh.png", "imgs/rayleigh.png", 0.001)
    assert rc is None

def test_oblique_shock():
    M = [1.1, 1.5, 2, 3, 5, 10, 1e9]
    gamma = 1.4

    # number of points for each Mach curve
    N = 100

    plt.figure()

    # colors
    jet = plt.get_cmap('hsv')
    cNorm  = colors.Normalize(vmin=0, vmax=len(M))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    c = [scalarMap.to_rgba(i) for i in range(len(M))]

    # labels
    lbls = [r"$M_{1}$ = " + str(M[i]) for  i in range(len(M))]
    lbls[-1] = r"$M_1$ = $\infty$"

    for i, m in enumerate(M):
        beta_min = np.rad2deg(np.arcsin(1 / m))
        betas = np.linspace(beta_min, 90, N)
        thetas = theta_from_mach_beta(m, betas, gamma)
        plt.plot(thetas, betas, color=c[i], linewidth=1, label=lbls[i])

    # compute the line passing through (M,theta_max) and the line M2 = 1
    M1 = np.logspace(0, 3, 5 * N)
    beta = beta_from_mach_max_theta(M1, gamma)
    beta_M2_equal_1, theta_max = beta_theta_max_for_unit_mach_downstream(M1, gamma)

    plt.plot(theta_max, beta, '--', color="0.2", linewidth=1)
    plt.plot(theta_max, beta_M2_equal_1, ':', color="0.3", linewidth=1)

    # select an index where to put the annotation
    i = 50
    plt.annotate("strong",
        (theta_max[i], beta[i]),
        (theta_max[i], beta[i] + 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-")
    )
    plt.annotate("weak",
        (theta_max[i], beta[i]),
        (theta_max[i], beta[i] - 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-"),
    )

    i = 20
    plt.annotate("$M_{2} < 1$",
        (theta_max[i], beta_M2_equal_1[i]),
        (theta_max[i], beta_M2_equal_1[i] + 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-", color="0.3"),
        color="0.3",
    )
    plt.annotate("$M_{2} > 1$",
        (theta_max[i], beta_M2_equal_1[i]),
        (theta_max[i], beta_M2_equal_1[i] - 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-", color="0.3"),
        color="0.3",
    )

    plt.suptitle("Oblique Shock Properties")
    plt.title(r"Mach - $\beta$ - $\theta$")
    plt.xlabel(r"Flow Deflection Angle, $\theta$ [deg]")
    plt.ylabel(r"Shock Wave Angle, $\beta$ [deg]")
    plt.xlim((0, 50))
    plt.ylim((0, 90))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', alpha=0.7)
    plt.grid(which='minor', linestyle=':', alpha=0.5)
    plt.legend(loc="lower right")
    plt.savefig("test_mach-beta-theta.png")

    # NOTE: by overlapping these two images with GIMP, there are minor
    # differences where the two black lines intersect on the right side
    # of the picture. Hence, here I increased the tolerance.
    rc = compare_images("test_mach-beta-theta.png", "imgs/mach-beta-theta.png", 0.09)
    assert rc is None


def test_conical_flow():
    Minf = [1.05, 1.2, 1.5, 2, 5, 10000]
    gamma = 1.4
    N = 200

    plt.figure()

    # colors
    jet = plt.get_cmap('hsv')
    cNorm  = colors.Normalize(vmin=0, vmax=len(Minf))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    c = [scalarMap.to_rgba(i) for i in range(len(Minf))]
    # labels
    lbls = [r"$M_{1}$ = " + str(Minf[i]) for  i in range(len(Minf))]
    lbls[-1] = r"$M_1$ = $\infty$"

    plt.figure()
    for j, M in enumerate(Minf):
        theta_c = np.zeros(N)
        # NOTE: to avoid errors in the integration process of Taylor-Maccoll equation,
        # beta should be different than Mach angle and 90deg, hence an offset is applied.
        offset = 1e-08
        theta_s = np.linspace(np.rad2deg(np.arcsin(1 / M)) + offset, 90 - offset, N)
        for i, ts in enumerate(theta_s):
            Mc, tc = mach_cone_angle_from_shock_angle(M, ts, gamma)
            theta_c[i] = tc
        theta_c = np.insert(theta_c, 0, 0)
        theta_c = np.append(theta_c, 0)
        theta_s = np.insert(theta_s, 0, np.rad2deg(np.arcsin(1 / M)))
        theta_s = np.append(theta_s, 90)
        plt.plot(theta_c, theta_s, color=c[j], label=lbls[j])

    # Compute the line passing through theta_c_max
    M = np.asarray([1.0005, 1.0025, 1.005, 1.025, 1.05, 1.07, 1.09,
                    1.12, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5,
                    1.6, 1.75, 2, 2.25, 3, 4, 5, 10, 100, 10000])
    b = np.zeros_like(M)
    tc = np.zeros_like(M)
    for i, m in enumerate(M):
        _, tc[i], b[i] = max_theta_c_from_mach(m, gamma)
    tc = np.insert(tc, 0, 0)
    b = np.insert(b, 0, 90)
    plt.plot(tc, b, '--', color="0.2", linewidth=1)

    # select an index where to put the annotation (chosen by trial and error)
    i = 16
    plt.annotate("strong",
        (tc[i], b[i]),
        (tc[i], b[i] + 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-")
    )
    plt.annotate("weak",
        (tc[i], b[i]),
        (tc[i], b[i] - 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-"),
    )

    M, beta, theta_c = load_data(gamma)
    plt.plot(np.asarray(theta_c), np.asarray(beta), ':', color="0.2", linewidth=1)

    i = 54
    plt.annotate("$M_{2} < 1$",
        (theta_c[i], beta[i]),
        (theta_c[i], beta[i] + 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-", color="0.3"),
        color="0.3",
    )
    plt.annotate("$M_{2} > 1$",
        (theta_c[i], beta[i]),
        (theta_c[i], beta[i] - 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-", color="0.3"),
        color="0.3",
    )

    # If there is no data for M2=1, we just need to generate it. IT IS SLOW!!!
    # M = np.asarray([1.05, 1.2, 1.35, 1.5, 2, 3, 5, 10000])
    # beta = np.zeros_like(M)
    # theta_c = np.zeros_like(M)
    # for i, m in enumerate(M):
    #     beta[i], theta_c[i] = beta_theta_c_for_unit_mach_downstream(m, gamma)
    # theta_c = np.insert(theta_c, 0, 0)
    # beta = np.insert(beta, 0, 90)
    # plt.plot(theta_c, beta, ':', color="0.2", linewidth=1)

    plt.suptitle("Conical Flow Properties")
    plt.title(r"Mach - $\theta_{s}$ - $\theta_{c}$")
    plt.xlabel(r"Cone Angle, $\theta_{c}$ [deg]")
    plt.ylabel(r"Shock Wave Angle, $\theta_{s}$ [deg]")
    plt.xlim((0, 60))
    plt.ylim((0, 90))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', alpha=0.7)
    plt.grid(which='minor', linestyle=':', alpha=0.5)
    plt.legend(loc="lower right")
    plt.savefig("test_conical_flow.png")

    rc = compare_images("test_conical_flow.png", "imgs/conical-flow.png", 0.001)
    assert rc is None
