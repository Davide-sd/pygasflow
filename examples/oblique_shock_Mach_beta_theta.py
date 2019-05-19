import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from pygasflow.shockwave import (
    Beta_From_Mach_Theta,
    Theta_From_Mach_Beta,
    Max_Theta_From_Mach,
    Beta_From_Mach_Max_Theta,
    Beta_Theta_Max_For_Unit_Mach_Downstream,
)

def Fast_Version(M, gamma):
    """
    To compute the Mach curves, this function discretize the Shock Wave Angle
    range and then compute the respective Deflection Angles.
    """

    # number of points for each Mach curve
    N = 100

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
        thetas = Theta_From_Mach_Beta(m, betas, gamma)
        plt.plot(thetas, betas, color=c[i], linewidth=1, label=lbls[i])

    # compute the line passing through (M,theta_max) and the line M2 = 1
    M1 = np.logspace(0, 3, 5 * N)
    beta = Beta_From_Mach_Max_Theta(M1, gamma)
    beta_M2_equal_1, theta_max = Beta_Theta_Max_For_Unit_Mach_Downstream(M1, gamma)
    
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
    plt.annotate("$M_{2} > 1$", 
        (theta_max[i], beta_M2_equal_1[i]),
        (theta_max[i], beta_M2_equal_1[i] + 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-", color="0.3"),
        color="0.3",
    )
    plt.annotate("$M_{2} < 1$", 
        (theta_max[i], beta_M2_equal_1[i]),
        (theta_max[i], beta_M2_equal_1[i] - 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-", color="0.3"),
        color="0.3",
    )

    plt.title(r"Mach - $\beta$ - $\theta$")
    plt.xlabel(r"Flow Deflection Angle, $\theta$ [deg]")
    plt.ylabel(r"Shock Wave Angle, $\beta$ [deg]")
    plt.xlim((0, 50))
    plt.ylim((0, 90))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', alpha=0.7)
    plt.grid(which='minor', linestyle=':', alpha=0.5)
    plt.legend(loc="lower right")
    plt.show()

def Slow_Version(M, gamma):
    """
    To compute the Mach curves, this function first find the maximum Deflection
    angle of each Mach curve, than discretize the deflection angle in the range
    0, theta_max and finally compute the shock wave angles.
    It is slower because it require a minimization procedure to find theta_max,
    and it needs a lot more discretization points to produce a nice result.
    """

    # number of points for each Mach curve
    N = 200

    # colors
    jet = plt.get_cmap('hsv')
    cNorm  = colors.Normalize(vmin=0, vmax=len(M))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    c = [scalarMap.to_rgba(i) for i in range(len(M))]

    # labels
    lbls = [r"$M_{1}$ = " + str(M[i]) for  i in range(len(M))]
    lbls[-1] = r"$M_1$ = $\infty$"

    # asdasdasdasd
    def logspace(a, b, N):
        # use log space (base=10, start value = 1 = 10^0, end value = 10 = 10^1)
        scale = np.logspace(0, 1, N)
        scale = scale[-1] - scale
        scale_diff = scale - scale[-1]
        return scale_diff / (scale[0] - scale[-1]) * (b - a) + a
    
    for i, m in enumerate(M):
        theta_max = Max_Theta_From_Mach(m)
        # pack the theta points around theta_max
        thetas = logspace(0, theta_max, N)
        betas = np.zeros(2 * N)
        for j, t in enumerate(thetas):
            b = Beta_From_Mach_Theta(m, t, gamma)
            betas[j] = b["weak"]
            betas[2 * N - 1 - j] = b["strong"]

        thetas = np.append(thetas, thetas[::-1])

        plt.plot(thetas, betas, color=c[i], linewidth=1, label=lbls[i])

    # compute the line passing through (M,theta_max) and the line M2 = 1
    M1 = np.logspace(0, 3, 2 * N)
    beta = Beta_From_Mach_Max_Theta(M1, gamma)
    beta_M2_equal_1, theta_max = Beta_Theta_Max_For_Unit_Mach_Downstream(M1, gamma)
    
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
    plt.annotate("$M_{2} > 1$", 
        (theta_max[i], beta_M2_equal_1[i]),
        (theta_max[i], beta_M2_equal_1[i] + 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-", color="0.3"),
        color="0.3",
    )
    plt.annotate("$M_{2} < 1$", 
        (theta_max[i], beta_M2_equal_1[i]),
        (theta_max[i], beta_M2_equal_1[i] - 10),
        horizontalalignment='center',
        arrowprops=dict(arrowstyle = "<-", color="0.3"),
        color="0.3",
    )

    plt.title(r"Mach - $\beta$ - $\theta$")
    plt.xlabel(r"Flow Deflection Angle, $\theta$ [deg]")
    plt.ylabel(r"Shock Wave Angle, $\beta$ [deg]")
    plt.xlim((0, 50))
    plt.ylim((0, 90))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', alpha=0.7)
    plt.grid(which='minor', linestyle=':', alpha=0.5)
    plt.legend(loc="lower right")
    plt.show()

def main():
    gamma = 1.4
    M1 = [1.1, 1.5, 2, 3, 5, 10, 1e9]

    Fast_Version(M1, gamma)
    # Slow_Version(M1, gamma)

if __name__ == "__main__":
    main()