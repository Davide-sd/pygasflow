import numpy as np

def sphere_solver(R, alpha, beta, phi_1=0, phi_2=2*np.pi):
    # substitutions to shorten the expressions
    lamb = np.cos(alpha) * np.cos(beta)
    tau = np.sqrt(1 - lamb**2)
    nu = -np.sin(beta)
    omega = np.sin(alpha) * np.cos(beta)

    def phi_1_func(sigma):
        # avoid NaN values due to rounding errors
        t = tau**2 - lamb**2 * cotan(sigma)**2
        t[np.abs(t) < 1e-12] = 0

        t2 = (-lamb * nu * cotan(sigma) + omega * np.sqrt(t)) / tau**2
        o = np.ones_like(t2)
        t2[np.isclose(t2, o)] = 1
        t2[np.isclose(t2, -o)] = -1
        return np.arcsin(t2)

    def phi_2_func(sigma):
        # avoid NaN values due to rounding errors
        t = tau**2 - lamb**2 * cotan(sigma)**2
        t[np.abs(t) < 1e-12] = 0

        print("phi_2_func", t)
        t2 = (-lamb * nu * cotan(sigma) + omega * np.sqrt(t)) / tau**2
        o = np.ones_like(t2)
        t2[np.isclose(t2, o)] = 1
        t2[np.isclose(t2, -o)] = -1
        return np.arcsin(t2)

    def CN_func_1(sigma):
        # phi_1 = phi_1_func(sigma)
        # phi_2 = phi_2_func(sigma)
        return (
            (1 / 4 * np.sin(4 * sigma) - sigma) * lamb**2 / 2 * np.sin(phi_1)
            - np.sin(phi_1) / 3 * (3 * omega**2 + (nu**2 - omega**2) * np.sin(phi_1)**2) * (3 * sigma / 2 - np.sin(2 * sigma) + 1 / 8 * np.sin(4 * sigma))
            - lamb * omega * (np.pi - phi_1 - 1/2 * np.sin(2 * phi_1) * np.sin(sigma)**4)
        )

    def CN_func_2(sigma):
        phi_1c = phi_1_func(sigma)
        phi_2c = phi_2_func(sigma)

        # avoid NaN values due to rounding errors
        t = tau**2 - np.cos(sigma)**2
        t[np.abs(t) < 1e-15] = 1e-15

        # avoid 1+eps or -(1+eps) values inside arcsin
        t2 = np.cos(sigma) / tau
        o = np.ones_like(t2)
        t2[np.isclose(t2, o)] = 1
        t2[np.isclose(t2, -o)] = -1

        return (
            (1 / 4 * np.sin(4 * sigma) - sigma) * lamb**2 / 4 * np.sin(phi_1)
            + (nu * omega / 3 * np.cos(phi_1)**3 - ((nu**2 - omega**2) * np.sin(phi_1)**2 + 3 * omega**2) * np.sin(phi_1) / 6) * (3 * sigma / 2 - np.sin(2 * sigma) + 1 / 8 * np.sin(4 * sigma))
            + (lamb * nu / 2 * np.sin(phi_1)**2 - lamb * omega / 2 * (2 * np.pi - phi_1 - 1 / 2 * np.sin(2 * phi_1))) * np.sin(sigma)**4
            - lamb** 3 * nu / (6 * tau**2) * np.cos(sigma)**4
            - lamb * omega * phi_1c * np.cos(sigma)**2
            - lamb * omega / 2 * phi_1c * np.cos(sigma)**4
            + omega / (6 * tau**2) * (3 * tau**2 - 1) * (t)**1.5 * np.cos(sigma)
            + omega / 2 * (lamb**2 + 1) * np.sqrt(t) * np.cos(sigma)
            + omega / 2 * np.arcsin(t2)
            - lamb * omega / 2 * np.arctan((lamb * np.cos(sigma)) / np.sqrt(t))
        )

    def CN_func_3(sigma):
        phi_1c = phi_1_func(sigma)
        phi_2c = phi_2_func(sigma)

        # avoid NaN values due to rounding errors
        t = tau**2 - np.cos(sigma)**2
        t[np.abs(t) < 1e-12] = 1e-12

        # avoid 1+eps or -(1+eps) values inside arcsin
        t2 = np.cos(sigma) / tau
        o = np.ones_like(t2)
        t2[np.isclose(t2, o)] = 1
        t2[np.isclose(t2, -o)] = -1

        return (
            omega / (3 * tau**2) * (3 * tau**2 - 1) * (t)**1.5 * np.cos(sigma)
            + omega * (lamb**2 + 1) * np.sqrt(t) * np.cos(sigma)
            + lamb * omega * (phi_2c - phi_1c) * np.cos(sigma)**2
            - lamb * omega / 2 * (phi_2c - phi_1c) * np.cos(sigma)**4
            + omega * np.arcsin(t2)
            - lamb * omega * np.arctan((lamb * np.cos(sigma)) / np.sqrt(t))
        )

    def sigma_func(phi):
        return arccotan((-nu * np.sin(phi) + omega * np.cos(phi)) / lamb)
    sigma_1 = np.pi / 2 - np.arctan(tau / lamb)
    sigma_2 = sigma_func(phi_1)
    sigma_3 = sigma_func(phi_2)
    sigma_4 = np.pi / 2 + np.arctan(tau / lamb)

    phi_3 = phi_1_func(sigma_3)
    phi_4 = phi_2_func(sigma_4)
    idx = (phi_4 > phi_3)
    sigma_4[idx] = sigma_3[idx]

    print("sigma_1", sigma_1, np.rad2deg(sigma_1))
    print("sigma_2", sigma_2, np.rad2deg(sigma_2))
    print("sigma_3", sigma_3, np.rad2deg(sigma_3))
    print("sigma_4", sigma_4, np.rad2deg(sigma_4))

    S = 4 * np.pi * R**2
    CN = -R**2 / S * (
        (CN_func_1(sigma_2) - CN_func_1(0))
        + (CN_func_2(sigma_3) - CN_func_2(sigma_2))
        + (CN_func_3(sigma_4) - CN_func_3(sigma_3))
    )
    return CN
