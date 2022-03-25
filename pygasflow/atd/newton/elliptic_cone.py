import numpy as np

def elliptic_cone(theta_xy, theta_xz):
    # substitutions to shorten the expressions
    m = np.tan(theta_xz) / np.tan(theta_xy)
    s = np.sin(theta_xz) / np.sin(theta_xy)
    lamb = np.cos(alpha) * np.cos(beta)
    tau = np.sqrt(1 - lamb**2)
    nu = -np.sin(beta)
    omega = np.sin(alpha) * np.cos(beta)

    Sigma = 1 + (m**2 - 1) * np.sin(phi)**2
    Psi = 1 + (m**2 * s**2 - 1) * np.sin(phi)**2
    Lambda = m * np.sqrt(s**2 - 1) * np.sin(phi)
    Lambda_p = m * np.sqrt(1 - s**2) * np.sin(phi)

    def CN_func_1(phi):
        return(
            (
                lamb**2 * np.sin(theta_xz)**2 * np.tan(theta_xy) / np.sqrt(1 - s**2)
                + s**2 * nu**2 * np.sin(2 * theta_xy) / (2 * (1 - s**2)**1.5)
                - s**2 * omega**2 * np.sin(2 * theta_xz) / (2 * m * (1 - s**2)**1.5)
            ) * np.log((np.sqrt(Sigma) + Lambda_p) / (np.sqrt(Sigma) - Lambda_p))
            + (2 * lamb * nu * np.sin(theta_xz)**2) / (s**2 - 1) * np.log(Psi / Sigma)
            + ((omega**2 * np.sin(2 * theta_xz)) / (1 - s**2) - (m * s**2 * nu**2 * np.sin(2 * theta_xy)) / (1 - s**2)) * np.sin(phi) / np.sqrt(Sigma)
            + (2 * lamb * omega * np.sin(2 * theta_xz) * np.tan(theta_xy)) / (1 - s**2) * (np.arctan(cotan(phi) / m) - s * np.arctan(cotan(phi) / (m * s)))
            - (4 * s * nu * omega * np.sin(theta_xy) * np.cos(xz)) / (1 - s**2) * (s / np.sqrt(1 - s**2) * np.arctan(np.cos(phi) * np.sqrt(1 - s**2) / (s * np.sqrt(Sigma))) - np.cos(phi) / np.sqrt(Sigma))
        )

    def CN_func_2(phi):
        return(
            (
                2 * lamb**2 * np.sin(theta_xz)**2 * np.tan(theta_xy) / np.sqrt(s**2 - 1)
                - s**2 * nu**2 * np.sin(2 * theta_xy) / ((s**2 - 1)**1.5)
                + s**2 * omega**2 * np.sin(2 * theta_xz) / (m * (s**2 - 1)**1.5)
            ) * np.arctan(Lambda / np.sqrt(Sigma))
            + (2 * lamb * nu * np.sin(theta_xz)**2) / (s**2 - 1) * np.log(Psi / Sigma)
            + ((omega**2 * np.sin(2 * theta_xz)) / (1 - s**2) - (m * s**2 * nu**2 * np.sin(2 * theta_xy)) / (1 - s**2)) * np.sin(phi) / np.sqrt(Sigma)
            + (2 * lamb * omega * np.sin(2 * theta_xz) * np.tan(theta_xy)) / (1 - s**2) * (np.arctan(cotan(phi) / m) - s * np.arctan(cotan(phi) / (m * s)))
            - (4 * s * nu * omega * np.sin(theta_xy) * np.cos(xz)) / (1 - s**2) * (s / (2 * np.sqrt(s**2 - 1)) * np.log((np.sqrt(Sigma) + np.cos(phi) * np.sqrt(s**2 - 1) / s) / (np.sqrt(Sigma) - np.cos(phi) * np.sqrt(s**2 - 1) / s)) - np.cos(phi) / np.sqrt(Sigma))
        )

    # eq (24a) and (24b)
    common_1 = s**2 * nu * omega * np.cos(theta_xy)**2
    common_2 = lamb * m * np.sin(theta_xz) * np.sqrt(s**2 * nu**2 * np.cos(theta_xy)**2 - (lamb**2 * np.sin(theta_xz)**2 - omega**2 * np.cos(theta_xz)**2))
    common_3 = lamb**2 * np.sin(theta_xz)**2 - omega**2 * np.cos(theta_xz)**2
    phi_i = arccotan((-common_1 - common_2) / common_3)
    phi_f = arccotan((-common_1 + common_2) / common_3)

    S = l**2 * np.tan(theta_xy) * np.tan(theta_xz) * (np.pi / 2 * np.arctan(cotan(phi_1) / m))
    if (m < 1) and (s < 1):
        return (CN_func_1(phi_f) - CN_func_1(phi_i))
    elif (m > 1) and (s > 1):
        return (CN_func_2(phi_f) - CN_func_2(phi_i))
    return None
