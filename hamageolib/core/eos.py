
def compute_compressible_eos_properties(P, T,
                           rho_ref, beta_ref, alpha_ref, T_ref, Kp):
    '''
    Compute equation of state (EOS) properties: correction factor f,
    compressibility beta, and density rho.

    Parameters:
        P (float): Pressure (Pa)
        T (float): Temperature (K)
        rho_ref (float): Reference density (kg/m^3)
        beta_ref (float): Reference isothermal compressibility (Pa^-1)
        alpha_ref (float): Reference thermal expansivity (K^-1)
        T_ref (float): Reference temperature (K)
        Kp (float): Pressure derivative of bulk modulus (dimensionless)

    Returns:
        tuple:
            f (float): Correction factor (dimensionless)
            beta (float): Isothermal compressibility (Pa^-1)
            rho (float): Density (kg/m^3)
    '''
    # compute thermal-pressure coupling coefficient ak = alpha_ref / beta_ref
    ak = alpha_ref / beta_ref

    # compute correction factor f
    f = 1.0 + (P - ak * (T - T_ref)) * Kp * beta_ref

    # compute compressibility beta
    beta = beta_ref / f

    # compute density rho
    rho = rho_ref * f**(1.0 / Kp)

    return f, beta, rho