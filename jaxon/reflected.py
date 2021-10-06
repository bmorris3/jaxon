from functools import partial

import numpy as np
from numpy import pi as pi64
from jax import config, jit, numpy as jnp

from .utils import trapz3d, floatX

__all__ = [
    'reflected_phase_curve',
    'reflected_phase_curve_inhomogeneous'
]

pi = np.cast[floatX](pi64)
h = np.cast[floatX](6.62607015e-34)  # J s
c = np.cast[floatX](299792458.0)  # m/s
k_B = np.cast[floatX](1.380649e-23)  # J/K
hc2 = np.cast[floatX](6.62607015e-34 * 299792458.0 ** 2)

zero = np.cast[floatX](0)
one = np.cast[floatX](1)
two = np.cast[floatX](2)
half = np.cast[floatX](0.5)


def linspace(start, stop, n):
    dx = (stop - start) / (n - 1)
    return jnp.arange(start, stop + dx, dx, dtype=floatX)


@jit
def mu(theta):
    r"""
    Angle :math:`\mu = \cos(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    """
    return jnp.cos(theta)


@jit
def tilda_mu(theta, alpha):
    r"""
    The normalized quantity
    :math:`\tilde{\mu} = \alpha \mu(\theta)`

    Parameters
    ----------
    theta : `~numpy.ndarray`
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`
    """
    return alpha * mu(theta)


@partial(jit, static_argnums=(0,))
def H(l, theta, alpha):
    r"""
    Hermite Polynomials in :math:`\tilde{\mu}(\theta)`.

    Parameters
    ----------
    l : int
        Implemented through :math:`\ell \leq 7`.
    theta : float
        Angle :math:`\theta`
    alpha : float
        Dimensionless fluid number :math:`\alpha`

    Returns
    -------
    result : `~numpy.ndarray`
        Hermite Polynomial evaluated at angles :math:`\theta`.
    """
    if l == 0:
        return jnp.ones_like(theta)
    elif l == 1:
        return two * tilda_mu(theta, alpha)
    elif l == 2:
        return (two + two) * tilda_mu(theta, alpha) ** 2 - two
    else:
        raise NotImplementedError()


@jit
def h_ml(omega_drag, alpha, theta, phi, C_11, m=1, l=1):
    r"""
    The :math:`h_{m\ell}` basis function.

    Parameters
    ----------
    omega_drag : float
        Dimensionless drag
    alpha : float
        Dimensionless fluid number
    m : int
        Spherical harmonic ``m`` index
    l : int
        Spherical harmonic ``l`` index
    theta : `~numpy.ndarray`
        Latitudinal coordinate
    phi : `~numpy.ndarray`
        Longitudinal coordinate
    C_11 : float
        Spherical harmonic coefficient

    Returns
    -------
    hml : `~numpy.ndarray`
        :math:`h_{m\ell}` basis function.
    """
    prefactor = (C_11 /
                 (jnp.power(omega_drag, two) *
                  jnp.power(alpha, two * two) +
                  jnp.power(m, two)) *
                 jnp.exp(-jnp.power(tilda_mu(theta, alpha), two) * half))

    result = prefactor * (
                mu(theta) * m * H(l, theta, alpha) * jnp.cos(m * phi) +
                alpha * omega_drag * (tilda_mu(theta, alpha) *
                                      H(l, theta, alpha) -
                                      H(l + one, theta, alpha)) *
                jnp.sin(m * phi))
    return result


@jit
def h_ml_sum_theano(hotspot_offset, omega_drag, alpha,
                    theta2d, phi2d, C_11):
    """
    Cythonized implementation of the quadruple loop over: theta's, phi's,
    l's and m's to compute the h_ml_sum term
    """
    phase_offset = half * pi

    hml_sum = h_ml(omega_drag, alpha,
                   theta2d,
                   phi2d +
                   phase_offset +
                   hotspot_offset,
                   C_11)

    return hml_sum


@jit
def blackbody_lambda(lam, temperature):
    """
    Compute the blackbody flux as a function of wavelength `lam` in mks units
    """
    return (two * hc2 * jnp.power(lam, -5.0) /
            jnp.expm1(jnp.divide(h * c, (lam * k_B * temperature))))


@jit
def blackbody2d(wavelengths, temperature):
    """
    Planck function evaluated for a vector of wavelengths in units of meters
    and temperature in units of Kelvin

    Parameters
    ----------
    wavelengths : `~numpy.ndarray`
        Wavelength array in units of meters
    temperature : `~numpy.ndarray`
        Temperature in units of Kelvin

    Returns
    -------
    pl : `~numpy.ndarray`
        Planck function evaluated at each wavelength
    """

    return blackbody_lambda(wavelengths, temperature)



@jit
def integrate_planck(filt_wavelength, filt_trans,
                     temperature):
    """
    Integrate the Planck function over wavelength for the temperature map of the
    planet `temperature` and the temperature of the host star `T_s`. If
    `return_interp`, returns the interpolation function for the integral over
    the ratio of the blackbodies over wavelength; else returns only the map
    (which can be used for trapezoidal approximation integration)
    """

    bb_num = blackbody2d(filt_wavelength, temperature)
    int_bb_num = trapz3d(bb_num * filt_trans, filt_wavelength)

    return int_bb_num


@jit
def reflected_phase_curve(phases, omega, g, a_rp):
    """
    Reflected light phase curve for a homogeneous sphere by
    Heng, Morris & Kitzmann (2021).

    Parameters
    ----------
    phases : `~np.ndarray`
        Orbital phases of each observation defined on (0, 1)
    omega : tensor-like
        Single-scattering albedo as defined in
    g : tensor-like
        Scattering asymmetry factor, ranges from (-1, 1).
    a_rp : float, tensor-like
        Semimajor axis scaled by the planetary radius

    Returns
    -------
    flux_ratio_ppm : tensor-like
        Flux ratio between the reflected planetary flux and the stellar flux in
        units of ppm.
    A_g : tensor-like
        Geometric albedo derived for the planet given {omega, g}.
    q : tensor-like
        Integral phase function
    """
    # Convert orbital phase on (0, 1) to "alpha" on (0, np.pi)
    alpha = jnp.where(
        phases != 0.5,
        jnp.asarray(2 * np.pi * phases - np.pi),
        1e-10
    )

    abs_alpha = jnp.abs(alpha)  # .astype(floatX)
    alpha_sort_order = jnp.argsort(alpha)
    sin_abs_sort_alpha = jnp.sin(
        abs_alpha[alpha_sort_order])  # .astype(floatX)
    sort_alpha = alpha[alpha_sort_order]  # .astype(floatX)

    gamma = jnp.sqrt(1 - omega)
    eps = (1 - gamma) / (1 + gamma)

    # Equation 34 for Henyey-Greestein
    P_star = (1 - g ** 2) / (1 + g ** 2 +
                             2 * g * jnp.cos(alpha)) ** 1.5
    # Equation 36
    P_0 = (1 - g) / (1 + g) ** 2

    # Equation 10:
    Rho_S = P_star - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_S_0 = P_0 - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_L = 0.5 * eps * (2 - eps) * (1 + eps) ** 2
    Rho_C = eps ** 2 * (1 + eps) ** 2

    alpha_plus = jnp.sin(abs_alpha / 2) + jnp.cos(abs_alpha / 2)
    alpha_minus = jnp.sin(abs_alpha / 2) - jnp.cos(abs_alpha / 2)

    # Equation 11:
    Psi_0 = jnp.where(
        (alpha_plus != 1.0) & (alpha_minus != -1.0),
        jnp.log((1 + alpha_minus) * (alpha_plus - 1) /
                (1 + alpha_plus) / (1 - alpha_minus)),
        0
    )

    Psi_S = 1 - 0.5 * (jnp.cos(abs_alpha / 2) -
                       1.0 / jnp.cos(abs_alpha / 2)) * Psi_0
    Psi_L = (jnp.sin(abs_alpha) + (np.pi - abs_alpha) *
             jnp.cos(abs_alpha)) / np.pi
    Psi_C = (-1 + 5 / 3 * jnp.cos(abs_alpha / 2) ** 2 - 0.5 *
             jnp.tan(abs_alpha / 2) * jnp.sin(abs_alpha / 2) ** 3 * Psi_0)

    # Equation 8:
    A_g = omega / 8 * (P_0 - 1) + eps / 2 + eps ** 2 / 6 + eps ** 3 / 24

    # Equation 9:
    Psi = ((12 * Rho_S * Psi_S + 16 * Rho_L *
            Psi_L + 9 * Rho_C * Psi_C) /
           (12 * Rho_S_0 + 16 * Rho_L + 6 * Rho_C))

    flux_ratio_ppm = 1e6 * (a_rp ** -2 * A_g * Psi)

    # q = _integral_phase_function(
    #    Psi, sin_abs_sort_alpha, sort_alpha, alpha_sort_order
    # )

    return flux_ratio_ppm, A_g  # , q


def rho(omega, P_0, P_star):
    """
    Equation 10
    """
    gamma = jnp.sqrt(1 - omega)
    eps = (1 - gamma) / (1 + gamma)

    Rho_S = P_star - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_S_0 = P_0 - 1 + 0.25 * ((1 + eps) * (2 - eps)) ** 2
    Rho_L = 0.5 * eps * (2 - eps) * (1 + eps) ** 2
    Rho_C = eps ** 2 * (1 + eps) ** 2

    return Rho_S, Rho_S_0, Rho_L, Rho_C


def I(alpha, Phi):
    """
    Equation 39
    """
    cos_alpha = jnp.cos(alpha)
    cos_alpha_2 = jnp.cos(alpha / 2)

    z = jnp.sin(alpha / 2 - Phi / 2) / jnp.cos(Phi / 2)

    # The following expression has the same behavior
    # as I_0 = jnp.arctanh(z), but it doesn't blow up at alpha=0
    I_0 = jnp.where(jnp.abs(z) < 1, 0.5 * (jnp.log1p(z) - jnp.log1p(-z)), 0)
    #     I_0 = jnp.arctanh(z)

    I_S = (-1 / (2 * cos_alpha_2) *
           (jnp.sin(alpha / 2 - Phi) +
            (cos_alpha - 1) * I_0))
    I_L = 1 / np.pi * (Phi * cos_alpha -
                       0.5 * jnp.sin(alpha - 2 * Phi))
    I_C = -1 / (24 * cos_alpha_2) * (
            -3 * jnp.sin(alpha / 2 - Phi) +
            jnp.sin(3 * alpha / 2 - 3 * Phi) +
            6 * jnp.sin(3 * alpha / 2 - Phi) -
            6 * jnp.sin(alpha / 2 + Phi) +
            24 * jnp.sin(alpha / 2) ** 4 * I_0
    )

    return I_S, I_L, I_C


@jit
def trapz1d(y_1d, x):
    """
    Trapezoid rule in one dimension. This only works if x is increasing.
    """
    s = 0.5 * ((x[1:] - x[:-1]) * (y_1d[1:] + y_1d[:-1]))
    return jnp.sum(s, axis=-1)


@jit
def _integral_phase_function(Psi, sin_abs_sort_alpha, sort_alpha, sort):
    """
    Integral phase function q for a generic, possibly asymmetric reflectivity
    map
    """
    return trapz1d(Psi[sort] * sin_abs_sort_alpha, sort_alpha)


def _g_from_ag(A_g, omega_0, omega_prime, x1, x2):
    """
    Compute the scattering asymmetry factor g for a given geometric albedo,
    and possibly asymmetric single scattering albedos.

    Parameters
    ----------
    A_g : tensor-like
        Geometric albedo on (0, 1)
    omega_0 : tensor-like
        Single-scattering albedo of the less reflective region.
        Defined on (0, 1).
    omega_prime : tensor-like
        Additional single-scattering albedo of the more reflective region,
        such that the single-scattering albedo of the reflective region is
        ``omega_0 + omega_prime``. Defined on (0, ``1-omega_0``).
    x1 : tensor-like
        Start longitude of the darker region [radians] on (-pi/2, pi/2)
    x2 : tensor-like
        Stop longitude of the darker region [radians] on (-pi/2, pi/2)

    Returns
    -------
    g : tensor-like
        Scattering asymmetry factor
    """
    gamma = jnp.sqrt(1 - omega_0)
    eps = (1 - gamma) / (1 + gamma)

    gamma_prime = jnp.sqrt(1 - omega_prime)
    eps_prime = (1 - gamma_prime) / (1 + gamma_prime)

    Rho_L = eps / 2 * (1 + eps) ** 2 * (2 - eps)
    Rho_L_prime = eps_prime / 2 * (1 + eps_prime) ** 2 * (2 - eps_prime)
    Rho_C = eps ** 2 * (1 + eps) ** 2
    Rho_C_prime = eps_prime ** 2 * (1 + eps_prime) ** 2
    C = -1 + 0.25 * (1 + eps) ** 2 * (2 - eps) ** 2
    C_prime = -1 + 0.25 * (1 + eps_prime) ** 2 * (2 - eps_prime) ** 2

    C_2 = 2 + jnp.sin(x1) - jnp.sin(x2)
    C_1 = (omega_0 * Rho_L * np.pi / 12 + omega_prime * Rho_L_prime / 12 *
           (x1 - x2 + np.pi + 0.5 * (jnp.sin(2 * x1) - jnp.sin(
               2 * x2))) +
           np.pi * omega_0 * Rho_C / 32 + 3 * np.pi * omega_prime *
           Rho_C_prime / 64 *
           (2 / 3 + 3 / 8 * (jnp.sin(x1) - jnp.sin(x2)) +
            1 / 24 * (jnp.sin(3 * x1) - jnp.sin(3 * x2))))
    C_3 = (16 * np.pi * A_g - 32 * C_1 - 2 * np.pi * omega_0 * C -
           np.pi * omega_prime * C_2 * C_prime
           ) / (2 * np.pi * omega_0 + np.pi * omega_prime * C_2)

    return - ((2 * C_3 + 1) - jnp.sqrt(1 + 8 * C_3)) / (2 * C_3)


def reflected_phase_curve_inhomogeneous(phases, omega_0, omega_prime, x1, x2,
                                        A_g, a_rp):
    """
    Reflected light phase curve for an inhomogeneous sphere by
    Heng, Morris & Kitzmann (2021), with inspiration from Hu et al. (2015).

    Parameters
    ----------
    phases : `~np.ndarray`
        Orbital phases of each observation defined on (0, 1)
    omega_0 : tensor-like
        Single-scattering albedo of the less reflective region.
        Defined on (0, 1).
    omega_prime : tensor-like
        Additional single-scattering albedo of the more reflective region,
        such that the single-scattering albedo of the reflective region is
        ``omega_0 + omega_prime``. Defined on (0, ``1-omega_0``).
    x1 : tensor-like
        Start longitude of the darker region [radians] on (-pi/2, pi/2)
    x2 : tensor-like
        Stop longitude of the darker region [radians] on (-pi/2, pi/2)
    a_rp : float, tensor-like
        Semimajor axis scaled by the planetary radius

    Returns
    -------
    flux_ratio_ppm : tensor-like
        Flux ratio between the reflected planetary flux and the stellar flux
        in units of ppm.
    g : tensor-like
        Scattering asymmetry factor on (-1, 1)
    q : tensor-like
        Integral phase function
    """

    g = _g_from_ag(A_g, omega_0, omega_prime, x1, x2)

    # Redefine alpha to be on (-pi, pi)
    alpha = (2 * np.pi * phases - np.pi).astype(floatX)
    abs_alpha = np.abs(alpha).astype(floatX)
    alpha_sort_order = np.argsort(alpha)
    sin_abs_sort_alpha = np.sin(abs_alpha[alpha_sort_order]).astype(floatX)
    sort_alpha = alpha[alpha_sort_order].astype(floatX)

    # Equation 34 for Henyey-Greestein
    P_star = (1 - g ** 2) / (1 + g ** 2 +
                             2 * g * jnp.cos(abs_alpha)) ** 1.5
    # Equation 36
    P_0 = (1 - g) / (1 + g) ** 2

    Rho_S, Rho_S_0, Rho_L, Rho_C = rho(omega_0, P_0, P_star)

    Rho_S_prime, Rho_S_0_prime, Rho_L_prime, Rho_C_prime = rho(
        omega_prime, P_0, P_star
    )

    alpha_plus = jnp.sin(abs_alpha / 2) + jnp.cos(abs_alpha / 2)
    alpha_minus = jnp.sin(abs_alpha / 2) - jnp.cos(abs_alpha / 2)

    # Equation 11:
    Psi_0 = jnp.where(
        (alpha_plus > -1) & (alpha_minus < 1),
        jnp.log((1 + alpha_minus) * (alpha_plus - 1) /
                (1 + alpha_plus) / (1 - alpha_minus)),
        0
    )

    Psi_S = 1 - 0.5 * (jnp.cos(abs_alpha / 2) -
                       1.0 / jnp.cos(abs_alpha / 2)) * Psi_0
    Psi_L = (jnp.sin(abs_alpha) + (np.pi - abs_alpha) *
             jnp.cos(abs_alpha)) / np.pi
    Psi_C = (-1 + 5 / 3 * jnp.cos(abs_alpha / 2) ** 2 -
             0.5 * jnp.tan(abs_alpha / 2) *
             jnp.sin(abs_alpha / 2) ** 3 * Psi_0)

    # Table 1:
    condition_a = (-np.pi / 2 <= alpha - np.pi / 2)
    condition_0 = ((alpha - np.pi / 2 <= np.pi / 2) &
                   (np.pi / 2 <= alpha + x1) &
                   (alpha + x1 <= alpha + x2))
    condition_1 = ((alpha - np.pi / 2 <= alpha + x1) &
                   (alpha + x1 <= np.pi / 2) &
                   (np.pi / 2 <= alpha + x2))
    condition_2 = ((alpha - np.pi / 2 <= alpha + x1) &
                   (alpha + x1 <= alpha + x2) &
                   (alpha + x2 <= np.pi / 2))

    condition_b = (alpha + np.pi / 2 <= np.pi / 2)
    condition_3 = ((alpha + x1 <= alpha + x2) &
                   (alpha + x2 <= -np.pi / 2) &
                   (-np.pi / 2 <= alpha + np.pi / 2))
    condition_4 = ((alpha + x1 <= -np.pi / 2) &
                   (-np.pi / 2 <= alpha + x2) &
                   (alpha + x2 <= alpha + np.pi / 2))
    condition_5 = ((-np.pi / 2 <= alpha + x1) &
                   (alpha + x1 <= alpha + x2) &
                   (alpha + x2 <= alpha + np.pi / 2))

    integration_angles = [
        [alpha - np.pi / 2, np.pi / 2],
        [alpha - np.pi / 2, alpha + x1],
        [alpha - np.pi / 2, alpha + x1, alpha + x2, np.pi / 2],
        [-np.pi / 2, alpha + np.pi / 2],
        [alpha + x2, alpha + np.pi / 2],
        [-np.pi / 2, alpha + x1, alpha + x2, alpha + np.pi / 2]
    ]

    conditions = [
        condition_a & condition_0,
        condition_a & condition_1,
        condition_a & condition_2,
        condition_b & condition_3,
        condition_b & condition_4,
        condition_b & condition_5,
    ]

    Psi_S_prime = 0
    Psi_L_prime = 0
    Psi_C_prime = 0

    for condition_i, angle_i in zip(conditions, integration_angles):
        for i, phi_i in enumerate(angle_i):
            sign = (-1) ** (i + 1)
            I_phi_S, I_phi_L, I_phi_C = I(alpha, phi_i)
            Psi_S_prime += jnp.where(condition_i, sign * I_phi_S, 0)
            Psi_L_prime += jnp.where(condition_i, sign * I_phi_L, 0)
            Psi_C_prime += jnp.where(condition_i, sign * I_phi_C, 0)

    # Compute everything for alpha=0
    angles_alpha0 = [-np.pi / 2, x1, x2, np.pi / 2]
    Psi_S_prime_alpha0 = 0
    Psi_L_prime_alpha0 = 0
    Psi_C_prime_alpha0 = 0
    for i, phi_i in enumerate(angles_alpha0):
        sign = (-1) ** (i + 1)
        I_phi_S_alpha0, I_phi_L_alpha0, I_phi_C_alpha0 = I(0, phi_i)

        Psi_S_prime_alpha0 += sign * I_phi_S_alpha0
        Psi_L_prime_alpha0 += sign * I_phi_L_alpha0
        Psi_C_prime_alpha0 += sign * I_phi_C_alpha0

    # Equation 37
    F_S = np.pi / 16 * (omega_0 * Rho_S * Psi_S +
                        omega_prime * Rho_S_prime * Psi_S_prime)
    F_L = np.pi / 12 * (omega_0 * Rho_L * Psi_L +
                        omega_prime * Rho_L_prime * Psi_L_prime)
    F_C = 3 * np.pi / 64 * (omega_0 * Rho_C * Psi_C +
                            omega_prime * Rho_C_prime * Psi_C_prime)

    sobolev_fluxes = F_S + F_L + F_C
    F_max = jnp.max(sobolev_fluxes)

    Psi = sobolev_fluxes / F_max

    flux_ratio_ppm = 1e6 * a_rp ** -2 * Psi * A_g

    q = _integral_phase_function(Psi, sin_abs_sort_alpha, sort_alpha,
                                 alpha_sort_order)

    # F_0 = F_S_alpha0 + F_L_alpha0 + F_C_alpha0

    return flux_ratio_ppm, g, q
