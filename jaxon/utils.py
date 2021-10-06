import numpy as np
from jax import jit, numpy as jnp, random

rng_key = random.PRNGKey(0)

floatX = 'float64'
zero = np.cast[floatX](0)
one = np.cast[floatX](1)
two = np.cast[floatX](2)
half = np.cast[floatX](0.5)


@jit
def sum2d(z):
    """
    Sum a 2d array over its axes
    """
    return jnp.sum(z)


@jit
def sum1d(z):
    """
    Sum a 1d array over its first axis
    """
    return jnp.sum(z)


@jit
def sinsq_2d(z):
    """
    The square of the sine of a 2d array
    """
    return jnp.power(jnp.sin(z), 2)


@jit
def cos_2d(z):
    """
    The cosine of a 2d array
    """
    return jnp.cos(z)


@jit
def trapz2d(z, x, y):
    """
    Integrates a regularly spaced 2D grid using the composite trapezium rule.

    Source: https://github.com/tiagopereira/python_tips/blob/master/code/trapz2d.py

    Parameters
    ----------
    z : `~numpy.ndarray`
        2D array
    x : `~numpy.ndarray`
        grid values for x (1D array)
    y : `~numpy.ndarray`
        grid values for y (1D array)

    Returns
    -------
    t : `~numpy.ndarray`
        Trapezoidal approximation to the integral under z
    """
    m = z.shape[0] - 1
    n = z.shape[1] - 1
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    s1 = z[0, 0, :] + z[m, 0, :] + z[0, n, :] + z[m, n, :]
    s2 = (jnp.sum(z[1:m, 0, :], axis=0) + jnp.sum(z[1:m, n, :], axis=0) +
          jnp.sum(z[0, 1:n, :], axis=0) + jnp.sum(z[m, 1:n, :], axis=0))
    s3 = jnp.sum(jnp.sum(z[1:m, 1:n, :], axis=0), axis=0)

    return dx * dy * (s1 + two * s2 + (two + two) * s3) / (two + two)


@jit
def trapz3d(y_3d, x):
    """
    Trapezoid rule in ~more dimensions~
    """
    s = half * ((x[..., 1:] - x[..., :-1]) * (y_3d[..., 1:] + y_3d[..., :-1]))
    return jnp.sum(s, axis=-1)


@jit
def interpolate(x0, y0, x):
    x = jnp.asarray(x)

    idx = jnp.searchsorted(x0, x)
    dl = jnp.asarray(x - x0[idx - 1])
    dr = jnp.asarray(x0[idx] - x)
    d = dl + dr
    wl = dr / d

    return wl * y0[idx - 1] + (1 - wl) * y0[idx]
