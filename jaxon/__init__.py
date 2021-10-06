# Licensed under a 3-clause BSD style license - see LICENSE.rst

from .version import __version__  # noqa
from jax.config import config
config.update('jax_enable_x64', True)
