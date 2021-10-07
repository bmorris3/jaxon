import os

import numpy as np
from jax import numpy as jnp, jit
from astropy.modeling.models import BlackBody
import astropy.units as u

from exojax.spec import contdb, molinfo, planck
from exojax.spec.rtransfer import nugrid, dtauCIA, rtrun


__all__ = [
    'exojax_spectrum'
]


# The next block of code must occur before relative imports to avoid circular
# import errors
mmw = 2.33  # mean molecular weight
mmrH2 = 0.74

nus_kepler, wav_kepler, res_kepler = nugrid(
    348, 970, 10, "nm", xsmode="modit"
)
nus_wfc3, wav_wfc3, res_wfc3 = nugrid(
    1120, 1650, 10, "nm", xsmode="modit"
)
nus_spitzer, wav_spitzer, res_spitzer = nugrid(
    3000, 5500, 10, "nm", xsmode="modit"
)

nus = jnp.concatenate([nus_spitzer, nus_wfc3, nus_kepler])
wav = jnp.concatenate([wav_kepler, wav_wfc3, wav_spitzer])
nus_vis, wav_vis, res_vis = nugrid(300, 6000, 100, "nm", xsmode="modit")

bb_star = BlackBody(temperature=6300*u.K)

bb_star_transformed = (bb_star(wav*u.nm)).to(
    u.erg/u.s/u.cm**2/u.Hz/u.sr, u.spectral_density(wav*u.nm)
) * np.pi

bb_star_transformed_vis = (bb_star(wav_vis*u.nm)).to(
    u.erg/u.s/u.cm**2/u.Hz/u.sr, u.spectral_density(wav_vis*u.nm)
) * np.pi

from .hatp7 import g
from .tp import get_Tarr
from .continuum import dtauHminusCtm

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
on_github = os.environ.get('GITHUB_ACTIONS', None) == 'True'

if (on_rtd or on_github) or not os.path.exists('/Users/brettmorris/'):
    prefix = ''
    os.makedirs(".database")
else:
    prefix = '/Users/brettmorris/git/exojax/'

cdbH2H2 = contdb.CdbCIA(
    prefix + '.database/H2-H2_2011.cia',
    [nus_vis.min(), nus_vis.max()]
)
cdbH2He = contdb.CdbCIA(
    prefix + '.database/H2-He_2011.cia',
    [nus_vis.min(), nus_vis.max()]
)
# mdbTiO = moldb.MdbExomol(
#     '/Users/brettmorris/git/exojax/.database/TiO/48Ti-16O/Toto/',
#     [nus_kepler.min(), nus_kepler.max()],
#     crit=1e-18
# )

# cnu_TiO, indexnu_TiO, R_TiO, pmarray_TiO = initspec.init_modit(
#     mdbTiO.nu_lines, nus
# )
# cnu_TiO_vis, indexnu_TiO_vis, R_TiO_vis, pmarray_TiO_vis = initspec.init_modit(
#     mdbTiO.nu_lines, nus_vis
# )


Pref = 1  # bar


# def _fT(T0, alpha):
#     return T0[:, None] * (Parr[None, :]/Pref) ** alpha[:, None]


T0_test = np.array([1000.0, 1700.0, 1000.0, 1700.0])
alpha_test = np.array([0.15, 0.15, 0.05, 0.05])
res = 0.2
# dgm_ngammaL_TiO = setdgm_exomol(
#     mdbTiO, fT, Parr, R_TiO, mdbTiO.molmass,
#     res, T0_test, alpha_test
# )


@jit
def exojax_spectrum(
        temperatures, vmr_prod, mmr_TiO, Parr, dParr, nus, wav,
        res):
    Tarr = get_Tarr(temperatures, Parr)

    molmassH2 = molinfo.molmass("H2")
    vmrH2 = (mmrH2 * mmw / molmassH2)  # VMR

    dtaucH2H2 = dtauCIA(
        nus, Tarr, Parr, dParr, vmrH2, vmrH2, mmw,
        g, cdbH2H2.nucia, cdbH2H2.tcia, cdbH2H2.logac
    )

    dtaucHeH2 = dtauCIA(
        nus, Tarr, Parr, dParr, vmrH2, vmrH2, mmw,
        g, cdbH2He.nucia, cdbH2He.tcia, cdbH2He.logac
    )

    dtau_hminus = dtauHminusCtm(
        nus, Tarr, Parr, dParr, vmr_prod, mmw, g
    )

    # SijM_TiO, ngammaLM_TiO, nsigmaDl_CO = exomol(
    #     mdbTiO, Tarr, Parr, res_kepler, mdbTiO.molmass
    # )
    # xsmdit3D = xsmatrix(
    #     cnu_TiO, indexnu_TiO, R_TiO, pmarray_TiO,
    #     nsigmaDl_CO, ngammaLM_TiO, SijM_TiO, nus,
    #     dgm_ngammaL_TiO
    # )
    # dtaum_TiO = dtauM(
    #     dParr, xsmdit3D, mmr_TiO * jnp.ones_like(Parr), mdbTiO.molmass, g
    # )

    dtau = dtau_hminus + dtaucH2H2 + dtaucHeH2  # + dtaum_TiO
    sourcef = planck.piBarr(Tarr, nus)
    F0 = rtrun(dtau, sourcef)

    ccgs = 29979245800.0
    Fcgs = F0 / ccgs  # convert to the unit of erg/cm2/s/Hz
    return Fcgs[::-1], [nus, dtau, Tarr, Parr, dParr], wav
