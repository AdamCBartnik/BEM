"""Physical constants and unit conversions.

Internally, the tracker works entirely in SI units (meters, seconds,
kilograms, Coulombs, Newtons). ParticleGroup (openPMD-beamphysics) stores
positions in [m], momenta in [eV/c], time in [s], and macro-charge weight
in [C]. The helpers below convert between the two.
"""

from scipy import constants as _c

ELEMENTARY_CHARGE = _c.e  # C
SPEED_OF_LIGHT = _c.c  # m/s
VACUUM_PERMITTIVITY = _c.epsilon_0  # F/m
COULOMB_CONSTANT = 1.0 / (4.0 * _c.pi * VACUUM_PERMITTIVITY)  # N m^2 / C^2


def ev_c_to_si_momentum(p_ev_c):
    """Convert momentum from [eV/c] (openPMD-beamphysics convention) to SI [kg m/s]."""
    return p_ev_c * ELEMENTARY_CHARGE / SPEED_OF_LIGHT


def si_momentum_to_ev_c(p_si):
    """Convert momentum from SI [kg m/s] to [eV/c]."""
    return p_si * SPEED_OF_LIGHT / ELEMENTARY_CHARGE


def ev_to_kg(mass_ev):
    """Convert a rest energy in [eV] to a rest mass in [kg]."""
    return mass_ev * ELEMENTARY_CHARGE / SPEED_OF_LIGHT**2
