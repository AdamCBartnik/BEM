"""Cathode shapes: each one bundles the external field, the image-charge
force, and the kill (re-entered-the-conductor) test for a particular
cathode geometry, behind one small interface (`Geometry`).

This is the extension point for new shapes. Two are implemented today:

- `FlatCathode`: the z=0 plane.
- `HemisphericalTip`: an infinite flat cathode with a hemispherical tip.

A future cylindrical tip, cylindrical well, or an arbitrary BEM-meshed
surface just needs a new class implementing the same three methods
(`field`, `image_force`, `kill_mask`) -- nothing in tracker.py,
batched_rk45.py, or parallel.py needs to know or care which shape it's
holding. `SpecificParticleTracer(geometry=..., ...)` takes one of these
directly, and each worker process in parallel.py can rebuild an equivalent
Geometry from its (picklable) constructor arguments, so this is also the
one place backend='cpu' + n_workers>1 needs to know about when a new shape
is added -- see `Geometry.worker_args`/`from_worker_args`.
"""

from abc import ABC, abstractmethod

import numpy as np

from .fields import GunField, HemisphericalTipField
from .forces import (
    coulomb_force,
    image_charge_force,
    hemispherical_tip_image_force,
    FlatCathodeImageSolution,
    HemisphericalTipImageSolution,
)

DEFAULT_Z0 = 3.0e-9  # m (3.0 nm), effective image-charge offset


class Geometry(ABC):
    """Interface a cathode shape must implement.

    Every method works on batched (n_groups, n_emit, ...) arrays, matching
    the shape the rest of the tracker already operates on -- a Geometry is
    never responsible for anything about grouping or timing, only "what is
    the field/image force/conductor boundary at these positions."
    """

    @abstractmethod
    def field(self, position):
        """External electric field [V/m] at `position`, shape (..., 3)."""

    @abstractmethod
    def image_force(self, position, charge, active, plummer_radius):
        """Image-charge force [N], shape (n_groups, n_emit, 3), restricted
        (like Coulomb) to interactions within one group."""

    @abstractmethod
    def kill_mask(self, position):
        """Boolean array, shape position.shape[:-1]: has each particle
        re-entered the conductor?"""

    @abstractmethod
    def worker_args(self):
        """A tuple of plain picklable values sufficient for `from_worker_args`
        to rebuild an equivalent Geometry (with xp=numpy) in a worker
        process. The first element must be a string naming the class, used
        by `from_worker_args` as a dispatch key."""

    @staticmethod
    def from_worker_args(args, xp=np):
        """Rebuild whichever Geometry subclass `worker_args()` came from."""
        kind = args[0]
        if kind not in _WORKER_REGISTRY:
            # BEM geometries (bem/geometry.py) self-register into
            # _WORKER_REGISTRY as an import-time side effect -- reliable
            # whenever *something* in the calling process already imported
            # that module, which is true for an ordinary script or a
            # notebook that constructs one directly. It is NOT reliable
            # inside a multiprocessing worker process, though: `parallel.py`
            # (the module a spawned worker actually reconstructs its state
            # from, since that's where the target function lives) never
            # imports bem/geometry.py itself, and Windows' "spawn" start
            # method does not re-run whatever an *interactive* (Jupyter)
            # session happened to import -- only a plain script run
            # directly gets that "for free", as a side effect of spawn
            # re-executing the launching script's own top-level imports.
            # So a BEM geometry built purely from a notebook fails here
            # with a bare KeyError the first time n_workers>1 is used, even
            # though the exact same code works from a .py script. Import it
            # explicitly (lazily, here, to avoid a circular import at
            # module load time -- bem/geometry.py itself imports this
            # module) so the registry is populated regardless of what the
            # calling context happened to import.
            from .bem import geometry as _bem_geometry  # noqa: F401
        cls = _WORKER_REGISTRY[kind]
        return cls._from_worker_args(args[1:], xp=xp)


class FlatCathode(Geometry):
    """An infinite flat cathode: the z=0 plane, solid conductor for z<=0.

    Parameters
    ----------
    E_gun : float
        z-component of the field [V/m] for z>=0 (see `fields.GunField`).
    z0 : float or None, optional
        Effective image-charge plane offset [m] (see
        `forces.image_charge_force`). Default 3.0 nm; None disables the
        image-charge force entirely.
    kill_z_below : float or None, optional
        A particle falling back to z <= this value is killed. Default 0.0;
        None disables killing.
    xp : module, optional
    """

    def __init__(self, E_gun, z0=DEFAULT_Z0, kill_z_below=0.0, xp=np):
        self.E_gun = float(E_gun)
        self.z0 = z0
        self.kill_z_below = kill_z_below
        self.xp = xp
        self._field = GunField(E_gun, xp=xp)

    @property
    def field_solver(self):
        """The underlying `fields.GunField` -- has its own
        `.potential(position)`, e.g. for
        `specific_particle_tracer.plotting.static_potential_grid`. Same
        property name as `bem.geometry.HemisphericalTipBEMGeometry`'s/
        `CylindricalWellBEMGeometry`'s, so a plotting snippet doesn't
        care which kind of Geometry it's handed."""
        return self._field

    @property
    def image_solution(self):
        """A `forces.FlatCathodeImageSolution` wrapping this geometry's
        own single-image system (the exact closed-form counterpart to
        `bem.image_charge.ImageChargeBEMSolution`), or None if `z0` is
        None (image force disabled) -- same `.image_potential(...)`
        interface as that BEM class, so plotting code doesn't need to
        know which kind of geometry it's looking at. See
        `bem.geometry.HemisphericalTipBEMGeometry.image_solution`."""
        if self.z0 is None:
            return None
        return FlatCathodeImageSolution(self.z0)

    def field(self, position):
        return self._field.evaluate(position)

    def image_force(self, position, charge, active, plummer_radius):
        if self.z0 is None:
            return self.xp.zeros_like(position)
        return image_charge_force(position, charge, active, self.z0, plummer_radius, xp=self.xp)

    def kill_mask(self, position):
        if self.kill_z_below is None:
            return self.xp.zeros(position.shape[:-1], dtype=bool)
        return position[..., 2] <= self.kill_z_below

    def worker_args(self):
        return ("flat", self.E_gun, self.z0, self.kill_z_below)

    @classmethod
    def _from_worker_args(cls, args, xp=np):
        E_gun, z0, kill_z_below = args
        return cls(E_gun, z0=z0, kill_z_below=kill_z_below, xp=xp)


class HemisphericalTip(Geometry):
    """An infinite flat cathode with a grounded hemispherical tip of radius
    R protruding from it at the origin (x=y=z=0).

    Parameters
    ----------
    E_gun : float
        Asymptotic field [V/m] far from the tip (see
        `fields.HemisphericalTipField`).
    R : float
        Tip radius [m].
    z0 : float or None, optional
        Effective image-charge offset [m]: image charges use the exact
        (3-image) solution for a tip of radius R - z0 rather than R (see
        `forces.hemispherical_tip_image_force`). Default 3.0 nm; None
        disables the image-charge force entirely. Must be < R.
    kill_z_below : float or None, optional
        A particle is always killed on falling back inside the tip
        (r <= R); this additionally kills one that falls to
        z <= kill_z_below away from the tip. Default 0.0; None disables
        that additional (flat-region) test.
    xp : module, optional
    """

    def __init__(self, E_gun, R, z0=DEFAULT_Z0, kill_z_below=0.0, xp=np):
        if z0 is not None and R <= z0:
            raise ValueError(
                f"R ({R!r}) must be greater than z0 ({z0!r}) -- a tip no bigger "
                "than the image-charge smoothing offset isn't a case this "
                "approximation is meant to cover."
            )
        self.E_gun = float(E_gun)
        self.R = float(R)
        self.z0 = z0
        self.kill_z_below = kill_z_below
        self.xp = xp
        self._field = HemisphericalTipField(E_gun, R, xp=xp)

    @property
    def field_solver(self):
        """The underlying `fields.HemisphericalTipField` (the closed-form
        analytic solution) -- has its own `.potential(position)`, e.g.
        for `specific_particle_tracer.plotting.static_potential_grid`.
        Same property name as `bem.geometry.HemisphericalTipBEMGeometry`'s/
        `CylindricalWellBEMGeometry`'s, so a plotting snippet doesn't
        care whether it's handed this analytic geometry or a BEM one."""
        return self._field

    @property
    def image_solution(self):
        """A `forces.HemisphericalTipImageSolution` wrapping this
        geometry's own exact 3-image system (the closed-form counterpart
        to `bem.image_charge.ImageChargeBEMSolution`), or None if `z0` is
        None (image force disabled) -- same `.image_potential(...)`
        interface as that BEM class, so plotting code doesn't need to
        know which kind of geometry it's looking at. See
        `bem.geometry.HemisphericalTipBEMGeometry.image_solution`."""
        if self.z0 is None:
            return None
        return HemisphericalTipImageSolution(self.R - self.z0, self.z0, real_radius=self.R)

    def field(self, position):
        return self._field.evaluate(position)

    def image_force(self, position, charge, active, plummer_radius):
        if self.z0 is None:
            return self.xp.zeros_like(position)
        image_radius = self.R - self.z0
        return hemispherical_tip_image_force(
            position, charge, active, image_radius, plummer_radius, plane_z0=self.z0, xp=self.xp
        )

    def kill_mask(self, position):
        xp = self.xp
        # Same tolerance as HemisphericalTipField.field's r >= R*(1-tol):
        # keeps the field-on and kill regions from overlapping right at
        # the numerical boundary (a particle placed via distributions.py's
        # sqrt-based mapping can land a hair below r=R in floating point).
        r_kill = self.R * (1.0 - 1e-6)
        inside_tip = xp.sum(position * position, axis=-1) <= r_kill * r_kill
        if self.kill_z_below is None:
            return inside_tip
        return inside_tip | (position[..., 2] <= self.kill_z_below)

    def worker_args(self):
        return ("hemispherical_tip", self.E_gun, self.R, self.z0, self.kill_z_below)

    @classmethod
    def _from_worker_args(cls, args, xp=np):
        E_gun, R, z0, kill_z_below = args
        return cls(E_gun, R, z0=z0, kill_z_below=kill_z_below, xp=xp)


_WORKER_REGISTRY = {
    "flat": FlatCathode,
    "hemispherical_tip": HemisphericalTip,
}


def make_accel_fn(charge, particle_mass, charge_to_mass, geometry, plummer_radius, xp=np):
    """Build accel(pos, vel, active, group_idx) -> accel, closing over the
    (already backend-resident) per-group `charge`/`particle_mass` arrays it
    should index with `group_idx`, and the given Geometry for field +
    image force.

    `charge_to_mass` (a scalar, species_charge / single-particle mass, e.g.
    -e/m_e for an electron) drives the *external field* term -- a
    macroparticle's acceleration under an external field doesn't depend on
    how many real particles its statistical weight represents, only on the
    species' intrinsic charge-to-mass ratio. `particle_mass` (per-particle,
    scaled the same way `charge`'s weight is: mass_species * weight /
    |species_charge|) is used for the *charge-charge* terms (Coulomb,
    image), where the force genuinely does scale with each macroparticle's
    own charge and so needs dividing by its own (equally scaled) mass to
    get back the right acceleration. Getting this distinction wrong --
    e.g. dividing the field force by the bare single-particle mass while
    using the full macro-charge -- inflates the field acceleration by the
    macroparticle's weighting factor, which is invisible whenever every
    particle happens to carry exactly one elementary charge of weight (as
    in the flat-weight test fixtures used throughout this project) and
    very wrong otherwise.
    """

    def accel(pos, vel, active, group_idx):
        c = charge[group_idx]
        m = particle_mass[group_idx]
        accel_field = charge_to_mass * geometry.field(pos)
        accel_coulomb = coulomb_force(pos, c, active, plummer_radius, xp=xp) / m[..., None]
        accel_image = geometry.image_force(pos, c, active, plummer_radius) / m[..., None]
        acc = accel_field + accel_coulomb + accel_image
        return xp.where(active[..., None], acc, 0.0)

    return accel


def make_kill_fn(geometry, z_max=None):
    """A Geometry's own kill_mask, optionally combined with a z_max cutoff:
    kill a particle once it flies past z_max too, e.g. well beyond the
    farthest screen/output time of interest. Since nothing in this
    tracker's model ever turns the external field off, an escaping
    particle would otherwise just keep coasting (taking ever-larger, but
    still non-free, adaptive steps) all the way to t_max for no purpose --
    z_max lets such particles stop being simulated once they're clearly
    done mattering, which is the same "already known to be irrelevant"
    savings kill_mask itself gives for particles that fall back into the
    conductor. None (default) disables this cutoff.
    """
    if z_max is None:
        return geometry.kill_mask

    def kill_fn(pos):
        return geometry.kill_mask(pos) | (pos[..., 2] >= z_max)

    return kill_fn
