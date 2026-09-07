"""BEM counterpart to `geometry.HemisphericalTip`: the same tip shape and
kill-mask/interface, but the external field comes from
`bem.fields.HemisphericalTipBEMField`'s boundary-element solve rather than
the closed-form one.

No image-charge force is implemented here yet (bem's recessed-geometry
image-force job hasn't been built) -- `image_force` always returns zero.
That's also exactly the setup this project's BEM roadmap wants for its
first end-to-end check: with image-charge and Coulomb forces both off,
every particle emitted (at rest) from the tip should gain nearly the same
kinetic energy reaching a far screen, since the screen sits close to an
equipotential of the (mostly uniform, far from the tip) asymptotic field --
so the *spread* in that energy is a direct, physical readout of how good
the field solve actually is, complementing the point-by-point accuracy
checks in bem.fields/bem.panel_field.
"""

import numpy as np

from .. import geometry as geometry_module
from .fields import HemisphericalTipBEMField


class HemisphericalTipBEMGeometry(geometry_module.Geometry):
    """A grounded hemispherical tip of radius R on an infinite flat
    cathode, in a uniform asymptotic field Ez, with the external field
    computed via `bem.fields.HemisphericalTipBEMField` instead of the
    closed-form `fields.HemisphericalTipField`.

    Parameters
    ----------
    E_gun : float
        Asymptotic field [V/m], same sign convention as
        `fields.GunField.Ez`.
    R : float
        Tip radius [m].
    n_theta, n_subdiv : optional
        Passed through to `HemisphericalTipBEMField` -- see there for what
        they trade off (profile resolution vs. solve/evaluate cost and
        accuracy).
    kill_z_below : float or None, optional
        Same meaning as `geometry.HemisphericalTip`'s.
    xp : module, optional
        numpy or cupy. The BEM solve always runs on numpy/CPU regardless
        of what's passed here (matching `HemisphericalTipBEMField`), but
        `field`'s evaluation runs on this backend -- the tracker rebuilds
        this Geometry via `worker_args`/`from_worker_args` on whatever
        backend it's actually using, which re-solves the BEM problem from
        scratch (see `worker_args`'s docstring below for the cost this
        implies).
    """

    def __init__(
        self,
        E_gun,
        R,
        n_theta=40,
        n_subdiv=8,
        kill_z_below=0.0,
        xp=np,
    ):
        self.E_gun = float(E_gun)
        self.R = float(R)
        self.n_theta = n_theta
        self.n_subdiv = n_subdiv
        self.kill_z_below = kill_z_below
        self.xp = xp
        self._field = HemisphericalTipBEMField(
            E_gun,
            R,
            n_theta=n_theta,
            n_subdiv=n_subdiv,
            xp=xp,
        )

    def field(self, position):
        return self._field.evaluate(position)

    def image_force(self, position, charge, active, plummer_radius):
        return self.xp.zeros_like(position)

    def kill_mask(self, position):
        # Same convention as geometry.HemisphericalTip.kill_mask.
        xp = self.xp
        r_kill = self.R * (1.0 - 1e-6)
        inside_tip = xp.sum(position * position, axis=-1) <= r_kill * r_kill
        if self.kill_z_below is None:
            return inside_tip
        return inside_tip | (position[..., 2] <= self.kill_z_below)

    def worker_args(self):
        """Note: `SpecificParticleTracer.__init__` always rebuilds its
        Geometry via `Geometry.from_worker_args(geometry.worker_args())`
        regardless of `n_workers` -- for this Geometry that means the BEM
        solve runs at least twice per tracker construction (once for the
        Geometry the caller built, once more inside the tracker), and once
        more per worker process if `n_workers > 1`. Not optimized away
        here, though the axisymmetric solve (bem.axisymmetric) this class
        now uses is cheap enough (direct linear solve over tens of profile
        nodes, not GMRES over thousands of mesh triangles) that this
        mostly doesn't matter in practice.
        """
        return (
            "hemispherical_tip_bem",
            self.E_gun,
            self.R,
            self.n_theta,
            self.n_subdiv,
            self.kill_z_below,
        )

    @classmethod
    def _from_worker_args(cls, args, xp=np):
        E_gun, R, n_theta, n_subdiv, kill_z_below = args
        return cls(
            E_gun,
            R,
            n_theta=n_theta,
            n_subdiv=n_subdiv,
            kill_z_below=kill_z_below,
            xp=xp,
        )


# geometry.Geometry.from_worker_args dispatches on this registry -- there's
# no public registration API (every shape so far has been built into
# geometry.py directly), so a BEM shape living in its own module registers
# itself here on import instead of geometry.py needing to know about bem/.
geometry_module._WORKER_REGISTRY["hemispherical_tip_bem"] = HemisphericalTipBEMGeometry
