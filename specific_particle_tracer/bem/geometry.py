"""BEM counterpart to `geometry.HemisphericalTip`: the same tip shape and
kill-mask/interface, but the external field comes from
`bem.fields.HemisphericalTipBEMField`'s boundary-element solve, and the
image-charge force from `bem.image_charge.ImageChargeBEMSolution` (an
azimuthal-Fourier-mode BEM solve on the recessed image surface, see that
module's docstring) rather than the closed-form `HemisphericalTipField`/
`forces.hemispherical_tip_image_force` this project already has an exact
solution for.

For *this* shape specifically, the closed-form solutions are exact and far
cheaper -- this class exists to validate the general (image-force-capable)
BEM machinery against them, not because this shape needs it. Confirmed
close agreement (a few percent, limited by the recessed profile's rounded
fillet differing from the closed form's sharp-ridge idealization, not by
the BEM solve itself) against `forces.hemispherical_tip_image_force` in
tests/test_bem_image_charge.py. Future, genuinely non-spherical tip shapes
with no closed-form image solution are the actual intended use.

This project's first end-to-end BEM check (before the image-charge force
was built) was: with image-charge and Coulomb forces both off, every
particle emitted (at rest) from the tip should gain nearly the same
kinetic energy reaching a far screen, since the screen sits close to an
equipotential of the (mostly uniform, far from the tip) asymptotic field --
the *spread* in that energy is a direct, physical readout of how good the
field solve is. See examples/bem_hemisphere_tip_example.py for both that
check and an image-force comparison.
"""

import numpy as np

from .. import geometry as geometry_module
from .fields import HemisphericalTipBEMField
from .mesh import hemisphere_tip_image_profile
from .image_charge import ImageChargeBEMSolution


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
    z0 : float or None, optional
        Image-charge offset [m] -- same role as `geometry.HemisphericalTip`'s
        `z0`, but here it also sets the recessed image surface's depth (see
        `bem.mesh.hemisphere_tip_image_profile`). Default *None* (image
        force off, matching this class's original behavior) -- unlike the
        closed-form path, turning this on is a real one-time cost (the
        image-charge BEM solve is O(n_profile^2) per mode-count-independent
        adaptive-quadrature assembly, tens of seconds to a few minutes
        depending on `image_n_theta` etc., not the cheap linear solve the
        static-field path uses), and `worker_args`/`from_worker_args`
        re-triggers it (see `worker_args`'s docstring) -- so it's opt-in
        rather than defaulting to on the way `HemisphericalTip.z0` does.
    image_n_theta, image_n_fillet, image_n_r : optional
        Recessed-image-profile resolution, passed to
        `bem.mesh.hemisphere_tip_image_profile`. Only matters if `z0` is
        not None. Standoffs smaller than the resulting profile's segment
        size lose accuracy (checked directly: the residual mirror-charge
        vs. bare-mode-expansion agreement only converges once mesh
        resolution comfortably resolves the standoff being evaluated, see
        tests/test_bem_image_charge.py) -- the defaults were sized to
        resolve standoffs down to `z0` itself, since closer than that the
        image force is dominated by the same z0 regularization every other
        image-charge path in this project already uses.
    image_n_max : int, optional
        Azimuthal Fourier modes kept for the image-charge solve (see
        `bem.image_charge`). Measured directly on this exact geometry
        (tests/test_bem_image_charge.py, and this class's own module
        history): with the mirror-charge blend below, n_max=16 keeps error
        under 1% (relative to n_max=40) down to a standoff of 0.06*R
        (comparable to the default `z0`); fewer modes are needed farther
        out. Since assembling the per-mode operators costs about the same
        regardless of n_max (the adaptive-quadrature recursion structure,
        not the mode count, dominates that cost), there's little reason to
        set this lower than the default unless evaluation-time cost (which
        *does* scale with n_max) matters more than solve-time cost.
    image_d_lo, image_d_hi : float, optional
        Mirror-charge blend distances (see `bem.image_charge`'s module
        docstring) -- fractions of R by default. Only change how fast the
        solve converges with `image_n_max`, never the converged answer.
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
        z0=None,
        image_n_theta=30,
        image_n_fillet=12,
        image_n_r=15,
        image_n_max=16,
        image_d_lo=0.1,
        image_d_hi=0.5,
        kill_z_below=0.0,
        xp=np,
    ):
        self.E_gun = float(E_gun)
        self.R = float(R)
        self.n_theta = n_theta
        self.n_subdiv = n_subdiv
        self.z0 = z0
        self.image_n_theta = image_n_theta
        self.image_n_fillet = image_n_fillet
        self.image_n_r = image_n_r
        self.image_n_max = image_n_max
        self.image_d_lo = image_d_lo
        self.image_d_hi = image_d_hi
        self.kill_z_below = kill_z_below
        self.xp = xp
        self._field = HemisphericalTipBEMField(
            E_gun,
            R,
            n_theta=n_theta,
            n_subdiv=n_subdiv,
            xp=xp,
        )
        self._image_solution = None
        if z0 is not None:
            image_profile = hemisphere_tip_image_profile(
                R,
                z0,
                plane_radius=5.0 * R,
                n_theta=image_n_theta,
                n_fillet=image_n_fillet,
                n_r=image_n_r,
            )
            self._image_solution = ImageChargeBEMSolution.solve(image_profile, n_max=image_n_max)

    def field(self, position):
        return self._field.evaluate(position)

    def image_force(self, position, charge, active, plummer_radius):
        if self._image_solution is None:
            return self.xp.zeros_like(position)
        return self._image_solution.image_force(
            position,
            charge,
            active,
            d_lo=self.image_d_lo * self.R,
            d_hi=self.image_d_hi * self.R,
            n_max=self.image_n_max,
            xp=self.xp,
        )

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
        solve(s) run at least twice per tracker construction (once for the
        Geometry the caller built, once more inside the tracker), and once
        more per worker process if `n_workers > 1`. Not optimized away
        here. The static-field axisymmetric solve is cheap enough (direct
        linear solve over tens of profile nodes) that this mostly doesn't
        matter; the image-charge solve (only run if `z0` is not None) is
        not cheap the same way (see its own docstring above) -- so leaving
        `z0=None` unless the image force is actually needed matters more
        here than it does for the static-field-only case.
        """
        return (
            "hemispherical_tip_bem",
            self.E_gun,
            self.R,
            self.n_theta,
            self.n_subdiv,
            self.z0,
            self.image_n_theta,
            self.image_n_fillet,
            self.image_n_r,
            self.image_n_max,
            self.image_d_lo,
            self.image_d_hi,
            self.kill_z_below,
        )

    @classmethod
    def _from_worker_args(cls, args, xp=np):
        (
            E_gun,
            R,
            n_theta,
            n_subdiv,
            z0,
            image_n_theta,
            image_n_fillet,
            image_n_r,
            image_n_max,
            image_d_lo,
            image_d_hi,
            kill_z_below,
        ) = args
        return cls(
            E_gun,
            R,
            n_theta=n_theta,
            n_subdiv=n_subdiv,
            z0=z0,
            image_n_theta=image_n_theta,
            image_n_fillet=image_n_fillet,
            image_n_r=image_n_r,
            image_n_max=image_n_max,
            image_d_lo=image_d_lo,
            image_d_hi=image_d_hi,
            kill_z_below=kill_z_below,
            xp=xp,
        )


# geometry.Geometry.from_worker_args dispatches on this registry -- there's
# no public registration API (every shape so far has been built into
# geometry.py directly), so a BEM shape living in its own module registers
# itself here on import instead of geometry.py needing to know about bem/.
geometry_module._WORKER_REGISTRY["hemispherical_tip_bem"] = HemisphericalTipBEMGeometry
