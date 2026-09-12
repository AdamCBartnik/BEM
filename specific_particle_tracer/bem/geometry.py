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
from .fields import HemisphericalTipBEMField, CylindricalWellBEMField, _cylindrical_well_contains
from .mesh import (
    hemisphere_tip_image_profile,
    hemisphere_tip_image_doubled_profile,
    hemisphere_tip_real_profile,
    cylindrical_well_image_profile,
    cylindrical_well_real_profile,
)
from .image_charge import ImageChargeBEMSolution, image_charge_weight
from ..constants import COULOMB_CONSTANT


def _blend_with_flat_fallback(near, far, z, z_lo, z_hi, xp):
    """Blend a near-field BEM quantity with its far-field flat-cathode
    equivalent, weighted by height z alone: `near` for z <= z_lo, `far`
    for z >= z_hi, a cubic smoothstep in between (see
    `bem.image_charge.image_charge_weight`, reused here for its shape,
    not its original near-surface-distance meaning).

    `near`/`far`/`z` broadcast against each other; `near`/`far` are
    typically (..., 3) fields or forces, `z` the matching (...) height.
    """
    w = image_charge_weight(z, z_lo, z_hi, xp=xp)
    return w[..., None] * near + (1.0 - w[..., None]) * far


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
    field_max_length, n_subdiv : optional
        Passed through to `HemisphericalTipBEMField` -- see there for what
        they trade off (profile resolution vs. solve/evaluate cost and
        accuracy). `field_max_length` defaults to that class's own
        default (`R*pi/39`, matching this class's old `n_theta=40`).
    z0 : float or None, optional
        Image-charge offset [m] -- same role as `geometry.HemisphericalTip`'s
        `z0`, but here it also sets the recessed image surface's depth (see
        `bem.mesh.hemisphere_tip_image_profile`). Default *None* (image
        force off, matching this class's original behavior) -- unlike the
        closed-form path, turning this on is a real one-time cost (the
        image-charge BEM solve is O(n_profile^2) per mode-count-independent
        adaptive-quadrature assembly, tens of seconds to a few minutes
        depending on `image_max_length` etc., not the cheap linear solve the
        static-field path uses), and `worker_args`/`from_worker_args`
        re-triggers it (see `worker_args`'s docstring) -- so it's opt-in
        rather than defaulting to on the way `HemisphericalTip.z0` does.
    image_max_length : float, optional
        Global cap [m] on every segment's length in the recessed image
        profile (and, reused, the real profile built alongside it purely
        for `image_potential`'s masking) -- see
        `bem.mesh.hemisphere_tip_image_profile`. Only matters if `z0` is
        not None. Default `(R - z0) * pi / 2 / 29` (matching this class's
        old `image_n_theta=30` cap resolution). Safe to apply uniformly
        out to the plane's far edge (5*R) despite that being a much
        larger scale than `R`/`z0`, since the plane region is *graded*
        (see `bem.mesh.graded_annulus_profile`): it only resolves this
        finely right at its inner edge, next to the cap, and coarsens
        geometrically from there. Standoffs smaller than the resulting
        profile's segment size lose accuracy (checked directly: the
        residual mirror-charge vs. bare-mode-expansion agreement only
        converges once mesh resolution comfortably resolves the standoff
        being evaluated, see tests/test_bem_image_charge.py) -- pass a
        smaller value if standoffs finer than `z0` itself matter, since
        closer than that the image force is dominated by the same z0
        regularization every other image-charge path in this project
        already uses.
    image_cap_max_length, image_plane_max_length : float, optional
        Per-region overrides for the sphere cap / plane, combined with
        `image_max_length` via `min(...)` -- see
        `bem.mesh._region_max_length`. No special default for either (the
        cap's default resolution already comes from `image_max_length`
        itself, and the plane's grading means it needs no separate,
        coarser default the way it would without grading).
    image_fillet_max_length : float, optional
        Per-region override for the ridge fillet, combined with
        `image_max_length` via `min(...)`. No special default needed:
        `bem.mesh`'s own `_MIN_FILLET_SEGMENTS` floor already guarantees
        the fillet is never resolved more coarsely than 5 segments over
        its 90 degree sweep, however `image_max_length` compares to the
        fillet's own (usually much smaller) radius z0 -- a pure
        length-based cap alone would otherwise happily accept 1-2
        segments there whenever z0 is small relative to the rest of the
        geometry's scale.
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
    image_mirror_symmetric : bool, optional
        Default False (the truncated-plane image profile above). If
        True, builds the image-charge solve on
        `bem.mesh.hemisphere_tip_image_doubled_profile` instead --
        mirror-doubled through z=-z0, no plane meshed or truncated at
        all (the same trick `HemisphericalTipBEMField` already uses for
        the *real* surface's own mirror plane at z=0, one level down).
        `image_plane_max_length` is unused in this mode (there's no
        plane region). `image_force` compensates via
        `_mirror_symmetric_image_force` (see that method's own docstring
        for the mechanics and for a real bug caught building it: the
        phantom mirror charge's *direct* Coulomb pull on the real
        particle is a genuine classical method-of-images term, not
        something the joint solve's own conductor-mediated response
        includes on its own). `image_solution.image_field`/
        `.image_potential` also handle this mode correctly, via their own
        `mirror_plane_z` parameter -- set automatically on
        `self._image_solution` here, so calling either directly on
        `image_solution` (e.g. from `specific_particle_tracer.plotting.
        image_potential_grid`) already does the right thing without the
        caller needing to know about any of this.
    z_handoff : float or None, optional
        Height [m] above which `field`/`image_force` blend over to a
        plain `geometry.FlatCathode` (same `z0`) instead of paying for a
        BEM evaluation -- by z_handoff the tip's perturbation has decayed
        to where the flat-plane answer is already an excellent
        approximation, so there's little accuracy to trade away for the
        speedup. A cubic smoothstep (see `bem.image_charge.
        image_charge_weight`) blends from pure BEM at 0.5*z_handoff to
        pure flat-cathode at z_handoff, rather than a hard cutoff, so a
        stepper never sees a force discontinuity. Default None -> 20*R.
    kill_z_below : float or None, optional
        Same meaning as `geometry.HemisphericalTip`'s.
    xp : module, optional
        numpy or cupy. Both BEM solves (static-field and, if `z0` is not
        None, image-charge) always run their one-time operator assembly
        on numpy/CPU regardless of what's passed here (matching
        `HemisphericalTipBEMField`) -- but `field`'s and `image_force`'s
        per-call evaluation both run on this backend, including
        `image_force`'s own per-mode linear solve (see
        `bem.image_charge.ImageChargeBEMSolution.image_force`'s docstring
        for the GPU-specific machinery -- custom RawKernels for its two
        Python-loop hot spots -- this enables). The tracker rebuilds this
        Geometry via `worker_args`/`from_worker_args` on whatever backend
        it's actually using, which re-solves the BEM problem(s) from
        scratch (see `worker_args`'s docstring below for the cost this
        implies).
    """

    def __init__(
        self,
        E_gun,
        R,
        field_max_length=None,
        n_subdiv=8,
        z0=None,
        image_max_length=None,
        image_cap_max_length=None,
        image_fillet_max_length=None,
        image_plane_max_length=None,
        image_n_max=16,
        image_d_lo=0.1,
        image_d_hi=0.5,
        image_mirror_symmetric=False,
        z_handoff=None,
        kill_z_below=0.0,
        xp=np,
    ):
        self.E_gun = float(E_gun)
        self.R = float(R)
        self.field_max_length = field_max_length
        self.n_subdiv = n_subdiv
        self.z0 = z0
        self.image_mirror_symmetric = image_mirror_symmetric
        # image_max_length matches the old cap resolution -- safe to also
        # apply out to the plane's far edge (5*R) because the plane region
        # is graded (bem.mesh.graded_annulus_profile), not uniform: it
        # only needs to start this fine, right where it meets the cap, and
        # is free to coarsen from there.
        self.image_max_length = (
            ((self.R - float(z0)) * np.pi / 2.0 / 29.0 if z0 is not None else None)
            if image_max_length is None
            else float(image_max_length)
        )
        self.image_cap_max_length = image_cap_max_length
        self.image_plane_max_length = image_plane_max_length
        # No special default for the fillet: bem.mesh's own
        # _MIN_FILLET_SEGMENTS floor already guarantees it's never
        # resolved more coarsely than 5 segments over its 90 degree
        # sweep, however image_max_length compares to z0.
        self.image_fillet_max_length = image_fillet_max_length
        self.image_n_max = image_n_max
        self.image_d_lo = image_d_lo
        self.image_d_hi = image_d_hi
        self.z_handoff = 20.0 * self.R if z_handoff is None else float(z_handoff)
        self.kill_z_below = kill_z_below
        self.xp = xp
        self._field = HemisphericalTipBEMField(
            E_gun,
            R,
            max_length=field_max_length,
            n_subdiv=n_subdiv,
            xp=xp,
        )
        self._flat_fallback = geometry_module.FlatCathode(E_gun, z0=z0, xp=xp)
        self._image_solution = None
        if z0 is not None:
            if image_mirror_symmetric:
                image_profile = hemisphere_tip_image_doubled_profile(
                    R,
                    z0,
                    max_length=self.image_max_length,
                    cap_max_length=self.image_cap_max_length,
                    fillet_max_length=self.image_fillet_max_length,
                )
            else:
                image_profile = hemisphere_tip_image_profile(
                    R,
                    z0,
                    plane_radius=5.0 * R,
                    max_length=self.image_max_length,
                    cap_max_length=self.image_cap_max_length,
                    fillet_max_length=self.image_fillet_max_length,
                    plane_max_length=self.image_plane_max_length,
                )
            # The real (unrecessed) surface, for image_potential's masking
            # only (see ImageChargeBEMSolution's docstring) -- not part of
            # the physics, so its own resolution doesn't need to match the
            # image profile's. No fillet region here, so no
            # image_fillet_max_length to pass through.
            real_profile = hemisphere_tip_real_profile(
                R,
                plane_radius=5.0 * R,
                max_length=self.image_max_length,
                cap_max_length=self.image_cap_max_length,
                plane_max_length=self.image_plane_max_length,
            )
            self._image_solution = ImageChargeBEMSolution.solve(
                image_profile, n_max=image_n_max, real_profile=real_profile
            )
            if image_mirror_symmetric:
                # Lets image_potential (and plotting.image_potential_grid,
                # which reads this automatically) apply the same phantom-
                # mirror-source correction _mirror_symmetric_image_force
                # applies for image_force -- see ImageChargeBEMSolution's
                # own docstring for why this isn't optional in this mode.
                self._image_solution.mirror_plane_z = -float(z0)

    @property
    def field_solver(self):
        """The underlying BEM field-solver object (e.g.
        `bem.fields.HemisphericalTipBEMField`/`CylindricalWellBEMField`)
        -- for `specific_particle_tracer.plotting.static_potential_grid`,
        which needs an object with its own `.potential(position)`."""
        return self._field

    @property
    def real_profile(self):
        """The (rho, z) generating profile of the real conductor surface
        -- the actual mesh nodes the field solve uses -- e.g. for
        `specific_particle_tracer.plotting.plot_profiles`."""
        return self._field.solution.profile

    @property
    def image_solution(self):
        """The underlying `bem.image_charge.ImageChargeBEMSolution`, or
        None if this geometry was built with `z0=None` (image force
        disabled). Has its own `.profile` (the recessed image surface)
        and `.image_potential(...)` -- e.g. for
        `specific_particle_tracer.plotting.plot_profiles`/
        `image_potential_grid`. If this geometry was built with
        `image_mirror_symmetric=True`, this solution's own
        `.image_field`/`.image_potential` already account for the
        mirror-image excitation too, via `.mirror_plane_z` (set here,
        read automatically by both methods) -- see this class's own
        docstring."""
        return self._image_solution

    def field(self, position):
        xp = self.xp
        near = self._field.evaluate(position)
        far = self._flat_fallback.field(position)
        z = position[..., 2]
        return _blend_with_flat_fallback(near, far, z, 0.5 * self.z_handoff, self.z_handoff, xp)

    def image_force(self, position, charge, active, plummer_radius):
        xp = self.xp
        far = self._flat_fallback.image_force(position, charge, active, plummer_radius)
        if self._image_solution is None:
            return far
        if self.image_mirror_symmetric:
            near = self._mirror_symmetric_image_force(position, charge, active, plummer_radius, _plane_force=far)
        else:
            near = self._image_solution.image_force(
                position,
                charge,
                active,
                d_lo=self.image_d_lo * self.R,
                d_hi=self.image_d_hi * self.R,
                n_max=self.image_n_max,
                xp=self.xp,
            )
        z = position[..., 2]
        return _blend_with_flat_fallback(near, far, z, 0.5 * self.z_handoff, self.z_handoff, xp)

    def _mirror_symmetric_image_force(self, position, charge, active, plummer_radius, _plane_force=None):
        """`image_force`'s `image_mirror_symmetric=True` path: the image
        solve was built on `bem.mesh.hemisphere_tip_image_doubled_profile`
        -- a closed surface with no plane meshed at all, valid only
        together with the classical method-of-images excitation for a
        "grounded conductor on a grounded plane" problem: the real charge
        *and* its negative mirror image through z=-z0, together.

        Rather than teach `bem.image_charge.ImageChargeBEMSolution` a new
        excitation shape, this reuses its existing joint multi-source
        solve (built for conductor-mediated cross-coupling between real
        group members) unchanged: append one phantom "particle" per real
        one -- same group, mirrored position, negated charge -- so the
        joint solve's combined right-hand side already *is* the source +
        mirror-image pair the doubled surface needs. The phantoms' own
        "forces" are meaningless, so `n_eval` keeps the reconstruction
        step from ever building them (see `image_force`'s own docs for
        that parameter). This part is exact (a
        genuine method-of-images construction, not an approximation on
        top of the mode truncation) precisely because the existing joint
        solve's own cross-coupling is already exact -- see
        bem.image_charge's module docstring's "Exactness" section for why
        *that* holds regardless of the mirror-charge blend parameters, an
        argument that applies here unchanged since a phantom source is
        just another source to that machinery, nothing about it assumes
        the sources given are all "real" particles.

        But `image_force`'s joint solve deliberately excludes *direct*
        Coulomb between the sources it's given (see its own docstring --
        that's `forces.coulomb_force`'s job for real particle pairs, kept
        separate on purpose). The phantom mirror charge's direct pull on
        the real particle is *not* optional here, though: it's exactly
        the classical "plane image" term in the textbook 3-image
        construction for a sphere-on-a-plane (`forces.
        hemispherical_tip_image_force`'s own `f_plane` term) -- so it has
        to be added back by hand. Confirmed directly this is the one
        missing piece: without it, this method's answer was a real,
        converged (not mesh- or mode-count-related) ~1-2% off from the
        exact analytic solution even on a bare sphere with no fillet
        involved at all; adding it brought that down to ~1e-4 (pure mode-
        truncation-level agreement).

        Costs roughly a doubled per-group source count in the joint
        solve (linear in the per-mode linear-solve step, quadratic in the
        direct mirror-charge cross-term) -- traded against the plane
        region's node count and truncation-radius error this mode
        removes entirely; see this class's own docstring for the
        `image_mirror_symmetric` option this implements.
        """
        xp = self.xp
        n_emit = position.shape[1]
        mirror_position = xp.array(position, copy=True)
        mirror_position[..., 2] = -2.0 * self.z0 - position[..., 2]
        extended_position = xp.concatenate([position, mirror_position], axis=1)
        extended_charge = xp.concatenate([charge, -charge], axis=1)
        extended_active = xp.concatenate([active, active], axis=1)

        conductor_mediated = self._image_solution.image_force(
            extended_position,
            extended_charge,
            extended_active,
            d_lo=self.image_d_lo * self.R,
            d_hi=self.image_d_hi * self.R,
            n_max=self.image_n_max,
            n_eval=n_emit,  # the phantoms' own "forces" are meaningless; don't pay to build them
            xp=xp,
        )

        # Direct Coulomb pull of the phantom mirror charge (-charge, at
        # mirror_position) on the real particle, Plummer-softened like
        # every other direct Coulomb-type term in this project. Geometry
        # alone keeps this well away from any singularity for a real,
        # physically valid particle: separation = 2*(z + z0), and z >= 0
        # for any particle actually outside the real conductor, so
        # separation >= 2*z0 always (the softening is just this
        # project's usual safety margin, not load-bearing here).
        # Include every active phantom in the group, not only i's own.
        direct_phantom = _plane_force
        if direct_phantom is None:
            direct_phantom = self._flat_fallback.image_force(position, charge, active, plummer_radius)

        return conductor_mediated + direct_phantom

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
            self.field_max_length,
            self.n_subdiv,
            self.z0,
            self.image_max_length,
            self.image_cap_max_length,
            self.image_fillet_max_length,
            self.image_plane_max_length,
            self.image_n_max,
            self.image_d_lo,
            self.image_d_hi,
            self.image_mirror_symmetric,
            self.z_handoff,
            self.kill_z_below,
        )

    @classmethod
    def _from_worker_args(cls, args, xp=np):
        (
            E_gun,
            R,
            field_max_length,
            n_subdiv,
            z0,
            image_max_length,
            image_cap_max_length,
            image_fillet_max_length,
            image_plane_max_length,
            image_n_max,
            image_d_lo,
            image_d_hi,
            image_mirror_symmetric,
            z_handoff,
            kill_z_below,
        ) = args
        return cls(
            E_gun,
            R,
            field_max_length=field_max_length,
            n_subdiv=n_subdiv,
            z0=z0,
            image_max_length=image_max_length,
            image_cap_max_length=image_cap_max_length,
            image_fillet_max_length=image_fillet_max_length,
            image_plane_max_length=image_plane_max_length,
            image_n_max=image_n_max,
            image_d_lo=image_d_lo,
            image_d_hi=image_d_hi,
            image_mirror_symmetric=image_mirror_symmetric,
            z_handoff=z_handoff,
            kill_z_below=kill_z_below,
            xp=xp,
        )


class CylindricalWellBEMGeometry(geometry_module.Geometry):
    """A grounded cylindrical well of radius R, depth H, cut into an
    infinite flat cathode, in a uniform asymptotic field Ez -- the
    external field from `bem.fields.CylindricalWellBEMField`, the image-
    charge force from `bem.image_charge.ImageChargeBEMSolution` on the
    recessed image surface `bem.mesh.cylindrical_well_image_profile`.
    Particles are emitted only from the well bottom (z=-H, rho<R).

    Unlike `HemisphericalTipBEMGeometry`, this shape has no closed-form
    solution to validate against -- this project's first BEM geometry
    with no analytic ground truth at all. See tests/test_bem_well.py for
    the validation strategy used instead: internal convergence (mesh
    resolution, `field_plane_radius`/`image_plane_radius` truncation,
    mode count) and the same uniform-emitted-energy screen check used for
    the hemisphere tip. The nearest-point mirror-charge term (see
    `bem.image_charge`) needs no special-casing for "which conductor
    piece is closest" -- a particle deep in the well finds its nearest
    point on the bottom or wall, one far above the plane finds it on the
    outer annulus, and the blend between those regimes falls out of the
    same generic closest-point search the hemisphere-tip geometry
    already uses.

    Parameters
    ----------
    E_gun : float
        Asymptotic field [V/m], same convention as `fields.GunField.Ez`.
    R, H : float
        Well radius and depth [m].
    field_plane_radius, field_max_length, n_subdiv : optional
        Passed through to `bem.fields.CylindricalWellBEMField` for the
        real-surface field solve. `field_plane_radius` defaults to
        `z_handoff` (no reason to mesh the plane farther out than the
        distance beyond which the flat-cathode fallback takes over
        anyway). `field_max_length` defaults to that class's own default
        (sized off `field_plane_radius`, with its own finer bottom/wall
        defaults -- see `bem.fields.CylindricalWellBEMField`).
    bottom_fillet_radius, rim_fillet_radius : float or None, optional
        Rounding radii [m] for the *real* bottom-of-well/rim corners --
        see `bem.mesh.cylindrical_well_real_profile`'s docstring for why
        a real fabricated well may not have perfectly sharp corners, and
        `bem.mesh.cylindrical_well_image_profile`'s for how the image
        profile derives its own (z0-eroded) fillet radii from these same
        real ones, rather than taking an independent image-side radius.
        Used for the real geometry everywhere it appears: the static-
        field solve's own mesh, and the real surface `image_potential`
        masks against.

        Default (only when z0 is not None -- otherwise both stay sharp,
        None): `bottom_fillet_radius` rounds a bit
        (`min(0.5*z0, 0.2*(H-z0))`) and `rim_fillet_radius` rounds more
        (`z0 + min(0.5*z0, 0.2*(H-z0))`, always > z0), scaled down
        together for a shallow well so neither default ever risks
        violating `bottom_fillet_radius + rim_fillet_radius < H`. The
        rim needing to end up bigger than z0 isn't an arbitrary
        asymmetry: the reflex bottom corner always survives z0 erosion
        as a real rounding (radius `bottom_fillet_radius + z0`, always
        positive) however small `bottom_fillet_radius` is, but the
        convex rim corner's eroded radius is `rim_fillet_radius - z0`,
        which needs `rim_fillet_radius > z0` just to stay a real
        rounding at all -- otherwise the image profile's rim reverts to
        the same sharp mitre an *unrounded* real corner would erode to,
        bringing back the closest-point-search discontinuity this
        default is specifically sized to avoid (see
        `cylindrical_well_image_profile`'s docstring for that failure
        mode).
    z0 : float or None, optional
        Image-charge offset [m], same role as elsewhere in this project.
        Default None (image force off).
    image_plane_radius : float, optional
        Defaults to 5*max(R, H), matching
        `HemisphericalTipBEMGeometry`'s image profile's 5*R.
    image_max_length : float, optional
        Global cap [m] on every segment's length in the recessed image
        profile (and, reused, the real profile built alongside it purely
        for `image_potential`'s masking) -- see
        `bem.mesh.cylindrical_well_image_profile`. Only matters if `z0`
        is not None. Default `min(R, H) / 14` (approximately matching
        this class's old `image_n_bottom=image_n_wall=image_n_r=15`, all
        historically equal). Safe to apply uniformly out to the plane's
        far edge (`image_plane_radius`) despite that being a much larger
        scale than `R`/`H`, since the plane region is *graded* (see
        `bem.mesh.graded_annulus_profile`): it only resolves this finely
        right at its inner edge, next to the well, and coarsens
        geometrically from there.
    image_bottom_max_length, image_wall_max_length,
    image_plane_max_length : float, optional
        Per-region overrides for the bottom disk / wall / outer plane,
        combined with `image_max_length` via `min(...)` (an override only
        refines, never coarsens -- see `bem.mesh._region_max_length`).
    image_fillet_max_length, image_rim_fillet_max_length : float, optional
        Per-region overrides for the bottom-of-well/rim corner fillets
        (both real and image use the same names, since they're the same
        corners just with different eroded radii), combined with
        `image_max_length` via `min(...)`. No special defaults needed:
        `bem.mesh`'s own `_MIN_FILLET_SEGMENTS` floor already guarantees
        at least 5 segments over each fillet's 90 degree sweep,
        regardless of how `image_max_length` compares to the fillet's
        own (usually much smaller) radius.
    image_n_max : int, optional
        Azimuthal Fourier modes kept for the image-charge solve -- see
        `bem.image_charge`.
    image_d_lo, image_d_hi : float, optional
        Mirror-charge blend distances -- fractions of R.
    z_handoff : float or None, optional
        Same role as `HemisphericalTipBEMGeometry.z_handoff`: height
        above which `field`/`image_force` blend over to a plain
        `geometry.FlatCathode` instead of paying for a BEM evaluation.
        Default None -> 20*max(R, H).
    xp : module, optional
        See `HemisphericalTipBEMGeometry.xp`'s docstring -- same
        division of labor between the one-time (always CPU) solve(s)
        and the per-call (this backend) evaluation.
    """

    def __init__(
        self,
        E_gun,
        R,
        H,
        field_plane_radius=None,
        field_max_length=None,
        n_subdiv=8,
        bottom_fillet_radius=None,
        rim_fillet_radius=None,
        z0=None,
        image_plane_radius=None,
        image_max_length=None,
        image_bottom_max_length=None,
        image_fillet_max_length=None,
        image_wall_max_length=None,
        image_rim_fillet_max_length=None,
        image_plane_max_length=None,
        image_n_max=16,
        image_d_lo=0.1,
        image_d_hi=0.5,
        z_handoff=None,
        xp=np,
    ):
        self.E_gun = float(E_gun)
        self.R = float(R)
        self.H = float(H)
        self.field_max_length = field_max_length
        self.n_subdiv = n_subdiv
        self.z0 = z0
        self.image_n_max = image_n_max
        self.image_d_lo = image_d_lo
        self.image_d_hi = image_d_hi
        self.z_handoff = 20.0 * max(self.R, self.H) if z_handoff is None else float(z_handoff)
        self.field_plane_radius = self.z_handoff if field_plane_radius is None else float(field_plane_radius)
        self.image_plane_radius = 5.0 * max(self.R, self.H) if image_plane_radius is None else float(image_plane_radius)

        # Real-geometry corner rounding (see cylindrical_well_real_profile's
        # docstring). Defaults round both corners a bit (only when z0 is
        # given -- otherwise there's no erosion motivating a default, so
        # both stay sharp/None), scaled down together for a shallow well.
        # rim_fillet_radius's default in particular is built to stay
        # > z0 -- see this class's own docstring for why that's not
        # optional: unlike the bottom corner (always rounds under
        # erosion), the rim corner's rounding doesn't survive erosion at
        # all unless it started out bigger than z0.
        if z0 is not None:
            available = self.H - float(z0)
            self.bottom_fillet_radius = (
                min(0.5 * float(z0), 0.2 * available) if bottom_fillet_radius is None else float(bottom_fillet_radius)
            )
            self.rim_fillet_radius = (
                float(z0) + min(0.5 * float(z0), 0.2 * available)
                if rim_fillet_radius is None
                else float(rim_fillet_radius)
            )
        else:
            self.bottom_fillet_radius = None if bottom_fillet_radius is None else float(bottom_fillet_radius)
            self.rim_fillet_radius = None if rim_fillet_radius is None else float(rim_fillet_radius)

        # image_max_length matches the old bottom/wall/plane resolution
        # (all equal historically) -- safe to also apply out to the
        # plane's far edge (image_plane_radius) because the plane region
        # is graded (bem.mesh.graded_annulus_profile), not uniform: it
        # only needs to start this fine, right where it meets the well,
        # and is free to coarsen from there.
        self.image_max_length = (
            min(self.R, self.H) / 14.0 if image_max_length is None else float(image_max_length)
        )
        self.image_bottom_max_length = image_bottom_max_length
        self.image_wall_max_length = image_wall_max_length
        self.image_plane_max_length = image_plane_max_length
        # No special defaults for either fillet's resolution: bem.mesh's
        # own _MIN_FILLET_SEGMENTS floor already guarantees at least 5
        # segments over each fillet's 90 degree sweep, however
        # image_max_length compares to the fillet's own radius.
        self.image_fillet_max_length = image_fillet_max_length
        self.image_rim_fillet_max_length = image_rim_fillet_max_length
        self.xp = xp
        self._field = CylindricalWellBEMField(
            E_gun,
            R,
            H,
            plane_radius=self.field_plane_radius,
            max_length=field_max_length,
            bottom_fillet_radius=self.bottom_fillet_radius,
            rim_fillet_radius=self.rim_fillet_radius,
            n_subdiv=n_subdiv,
            xp=xp,
        )
        self._flat_fallback = geometry_module.FlatCathode(E_gun, z0=z0, xp=xp)
        self._image_solution = None
        if z0 is not None:
            image_profile = cylindrical_well_image_profile(
                R,
                H,
                z0,
                self.image_plane_radius,
                self.image_max_length,
                bottom_fillet_radius=self.bottom_fillet_radius,
                rim_fillet_radius=self.rim_fillet_radius,
                bottom_max_length=self.image_bottom_max_length,
                bottom_fillet_max_length=self.image_fillet_max_length,
                wall_max_length=self.image_wall_max_length,
                rim_fillet_max_length=self.image_rim_fillet_max_length,
                plane_max_length=self.image_plane_max_length,
            )
            # The real (unrecessed) surface, for image_potential's masking
            # only (see ImageChargeBEMSolution's docstring) -- not part of
            # the physics, so its own resolution doesn't need to match the
            # image profile's, but its SHAPE (including any corner
            # rounding) does, so masking is against the same real geometry
            # everything else here uses.
            real_profile = cylindrical_well_real_profile(
                R,
                H,
                self.image_plane_radius,
                self.image_max_length,
                bottom_fillet_radius=self.bottom_fillet_radius,
                rim_fillet_radius=self.rim_fillet_radius,
                bottom_max_length=self.image_bottom_max_length,
                bottom_fillet_max_length=self.image_fillet_max_length,
                wall_max_length=self.image_wall_max_length,
                rim_fillet_max_length=self.image_rim_fillet_max_length,
                plane_max_length=self.image_plane_max_length,
            )
            self._image_solution = ImageChargeBEMSolution.solve(
                image_profile, n_max=image_n_max, real_profile=real_profile
            )

    @property
    def field_solver(self):
        """The underlying BEM field-solver object (e.g.
        `bem.fields.HemisphericalTipBEMField`/`CylindricalWellBEMField`)
        -- for `specific_particle_tracer.plotting.static_potential_grid`,
        which needs an object with its own `.potential(position)`."""
        return self._field

    @property
    def real_profile(self):
        """The (rho, z) generating profile of the real conductor surface
        -- the actual mesh nodes the field solve uses -- e.g. for
        `specific_particle_tracer.plotting.plot_profiles`."""
        return self._field.solution.profile

    @property
    def image_solution(self):
        """The underlying `bem.image_charge.ImageChargeBEMSolution`, or
        None if this geometry was built with `z0=None` (image force
        disabled). Has its own `.profile` (the recessed image surface)
        and `.image_potential(...)` -- e.g. for
        `specific_particle_tracer.plotting.plot_profiles`/
        `image_potential_grid`."""
        return self._image_solution

    def field(self, position):
        xp = self.xp
        near = self._field.evaluate(position)
        far = self._flat_fallback.field(position)
        z = position[..., 2]
        return _blend_with_flat_fallback(near, far, z, 0.5 * self.z_handoff, self.z_handoff, xp)

    def image_force(self, position, charge, active, plummer_radius):
        xp = self.xp
        far = self._flat_fallback.image_force(position, charge, active, plummer_radius)
        if self._image_solution is None:
            return far
        near = self._image_solution.image_force(
            position,
            charge,
            active,
            d_lo=self.image_d_lo * self.R,
            d_hi=self.image_d_hi * self.R,
            n_max=self.image_n_max,
            xp=self.xp,
        )
        z = position[..., 2]
        return _blend_with_flat_fallback(near, far, z, 0.5 * self.z_handoff, self.z_handoff, xp)

    def kill_mask(self, position):
        # A particle is killed once it's inside the real conductor --
        # see _cylindrical_well_contains for the exact (possibly
        # corner-rounded) boundary this tests against.
        xp = self.xp
        rho = xp.linalg.norm(position[..., :2], axis=-1)
        z = position[..., 2]
        return _cylindrical_well_contains(
            rho, z, self.R, self.H, self.bottom_fillet_radius or 0.0, self.rim_fillet_radius or 0.0, xp
        )

    def worker_args(self):
        return (
            "cylindrical_well_bem",
            self.E_gun,
            self.R,
            self.H,
            self.field_plane_radius,
            self.field_max_length,
            self.n_subdiv,
            self.bottom_fillet_radius,
            self.rim_fillet_radius,
            self.z0,
            self.image_plane_radius,
            self.image_max_length,
            self.image_bottom_max_length,
            self.image_fillet_max_length,
            self.image_wall_max_length,
            self.image_rim_fillet_max_length,
            self.image_plane_max_length,
            self.image_n_max,
            self.image_d_lo,
            self.image_d_hi,
            self.z_handoff,
        )

    @classmethod
    def _from_worker_args(cls, args, xp=np):
        (
            E_gun,
            R,
            H,
            field_plane_radius,
            field_max_length,
            n_subdiv,
            bottom_fillet_radius,
            rim_fillet_radius,
            z0,
            image_plane_radius,
            image_max_length,
            image_bottom_max_length,
            image_fillet_max_length,
            image_wall_max_length,
            image_rim_fillet_max_length,
            image_plane_max_length,
            image_n_max,
            image_d_lo,
            image_d_hi,
            z_handoff,
        ) = args
        return cls(
            E_gun,
            R,
            H,
            field_plane_radius=field_plane_radius,
            field_max_length=field_max_length,
            n_subdiv=n_subdiv,
            bottom_fillet_radius=bottom_fillet_radius,
            rim_fillet_radius=rim_fillet_radius,
            z0=z0,
            image_plane_radius=image_plane_radius,
            image_max_length=image_max_length,
            image_bottom_max_length=image_bottom_max_length,
            image_fillet_max_length=image_fillet_max_length,
            image_wall_max_length=image_wall_max_length,
            image_rim_fillet_max_length=image_rim_fillet_max_length,
            image_plane_max_length=image_plane_max_length,
            image_n_max=image_n_max,
            image_d_lo=image_d_lo,
            image_d_hi=image_d_hi,
            z_handoff=z_handoff,
            xp=xp,
        )


# geometry.Geometry.from_worker_args dispatches on this registry -- there's
# no public registration API (every shape so far has been built into
# geometry.py directly), so a BEM shape living in its own module registers
# itself here on import instead of geometry.py needing to know about bem/.
geometry_module._WORKER_REGISTRY["hemispherical_tip_bem"] = HemisphericalTipBEMGeometry
geometry_module._WORKER_REGISTRY["cylindrical_well_bem"] = CylindricalWellBEMGeometry
