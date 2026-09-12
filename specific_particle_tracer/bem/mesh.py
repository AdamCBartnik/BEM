"""Axisymmetric surface meshing for BEM geometries.

A conductor's *real* surface sets the static-field boundary condition and
kills particles that cross it. Its *image* (or "recessed") surface is the
same shape offset inward by the image-plane depth ``z0``, and is what a BEM
solver should use as the source surface for the image-charge correction (the
same regularization idea as ``geometry.FlatCathode``'s and
``geometry.HemisphericalTip``'s analytic ``z0`` image-plane offset).

Offsetting a surface inward is a Minkowski erosion: convex pieces (a sphere,
a plane) simply shrink/translate, but a reflex ("concave", as seen from the
solid) edge -- like the ridge where a hemispherical tip meets the surrounding
plane -- gets rounded into a fillet, because a ball of radius z0 can't be
pushed into a sharp reflex corner without rounding it. For the hemisphere-
on-plane case that fillet is exactly a quarter-torus: tube radius z0, major
radius R, centered on the original ridge circle. A convex corner (e.g. a
well's rim) instead mitres to a sharp corner under this same erosion --
geometrically exact, but a real problem for the image-charge closest-point
search (see `cylindrical_well_image_profile`'s docstring for the failure
this causes).

A real fabricated conductor never has infinitely sharp edges, though, so
`cylindrical_well_real_profile` also accepts an optional rounding radius for
either of its corners -- `bottom_fillet_radius`/`rim_fillet_radius` -- and
`cylindrical_well_image_profile` takes the *same* two parameters (meaning
the same real, physical radii, not an independent image-side choice) and
derives its own fillet radii by applying the *same* z0 erosion to this
already-rounded real shape, rather than to the idealized sharp one. That
erosion, applied to an already-rounded corner, doesn't just repeat the
sharp-corner story with the numbers shifted:

- At the reflex corner, eroding an already-rounded fillet of radius rho by
  z0 grows it to rho + z0 (same center) -- always a real, valid rounding,
  same as the sharp case (rho=0) degenerating to z0 exactly.
- At the convex corner, eroding an already-rounded fillet of radius rho by
  z0 shrinks it to rho - z0 (same center) -- only a real rounding if
  rho > z0. If the real rounding is too small (rho <= z0), erosion doesn't
  leave a smaller-but-still-rounded corner: it fully consumes the fillet
  and keeps eating into the neighboring wall/plane, converging on exactly
  the same sharp mitre an *unrounded* (rho=0) real corner would erode to.
  So a real rim rounding only fixes the closest-point-search discontinuity
  if it's genuinely bigger than z0, not just present.

Every profile-assembling function here (everything except the low-level
per-region primitives just below, and the raw `sphere_profile`/`sphere_mesh`)
takes a `max_length` argument (plus optional per-region overrides) instead of
a point count directly -- the number of points in each region is whatever
satisfies `max_length` (see `_n_for_length`/`_region_max_length`). This
keeps element size, and so BEM step accuracy, roughly uniform across a whole
geometry by default, rather than implicitly coupling it to how many
points happened to be requested for a given region; ask for a smaller
`max_length` in a specific region (a fillet, say) via that region's own
override when it alone needs finer resolution.

All geometries here are axisymmetric, so each is built as a 2D generating
profile (r, z) in the half-plane r >= 0, then revolved around the z-axis
into a triangulated 3D mesh. Mesh arrays follow BEMpp's convention:
``vertices`` has shape (3, n_vertices), ``elements`` has shape (3,
n_elements) of vertex indices per triangle.
"""

import numpy as np


def sphere_cap_profile(radius, n_theta, theta_max=np.pi / 2):
    """Profile of a sphere cap, from the pole (r=0) to polar angle theta_max.

    Returns an (n_theta, 2) array of (r, z) points, pole first.
    """
    theta = np.linspace(0.0, theta_max, n_theta)
    r = radius * np.sin(theta)
    z = radius * np.cos(theta)
    return np.column_stack([r, z])


def flat_annulus_profile(r_inner, r_outer, z, n_r):
    """Profile of a flat annulus at height z, from r_inner to r_outer,
    uniformly spaced."""
    r = np.linspace(r_inner, r_outer, n_r)
    return np.column_stack([r, np.full_like(r, z)])


def graded_annulus_profile(r_inner, r_outer, z, max_length, growth=1.15):
    """Profile of a flat annulus at height z, from r_inner to r_outer, with
    node spacing starting at `max_length` right at r_inner and growing
    geometrically (by a factor of `growth` per step) out to r_outer.

    Used for the far outer-plane truncation region of every profile below:
    that region needs to resolve the physics near the feature (its inner
    edge, right where the actual bump/well/tip meets it) about as finely
    as everything else, but not all the way out to `r_outer`, which can be
    many multiples of the feature scale (`plane_radius` defaults to
    5-20x the feature radius) -- meshing it uniformly at `max_length`
    would make its point count scale with plane_radius directly, quickly
    dwarfing the rest of the profile for no accuracy benefit (confirmed
    directly: this is what an earlier version of this module's per-class
    defaults did before switching to grading here)."""
    r = [r_inner]
    step = max_length
    while r[-1] + step < r_outer:
        r.append(r[-1] + step)
        step *= growth
    # Snap the last interior node onto r_outer instead of leaving a sliver
    # segment behind it: the loop above stops as soon as one more `step`
    # would overshoot, which can leave r[-1] arbitrarily close to r_outer,
    # and a near-zero-length final segment is a degenerate BEM panel (its
    # tangent normalizes to 0/0 in `_segment_quadrature_geometry`). Half a
    # step is the natural threshold -- it keeps every segment within
    # [step/2, 1.5*step] of the intended grading either way.
    if len(r) > 1 and r_outer - r[-1] < 0.5 * (r[-1] - r[-2]):
        r[-1] = r_outer
    else:
        r.append(r_outer)
    r = np.asarray(r, dtype=float)
    return np.column_stack([r, np.full_like(r, z)])


def torus_fillet_profile(major_radius, tube_radius, phi_start, phi_end, n_phi):
    """Profile of a torus tube slice: (major_radius + tube_radius*cos(phi),
    tube_radius*sin(phi)) for phi in [phi_start, phi_end]."""
    phi = np.linspace(phi_start, phi_end, n_phi)
    r = major_radius + tube_radius * np.cos(phi)
    z = tube_radius * np.sin(phi)
    return np.column_stack([r, z])


def cylinder_wall_profile(rho, z_bottom, z_top, n_z):
    """Profile of a vertical cylindrical wall at radius `rho`, from
    `z_bottom` to `z_top`."""
    z = np.linspace(z_bottom, z_top, n_z)
    return np.column_stack([np.full_like(z, rho), z])


def sphere_profile(radius, n_theta):
    """Profile of a full closed sphere, pole to pole (theta: 0 to pi)."""
    return sphere_cap_profile(radius, n_theta, theta_max=np.pi)


def sphere_mesh(radius, n_theta=40, n_phi=48):
    """Triangulated mesh of a full closed sphere."""
    return revolve_profile(sphere_profile(radius, n_theta), n_phi)


def _n_for_length(arc_length, max_length, min_points=2):
    """Number of points needed along a path of the given total arc length
    so that every one of its (n-1) segments is no longer than max_length."""
    if max_length <= 0.0:
        raise ValueError(f"max_length must be positive (got {max_length!r})")
    n_segments = max(1, int(np.ceil(arc_length / max_length)))
    return max(min_points, n_segments + 1)


def _region_max_length(max_length, region_max_length):
    """A region's effective segment-length cap: `max_length` itself, or
    the tighter of `max_length` and `region_max_length` when the caller
    gave a per-region override -- an override can only refine a region
    further, never relax it past the global cap (so a fillet, say, is
    always resolved at least as finely as everything else)."""
    return max_length if region_max_length is None else min(max_length, region_max_length)


# Every fillet in this module sweeps exactly 90 degrees. A fillet's own
# radius is usually much smaller than the rest of the geometry (it's set
# by z0 or rim_fillet_radius, not R/H), so a max_length picked for the
# overall geometry's scale would often let `_n_for_length` alone resolve
# it with very few segments -- fine for the length constraint, too coarse
# to actually look like an arc. This floor guarantees a fillet is never
# resolved more coarsely than 5 segments over its 90 degree sweep,
# regardless of max_length (default or user-supplied).
_MIN_FILLET_SEGMENTS = 5


def hemisphere_tip_real_profile(
    R, plane_radius, max_length, cap_max_length=None, plane_max_length=None, plane_growth=1.15,
):
    """Generating profile of the real (physical) hemisphere-on-plane
    conductor surface: sphere cap of radius R, then the surrounding plane
    (z=0) out to plane_radius.

    Parameters
    ----------
    max_length : float
        Global cap [m] on every segment's length in the profile.
    cap_max_length, plane_max_length : float, optional
        Per-region caps for the sphere cap / outer plane; each is
        combined with `max_length` via `min(...)` -- see
        `_region_max_length`.
    plane_growth : float, optional
        The outer plane is graded (see `graded_annulus_profile`), not
        uniform: `plane_growth` is its per-step growth factor.
    """
    n_theta = _n_for_length(R * (np.pi / 2), _region_max_length(max_length, cap_max_length))
    cap = sphere_cap_profile(R, n_theta)
    plane = graded_annulus_profile(
        R, plane_radius, 0.0, _region_max_length(max_length, plane_max_length), growth=plane_growth
    )[1:]
    return np.concatenate([cap, plane], axis=0)


def hemisphere_tip_image_profile(
    R, z0, plane_radius, max_length, cap_max_length=None, fillet_max_length=None, plane_max_length=None,
    plane_growth=1.15,
):
    """Generating profile of the recessed image-charge surface for a
    hemisphere-on-plane conductor: sphere cap of radius R-z0, a quarter-torus
    fillet (tube radius z0, major radius R) rounding the ridge, then the
    plane at z=-z0 out to plane_radius.

    Requires 0 < z0 < R (the cap must not shrink to a point or invert).
    The fillet's tube radius is always exactly z0 (not independently
    configurable): it has to be, for the fillet to meet the cap and the
    plane tangentially -- any other radius would leave a gap or overlap
    at one of those two seams (this ridge is a *reflex* corner, so
    rounding it is the only geometrically valid z0-erosion in the first
    place; see the module docstring).

    Parameters
    ----------
    max_length : float
        Global cap [m] on every segment's length in the profile.
    cap_max_length, fillet_max_length, plane_max_length : float, optional
        Per-region caps for the sphere cap / ridge fillet / outer plane;
        each is combined with `max_length` via `min(...)` -- see
        `_region_max_length`.
    plane_growth : float, optional
        The outer plane is graded (see `graded_annulus_profile`), not
        uniform: `plane_growth` is its per-step growth factor.
    """
    if not (0.0 < z0 < R):
        raise ValueError(f"z0 must satisfy 0 < z0 < R (got z0={z0!r}, R={R!r})")
    n_theta = _n_for_length((R - z0) * (np.pi / 2), _region_max_length(max_length, cap_max_length))
    n_fillet = _n_for_length(
        z0 * (np.pi / 2), _region_max_length(max_length, fillet_max_length), min_points=_MIN_FILLET_SEGMENTS + 1
    )
    cap = sphere_cap_profile(R - z0, n_theta)
    fillet = torus_fillet_profile(R, z0, np.pi, 1.5 * np.pi, n_fillet)[1:]
    plane = graded_annulus_profile(
        R, plane_radius, -z0, _region_max_length(max_length, plane_max_length), growth=plane_growth
    )[1:]
    return np.concatenate([cap, fillet, plane], axis=0)


def hemisphere_tip_image_doubled_profile(R, z0, max_length, cap_max_length=None, fillet_max_length=None):
    """Generating profile of the recessed image-charge surface for a
    hemisphere-on-plane conductor, *mirror-doubled* through z=-z0 instead
    of truncated at some `plane_radius`: sphere cap (radius R-z0) + ridge
    fillet (as in `hemisphere_tip_image_profile`), then that same
    cap+fillet piece again, reflected through z=-z0, so the profile goes
    pole-to-pole with no plane at all -- exactly the same trick
    `bem.fields.HemisphericalTipBEMField` already uses for the *real*
    surface (mirror through z=0 gives the exact, complete sphere, no
    plane meshed either), just applied one level down to the *recessed*
    surface's own mirror plane at z=-z0 instead of the real surface's at
    z=0.

    This is valid because the recessed surface (cap + fillet, ending at
    the point (R, -z0)) is itself smoothly (C1) mirror-symmetric about
    z=-z0: the fillet's tangent there is
    exactly horizontal (checked directly, phi=1.5*pi in
    `torus_fillet_profile`'s parametrization gives dz/dphi=0), the same
    condition that makes the real hemisphere's own mirror trick work.
    Unlike that case, the doubled shape here isn't a literal sphere (the
    fillet is a torus arc, not spherical), but sphericity was never what
    the trick actually needed -- only a smooth, closed mirror-symmetric
    surface, which this is regardless.

    Using this profile with `bem.image_charge.ImageChargeBEMSolution`
    needs one more piece beyond just meshing it, though: the induced-
    charge solve must be excited by the real source *and* its negative
    mirror image through z=-z0 together (the classical "grounded
    conductor on a grounded plane" method-of-images construction), not
    the bare source alone -- see `bem.geometry.HemisphericalTipBEMGeometry
    .image_force`'s docstring for how that's done (a phantom mirror
    "particle" added to the joint multi-source solve already built for
    conductor-mediated cross-coupling, not a change to
    `ImageChargeBEMSolution` itself). `image_field`/`image_potential`
    called directly on a solution built from *this* profile handle this
    too, via their own `mirror_plane_z` parameter -- auto-detected from
    `ImageChargeBEMSolution.mirror_plane_z`, which
    `HemisphericalTipBEMGeometry` sets for you when
    `image_mirror_symmetric=True`, so no extra care is needed at the
    call site either way.

    Trades away the finite-`plane_radius` truncation error and the
    plane's own point count entirely (no plane region at all) for a
    second, phantom "source" per real particle in `image_force`'s joint
    solve (see that method's own cost discussion).

    Requires 0 < z0 < R (same as `hemisphere_tip_image_profile`).

    Parameters
    ----------
    max_length : float
        Global cap [m] on every segment's length in the profile.
    cap_max_length, fillet_max_length : float, optional
        Per-region caps for the sphere cap / ridge fillet, combined with
        `max_length` via `min(...)` -- see `_region_max_length`.
    """
    if not (0.0 < z0 < R):
        raise ValueError(f"z0 must satisfy 0 < z0 < R (got z0={z0!r}, R={R!r})")
    n_theta = _n_for_length((R - z0) * (np.pi / 2), _region_max_length(max_length, cap_max_length))
    n_fillet = _n_for_length(
        z0 * (np.pi / 2), _region_max_length(max_length, fillet_max_length), min_points=_MIN_FILLET_SEGMENTS + 1
    )
    cap = sphere_cap_profile(R - z0, n_theta)
    fillet = torus_fillet_profile(R, z0, np.pi, 1.5 * np.pi, n_fillet)[1:]
    upper = np.concatenate([cap, fillet], axis=0)  # pole (0, R-z0) ... corner (R, -z0)

    # Mirror everything except the shared corner point through z=-z0,
    # continuing from just past the corner out to the mirrored pole --
    # not re-including the corner itself (upper already has it).
    mirror = upper[-2::-1].copy()
    mirror[:, 1] = -2.0 * z0 - mirror[:, 1]
    return np.concatenate([upper, mirror], axis=0)


def cylindrical_well_real_profile(
    R, H, plane_radius, max_length,
    bottom_fillet_radius=None, rim_fillet_radius=None,
    bottom_max_length=None, bottom_fillet_max_length=None, wall_max_length=None,
    rim_fillet_max_length=None, plane_max_length=None, plane_growth=1.15,
):
    """Generating profile of the real (physical) cylindrical-well conductor
    surface: a flat bottom disk of radius R at z=-H, a cylindrical wall at
    rho=R from z=-H to z=0, then the surrounding plane (z=0) out to
    plane_radius.

    Both corners (rim at rho=R,z=0 and the bottom-of-well corner at
    rho=R,z=-H) are sharp by default, matching the idealized shape as
    originally specified. But a real fabricated well never has infinitely
    sharp edges, so either can be rounded in place, via
    `bottom_fillet_radius`/`rim_fillet_radius`: R and H stay the nominal
    disk radius and wall span exactly as given -- the flat/cylindrical
    pieces just stop a little short of where the sharp corner would have
    been, on both sides, and a tangent arc of the given radius bridges the
    gap. See the module docstring for the reflex/convex distinction
    between these two corners, and for how `cylindrical_well_image_profile`
    derives its own (z0-eroded) fillet radii from these same real ones
    rather than taking an independent image-side radius.

    Requires bottom_fillet_radius + rim_fillet_radius < H (the wall
    segment between the two corners' tangent points must not vanish or
    invert).

    Parameters
    ----------
    max_length : float
        Global cap [m] on every segment's length in the profile.
    bottom_fillet_radius, rim_fillet_radius : float or None, optional
        Rounding radii [m] for the bottom-of-well and rim corners.
        Default None (0) for either -- sharp, matching this function's
        original behavior.
    bottom_max_length, bottom_fillet_max_length, wall_max_length,
    rim_fillet_max_length, plane_max_length : float, optional
        Per-region caps for the bottom disk / bottom-corner fillet / wall
        / rim fillet / outer plane; each is combined with `max_length`
        via `min(...)` -- see `_region_max_length`.
    plane_growth : float, optional
        The outer plane is graded (see `graded_annulus_profile`), not
        uniform: `plane_growth` is its per-step growth factor.
    """
    bottom_fillet_radius = 0.0 if bottom_fillet_radius is None else float(bottom_fillet_radius)
    rim_fillet_radius = 0.0 if rim_fillet_radius is None else float(rim_fillet_radius)
    if bottom_fillet_radius < 0.0:
        raise ValueError(f"bottom_fillet_radius must be >= 0 (got {bottom_fillet_radius!r})")
    if rim_fillet_radius < 0.0:
        raise ValueError(f"rim_fillet_radius must be >= 0 (got {rim_fillet_radius!r})")
    if bottom_fillet_radius + rim_fillet_radius >= H:
        raise ValueError(
            "bottom_fillet_radius + rim_fillet_radius must be < H "
            f"(got {bottom_fillet_radius!r} + {rim_fillet_radius!r} >= {H!r})"
        )

    bottom_r = R - bottom_fillet_radius
    n_bottom = _n_for_length(bottom_r, _region_max_length(max_length, bottom_max_length))
    pieces = [flat_annulus_profile(0.0, bottom_r, -H, n_bottom)]

    if bottom_fillet_radius > 0.0:
        n_bf = _n_for_length(
            bottom_fillet_radius * (np.pi / 2),
            _region_max_length(max_length, bottom_fillet_max_length),
            min_points=_MIN_FILLET_SEGMENTS + 1,
        )
        bf = torus_fillet_profile(bottom_r, bottom_fillet_radius, -0.5 * np.pi, 0.0, n_bf)[1:]
        bf[:, 1] += -H + bottom_fillet_radius
        pieces.append(bf)

    wall_z_lo = -H + bottom_fillet_radius
    wall_z_hi = -rim_fillet_radius
    n_wall = _n_for_length(wall_z_hi - wall_z_lo, _region_max_length(max_length, wall_max_length))
    pieces.append(cylinder_wall_profile(R, wall_z_lo, wall_z_hi, n_wall)[1:])

    if rim_fillet_radius > 0.0:
        n_rf = _n_for_length(
            rim_fillet_radius * (np.pi / 2),
            _region_max_length(max_length, rim_fillet_max_length),
            min_points=_MIN_FILLET_SEGMENTS + 1,
        )
        rf = torus_fillet_profile(R + rim_fillet_radius, rim_fillet_radius, np.pi, 0.5 * np.pi, n_rf)[1:]
        rf[:, 1] += -rim_fillet_radius
        pieces.append(rf)

    plane_r0 = R + rim_fillet_radius
    pieces.append(
        graded_annulus_profile(
            plane_r0, plane_radius, 0.0, _region_max_length(max_length, plane_max_length), growth=plane_growth
        )[1:]
    )
    return np.concatenate(pieces, axis=0)


def cylindrical_well_image_profile(
    R, H, z0, plane_radius, max_length,
    bottom_fillet_radius=None, rim_fillet_radius=None,
    bottom_max_length=None, bottom_fillet_max_length=None, wall_max_length=None,
    rim_fillet_max_length=None, plane_max_length=None, plane_growth=1.15,
):
    """Generating profile of the recessed image-charge surface for a
    cylindrical well of radius R, depth H, cut into an infinite flat
    cathode -- the real surface (`cylindrical_well_real_profile`, with the
    *same* `bottom_fillet_radius`/`rim_fillet_radius`), offset inward by z0
    everywhere (see the module docstring for what that erosion does to an
    already-rounded corner of each kind).

    - The bottom-of-well (reflex) corner's fillet radius becomes
      `bottom_fillet_radius + z0` -- always a real rounding, same center
      as the real fillet (rho=R-bottom_fillet_radius, z=-H+bottom_fillet_radius).
    - The rim (convex) corner's fillet radius becomes
      `rim_fillet_radius - z0` -- only a real rounding if
      `rim_fillet_radius > z0`; otherwise erosion fully consumes it and
      this profile reverts to the sharp mitre at (rho=R+z0, z=-z0), same
      as an unrounded (rim_fillet_radius=0) real corner would erode to --
      *not* some intermediate smaller-but-still-rounded corner. A real
      rim rounding only fixes the closest-point-search discontinuity that
      motivated this in the first place (see below) if it's genuinely
      bigger than z0, not just present.

      Why that discontinuity matters: at a sharp mitred corner, a query
      point is equidistant from both adjacent faces, and the assigned
      surface normal jumps discontinuously (not just inaccurately --
      discontinuously) depending on which face's argmin wins the tie. A
      particle whose trajectory passes near that knife-edge sees a
      genuine jump discontinuity in its force, which no amount of
      adaptive step-size shrinking can resolve -- RK45 spins forever
      trying anyway (confirmed directly: a particle escaping a shallow
      well over its rim can hang indefinitely this way).

    Requires 0 < z0 < min(R, H) and
    bottom_fillet_radius + max(rim_fillet_radius, z0) < H (the wall
    segment between the two eroded fillets' tangent points must not
    vanish or invert -- using max(rim_fillet_radius, z0) rather than
    plain rim_fillet_radius since an under-sized rim rounding still gets
    eroded back by the full z0, per the mitre-reversion behavior above).

    Parameters
    ----------
    max_length : float
        Global cap [m] on every segment's length in the profile.
    bottom_fillet_radius, rim_fillet_radius : float or None, optional
        The *real* corners' rounding radii [m] -- not independent
        image-side choices, see above. Default None (0) for either.
    bottom_max_length, bottom_fillet_max_length, wall_max_length,
    rim_fillet_max_length, plane_max_length : float, optional
        Per-region caps for the bottom disk / bottom-corner fillet / wall
        / rim fillet / outer plane; each is combined with `max_length`
        via `min(...)` -- see `_region_max_length`. Same names as
        `cylindrical_well_real_profile`'s, since they name the same
        regions of the same shape.
    plane_growth : float, optional
        The outer plane is graded (see `graded_annulus_profile`), not
        uniform: `plane_growth` is its per-step growth factor.
    """
    if not (0.0 < z0 < min(R, H)):
        raise ValueError(
            f"z0 must satisfy 0 < z0 < min(R, H) (got z0={z0!r}, R={R!r}, H={H!r})"
        )
    bottom_fillet_radius = 0.0 if bottom_fillet_radius is None else float(bottom_fillet_radius)
    rim_fillet_radius = 0.0 if rim_fillet_radius is None else float(rim_fillet_radius)
    if bottom_fillet_radius < 0.0:
        raise ValueError(f"bottom_fillet_radius must be >= 0 (got {bottom_fillet_radius!r})")
    if rim_fillet_radius < 0.0:
        raise ValueError(f"rim_fillet_radius must be >= 0 (got {rim_fillet_radius!r})")
    wall_top_extent = max(rim_fillet_radius, z0)
    if bottom_fillet_radius + wall_top_extent >= H:
        raise ValueError(
            "bottom_fillet_radius + max(rim_fillet_radius, z0) must be < H "
            f"(got {bottom_fillet_radius!r} + {wall_top_extent!r} >= {H!r})"
        )

    image_bottom_radius = bottom_fillet_radius + z0
    image_rim_radius = rim_fillet_radius - z0  # <= 0 -> sharp mitre, see docstring

    bottom_r = R - bottom_fillet_radius
    n_bottom = _n_for_length(bottom_r, _region_max_length(max_length, bottom_max_length))
    pieces = [flat_annulus_profile(0.0, bottom_r, -H - z0, n_bottom)]

    n_fillet = _n_for_length(
        image_bottom_radius * (np.pi / 2),
        _region_max_length(max_length, bottom_fillet_max_length),
        min_points=_MIN_FILLET_SEGMENTS + 1,
    )
    bf = torus_fillet_profile(bottom_r, image_bottom_radius, -0.5 * np.pi, 0.0, n_fillet)[1:]
    bf[:, 1] += -H + bottom_fillet_radius
    pieces.append(bf)

    wall_z_lo = -H + bottom_fillet_radius
    wall_z_hi = -wall_top_extent
    n_wall = _n_for_length(wall_z_hi - wall_z_lo, _region_max_length(max_length, wall_max_length))
    pieces.append(cylinder_wall_profile(R + z0, wall_z_lo, wall_z_hi, n_wall)[1:])

    if image_rim_radius > 0.0:
        n_rf = _n_for_length(
            image_rim_radius * (np.pi / 2),
            _region_max_length(max_length, rim_fillet_max_length),
            min_points=_MIN_FILLET_SEGMENTS + 1,
        )
        rf = torus_fillet_profile(R + rim_fillet_radius, image_rim_radius, np.pi, 0.5 * np.pi, n_rf)[1:]
        rf[:, 1] += -rim_fillet_radius
        pieces.append(rf)
        plane_r0 = R + rim_fillet_radius
    else:
        plane_r0 = R + z0  # sharp mitre: wall meets plane directly, no fillet piece

    pieces.append(
        graded_annulus_profile(
            plane_r0, plane_radius, -z0, _region_max_length(max_length, plane_max_length), growth=plane_growth
        )[1:]
    )
    return np.concatenate(pieces, axis=0)


def revolve_profile(profile_rz, n_phi, pole_tol=1e-12):
    """Revolve a 2D generating profile (r, z), r>=0, ordered along the
    surface, around the z-axis into a triangulated 3D mesh.

    A profile point with r ~ 0 is treated as a pole (a single shared vertex
    rather than a full ring of n_phi vertices).

    Returns (vertices, elements): vertices has shape (3, n_vertices),
    elements has shape (3, n_elements) of vertex indices (BEMpp convention).
    """
    profile_rz = np.asarray(profile_rz, dtype=float)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    vertices = []
    ring_indices = []
    for r, z in profile_rz:
        if r < pole_tol:
            idx = len(vertices)
            vertices.append((0.0, 0.0, z))
            ring_indices.append(np.full(n_phi, idx, dtype=int))
        else:
            start = len(vertices)
            vertices.extend((r * c, r * s, z) for c, s in zip(cos_phi, sin_phi))
            ring_indices.append(np.arange(start, start + n_phi))

    elements = []
    for i in range(len(ring_indices) - 1):
        ring_a, ring_b = ring_indices[i], ring_indices[i + 1]
        for k in range(n_phi):
            a, b = ring_a[k], ring_a[(k + 1) % n_phi]
            c, d = ring_b[k], ring_b[(k + 1) % n_phi]
            if a == b:  # ring i is a pole
                elements.append((a, d, c))
            elif c == d:  # ring i+1 is a pole
                elements.append((a, b, c))
            else:
                elements.append((a, b, d))
                elements.append((a, d, c))

    vertices = np.asarray(vertices, dtype=float).T
    elements = np.asarray(elements, dtype=int).T
    return vertices, elements


def hemisphere_tip_real_mesh(R, plane_radius, max_length, n_phi=48, **profile_kwargs):
    """Triangulated mesh of the real hemisphere-on-plane conductor surface.
    `max_length`/`profile_kwargs` are passed straight through to
    `hemisphere_tip_real_profile`; `n_phi` is the (unrelated) azimuthal
    resolution of the revolve itself."""
    profile = hemisphere_tip_real_profile(R, plane_radius, max_length, **profile_kwargs)
    return revolve_profile(profile, n_phi)


def hemisphere_tip_image_mesh(R, z0, plane_radius, max_length, n_phi=48, **profile_kwargs):
    """Triangulated mesh of the recessed image-charge surface for a
    hemisphere-on-plane conductor. `max_length`/`profile_kwargs` are passed
    straight through to `hemisphere_tip_image_profile`; `n_phi` is the
    (unrelated) azimuthal resolution of the revolve itself."""
    profile = hemisphere_tip_image_profile(R, z0, plane_radius, max_length, **profile_kwargs)
    return revolve_profile(profile, n_phi)


def cylindrical_well_real_mesh(R, H, plane_radius, max_length, n_phi=48, **profile_kwargs):
    """Triangulated mesh of the real cylindrical-well conductor surface.
    `max_length`/`profile_kwargs` are passed straight through to
    `cylindrical_well_real_profile`; `n_phi` is the (unrelated) azimuthal
    resolution of the revolve itself."""
    profile = cylindrical_well_real_profile(R, H, plane_radius, max_length, **profile_kwargs)
    return revolve_profile(profile, n_phi)


def cylindrical_well_image_mesh(R, H, z0, plane_radius, max_length, n_phi=48, **profile_kwargs):
    """Triangulated mesh of the recessed image-charge surface for a
    cylindrical well. `max_length`/`profile_kwargs` are passed straight
    through to `cylindrical_well_image_profile`; `n_phi` is the (unrelated)
    azimuthal resolution of the revolve itself."""
    profile = cylindrical_well_image_profile(R, H, z0, plane_radius, max_length, **profile_kwargs)
    return revolve_profile(profile, n_phi)


def min_distance_to_profile(points_rz, profile_rz):
    """For each (r, z) in points_rz, the minimum Euclidean distance to the
    polyline through profile_rz. Used to numerically check that an offset
    profile really does sit a constant distance from the original."""
    points_rz = np.asarray(points_rz, dtype=float)
    profile_rz = np.asarray(profile_rz, dtype=float)

    a = profile_rz[:-1][None, :, :]
    b = profile_rz[1:][None, :, :]
    p = points_rz[:, None, :]

    ab = b - a
    ab_len_sq = np.sum(ab**2, axis=-1)
    t = np.sum((p - a) * ab, axis=-1) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t[..., None] * ab
    dist = np.linalg.norm(p - closest, axis=-1)
    return dist.min(axis=-1)
