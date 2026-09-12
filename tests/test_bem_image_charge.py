import numpy as np
import pytest

from specific_particle_tracer.bem.image_charge import (
    ImageChargeBEMSolution,
    assemble_mode_operators,
    image_charge_weight,
    mirror_point,
)
from specific_particle_tracer.constants import VACUUM_PERMITTIVITY

# Assembling the per-mode operators is by far the most expensive thing in
# this file (adaptive quadrature; the solves themselves are trivial next to
# it), and the tests below deliberately reuse a handful of profiles over and
# over -- so cache the assembly per profile instead of repeating it ~19
# times. Two things make this safe and simple:
#   * `A` is purely geometric, so it depends only on `profile` -- never on
#     n_max beyond how many modes get assembled, nor on `real_profile`
#     (which only ever affects `image_potential`'s masking).
#   * every consumer slices `self.A[: n_max + 1, ...]`, so an `A` assembled
#     for *more* modes than a caller asked for behaves identically to one
#     assembled for exactly that many. That means one assembly at the
#     largest n_max a profile is used with serves every smaller request,
#     including the n_max-convergence sweeps.
_OPERATOR_CACHE = {}


def _cached_solve(profile, n_max, real_profile=None):
    """`ImageChargeBEMSolution.solve` with the operator assembly cached
    across tests -- see `_OPERATOR_CACHE` above. Returns a solution whose
    `.n_max` is the requested one regardless of how many modes the cached
    `A` actually holds."""
    profile = np.asarray(profile, dtype=float)
    key = profile.tobytes()
    assembled, A = _OPERATOR_CACHE.get(key, (-1, None))
    if assembled < n_max:
        A = assemble_mode_operators(profile, n_max)
        _OPERATOR_CACHE[key] = (n_max, A)
    return ImageChargeBEMSolution(profile, A, n_max, real_profile=real_profile)


_GEOMETRY_CACHE = {}


def _cached_geometry(**kwargs):
    """A `HemisphericalTipBEMGeometry` cached on its constructor arguments:
    building one runs a static-field solve *and* an image-charge assembly,
    and several tests below want byte-identical geometries. None of them
    mutate the object (the one test that needs a tweaked copy makes its own
    `copy.copy`), so sharing is safe."""
    from specific_particle_tracer.bem.geometry import HemisphericalTipBEMGeometry

    key = tuple(sorted(kwargs.items()))
    if key not in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE[key] = HemisphericalTipBEMGeometry(**kwargs)
    return _GEOMETRY_CACHE[key]


def _graded_flat_profile(r_max, r_min=0.01, growth=1.15):
    """A flat (z=0) generating profile, node spacing growing geometrically
    from r_min near the pole out to r_max -- fine enough near the origin
    to resolve a near-field point charge without needing hundreds of
    uniformly-spaced nodes out to r_max."""
    rs = [0.0]
    r = r_min
    while r < r_max:
        rs.append(r)
        r *= growth
    rs.append(r_max)
    rs = np.array(rs)
    return np.column_stack([rs, np.zeros_like(rs)])


def test_image_charge_weight_endpoints_and_monotonic():
    assert image_charge_weight(0.0, 1.0, 2.0) == 1.0
    assert image_charge_weight(2.0, 1.0, 2.0) == 0.0
    assert image_charge_weight(3.0, 1.0, 2.0) == 0.0
    ds = np.linspace(0.0, 2.0, 20)
    ws = [image_charge_weight(d, 1.0, 2.0) for d in ds]
    assert all(w1 >= w2 for w1, w2 in zip(ws, ws[1:]))  # non-increasing


def test_mirror_point_reflects_through_tangent_line():
    # Flat tangent along the "rho" direction at closest=(1,0): reflecting a
    # point directly above should just flip z and leave rho fixed.
    closest = np.array([1.0, 0.0])
    tangent_unit = np.array([1.0, 0.0])
    normal_unit = np.array([0.0, 1.0])
    rho_img, z_img, side = mirror_point(1.0, 0.5, closest, tangent_unit, normal_unit)
    assert abs(rho_img - 1.0) < 1e-12
    assert abs(z_img - (-0.5)) < 1e-12
    assert side == 1


def test_flat_plane_full_mirror_reproduces_classical_image_charge_on_axis():
    """A point charge above a large, finely-resolved flat grounded plane:
    with the mirror-charge trick fully engaged (w=1 everywhere), even the
    m=0-only residual should reproduce the classical exact mirror-charge
    field essentially exactly -- the residual really is ~zero for a
    (nearly) infinite flat plane, so no modes are needed at all."""
    Q = 1.0
    d = 0.3
    profile = _graded_flat_profile(r_max=200.0, r_min=0.01, growth=1.2)
    position = np.array([1e-9, 0.0, d])

    E_exact = np.array([0.0, 0.0, -Q / (4 * np.pi * (2 * d) ** 2)]) / VACUUM_PERMITTIVITY

    sol = _cached_solve(profile, n_max=0)
    E = sol.image_field(position, Q, d_lo=1e6, d_hi=1e6 + 1.0)
    assert np.linalg.norm(E - E_exact) / np.linalg.norm(E_exact) < 1e-3


def test_flat_plane_full_mirror_reproduces_classical_image_charge_off_axis():
    """Same as the on-axis version, but for an off-axis particle position
    -- exercises the phi-rotation bookkeeping in `image_field` (particle
    not at phi=0) and the mirror-point reflection/side bookkeeping."""
    Q = 1.0
    d = 0.3
    profile = _graded_flat_profile(r_max=200.0, r_min=0.01, growth=1.2)
    position = np.array([0.5, 0.2, d])

    mirror = np.array([position[0], position[1], -d])
    r = position - mirror
    E_exact = -Q * r / (4 * np.pi * np.linalg.norm(r) ** 3) / VACUUM_PERMITTIVITY

    sol = _cached_solve(profile, n_max=0)
    E = sol.image_field(position, Q, d_lo=1e6, d_hi=1e6 + 1.0)
    assert np.linalg.norm(E - E_exact) / np.linalg.norm(E_exact) < 1e-3


def test_bare_mode_expansion_converges_to_the_same_flat_plane_answer():
    """With the mirror-charge trick fully *disabled* (w=0, pure Fourier-
    mode BEM), the answer should converge to the same classical result as
    more modes are kept -- the point of the "exactness" argument in
    bem.image_charge's module docstring: the mirror-charge trick only
    changes how fast this converges, not what it converges to."""
    Q = 1.0
    d = 0.3
    profile = _graded_flat_profile(r_max=200.0, r_min=0.01, growth=1.2)
    position = np.array([0.5, 0.2, d])

    mirror = np.array([position[0], position[1], -d])
    r = position - mirror
    E_exact = -Q * r / (4 * np.pi * np.linalg.norm(r) ** 3) / VACUUM_PERMITTIVITY

    errs = []
    for n_max in [0, 1, 2, 4]:
        sol = _cached_solve(profile, n_max=n_max)
        E = sol.image_field(position, Q, d_lo=0.0, d_hi=0.0)
        errs.append(np.linalg.norm(E - E_exact) / np.linalg.norm(E_exact))

    assert errs[1] < errs[0]
    assert errs[2] < errs[1]
    assert errs[3] < errs[2]


def test_mirror_charge_trick_needs_far_fewer_modes_for_the_same_accuracy():
    """The actual motivation for the mirror-charge subtraction: at a fixed
    (small) n_max, engaging it should give dramatically better accuracy
    than the bare mode expansion, for a particle close to the surface."""
    Q = 1.0
    d = 0.3
    profile = _graded_flat_profile(r_max=200.0, r_min=0.01, growth=1.2)
    position = np.array([0.5, 0.2, d])

    mirror = np.array([position[0], position[1], -d])
    r = position - mirror
    E_exact = -Q * r / (4 * np.pi * np.linalg.norm(r) ** 3) / VACUUM_PERMITTIVITY

    n_max = 1
    sol = _cached_solve(profile, n_max=n_max)
    E_bare = sol.image_field(position, Q, d_lo=0.0, d_hi=0.0)
    E_mirror = sol.image_field(position, Q, d_lo=1.0, d_hi=2.0)

    err_bare = np.linalg.norm(E_bare - E_exact) / np.linalg.norm(E_exact)
    err_mirror = np.linalg.norm(E_mirror - E_exact) / np.linalg.norm(E_exact)
    assert err_mirror < err_bare / 10.0


def test_real_hemisphere_tip_matches_exact_three_image_analytic_solution():
    """The strongest available check: for the project's actual hemisphere-
    on-plane recessed image geometry, compare against
    `forces.hemispherical_tip_image_force` -- an *exact* (not approximate)
    closed-form 3-image solution for a sharp-ridged hemisphere-on-infinite-
    plane conductor. `bem.mesh.hemisphere_tip_image_profile` instead rounds
    that ridge into a physically-motivated fillet (see its module
    docstring), so exact agreement isn't expected -- but near the pole,
    away from the fillet, the two geometries are essentially identical and
    the fields should agree closely. This is also what caught the
    units mismatch this module originally had (this package's kernels are
    all G=1/(4*pi*r), i.e. natural units with vacuum permittivity set to
    1, whereas `forces.py` works in SI with an explicit
    1/(4*pi*epsilon_0) -- `image_field` divides by VACUUM_PERMITTIVITY to
    match)."""
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile
    from specific_particle_tracer.forces import hemispherical_tip_image_force

    R = 50e-9
    z0 = 3e-9
    plane_radius = 5 * R
    profile = hemisphere_tip_image_profile(R, z0, plane_radius, max_length=R / 40, fillet_max_length=z0 / 12)

    n_max = 24
    sol = _cached_solve(profile, n_max=n_max)

    Q = -1.0
    d_lo, d_hi = 0.1 * R, 0.5 * R

    for theta_deg, d_over_R in [(0.0, 0.1), (0.0, 0.3), (10.0, 0.1), (30.0, 0.1)]:
        theta = np.radians(theta_deg)
        direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
        position = (R + d_over_R * R) * direction

        F_bem = Q * sol.image_field(position, Q, d_lo, d_hi, n_max=n_max)

        F_exact = hemispherical_tip_image_force(
            position.reshape(1, 1, 3), np.array([[Q]]), np.array([[True]]), R - z0, plummer_radius=1e-12, plane_z0=z0
        )[0, 0]

        rel_err = np.linalg.norm(F_bem - F_exact) / np.linalg.norm(F_exact)
        assert rel_err < 0.02


def test_mirror_symmetric_hemisphere_tip_matches_exact_three_image_analytic_solution():
    """`HemisphericalTipBEMGeometry(image_mirror_symmetric=True)` solves the
    image-charge problem on a *closed*, mirror-doubled profile (see
    `bem.mesh.hemisphere_tip_image_doubled_profile`) with no plane meshed
    at all, instead of the truncated-plane profile the default path uses
    -- via a phantom mirror "particle" added to the existing joint
    multi-source solve (`bem.geometry.HemisphericalTipBEMGeometry.
    _mirror_symmetric_image_force`). Should match the exact 3-image
    solution about as well as the default (truncated-plane) path does.

    Regression test for a real, once-shipped bug: `image_force`'s joint
    solve deliberately excludes *direct* Coulomb between the sources it's
    given (that's `forces.coulomb_force`'s job for real particle pairs)
    -- but the phantom mirror charge's direct pull on the real particle
    is exactly the classical "plane image" term in the 3-image
    construction and isn't optional. Missing it gave a stable, mesh- and
    mode-count-independent ~1-2% error (confirmed directly, including on
    a bare sphere with no fillet involved at all, to rule out any
    geometry-specific explanation) -- adding it back brought this well
    under 1%, matching the truncated-plane path's own accuracy."""
    from specific_particle_tracer.forces import hemispherical_tip_image_force

    R = 50e-9
    z0 = 3e-9
    geom = _cached_geometry(E_gun=-1e8, R=R, z0=z0, image_n_max=24, image_mirror_symmetric=True)

    Q = -1.602176634e-19
    for theta_deg, d_over_R in [(0.0, 0.1), (10.0, 0.1), (30.0, 0.1), (45.0, 0.1)]:
        theta = np.radians(theta_deg)
        position = (R + d_over_R * R) * np.array([np.sin(theta), 0.0, np.cos(theta)])
        pos3, ch2, act2 = position.reshape(1, 1, 3), np.array([[Q]]), np.array([[True]])

        F_bem = geom.image_force(pos3, ch2, act2, plummer_radius=1e-12)[0, 0]
        F_exact = hemispherical_tip_image_force(pos3, ch2, act2, R - z0, plummer_radius=1e-12, plane_z0=z0)[0, 0]

        rel_err = np.linalg.norm(F_bem - F_exact) / np.linalg.norm(F_exact)
        assert rel_err < 0.02


def test_mirror_symmetric_hemisphere_tip_cross_coupling_matches_exact_multi_image():
    """Same cross-coupling check as the truncated-plane path's own
    multi-particle test, but for `image_mirror_symmetric=True`."""
    from specific_particle_tracer.forces import hemispherical_tip_image_force

    R = 50e-9
    z0 = 3e-9
    geom = _cached_geometry(E_gun=-1e8, R=R, z0=z0, image_n_max=24, image_mirror_symmetric=True)

    Q = -1.602176634e-19
    theta1, theta2 = np.radians(10.0), np.radians(25.0)
    p1 = 1.1 * R * np.array([np.sin(theta1), 0.0, np.cos(theta1)])
    p2 = 1.15 * R * np.array([0.0, np.sin(theta2), np.cos(theta2)])
    positions = np.stack([p1, p2])[None, :, :]
    charges = np.array([[Q, Q]])
    active = np.array([[True, True]])

    F_bem = geom.image_force(positions, charges, active, plummer_radius=1e-12)[0]
    F_exact = hemispherical_tip_image_force(positions, charges, active, R - z0, plummer_radius=1e-12, plane_z0=z0)[0]

    rel_err = np.linalg.norm(F_bem - F_exact, axis=-1) / np.linalg.norm(F_exact, axis=-1)
    assert np.max(rel_err) < 0.03


def test_doubled_sphere_includes_every_direct_phantom_image():
    """An exact sphere/plane geometry isolates cross images from erosion
    and fillet errors in the production geometry. Test both backends.
    """
    from specific_particle_tracer.bem.geometry import HemisphericalTipBEMGeometry
    from specific_particle_tracer.bem.mesh import sphere_profile
    from specific_particle_tracer.geometry import FlatCathode
    from specific_particle_tracer.forces import hemispherical_tip_image_force
    R = 50e-9
    sol = _cached_solve(sphere_profile(R, 40), n_max=8)
    sol.mirror_plane_z = 0.
    pos = R*np.array([[[.2,0.,1.3],[-.2,0.,1.5]], [[.2,0.,1.3],[-.2,0.,1.5]]])
    charge = np.full((2,2), -1.602176634e-19)
    active = np.array([[True,True],[True,False]])
    expected = hemispherical_tip_image_force(pos,charge,active,R,1e-12,plane_z0=0.)
    obj = HemisphericalTipBEMGeometry.__new__(HemisphericalTipBEMGeometry)
    obj.R=R; obj.z0=0.; obj.image_d_lo=0.; obj.image_d_hi=0.; obj.image_n_max=8
    obj._image_solution=sol
    backends = [np]
    try:
        import cupy as cp
        backends.append(cp)
    except ImportError:
        pass
    for xp in backends:
        obj.xp=xp
        obj._flat_fallback=FlatCathode(0.,z0=0.,xp=xp)
        actual=obj._mirror_symmetric_image_force(xp.asarray(pos),xp.asarray(charge),xp.asarray(active),1e-12)
        if xp is not np:
            actual=xp.asnumpy(actual)
        error=np.linalg.norm(actual-expected)/np.linalg.norm(expected)
        assert error < .003


def test_mirror_symmetric_matches_truncated_plane_past_the_rim():
    """The two image-charge modes solve the same physical problem two ways,
    so they have to agree everywhere a particle can actually be -- including
    out past the tip's rim (rho > R, small z), which is precisely where an
    emitted particle escapes and which every other test in this file misses
    by only ever sampling theta <= 45 degrees.

    Regression test for a real shipped bug found by scanning this domain:
    `bem.mesh.hemisphere_tip_image_doubled_profile` closes at its belt
    (rho=R, z=-z0) as a tangential cusp, and the local mirror-charge trick
    breaks down there in two separate ways (the mirror charge escapes the
    conductor, and the real/phantom pair stops being antisymmetric, which
    is what the whole doubled construction rests on) -- see
    `ImageChargeBEMSolution._local_mirror_charge_is_valid`. Measured before
    the guard: a factor of 33, with the force pointing the wrong way
    (repulsive instead of attractive), stable under both n_max and mesh
    refinement -- this project's own established tell for a real bug rather
    than under-convergence.

    Deliberately checks the force *direction* too: the mode failed by
    flipping the sign of the dominant component, which a loose
    relative-magnitude tolerance alone can let through.
    """

    R = 50e-9
    z0 = 3e-9
    Q = -1.602176634e-19
    doubled = _cached_geometry(E_gun=-1e8, R=R, z0=z0, image_n_max=24, image_mirror_symmetric=True)
    truncated = _cached_geometry(E_gun=-1e8, R=R, z0=z0, image_n_max=24, image_mirror_symmetric=False)

    for r_over_R in (1.02, 1.10, 1.30):
        for theta_deg in (60.0, 75.0, 85.0, 88.0, 89.0):
            theta = np.radians(theta_deg)
            position = r_over_R * R * np.array([np.sin(theta), 0.0, np.cos(theta)])
            pos3, ch2, act2 = position.reshape(1, 1, 3), np.array([[Q]]), np.array([[True]])

            F_doubled = doubled.image_force(pos3, ch2, act2, plummer_radius=1e-12)[0, 0]
            F_truncated = truncated.image_force(pos3, ch2, act2, plummer_radius=1e-12)[0, 0]

            rel_err = np.linalg.norm(F_doubled - F_truncated) / np.linalg.norm(F_truncated)
            assert rel_err < 0.05, f"r/R={r_over_R}, theta={theta_deg}: rel_err={rel_err}"

            # Attraction toward the cathode, in both modes: a grounded
            # conductor never pushes a charge away from itself.
            assert F_doubled[2] < 0.0, f"r/R={r_over_R}, theta={theta_deg}: repulsive Fz={F_doubled[2]}"
            assert F_truncated[2] < 0.0


def test_mirror_symmetric_image_field_matches_image_force_single_particle():
    """`ImageChargeBEMSolution.image_field`/`.image_potential` had the same
    missing-mirror-excitation gap `_mirror_symmetric_image_force` was fixed
    for (see the two tests above): called directly on a mirror-symmetric
    solution, they only solved for the induced response to the bare source,
    never adding the phantom mirror source's own contribution to either the
    solve's RHS or the direct (non-conductor-mediated) field/potential at
    the query point. Fixed the same way `image_force` was, via a
    `mirror_plane_z` parameter (defaulting to the solution's own
    `.mirror_plane_z`, which `HemisphericalTipBEMGeometry` sets
    automatically in this mode -- see its docstring) that adds both pieces
    back.

    For a single active particle (no other real particles to cross-couple
    with), `image_force`'s joint real+phantom solve and `image_field`'s
    single-source-plus-RHS-correction solve are two independently-coded
    routes to the same physics, so they should agree closely -- this is
    also a regression test for `image_potential_grid`, which calls
    `image_potential` exactly this way (no explicit `mirror_plane_z`) and
    silently returned a wrong answer before this fix.

    Position chosen near the rim (theta=85 deg, off the pole) rather than
    near the tip apex used elsewhere in this file: the phantom mirror
    source sits close to *this* real source only there (both near z~0,
    the mirror plane at z=-z0), which is where the missing-mirror-term bug
    actually bites -- near the apex the real and phantom sources end up
    almost 2R apart, so the "disable the fix" comparison below would pass
    by coincidence (checked directly: <0.5% difference at theta=25 deg,
    where the fix barely matters) rather than actually exercising it."""

    R = 50e-9
    z0 = 3e-9
    geom = _cached_geometry(E_gun=-1e8, R=R, z0=z0, image_n_max=24, image_mirror_symmetric=True)
    sol = geom.image_solution
    assert sol.mirror_plane_z == -z0

    theta = np.radians(85.0)
    phi_particle = 0.6
    Q = -1.602176634e-19
    position = 1.05 * R * np.array(
        [np.sin(theta) * np.cos(phi_particle), np.sin(theta) * np.sin(phi_particle), np.cos(theta)]
    )
    pos3, ch2, act2 = position.reshape(1, 1, 3), np.array([[Q]]), np.array([[True]])

    F_bem = geom.image_force(pos3, ch2, act2, plummer_radius=1e-12)[0, 0]

    d_lo, d_hi = geom.image_d_lo * R, geom.image_d_hi * R
    F_field = Q * sol.image_field(position, Q, d_lo, d_hi, n_max=geom.image_n_max)

    rel_err = np.linalg.norm(F_bem - F_field) / np.linalg.norm(F_bem)
    assert rel_err < 1e-3

    # Disabling the mirror correction (mirror_plane_z=None means "use
    # self.mirror_plane_z", so force it off by clearing that attribute
    # on a shallow copy) must give a substantially different, and
    # therefore wrong, answer -- otherwise this test wouldn't actually be
    # exercising the fix.
    import copy

    sol_unmirrored = copy.copy(sol)
    sol_unmirrored.mirror_plane_z = None
    F_field_unmirrored = Q * sol_unmirrored.image_field(position, Q, d_lo, d_hi, n_max=geom.image_n_max)
    assert np.linalg.norm(F_field_unmirrored - F_bem) / np.linalg.norm(F_bem) > 0.1


def test_image_force_batch_matches_single_particle_image_field():
    """image_force (batched, matching geometry.Geometry's calling
    convention) must agree with charge * image_field for each active
    particle, and leave inactive particles at exactly zero force.

    Deliberately off-axis and off the phi=0 symmetry plane: an exactly
    on-axis (or exactly phi=0) position makes the y (or x/y) component
    mathematically zero by symmetry, and the two code paths round to
    *different* floating-point-noise values around that true zero (checked
    directly: away from these special positions the two paths agree to
    machine precision; a naive component-wise comparison right at them
    fails only because it's effectively comparing two near-zero noise
    floors to each other -- the same "small-scale atol trap" this
    project's own conventions warn about elsewhere).

    n_subdiv=1 explicitly: image_force's field-reconstruction quadrature
    defaults to n_subdiv=4 (finer than image_field's -- effectively
    n_subdiv=1 -- own per-segment quadrature), which is a genuine (if
    tiny, ~1e-8 relative here) refinement, not a bug (checked directly:
    the two match to 0.0 exactly at n_subdiv=1). Pinning n_subdiv=1 here
    keeps this test's job -- confirming N=1 batched reduces to exactly the
    single-particle math -- distinct from a quadrature-refinement check."""
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile

    R = 50e-9
    z0 = 3e-9
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 16, fillet_max_length=z0 / 6)
    n_max = 6
    sol = _cached_solve(profile, n_max=n_max)

    theta = np.radians(25.0)
    phi_particle = 0.6
    active_position = 1.1 * R * np.array(
        [np.sin(theta) * np.cos(phi_particle), np.sin(theta) * np.sin(phi_particle), np.cos(theta)]
    )
    inactive_position = np.array([0.3 * R, 0.1 * R, np.sqrt(R**2 - 0.1**2 * R**2) * 1.05])
    positions = np.array([[active_position, inactive_position]])
    charges = np.array([[-1.0, -1.0]])
    active = np.array([[True, False]])

    force = sol.image_force(positions, charges, active, d_lo=0.1 * R, d_hi=0.5 * R, n_max=n_max, n_subdiv=1)

    assert force.shape == positions.shape
    assert np.all(force[0, 1] == 0.0)

    expected0 = charges[0, 0] * sol.image_field(positions[0, 0], charges[0, 0], 0.1 * R, 0.5 * R, n_max=n_max)
    assert np.linalg.norm(force[0, 0] - expected0) / np.linalg.norm(expected0) < 1e-10


def test_image_force_cross_coupling_matches_exact_flat_plane_multi_image():
    """The whole point of the joint solve: with >1 active particle,
    image_force must include the conductor-mediated cross-term (particle
    j's presence changing the force on particle i), not just each
    particle's own self-image. On a flat plane the exact multi-particle
    answer is `forces.image_charge_force`'s own all-pairs treatment
    (mirror charges only, no BEM) -- with the mirror-charge trick fully
    engaged (near-zero residual, as in the single-particle flat-plane
    tests above) this is close to a pure test of the cross-mirror-charge
    sum, and should match to near machine precision."""
    from specific_particle_tracer.forces import image_charge_force

    profile = _graded_flat_profile(r_max=200.0, r_min=0.01, growth=1.2)
    n_max = 4
    sol = _cached_solve(profile, n_max=n_max)

    # Both particles in the SAME group (n_groups=1, n_emit=2): only
    # particles within a group interact, matching image_charge_force's own
    # (n_groups, n_emit, 3) convention below.
    positions = np.array([[[0.5, 0.2, 0.4], [-0.3, 0.6, 0.7]]])
    charges = np.array([[-1.0, -1.5]])
    active = np.array([[True, True]])

    F_bem = sol.image_force(positions, charges, active, d_lo=1e6, d_hi=1e6 + 1.0, n_max=n_max)[0]
    F_exact = image_charge_force(positions, charges, active, z0=0.0, plummer_radius=1e-12)[0]

    assert np.max(np.abs(F_bem - F_exact) / np.abs(F_exact)) < 1e-10

    # And the cross term must actually matter: computing particle 0 as if
    # it were alone (its own group of one) must NOT match the joint (or
    # exact) answer -- a single charge above an infinite flat plane feels
    # a purely normal force, but particle 1's presence breaks that symmetry.
    F0_alone = sol.image_force(positions[:, :1], charges[:, :1], active[:, :1], d_lo=1e6, d_hi=1e6 + 1.0, n_max=n_max)[0, 0]
    assert F0_alone[0] == 0.0 and F0_alone[1] == 0.0
    assert abs(F_bem[0, 0]) > 0.1 * abs(F_bem[0, 2])


def test_image_force_cross_coupling_matches_exact_hemisphere_tip_multi_image():
    """Same cross-coupling check as the flat-plane version above, but on
    the project's actual (curved, recessed) hemisphere-tip geometry,
    against `forces.hemispherical_tip_image_force`'s own all-pairs exact
    3-image solution -- agreement at the same few-percent level the
    single-particle case already showed (limited by the recessed
    profile's rounded fillet vs. that formula's sharp-ridge idealization,
    not by the joint-solve machinery)."""
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile
    from specific_particle_tracer.forces import hemispherical_tip_image_force

    R = 50e-9
    z0 = 3e-9
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 40, fillet_max_length=z0 / 12)
    n_max = 24
    sol = _cached_solve(profile, n_max=n_max)

    d_lo, d_hi = 0.1 * R, 0.5 * R
    theta1, theta2 = np.radians(10.0), np.radians(25.0)
    p1 = 1.1 * R * np.array([np.sin(theta1), 0.0, np.cos(theta1)])
    p2 = 1.15 * R * np.array([0.0, np.sin(theta2), np.cos(theta2)])
    # One group of two (only particles within a group interact).
    positions = np.stack([p1, p2])[None, :, :]
    charges = np.array([[-1.0, -1.0]])
    active = np.array([[True, True]])

    F_bem = sol.image_force(positions, charges, active, d_lo, d_hi, n_max=n_max)[0]
    F_exact = hemispherical_tip_image_force(positions, charges, active, R - z0, plummer_radius=1e-12, plane_z0=z0)[0]

    rel_err = np.linalg.norm(F_bem - F_exact, axis=-1) / np.linalg.norm(F_exact, axis=-1)
    assert np.max(rel_err) < 0.02

    F0_alone = sol.image_force(positions[:, :1], charges[:, :1], active[:, :1], d_lo, d_hi, n_max=n_max)[0, 0]
    assert np.linalg.norm(F_bem[0] - F0_alone) > 0.1 * np.linalg.norm(F_exact[0])


def test_image_force_gpu_matches_cpu():
    """image_force's whole pipeline (closest-point search, mirror-charge
    RHS, per-mode solve, joint field reconstruction, all-pairs mirror
    cross-term) run with xp=cupy must reproduce the xp=numpy answer to
    near machine precision -- every array op in the method was rewritten
    to go through `xp` for this (previously it forced everything to numpy
    internally, see git history), so this is the end-to-end check that the
    rewrite didn't silently drop back to numpy semantics or mix array
    modules somewhere."""
    cp = pytest.importorskip("cupy")
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile

    R = 50e-9
    z0 = 3e-9
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 16, fillet_max_length=z0 / 6)
    n_max = 8
    sol = _cached_solve(profile, n_max=n_max)

    theta1, theta2 = np.radians(15.0), np.radians(35.0)
    p1 = 1.1 * R * np.array([np.sin(theta1), 0.0, np.cos(theta1)])
    p2 = 1.2 * R * np.array([0.0, np.sin(theta2), np.cos(theta2)])
    positions = np.stack([p1, p2])[None, :, :]
    charges = np.array([[-1.0, -1.3]])
    active = np.array([[True, True]])
    d_lo, d_hi = 0.1 * R, 0.5 * R

    F_cpu = sol.image_force(positions, charges, active, d_lo, d_hi, n_max=n_max, xp=np)
    F_gpu = sol.image_force(
        cp.asarray(positions), cp.asarray(charges), cp.asarray(active), d_lo, d_hi, n_max=n_max, xp=cp
    )

    assert np.max(np.abs(cp.asnumpy(F_gpu) - F_cpu) / np.abs(F_cpu)) < 1e-8


def test_image_force_requires_explicit_group_axis():
    """position must be (n_groups, n_emit, 3), not some other leading batch
    shape -- an earlier version of this method accepted (and silently
    mishandled) any leading shape by flattening it away before solving,
    which broke group isolation (see the next test). A flat (N, 3) array
    should now be rejected outright rather than silently doing the wrong
    thing again."""
    profile = _graded_flat_profile(r_max=200.0, r_min=0.01, growth=1.2)
    sol = _cached_solve(profile, n_max=2)
    positions_flat = np.array([[0.5, 0.2, 0.4], [-0.3, 0.6, 0.7]])  # (2, 3), missing the group axis
    with pytest.raises(ValueError):
        sol.image_force(positions_flat, np.array([-1.0, -1.0]), np.array([True, True]), 0.1, 0.5, n_max=2)


def test_image_force_groups_never_interact():
    """The core architectural invariant this project's whole per-group
    adaptive-stepping design rests on (see tracker.SpecificParticleTracer's
    module docstring and forces._pairwise_force's group-preserving einsum):
    a particle's image-charge force must depend only on the other
    particles in *its own* emission group, never on particles in a
    different group, however physically close they happen to be. Caught a
    real bug this way during development: an earlier version flattened the
    group axis away before doing the joint solve, so a second, unrelated,
    physically distant group changed a first group's own particle's force
    by ~14% instead of by exactly zero."""
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile

    R = 50e-9
    z0 = 3e-9
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 16, fillet_max_length=z0 / 6)
    n_max = 8
    sol = _cached_solve(profile, n_max=n_max)
    d_lo, d_hi = 0.1 * R, 0.5 * R

    theta1 = np.radians(15.0)
    p1 = 1.1 * R * np.array([np.sin(theta1), 0.0, np.cos(theta1)])
    charge1 = -1.0

    positions_alone = p1.reshape(1, 1, 3)
    F_alone = sol.image_force(
        positions_alone, np.array([[charge1]]), np.array([[True]]), d_lo, d_hi, n_max=n_max
    )[0, 0]

    theta_far = np.radians(80.0)
    p_far = 1.1 * R * np.array([np.sin(theta_far), 0.3, np.cos(theta_far)])
    positions_two_groups = np.stack([positions_alone[0], np.array([p_far])], axis=0)  # (2, 1, 3)
    charges_two = np.array([[charge1], [-2.0]])
    active_two = np.array([[True], [True]])
    F_two_groups = sol.image_force(positions_two_groups, charges_two, active_two, d_lo, d_hi, n_max=n_max)

    assert np.array_equal(F_two_groups[0, 0], F_alone)


# ----------------------------------------------------------------------
# image_potential
# ----------------------------------------------------------------------


def test_image_potential_gradient_matches_image_field_at_the_source():
    """image_potential is a decoupled-query generalization of image_field
    (see its docstring): -grad(image_potential) evaluated back at the
    source's own position must reproduce image_field's directly-returned
    E there, since they share the exact same underlying mode solve."""
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile

    R, z0 = 50e-9, 3e-9
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 25, fillet_max_length=z0 / 10)
    sol = _cached_solve(profile, n_max=16)

    theta = np.radians(20.0)
    source = (1.2 * R) * np.array([np.sin(theta), 0.0, np.cos(theta)])
    Q = -1.0
    d_lo, d_hi = 0.1 * R, 0.5 * R

    E_direct = sol.image_field(source, Q, d_lo, d_hi)

    h = 1e-4 * R

    def V(p):
        return sol.image_potential(source, Q, p, d_lo, d_hi)

    grads = []
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = h
        grads.append(-(V(source + dp) - V(source - dp)) / (2.0 * h))
    E_fd = np.array(grads)

    assert np.allclose(E_fd, E_direct, rtol=1e-5)


def test_image_potential_scales_linearly_with_charge():
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile

    R, z0 = 50e-9, 3e-9
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 16, fillet_max_length=z0 / 6)
    sol = _cached_solve(profile, n_max=10)

    source = np.array([0.3 * R, 0.0, 1.1 * R])
    query = np.array([[0.1 * R, 0.0, 1.0 * R], [0.5 * R, 0.2 * R, 0.5 * R]])
    d_lo, d_hi = 0.1 * R, 0.5 * R

    V1 = sol.image_potential(source, 1.0, query, d_lo, d_hi)
    V3 = sol.image_potential(source, 3.0, query, d_lo, d_hi)
    assert np.allclose(V3, 3.0 * V1, rtol=1e-10)


def test_image_potential_query_shape_is_preserved():
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile

    R, z0 = 50e-9, 3e-9
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 16, fillet_max_length=z0 / 6)
    sol = _cached_solve(profile, n_max=10)

    source = np.array([0.3 * R, 0.0, 1.1 * R])
    d_lo, d_hi = 0.1 * R, 0.5 * R

    r = np.linspace(0.01 * R, 2 * R, 5)
    z = np.linspace(0.0, 2 * R, 7)
    Rg, Zg = np.meshgrid(r, z)
    query = np.stack([Rg, np.zeros_like(Rg), Zg], axis=-1)  # (7, 5, 3)

    V = sol.image_potential(source, -1.0, query, d_lo, d_hi)
    assert V.shape == (7, 5)


def test_image_potential_masks_against_real_profile_when_given():
    """Without a real_profile, image_potential masks against this
    solution's own recessed surface -- but a query point between the
    recessed and real surfaces is physically inside the real conductor,
    so passing the real profile should mask it too (see the class
    docstring's `real_profile` parameter)."""
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile, hemisphere_tip_real_profile

    R, z0 = 50e-9, 3e-9
    image_profile = hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 25, fillet_max_length=z0 / 10)
    real_profile = hemisphere_tip_real_profile(R, 5 * R, max_length=R / 25)

    sol_no_real = _cached_solve(image_profile, n_max=10)
    sol_with_real = _cached_solve(image_profile, n_max=10, real_profile=real_profile)

    source = np.array([0.3 * R, 0.0, 1.1 * R])
    d_lo, d_hi = 0.1 * R, 0.5 * R

    # Just inside the real dome (r < R) but outside the recessed cap
    # (r > R - z0) -- real material, not vacuum.
    theta = np.radians(40.0)
    point_in_shell = 0.99 * R * np.array([np.sin(theta), 0.0, np.cos(theta)])

    V_no_real = sol_no_real.image_potential(source, -1.0, point_in_shell, d_lo, d_hi)
    V_with_real = sol_with_real.image_potential(source, -1.0, point_in_shell, d_lo, d_hi)

    assert V_no_real != 0.0  # old behavior: incorrectly left unmasked
    assert V_with_real == 0.0  # masked against the real surface
