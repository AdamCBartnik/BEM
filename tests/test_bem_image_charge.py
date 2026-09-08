import numpy as np
import pytest

from specific_particle_tracer.bem.image_charge import (
    ImageChargeBEMSolution,
    image_charge_weight,
    mirror_point,
)
from specific_particle_tracer.constants import VACUUM_PERMITTIVITY


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

    sol = ImageChargeBEMSolution.solve(profile, n_max=0)
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

    sol = ImageChargeBEMSolution.solve(profile, n_max=0)
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
        sol = ImageChargeBEMSolution.solve(profile, n_max=n_max)
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
    sol = ImageChargeBEMSolution.solve(profile, n_max=n_max)
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
    profile = hemisphere_tip_image_profile(R, z0, plane_radius, n_theta=45, n_fillet=15, n_r=20)

    n_max = 24
    sol = ImageChargeBEMSolution.solve(profile, n_max=n_max)

    Q = -1.0
    d_lo, d_hi = 0.1 * R, 0.5 * R

    for theta_deg, d_over_R in [(0.0, 0.1), (0.0, 0.3), (10.0, 0.1), (30.0, 0.1)]:
        theta = np.radians(theta_deg)
        direction = np.array([np.sin(theta), 0.0, np.cos(theta)])
        position = (R + d_over_R * R) * direction

        F_bem = Q * sol.image_field(position, Q, d_lo, d_hi, n_max=n_max)

        F_exact = hemispherical_tip_image_force(
            position.reshape(1, 1, 3), np.array([[Q]]), np.array([[True]]), R - z0, plummer_radius=1e-12
        )[0, 0]

        rel_err = np.linalg.norm(F_bem - F_exact) / np.linalg.norm(F_exact)
        assert rel_err < 0.02


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
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, n_theta=20, n_fillet=8, n_r=10)
    n_max = 6
    sol = ImageChargeBEMSolution.solve(profile, n_max=n_max)

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
    sol = ImageChargeBEMSolution.solve(profile, n_max=n_max)

    positions = np.array([[0.5, 0.2, 0.4], [-0.3, 0.6, 0.7]])
    charges = np.array([-1.0, -1.5])
    active = np.array([True, True])

    F_bem = sol.image_force(positions, charges, active, d_lo=1e6, d_hi=1e6 + 1.0, n_max=n_max)
    F_exact = image_charge_force(positions[None, :, :], charges[None, :], active[None, :], z0=0.0, plummer_radius=1e-12)[0]

    assert np.max(np.abs(F_bem - F_exact) / np.abs(F_exact)) < 1e-10

    # And the cross term must actually matter: computing particle 0 as if
    # it were alone must NOT match the joint (or exact) answer -- a single
    # charge above an infinite flat plane feels a purely normal force, but
    # particle 1's presence breaks that symmetry.
    F0_alone = sol.image_force(positions[:1], charges[:1], active[:1], d_lo=1e6, d_hi=1e6 + 1.0, n_max=n_max)[0]
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
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, n_theta=45, n_fillet=15, n_r=20)
    n_max = 24
    sol = ImageChargeBEMSolution.solve(profile, n_max=n_max)

    d_lo, d_hi = 0.1 * R, 0.5 * R
    theta1, theta2 = np.radians(10.0), np.radians(25.0)
    p1 = 1.1 * R * np.array([np.sin(theta1), 0.0, np.cos(theta1)])
    p2 = 1.15 * R * np.array([0.0, np.sin(theta2), np.cos(theta2)])
    positions = np.stack([p1, p2])
    charges = np.array([-1.0, -1.0])
    active = np.array([True, True])

    F_bem = sol.image_force(positions, charges, active, d_lo, d_hi, n_max=n_max)
    F_exact = hemispherical_tip_image_force(
        positions[None, :, :], charges[None, :], active[None, :], R - z0, plummer_radius=1e-12
    )[0]

    rel_err = np.linalg.norm(F_bem - F_exact, axis=-1) / np.linalg.norm(F_exact, axis=-1)
    assert np.max(rel_err) < 0.02

    F0_alone = sol.image_force(positions[:1], charges[:1], active[:1], d_lo, d_hi, n_max=n_max)[0]
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
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, n_theta=20, n_fillet=8, n_r=10)
    n_max = 8
    sol = ImageChargeBEMSolution.solve(profile, n_max=n_max)

    theta1, theta2 = np.radians(15.0), np.radians(35.0)
    p1 = 1.1 * R * np.array([np.sin(theta1), 0.0, np.cos(theta1)])
    p2 = 1.2 * R * np.array([0.0, np.sin(theta2), np.cos(theta2)])
    positions = np.stack([p1, p2])
    charges = np.array([-1.0, -1.3])
    active = np.array([True, True])
    d_lo, d_hi = 0.1 * R, 0.5 * R

    F_cpu = sol.image_force(positions, charges, active, d_lo, d_hi, n_max=n_max, xp=np)
    F_gpu = sol.image_force(
        cp.asarray(positions), cp.asarray(charges), cp.asarray(active), d_lo, d_hi, n_max=n_max, xp=cp
    )

    assert np.max(np.abs(cp.asnumpy(F_gpu) - F_cpu) / np.abs(F_cpu)) < 1e-8
