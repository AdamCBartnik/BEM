import numpy as np

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
    particle, and leave inactive particles at exactly zero force."""
    from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile

    R = 50e-9
    z0 = 3e-9
    profile = hemisphere_tip_image_profile(R, z0, 5 * R, n_theta=20, n_fillet=8, n_r=10)
    n_max = 6
    sol = ImageChargeBEMSolution.solve(profile, n_max=n_max)

    positions = np.array(
        [[[0.0, 0.0, 1.2 * R], [0.3 * R, 0.1 * R, np.sqrt(R**2 - 0.1**2 * R**2) * 1.05]]]
    )
    charges = np.array([[-1.0, -1.0]])
    active = np.array([[True, False]])

    force = sol.image_force(positions, charges, active, d_lo=0.1 * R, d_hi=0.5 * R, n_max=n_max)

    assert force.shape == positions.shape
    assert np.all(force[0, 1] == 0.0)

    expected0 = charges[0, 0] * sol.image_field(positions[0, 0], charges[0, 0], 0.1 * R, 0.5 * R, n_max=n_max)
    assert np.allclose(force[0, 0], expected0)
