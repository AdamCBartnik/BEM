import numpy as np

from specific_particle_tracer.bem.mesh import sphere_mesh
from specific_particle_tracer.bem.panel_field import evaluate_panel_field


def test_uniform_charge_density_gives_the_coulomb_field_of_its_total_charge():
    """A uniform single-layer (charge) density t0 over a closed sphere is,
    from outside, indistinguishable from a point charge equal to the
    total charge t0 * surface_area -- the basic sanity check for the
    Coulomb-law term."""
    R = 50e-9
    t0 = 3.0
    vertices, elements = sphere_mesh(R, n_theta=30, n_phi=36)
    g_nodal = np.zeros(vertices.shape[1])
    t_nodal = np.full(vertices.shape[1], t0)

    r = 3.0 * R
    point = np.array([[0.0, 0.0, r]])
    E = evaluate_panel_field(vertices, elements, g_nodal, t_nodal, point)

    total_charge = t0 * 4.0 * np.pi * R**2
    E_expected = total_charge / (4.0 * np.pi * r**2)
    assert abs(E[0, 2] - E_expected) / E_expected < 1e-2
    assert np.allclose(E[0, :2], 0.0, atol=1e-6 * E_expected)


def test_uniform_dipole_density_gives_exactly_zero_field_outside_a_closed_surface():
    """A uniform double-layer (dipole) density over a closed surface has an
    exactly constant (zero, from outside) potential everywhere exterior --
    the standard solid-angle identity -- so its field must vanish exactly,
    for any g0. `evaluate_panel_field` shifts g by a reference value before
    integrating specifically to make this identity exact rather than
    merely approximate (see the module docstring); for a uniform g that
    reference IS g itself, so this case is almost tautological -- the real
    regression coverage for the double-layer term's correctness (a mesh-
    winding bug and a sign error, both found and fixed during development)
    is the non-uniform, physically solved case in test_bem_laplace.py,
    which this test can't substitute for."""
    R = 50e-9
    g0 = 5.0
    vertices, elements = sphere_mesh(R, n_theta=30, n_phi=36)
    g_nodal = np.full(vertices.shape[1], g0)
    t_nodal = np.zeros(vertices.shape[1])

    point = np.array([[0.0, 0.0, 3.0 * R], [1.5 * R, 0.0, 2.0 * R]])
    E = evaluate_panel_field(vertices, elements, g_nodal, t_nodal, point)

    assert np.allclose(E, 0.0, atol=1e-9)
