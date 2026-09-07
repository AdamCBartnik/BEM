import numpy as np

from specific_particle_tracer.bem.mesh import sphere_mesh
from specific_particle_tracer.bem.panel_field import evaluate_coulomb_field


def test_uniform_charge_density_gives_the_coulomb_field_of_its_total_charge():
    """A uniform surface charge density sigma0 over a closed sphere is,
    from outside, indistinguishable from a point charge equal to the
    total charge sigma0 * surface_area -- the basic sanity check for this
    module's Coulomb-law panel integration."""
    R = 50e-9
    sigma0 = 3.0
    vertices, elements = sphere_mesh(R, n_theta=30, n_phi=36)
    sigma_nodal = np.full(vertices.shape[1], sigma0)

    r = 3.0 * R
    point = np.array([[0.0, 0.0, r]])
    E = evaluate_coulomb_field(vertices, elements, sigma_nodal, point)

    total_charge = sigma0 * 4.0 * np.pi * R**2
    E_expected = total_charge / (4.0 * np.pi * r**2)
    assert abs(E[0, 2] - E_expected) / E_expected < 1e-2
    assert np.allclose(E[0, :2], 0.0, atol=1e-6 * E_expected)
