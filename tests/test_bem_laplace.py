import numpy as np

from specific_particle_tracer.bem.mesh import sphere_mesh
from specific_particle_tracer.bem.laplace import ExteriorLaplaceSolution


def test_exterior_dirichlet_solve_matches_grounded_sphere_in_uniform_field():
    """Grounded sphere of radius a in a uniform asymptotic field E0*z_hat:
    the classic closed-form case. Perturbation potential phi_pert =
    phi_total - phi_inf = E0*a^3*z/r^3; Dirichlet data forcing phi_total=0
    on the sphere is g = -phi_inf = E0*z.

    This exercises the general 3D (non-axisymmetric) machinery directly --
    see bem.axisymmetric for the much cheaper, more accurate path this
    project actually uses for axisymmetric shapes like the hemispherical
    tip (bem.fields.HemisphericalTipBEMField)."""
    a = 50e-9
    E0 = -1e8

    vertices, elements = sphere_mesh(a, n_theta=40, n_phi=48)
    solution = ExteriorLaplaceSolution.solve(vertices, elements, dirichlet_fn=lambda x, y, z: E0 * z)

    theta = np.array([0.0, 0.3, 0.9, 1.5, 2.5, np.pi])
    r = 3.0 * a
    points = np.column_stack([r * np.sin(theta), np.zeros_like(theta), r * np.cos(theta)])

    phi_bem = solution.potential(points)
    phi_analytic = E0 * a**3 * points[:, 2] / r**3

    assert np.max(np.abs(phi_bem - phi_analytic)) / np.abs(E0 * a) < 1e-2
