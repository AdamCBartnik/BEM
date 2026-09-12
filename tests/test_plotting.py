import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from specific_particle_tracer.fields import GunField  # noqa: E402
from specific_particle_tracer.bem.mesh import hemisphere_tip_image_profile  # noqa: E402
from specific_particle_tracer.bem.image_charge import ImageChargeBEMSolution  # noqa: E402
from specific_particle_tracer.plotting import (  # noqa: E402
    plot_profiles,
    static_potential_grid,
    image_potential_grid,
)

R = 50e-9
EZ = -1e8


def test_static_potential_grid_shape_and_conductor_value():
    field = GunField(EZ)
    R_, Z_, V = static_potential_grid(field, r_max=2 * R, z_range=(-R, 2 * R), n_r=10, n_z=12)
    assert R_.shape == Z_.shape == V.shape == (12, 10)
    # Below z=0 is inside the (grounded) conductor: exactly 0 V.
    assert np.all(V[Z_ < 0.0] == 0.0)
    # Above it, matches the analytic Phi = -Ez*z directly.
    assert np.allclose(V[Z_ >= 0.0], -EZ * Z_[Z_ >= 0.0])


def test_image_potential_grid_shape_and_linearity():
    profile = hemisphere_tip_image_profile(R, 3e-9, 5 * R, max_length=R / 16, fillet_max_length=3e-9 / 6)
    sol = ImageChargeBEMSolution.solve(profile, n_max=10)

    R_, Z_, V1 = image_potential_grid(sol, source_r0=0.3 * R, source_z0=1.1 * R, r_max=2 * R, z_range=(0, 2 * R), charge=1.0, n_r=8, n_z=9)
    _, _, V2 = image_potential_grid(sol, source_r0=0.3 * R, source_z0=1.1 * R, r_max=2 * R, z_range=(0, 2 * R), charge=2.0, n_r=8, n_z=9)
    assert R_.shape == Z_.shape == V1.shape == (9, 8)
    assert np.allclose(V2, 2.0 * V1, rtol=1e-10)


def test_plot_profiles_runs_without_error():
    fig, ax = plt.subplots()
    real = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]])
    image = np.array([[0.0, -0.1], [1.0, -0.1], [2.0, 0.3]])
    plot_profiles(ax, real, image)
    plt.close(fig)


def test_plot_profiles_handles_only_one_profile():
    fig, ax = plt.subplots()
    real = np.array([[0.0, 0.0], [1.0, 0.0]])
    plot_profiles(ax, real_profile=real, show_nodes=False)
    plt.close(fig)
