"""Example: visualizing meshes and potentials with
specific_particle_tracer.plotting.

That module is deliberately data-only for the two potential grids
(`static_potential_grid`/`image_potential_grid` return plain numpy
`(R, Z, V)` arrays, no matplotlib import at all) -- plot styling
(colormap, contour levels, colorbar, log scale or not) is exactly the
kind of detail worth owning in your own code rather than fighting
someone else's defaults for. `plot_profiles` is kept as a small
matplotlib convenience since it has no such judgment calls of its own
(it just draws given line data).

Requires matplotlib (`pip install -e .[plot]`) to run this script.
Produces three PNGs in the current directory: profiles.png,
static_potential.png, image_potential.png.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from specific_particle_tracer.constants import ELEMENTARY_CHARGE
from specific_particle_tracer.fields import GunField, HemisphericalTipField
from specific_particle_tracer.bem.fields import HemisphericalTipBEMField, CylindricalWellBEMField
from specific_particle_tracer.bem.mesh import (
    hemisphere_tip_real_profile,
    hemisphere_tip_image_profile,
    cylindrical_well_real_profile,
    cylindrical_well_image_profile,
)
from specific_particle_tracer.bem.image_charge import ImageChargeBEMSolution
from specific_particle_tracer.plotting import plot_profiles, static_potential_grid, image_potential_grid

R = 50e-9  # tip/well radius [m]
H = 5e-9  # well depth [m]
z0 = 3e-9  # image-charge offset [m]
E_gun = -1e8  # asymptotic field [V/m]

# --- 1. (r, z) mesh/profile plots -------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

# BEM hemisphere tip: the actual discretization used by a solve.
htbf = HemisphericalTipBEMField(E_gun, R)
# real_profile is passed only so image_potential's masking (see its
# docstring) uses the real conductor boundary rather than this recessed
# one -- no real particle occupies the thin sliver in between.
htbf_image_solution = ImageChargeBEMSolution.solve(
    hemisphere_tip_image_profile(R, z0, 5 * R, max_length=R / 25, fillet_max_length=z0 / 10),
    n_max=16,
    real_profile=hemisphere_tip_real_profile(R, 5 * R, max_length=R / 25),
)
plot_profiles(axes[0], htbf.solution.profile, htbf_image_solution.profile)
axes[0].set_title("BEM hemisphere tip (actual mesh nodes)")

# BEM cylindrical well.
cwbf = CylindricalWellBEMField(E_gun, R, H, max_length=min(R, H) / 24)
well_image_solution = ImageChargeBEMSolution.solve(
    cylindrical_well_image_profile(R, H, z0, 5 * R, max_length=min(R, H) / 14, fillet_max_length=z0 / 10), n_max=16
)
plot_profiles(axes[1], cwbf.solution.profile, well_image_solution.profile)
axes[1].set_title("BEM cylindrical well (actual mesh nodes)")
# H << R here, so an equal-aspect view over the full outer plane makes the
# well itself look like a hairline -- zoom in if you actually want to see
# its shape:
#     axes[1].set_xlim(0, 2 * R); axes[1].set_ylim(-2 * H, 0.5 * H)

# Analytic hemisphere tip: no solver mesh exists at all -- the reasonable
# default is the SAME geometry-only profile functions, at high resolution
# (they're just curves here, not a discretization to inspect), no nodes.
real_analytic = hemisphere_tip_real_profile(R, 3 * R, max_length=R / 300)
image_analytic = hemisphere_tip_image_profile(R, z0, 3 * R, max_length=R / 300, fillet_max_length=z0 / 150)
plot_profiles(axes[2], real_analytic, image_analytic, show_nodes=False)
axes[2].set_title("analytic hemisphere tip (geometry only)")

fig.savefig("profiles.png", dpi=120)
print("wrote profiles.png")

# --- 2. static potential density plots ---------------------------------


def plot_density(ax, R, Z, V, title, cmap="RdBu_r", levels=40):
    cf = ax.contourf(R, Z, V, levels=levels, cmap=cmap)
    ax.figure.colorbar(cf, ax=ax, label="V [V]")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.set_title(title)
    return cf


fig, axes = plt.subplots(2, 2, figsize=(10, 10))

htf = HemisphericalTipField(E_gun, R)
plot_density(axes[0, 0], *static_potential_grid(htf, r_max=3 * R, z_range=(-R, 3 * R)), "hemisphere tip (analytic)")
plot_density(axes[0, 1], *static_potential_grid(htbf, r_max=3 * R, z_range=(-R, 3 * R)), "hemisphere tip (BEM)")

gf = GunField(E_gun)
plot_density(axes[1, 0], *static_potential_grid(gf, r_max=3 * R, z_range=(-R, 3 * R)), "flat cathode (analytic)")
plot_density(axes[1, 1], *static_potential_grid(cwbf, r_max=2 * R, z_range=(-2 * H, 2 * R)), "cylindrical well (BEM)")

fig.tight_layout()
fig.savefig("static_potential.png", dpi=120)
print("wrote static_potential.png")

# Just the numbers, no figure at all -- e.g. to check a specific value:
R_grid, Z_grid, V_grid = static_potential_grid(htbf, r_max=3 * R, z_range=(-R, 3 * R), n_r=60, n_z=60)
print(f"\nstatic potential at r~0, z~1.5R: {V_grid[np.argmin(np.abs(Z_grid[:, 0] - 1.5 * R)), 0]:.3f} V")

# --- 3. image potential density plot ------------------------------------

# A single electron at (r0, phi=0, z0) = (0.3R, 0, 1.1R) -- see
# ImageChargeBEMSolution.image_potential's docstring: unlike the static
# potential above, this is NOT axisymmetric (it depends on query_phi too).
#
# This potential diverges like 1/distance approaching the source itself,
# which washes out a linear color scale -- a log-scaled (SymLogNorm) map
# handles that, but building its levels needs a little care: only include
# a sign's worth of levels if the data actually has values of that sign,
# or half the colorbar goes to a color that never appears (the bug that
# prompted dropping this project's own plot_image_potential wrapper).


def symlog_levels(V, linthresh, n_levels=40):
    vmax = max(abs(V.min()), abs(V.max()), linthresh * 10.0)
    pos = np.geomspace(linthresh, vmax, max(n_levels // 2, 2))
    levels = [0.0]
    if V.max() > 0.0:
        levels = list(pos) + levels
    if V.min() < 0.0:
        levels = levels + list(-pos[::-1])
    return np.unique(levels)


def plot_image_density(ax, R, Z, V, title, source_r, source_z, linthresh=1e-3):
    levels = symlog_levels(V, linthresh)
    norm = SymLogNorm(linthresh=linthresh, vmin=V.min(), vmax=V.max())
    cf = ax.contourf(R, Z, V, levels=levels, cmap="RdBu_r", norm=norm)
    ax.figure.colorbar(cf, ax=ax, label="V [V]")
    ax.plot([source_r], [source_z], marker="*", color="k", ms=14, ls="none", label="source charge")
    ax.legend()
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    ax.set_title(title)
    return cf


source_r0, source_z0 = 0.3 * R, 1.1 * R

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
grid0 = image_potential_grid(
    htbf_image_solution, source_r0=source_r0, source_z0=source_z0,
    r_max=2 * R, z_range=(0, 2 * R), charge=-ELEMENTARY_CHARGE, query_phi=0.0,
)
plot_image_density(axes[0], *grid0, "image potential, query_phi=0 (source's own plane)", source_r0, source_z0)

grid_half_pi = image_potential_grid(
    htbf_image_solution, source_r0=source_r0, source_z0=source_z0,
    r_max=2 * R, z_range=(0, 2 * R), charge=-ELEMENTARY_CHARGE, query_phi=np.pi / 2,
)
ax = axes[1]
levels = symlog_levels(grid_half_pi[2], 1e-3)
norm = SymLogNorm(linthresh=1e-3, vmin=grid_half_pi[2].min(), vmax=grid_half_pi[2].max())
cf = ax.contourf(*grid_half_pi, levels=levels, cmap="RdBu_r", norm=norm)
ax.figure.colorbar(cf, ax=ax, label="V [V]")
ax.set_xlabel("r [m]")
ax.set_ylabel("z [m]")
ax.set_aspect("equal")
ax.set_title("image potential, query_phi=pi/2")
# (the source itself isn't in this slice at all -- it sits at query_phi=0
# -- so there's no marker to draw here)

fig.tight_layout()
fig.savefig("image_potential.png", dpi=120)
print("wrote image_potential.png")

# Just the numbers again:
R_grid, Z_grid, V_grid = image_potential_grid(
    htbf_image_solution, source_r0=source_r0, source_z0=source_z0,
    r_max=2 * R, z_range=(0, 2 * R), charge=-ELEMENTARY_CHARGE, n_r=60, n_z=60,
)
print(f"image potential at r~0.3R, z~1.0R (query_phi=0): {V_grid[np.argmin(np.abs(Z_grid[:,0]-1.0*R)), np.argmin(np.abs(R_grid[0,:]-0.3*R))]:.4f} V")
