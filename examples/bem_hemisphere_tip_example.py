"""Example: the axisymmetric BEM solver for a hemispherical-tip cathode.

Three parts:
1. Solve the field with bem.fields.HemisphericalTipBEMField and compare it
   directly to the closed-form fields.HemisphericalTipField.
2. Push a handful of particles (zero initial velocity, no image-charge or
   Coulomb force) from the tip to a far screen with the actual tracker,
   using bem.geometry.HemisphericalTipBEMGeometry -- this is this
   project's own convergence check: with those forces off, every particle
   should gain nearly the same energy, since the screen sits close to an
   equipotential of the (mostly uniform, far from the tip) asymptotic
   field. See specific_particle_tracer/bem/*.py for the physics and the
   numerical story behind this solver (indirect/charge-simulation BEM,
   reduced to a 1D axisymmetric problem via elliptic integrals, fully
   vectorized -- runs on numpy or, for large particle counts, cupy).
3. Turn the image-charge force on (bem.geometry.HemisphericalTipBEMGeometry's
   `z0` argument) and compare its BEM-computed image force -- an azimuthal-
   Fourier-mode BEM solve on the recessed image surface, see
   bem/image_charge.py's module docstring -- against this project's exact
   closed-form 3-image solution for this same hemisphere-on-plane shape
   (forces.hemispherical_tip_image_force), first for one particle and
   then for two at once (exercising the joint multi-particle solve's
   conductor-mediated cross-coupling, since hemispherical_tip_image_force
   includes that too via its own all-pairs treatment). This shape has an
   exact solution already, so the BEM path isn't the better choice *here*
   -- this comparison exists to validate the general (image-force-capable)
   machinery a genuinely non-spherical tip shape would actually need.
"""

import time

import numpy as np

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

from specific_particle_tracer.constants import ELEMENTARY_CHARGE, ev_c_to_si_momentum
from specific_particle_tracer.distributions import flat_distribution_to_hemisphere
from specific_particle_tracer.fields import HemisphericalTipField
from specific_particle_tracer.geometry import HemisphericalTip
from specific_particle_tracer.bem.fields import HemisphericalTipBEMField
from specific_particle_tracer.bem.geometry import HemisphericalTipBEMGeometry
from specific_particle_tracer.tracker import SpecificParticleTracer

R = 50e-9  # tip radius [m]
E_gun = -1e8  # asymptotic field [V/m] (negative: accelerates electrons in +z)

# --- Part 1: field values, BEM vs. the closed-form solution ---------------

bem_field = HemisphericalTipBEMField(E_gun, R, n_theta=40)  # xp=np by default; pass xp=cupy for GPU
analytic_field = HemisphericalTipField(E_gun, R)

theta = np.radians([10, 30, 50, 70])  # angle from the pole
points = np.column_stack([1.5 * R * np.sin(theta), np.zeros_like(theta), 1.5 * R * np.cos(theta)])

E_bem = bem_field.evaluate(points)
E_analytic = analytic_field.evaluate(points)
print("Field at r=1.5R, a few angles from the pole:")
for th, eb, ea in zip(np.degrees(theta), E_bem, E_analytic):
    print(f"  theta={th:5.1f} deg   BEM Ez={eb[2]:.4e} V/m   analytic Ez={ea[2]:.4e} V/m")

# --- Part 2: push particles from the tip to a screen -----------------------

N = 20
z_screen = 3.0 * R

# Fibonacci-disk sampling within radius 0.85R, mapped onto the hemisphere
# (flat_distribution_to_hemisphere projects (x, y, 0) straight up onto the
# dome and rotates momentum to match -- see its docstring).
golden_angle = np.pi * (3.0 - np.sqrt(5.0))
i = np.arange(N)
r_flat = 0.85 * R * np.sqrt((i + 0.5) / N)
phi_flat = golden_angle * i
x, y = r_flat * np.cos(phi_flat), r_flat * np.sin(phi_flat)

flat_particles = ParticleGroup(
    data=dict(
        x=x, y=y, z=np.zeros(N),
        px=np.zeros(N), py=np.zeros(N), pz=np.zeros(N),  # start at rest
        t=np.zeros(N),
        weight=np.full(N, ELEMENTARY_CHARGE),
        status=np.ones(N),
        id=np.arange(N),
        species="electron",
    )
)
particles = flat_distribution_to_hemisphere(flat_particles, R)


def energy_ev(screen, mass_kg):
    px = ev_c_to_si_momentum(np.asarray(screen.px))
    py = ev_c_to_si_momentum(np.asarray(screen.py))
    pz = ev_c_to_si_momentum(np.asarray(screen.pz))
    return (px**2 + py**2 + pz**2) / (2.0 * mass_kg) / ELEMENTARY_CHARGE


def run(label, geometry, **tracker_kwargs):
    t0 = time.time()
    tracer = SpecificParticleTracer(
        particles, n_emit=N, geometry=geometry, screens=[z_screen], z_max=1.2 * z_screen, **tracker_kwargs
    )
    screens, _ = tracer.run()
    ke = energy_ev(screens[0], tracer.mass)
    print(
        f"{label}: {time.time() - t0:.2f}s, {len(screens[0])}/{N} arrived, "
        f"KE mean={ke.mean():.4f} eV, spread={100 * ke.std() / ke.mean():.3f}%"
    )


print("\nPushing particles to a screen at z=3R (no image-charge/Coulomb force):")
run("analytic", HemisphericalTip(E_gun, R, z0=None))
run("BEM (cpu)", HemisphericalTipBEMGeometry(E_gun, R, n_theta=40))

# For a GPU run (worth it once you have hundreds-to-thousands of particles
# or a fine mesh -- see bem/axisymmetric.py's module docstring for the
# crossover point measured on this project's own hardware):
#
#     run("BEM (gpu)", HemisphericalTipBEMGeometry(E_gun, R, n_theta=40), backend="gpu")

# --- Part 3: image-charge force, BEM vs. the exact 3-image analytic form --

from specific_particle_tracer.forces import hemispherical_tip_image_force  # noqa: E402
from specific_particle_tracer.geometry import DEFAULT_Z0  # noqa: E402

print(f"\nBuilding the image-charge BEM solve (z0={DEFAULT_Z0:.1e} m) -- this is the")
print("expensive one-time step (a per-mode adaptive-quadrature operator")
print("assembly, not a cheap linear solve): tens of seconds to a few minutes")
print("depending on image_n_theta/n_fillet/n_r.")

bem_geom_with_image = HemisphericalTipBEMGeometry(
    E_gun, R, n_theta=40, z0=DEFAULT_Z0, image_n_theta=30, image_n_fillet=12, image_n_r=15, image_n_max=16
)

print("\nImage-charge force, BEM vs. exact 3-image analytic (this shape has an")
print("exact solution already -- this is a validation, not a use case). One")
print("particle at a time first:")
for theta_deg, d_over_R in [(5.0, 0.1), (20.0, 0.1)]:
    theta = np.radians(theta_deg)
    position = (1 + d_over_R) * R * np.array([np.sin(theta), 0.0, np.cos(theta)])
    charge = np.array([-ELEMENTARY_CHARGE])
    active = np.array([True])

    F_bem = bem_geom_with_image.image_force(position.reshape(1, 3), charge, active, plummer_radius=1e-12)[0]
    F_exact = hemispherical_tip_image_force(
        position.reshape(1, 1, 3), charge.reshape(1, 1), active.reshape(1, 1), R - DEFAULT_Z0, plummer_radius=1e-12
    )[0, 0]

    rel_err = np.linalg.norm(F_bem - F_exact) / np.linalg.norm(F_exact)
    print(f"  theta={theta_deg:5.1f} deg  d/R={d_over_R:.2f}   |F_bem|={np.linalg.norm(F_bem):.4e} N   rel_err={rel_err:.2%}")

print("\nNow two particles at once: bem.image_charge.image_force solves a")
print("*joint* problem, including the conductor-mediated cross-term (one")
print("particle's presence changing the induced-charge force on the other)")
print("-- not just each particle's own self-image. hemispherical_tip_image_force")
print("includes that cross-term too (its own all-pairs treatment), so this is")
print("still an apples-to-apples comparison, just with N=2 this time.")
theta_pair = np.radians([5.0, 20.0])
positions_pair = 1.1 * R * np.column_stack([np.sin(theta_pair), np.zeros(2), np.cos(theta_pair)])
charges_pair = np.full(2, -ELEMENTARY_CHARGE)
active_pair = np.ones(2, dtype=bool)

F_bem_pair = bem_geom_with_image.image_force(positions_pair, charges_pair, active_pair, plummer_radius=1e-12)
F_exact_pair = hemispherical_tip_image_force(
    positions_pair[None, :, :], charges_pair[None, :], active_pair[None, :], R - DEFAULT_Z0, plummer_radius=1e-12
)[0]
for i in range(2):
    rel_err = np.linalg.norm(F_bem_pair[i] - F_exact_pair[i]) / np.linalg.norm(F_exact_pair[i])
    print(f"  particle {i}: |F_bem|={np.linalg.norm(F_bem_pair[i]):.4e} N   rel_err={rel_err:.2%}")

# image_force (unlike the one-time operator assembly above) runs on
# whatever backend the Geometry was built with, including custom
# RawKernels for its two GPU-hostile Python loops -- see
# bem/image_charge.py's module docstring for the measured crossover point
# (worth a few hundred to a few thousand simultaneously active particles
# before GPU actually wins over CPU for this specific operation):
#
#     bem_geom_gpu = HemisphericalTipBEMGeometry(
#         E_gun, R, n_theta=40, z0=DEFAULT_Z0, image_n_max=16, xp=cupy
#     )
#     force_gpu = bem_geom_gpu.image_force(positions_pair, charges_pair, active_pair, plummer_radius=1e-12)
