"""Example: the axisymmetric BEM solver for a cylindrical-well cathode.

Unlike `examples/bem_hemisphere_tip_example.py`'s hemispherical tip, a
cylindrical well has no closed-form solution to compare against -- see
`bem.mesh`'s and `bem.fields`'s module docstrings for why a *depression*
can't reuse the tip's exact "mirror through z=0, never mesh the plane"
trick (a depression's mirror image overlaps its own real domain instead of
tiling space), so this shape's field solve genuinely needs an open,
truncated-plane profile, same as the image-charge solves already used
elsewhere in this project.

With no analytic ground truth, this example leans on the two internal
diagnostics this project already trusts:

1. Convergence under mesh refinement (does the answer stabilize as
   n_bottom/n_wall/n_r grow, and as the truncation radius grows?) --
   see tests/test_bem_well.py for the actual convergence checks; here we
   just show the numbers once.
2. The uniform-emitted-energy screen check, but in an even cleaner form
   than the hemisphere-tip case: since the *entire* well surface (bottom,
   wall) and the surrounding plane are all one grounded (V=0) conductor,
   energy conservation alone guarantees every particle emitted at rest
   from anywhere on that surface gains exactly the same kinetic energy by
   the time it reaches any shared, sufficiently-equipotential screen --
   regardless of how complicated the field looks near the well itself.
   Any spread measured here is purely numerical error, not physics.

Also demonstrated: the field at the well bottom can point *opposite* the
external field for a deep-enough well (confirmed directly, not a bug --
see tests/test_bem_well.py's `test_field_can_reverse_sign_deep_in_a_well_
and_is_mesh_stable_there`), and the far-field handoff that blends BEM back
to a plain flat cathode beyond `z_handoff` (default 20*max(R, H)) so a
particle far downstream doesn't keep paying for a full BEM evaluation.
"""

import time

import numpy as np

try:
    from beamphysics import ParticleGroup
except ImportError:
    from pmd_beamphysics import ParticleGroup

from specific_particle_tracer.constants import ELEMENTARY_CHARGE, ev_c_to_si_momentum
from specific_particle_tracer.bem.fields import CylindricalWellBEMField
from specific_particle_tracer.bem.geometry import CylindricalWellBEMGeometry
from specific_particle_tracer.tracker import SpecificParticleTracer

R = 50e-9  # well radius [m]
H = 5e-9  # well depth [m] -- shallow enough (H/R=0.1) that every emission
# point on the bottom keeps the same field sign as the asymptotic field;
# see Part 1 below for what happens at H/R past ~0.35-0.4, where deeper
# points reverse and an electron emitted at rest there is pushed back into
# the conductor instead of out (not usable for the Part 2/3 escape demo).
E_gun = -1e8  # asymptotic field [V/m]

# --- Part 1: field inside the well, and the depth-dependent sign flip -----

print("Field on-axis at the well bottom, as a function of depth H/R")
print("(sign flips around H/R ~ 0.35-0.4 for this radius -- confirmed")
print("mesh-stable, not an artifact; see the module docstring above):\n")
for H_over_R in [0.02, 0.1, 0.2, 0.4, 0.6, 1.0]:
    H_ = H_over_R * R
    field = CylindricalWellBEMField(E_gun, R, H_, max_length=min(R, H_) / 29)
    Ez = field.evaluate(np.array([[0.0, 0.0, -H_]]))[0, 2]
    print(f"  H/R={H_over_R:.2f}   Ez(bottom center)={Ez:+.3e} V/m   (E_gun={E_gun:.1e})")

# --- Part 2: push particles from the well bottom to a screen --------------

N = 24
z_screen = 3.0 * R

golden_angle = np.pi * (3.0 - np.sqrt(5.0))
i = np.arange(N)
r_flat = 0.85 * R * np.sqrt((i + 0.5) / N)
phi_flat = golden_angle * i
x, y = r_flat * np.cos(phi_flat), r_flat * np.sin(phi_flat)

particles = ParticleGroup(
    data=dict(
        x=x,
        y=y,
        z=np.full(N, -H),  # emitted from the flat well bottom -- no distribution-mapping helper needed
        px=np.zeros(N),
        py=np.zeros(N),
        pz=np.zeros(N),  # start at rest
        t=np.zeros(N),
        weight=np.full(N, ELEMENTARY_CHARGE),
        status=np.ones(N),
        id=np.arange(N),
        species="electron",
    )
)


def energy_ev(screen, mass_kg):
    px = ev_c_to_si_momentum(np.asarray(screen.px))
    py = ev_c_to_si_momentum(np.asarray(screen.py))
    pz = ev_c_to_si_momentum(np.asarray(screen.pz))
    return (px**2 + py**2 + pz**2) / (2.0 * mass_kg) / ELEMENTARY_CHARGE


print(f"\nPushing {N} particles from the well bottom (z=-H) to a screen at z=3R")
print("(no image-charge/Coulomb force -- energy spread here is a direct")
print("readout of field-solve accuracy, via energy conservation on this")
print("single equipotential (grounded) conductor):")

t0 = time.time()
geometry = CylindricalWellBEMGeometry(E_gun, R, H, field_max_length=min(R, H) / 29)
tracer = SpecificParticleTracer(particles, n_emit=N, geometry=geometry, screens=[z_screen], z_max=1.2 * z_screen)
screens, _ = tracer.run()
ke = energy_ev(screens[0], tracer.mass)
print(
    f"BEM well: {time.time() - t0:.2f}s, {len(screens[0])}/{N} arrived, "
    f"KE mean={ke.mean():.4f} eV, spread={100 * ke.std() / ke.mean():.4f}%"
)

# --- Part 3: image-charge force and the far-field flat-cathode handoff ----

from specific_particle_tracer.geometry import DEFAULT_Z0  # noqa: E402

print(f"\nBuilding the image-charge BEM solve (z0={DEFAULT_Z0:.1e} m) -- same")
print("one-time-cost caveat as the hemisphere-tip case.")

geometry_with_image = CylindricalWellBEMGeometry(
    E_gun,
    R,
    H,
    field_max_length=min(R, H) / 29,
    z0=DEFAULT_Z0,
    image_n_max=16,
)

charge = np.array([[-ELEMENTARY_CHARGE]])
active = np.array([[True]])
print("\nImage-charge force magnitude vs. height above the well bottom")
print(f"(z_handoff={geometry_with_image.z_handoff:.2e} m -- beyond that, this")
print("blends over to a plain flat-cathode image force instead of paying")
print("for a full BEM evaluation):")
for z in [-H + 0.5e-9, -0.02 * R, 0.5 * R, 2.0 * R, 0.7 * geometry_with_image.z_handoff, 3.0 * geometry_with_image.z_handoff]:
    position = np.array([[[0.3 * R, 0.0, z]]])
    F = geometry_with_image.image_force(position, charge, active, plummer_radius=1e-12)
    print(f"  z={z: .3e} m   |F|={np.linalg.norm(F):.4e} N")
