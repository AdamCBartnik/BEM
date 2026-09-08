"""Image-charge BEM force for an off-axis point charge near an axisymmetric
grounded conductor, via azimuthal Fourier-mode decomposition (bem.toroidal,
bem.ring_modes) plus a local planar mirror-charge subtraction.

Why a mirror-charge subtraction at all
---------------------------------------
The excitation here (a real particle at an arbitrary, generally off-axis 3D
position) is not axisymmetric, so bem.axisymmetric's m=0-only reduction
doesn't apply -- see that module's docstring and
[[project-bem-symmetry-scope]] in this project's memory. The azimuthal
Fourier-mode approach lets each mode still be a cheap 1D (generating-
profile) solve, but the *number of modes* needed to represent the induced
charge grows as the excitation becomes more spatially localized in phi --
and a point charge's induced response is most localized exactly when the
charge is close to the surface (checked directly: reconstructing a bare
point-charge potential from `ring_modes.point_charge_potential_modes` at a
separation comparable to the surface's own radius of curvature needed
several tens of modes for 1e-6 relative accuracy -- see
tests/test_bem_image_charge.py) -- which is precisely the regime where the
image-charge force matters most for this project.

The fix: for a truly flat, infinite, grounded plane, the *exact* induced
field in the exterior is reproduced everywhere by a single classical
mirror point charge (no mode decomposition needed at all). Our surface is
curved and finite, so that's only a local approximation -- but it's exactly
the *local, near-field, most-localized* part of the true induced response,
which is exactly the part that's expensive to represent in Fourier modes.
Subtracting it out before solving leaves a *residual* problem whose
excitation is smoother (less localized in phi) than the bare point charge
alone, needing many fewer modes for the same accuracy -- while the physics
recovers *exactly* once the subtracted mirror charge's own field is added
back at the end (not an approximation on top of the mode truncation, see
"Exactness" below).

Where the mirror charge goes
------------------------------
At the true closest point C on the surface to the particle's position P
(found via the same closest-point-on-generating-profile search already
used for near-surface field regularization in bem.axisymmetric -- valid
here too since the true closest point to any 3D P on an axisymmetric
surface stays in P's own meridian half-plane, by the surface's rotational
symmetry), reflect P through the *tangent line* to the profile at C (in
that meridian plane; the tangent *plane* in 3D also contains the local
phi-hat direction, which reflection through it leaves unchanged, so this
2D reflection is exact, not an approximation) to get the mirror point.

Cutoff / blending
------------------
The mirror charge is scaled by a weight w(d) in [0, 1] (d = distance from
P to C): 1 at d=0 (trust the local-plane approximation fully -- this is
where it matters most and is most accurate, since d is then small compared
to the surface's local radius of curvature), smoothly falling to 0 by
d = d_hi (a few local-curvature-radii out, where the local-plane
approximation has broken down and there's no benefit to forcing it). The
transition is a cubic smoothstep, not a sharp cutoff: a discontinuous w(d)
would put a kink in the force as the particle moves across the cutoff
distance, which is exactly the kind of thing that stalls an adaptive ODE
integrator (see this project's own history with the pre-regularization
near-surface field, in bem.axisymmetric's development). d_lo, d_hi are
free parameters (see `solve_residual`/`image_field`) with no fixed "right"
answer yet -- they trade off against the number of modes needed, which is
exactly the empirical question this module exists to let us measure (see
tests/test_bem_image_charge.py's mode-count-vs-standoff experiments).

Exactness of the subtraction (why d_lo/d_hi don't need to be "correct")
--------------------------------------------------------------------------
Let sigma_true be the *true* induced surface density (unknown, and the
whole point of doing a BEM solve for it). Define sigma_residual by

    V[sigma_residual](x) = -[Q*G(x,r0) + Q_img*G(x,r_img)]   for x on S

(a directly solvable BIE -- Q_img = -Q*w(d), fixed once P's position is
fixed). Since V[sigma_true](x) = -Q*G(x,r0) on S by definition of the true
problem, subtracting gives V[sigma_residual - sigma_true](x) = Q_img*G(x,
r_img) on S. Both S[sigma_residual - sigma_true] and Q_img*G(., r_img) are
harmonic in the exterior and decay at infinity, and they agree on the
boundary S -- so by uniqueness of the exterior Dirichlet problem, they're
equal *everywhere* in the exterior, not just on S. Hence, evaluated at any
exterior point x (in particular x = P, since w/Q_img/r_img were fixed
using P's own position, not re-derived as x moves):

    E_true_induced(x) = E_residual(x) + E_mirror_direct(x)

exactly, for *any* choice of w(d)/d_lo/d_hi/r_img -- the only thing they
affect is how many Fourier modes `solve_residual`/`image_field` need to
reach a given accuracy, never the correctness of the converged answer. This
gives a strong built-in test: the final field must (for large enough
n_max) agree regardless of the blending parameters, including w=0
(equivalent to bypassing the whole mirror-charge trick).

Measured mode counts (this project's actual geometry)
--------------------------------------------------------
For the real hemisphere-tip recessed image profile (R=50nm, z0=3nm, a
mesh resolved finely enough that these numbers reflect mode truncation
rather than surface discretization error -- checked directly: on a plain
sphere, the same measurement redone with a coarser mesh gives a *worse*-
looking bare/mirror mismatch that has nothing to do with mode count and
shrinks as the mesh is refined), for an off-axis particle at ~40 degrees
from the pole, modes needed for 1% accuracy (relative to n_max=40):

    standoff/R    bare (no mirror)    with mirror (d_lo=0.1R, d_hi=0.5R)
      0.06              14                          9
      0.10              11                          8
      0.20               7                          6
      0.50               4                          4  (mirror weight ~0 here)
      1.00               2                          2
      2.00               2                          2

The mirror-charge subtraction genuinely helps (roughly a third fewer
modes near the surface), but *not* the order-of-magnitude reduction it
gives on a truly flat plane or a sphere approached much closer than its
own radius (checked separately: on a sphere at standoff 0.02*radius, the
mirror trick alone -- n_max=0 -- gets within ~1% of the exact grounded-
sphere image-charge answer, vs. needing 20+ modes bare). The reason: the
mirror-charge approximation is a *locally flat* approximation, good when
the standoff is small *compared to the local radius of curvature* -- and
the physically relevant standoffs here (down to z0, since closer than
that the existing z0 image-plane regularization already applies, the same
one `forces.hemispherical_tip_image_force` uses) are a sizeable fraction
of the tip's own radius of curvature (R - z0), not orders of magnitude
smaller. So the benefit is real but modest for *this* shape; it would be
larger for a sharper (smaller local radius of curvature) feature, and
smaller still for a flatter one -- something to keep in mind when this
machinery is pointed at a genuinely different tip shape.

A bug this measurement caught along the way, worth remembering: the two
columns above (bare vs. mirror) initially disagreed by several percent
*even at full mode convergence*, which the exactness argument above says
should never happen. Root cause was a mesh-resolution artifact, not a
bug in the trick itself -- confirmed by reproducing the same symptom on a
plain sphere and watching it shrink as the mesh was refined (8.3% -> 2.6%
-> 1.0% -> 0.45% as n_theta doubled from 30 to 240, at a standoff smaller
than the coarser meshes' own segment size). Lesson: a "both sides should
agree" invariant test can look like it's failing from a real bug when
it's actually just underresolved -- refine the mesh before concluding
the two code paths disagree.

Units
-----
Every kernel in this module (and in bem.ring_modes/bem.toroidal/
bem.axisymmetric) works in G = 1/(4*pi*r), i.e. natural units with vacuum
permittivity implicitly 1 -- fine for the static-field solver (its
boundary data is given directly in volts, no explicit charge anywhere),
but a real point charge's field needs the physical 1/(4*pi*epsilon_0);
`ImageChargeBEMSolution.image_field`/`.image_force` divide by
VACUUM_PERMITTIVITY before returning so their output is directly
comparable to `forces.py`'s COULOMB_CONSTANT-based force functions. This
was caught (not designed in from the start) by comparing against
`forces.hemispherical_tip_image_force`'s exact 3-image solution for this
project's actual hemisphere-tip shape and finding a suspiciously exact
~1/epsilon_0 ratio between the two answers.

Multiple particles: conductor-mediated cross-coupling
-------------------------------------------------------
`image_field` (above) treats one particle at a time, as if it were the
only charge present. `ImageChargeBEMSolution.image_force` instead solves
a *joint* problem over every active particle at once: by linear
superposition, the true combined induced response equals the sum of what
each particle would induce alone, so summing every particle's own
excitation (real charge + its own local mirror charge) into one shared
per-mode right-hand side and solving once gives, in a single solve, the
same total residual density N independent single-particle solves would
sum to -- cheaper (one solve, O(N) field evaluations, vs. N solves) and
it's what lets the *cross* term (particle j's presence changing the force
felt by particle i, exactly analogous to `forces.image_charge_force`'s
all-pairs real-times-mirror sum on a flat plane) fall out for free. The
one complication: `image_field`'s trick of rotating its lone source to
phi=0 (so only a cosine series is needed) doesn't generalize to several
sources at different azimuths, so the joint right-hand side (and the
solved density) needs both a cos(m*phi) and a sin(m*phi) part per mode --
`image_force` carries both explicitly rather than reusing `image_field`'s
single-series machinery. Validated in
tests/test_bem_image_charge.py three ways: N=1 reduces to exactly
`image_field`'s answer; on a flat plane (residual ~0, so this mostly
tests the cross-mirror-charge sum) it matches `forces.image_charge_force`
to machine precision; on the real hemisphere-tip geometry it matches
`forces.hemispherical_tip_image_force`'s all-pairs 3-image solution to
the same few-percent level the single-particle case already did.

GPU support
-----------
`image_force` (not `image_field`, kept CPU/numpy-only as a simpler
validated reference) runs its whole per-call pipeline -- closest-point
search, mirror-charge reflection, RHS assembly, the per-mode linear solve,
joint field reconstruction, the all-pairs mirror cross-term -- on whatever
`xp` is passed (the one-time operator *assembly*, `ImageChargeBEMSolution.
solve`, stays numpy/CPU-only regardless, like the rest of this project's
BEM solves; see bem.geometry.HemisphericalTipBEMGeometry's docstring).
Field reconstruction was also rewritten to use a single fixed, precomputed
quadrature (`_segment_quadrature_geometry`/`_interp_nodal_density`) instead
of a Python loop over profile segments -- both faster on CPU and, more
importantly, GPU-friendly (one batch of array ops instead of one round of
kernel launches per segment).

Two remaining Python-level loops turned out to dominate GPU cost even
after that, at the small-to-medium particle counts most simulations
actually use, and got custom `cupy.RawKernel`s:

- `bem.toroidal.toroidal_Q`'s downward recursion (see that module) --
  every mode-kernel call (`point_charge_potential_modes` for the RHS,
  `ring_field_modes` for field reconstruction) goes through it, and its
  loop length is set by chi (how close field and source points are), not
  by batch size, so it doesn't amortize away for a big batch of particles
  the way ordinary vectorized work does.
- `bem.axisymmetric._closest_point_on_profile` (shared with that module's
  own near-surface field regularization) -- its loop length is the number
  of profile segments (tens), and even without any data-dependent
  branching, tens of Python-level iterations each launching several tiny
  elementwise kernels (clip, compare, three `xp.where`s) turned out to
  dominate GPU cost at query-point counts as small as 10.

Both fuse their whole per-element loop into one kernel launch (one CUDA
thread per chi / per query point, looping internally in device code) and
fall back to the original per-xp Python loop when the fast path doesn't
apply (numpy always; cupy above `toroidal._TOROIDAL_MAX_KERNEL_M`, a mode
count the per-thread local array size needs bounding, which the profile
closest-point kernel doesn't need since it just reads global memory).
Measured impact on `image_force` itself (this project's actual hemisphere-
tip geometry, n_max=12, RTX 5070 Ti, before -> after both kernels):

    N        CPU          GPU before    GPU after    speedup before/after
    1        4.2 ms        68.2 ms        8.4 ms      0.06x  ->  0.50x
    10       6.9 ms        66.7 ms        8.5 ms      0.10x  ->  0.81x
    100     56.1 ms        76.8 ms        9.7 ms      0.72x  ->  5.81x
    1000  1267.0 ms        86.6 ms       32.0 ms     15.65x  -> 39.62x

The GPU/CPU crossover point (where GPU stops being a net loss) moved from
somewhere around N~300-1000 down to N~10-20 -- worth knowing before
assuming "GPU" automatically means "fast" for a small particle batch.

(The table's N is a single group's own particle count, i.e. this measures
G=1, E=N -- no group boundary is involved when there's only one group, so
these numbers are unaffected by the group-isolation fix below.)

A RawKernel argument gotcha worth remembering if this code is touched
again: it receives array arguments as flat pointers with *no stride
information*. Passing a non-contiguous view (e.g. one column of a
(n, 2) C-order array, a stride-2 view) reads silently-wrong-but-finite
data with no exception at all -- caught exactly this way while writing
the closest-point kernel (segment index, t, and normal disagreed with the
CPU reference on some points while the reported distance coincidentally
still matched, because the kernel was reading interleaved rho/z-like
garbage instead of the intended column). Every array handed to
`kernel(...)` must be made contiguous explicitly first
(`cp.ascontiguousarray`), even if it was already a "real" array a moment
before slicing.

Groups never interact -- a bug found and fixed here
-----------------------------------------------------
Every other force in this project (forces._pairwise_force, which backs
coulomb_force/image_charge_force/hemispherical_tip_image_force) treats its
input as (n_groups, n_emit, 3) and only ever pairs particles *within* the
same group -- the whole basis for tracker.SpecificParticleTracer's
per-group independent adaptive stepping (see its own module docstring).
An earlier version of `image_force` here didn't respect this: it accepted
any leading batch shape and flattened it away before doing the joint
solve, so two particles from *different* emission groups -- meant to be
physically non-interacting, however close they happened to be -- would
spuriously polarize the same patch of conductor together. Caught directly
(not theoretically): adding a second, distant, unrelated group changed a
first group's own particle's force by ~14%, when it should have changed
it by exactly zero. Fixed by requiring `position` to carry the group axis
explicitly (a bare (N, 3) now raises, rather than silently doing the wrong
thing again) and doing the per-mode solve, field reconstruction, and
mirror-charge cross-term all *per group* -- see `image_force`'s own
docstring for how (batching G independent solves into one call's "many
right-hand-sides" axis, not a Python loop over groups) and
tests/test_bem_image_charge.py's test_image_force_groups_never_interact
for the regression check.

Cost implication and multi-process CPU scaling
-------------------------------------------------
Doing this correctly costs more than the bug did whenever more than one
group shares a step: the buggy version's single shared right-hand side
made its per-mode solve O(n_max * P^3), independent of how many groups
were flattened together, which was cheap only because it was quietly
ignoring group boundaries. The fixed version's cost genuinely grows with
G (number of groups sharing a step) -- measured on this project's actual
geometry (n_theta=20 image profile, n_max=8): ~2 ms/group at G=10,
settling to ~6-7 ms/group by G>=50 (small-batch fixed overhead
amortizing away, the same pattern this project's own per-group tracker
cost already showed for the *field*-only case). For a near-surface start
(every particle begins at rest right where the image force is strongest
and most rapidly varying, needing many small adaptive steps to resolve),
that per-call cost gets paid many times over a single trajectory, so it
dominates far more than a naive "per-call cost x expected number of
calls" estimate suggests.

This is exactly where multi-process CPU parallelism (tracker.
SpecificParticleTracer's `n_workers`, unaffected by anything in this
module or bem.toroidal/bem.axisymmetric's GPU work -- it's a separate,
already-existing mechanism, see parallel.py) earns its keep: splitting
groups across worker processes doesn't just parallelize the work, it
directly *shrinks* the G each worker's own image_force calls have to
solve for, which is worth more than proportionally given the per-group
cost above only fully amortizes once G is already largish. Measured
directly (this project's own geometry, n_theta=20 image profile, n_max=8,
n_emit=5, 50 groups sharing steps, RTX-5070-Ti-equipped machine, all on
CPU): n_workers=1 -> 489.0s, 2 -> 165.1s (2.96x), 4 -> 90.8s (5.39x),
8 -> 62.7s (7.79x) -- scaling at least as well as, and here somewhat
better than, plain linear, unlike the field-only case at the same group
count (which needs tens of thousands of groups before multi-process CPU
parallelism clearly pays for itself, since its own per-group cost is
much smaller to begin with -- see tracker.SpecificParticleTracer's own
docstring). Multi-process CPU and the GPU port above are independent,
complementary options for the same underlying cost, not alternatives to
pick between in general -- which one wins depends on G, n_emit, n_max,
and how many physical cores vs. how capable a GPU is available.
"""

import numpy as np

from ..constants import VACUUM_PERMITTIVITY
from .axisymmetric import _GAUSS_X, _GAUSS_W, _segment_endpoints, _closest_point_on_profile
from .ring_modes import ring_potential_modes, ring_field_modes, point_charge_potential_modes


def _to_xp(arr, xp, dtype=None):
    """Move arr onto array module xp, safely: numpy can't implicitly
    convert a cupy array (it raises rather than silently round-tripping
    through the host), so route through `backend.to_numpy` first whenever
    the target is numpy; going the other way, `xp.asarray` handles both a
    numpy-array input (host->device copy) and an already-on-xp input (no-op)."""
    if xp is np:
        from ..backend import to_numpy

        arr = to_numpy(arr)
    return xp.asarray(arr, dtype=dtype) if dtype is not None else xp.asarray(arr)


def _smoothstep(t, xp=np):
    t = xp.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def image_charge_weight(d, d_lo, d_hi, xp=np):
    """1 at d <= d_lo, 0 at d >= d_hi, cubic smoothstep in between."""
    if d_hi <= d_lo:
        raise ValueError(f"d_hi must be > d_lo (got d_lo={d_lo!r}, d_hi={d_hi!r})")
    return _smoothstep((d_hi - d) / (d_hi - d_lo), xp=xp)


def mirror_point(p_rho, p_z, closest, tangent_unit, normal_unit):
    """Reflect (p_rho, p_z) through the line at `closest` with direction
    `tangent_unit`, in the 2D generating-profile (meridian) plane -- see
    module docstring's "Where the mirror charge goes". Returns
    (rho_img, z_img, sign): sign = -1 if the reflection crosses the axis
    (rho would go negative -- physically means the image sits in the
    meridian half-plane at phi = phi_particle + pi, not phi_particle), in
    which case rho_img is returned as the (positive) magnitude.
    """
    delta = np.array([p_rho, p_z]) - closest
    d_t = delta @ tangent_unit
    d_n = delta @ normal_unit
    image = closest + d_t * tangent_unit - d_n * normal_unit
    rho_img, z_img = image
    if rho_img < 0.0:
        return -rho_img, z_img, -1
    return rho_img, z_img, 1


def mirror_points_batch(p_rho, p_z, closest, tangent_unit, normal_unit, xp=np):
    """Vectorized `mirror_point`: p_rho, p_z shape (N,); closest,
    tangent_unit, normal_unit shape (N, 2) (one closest-point/tangent/
    normal per particle, e.g. from `_closest_points_batch`). Returns
    (rho_img, z_img, side), each shape (N,)."""
    delta = xp.stack([p_rho, p_z], axis=-1) - closest  # (N, 2)
    d_t = xp.sum(delta * tangent_unit, axis=-1)
    d_n = xp.sum(delta * normal_unit, axis=-1)
    image = closest + d_t[:, None] * tangent_unit - d_n[:, None] * normal_unit
    rho_img, z_img = image[:, 0], image[:, 1]
    side = xp.where(rho_img < 0.0, -1.0, 1.0)
    rho_img = xp.abs(rho_img)
    return rho_img, z_img, side


def _segment_potential_modes(rho_f, z_f, p0, p1, s0, s1, n_max, refine_ratio, max_depth, depth=0):
    """Vector-valued (all modes 0..n_max at once) analogue of
    `bem.axisymmetric._segment_potential`, for assembling the per-mode
    single-layer operator matrices. The adaptive-subdivision *decisions*
    (whether to recurse) don't depend on which mode -- only the kernel
    evaluated at each leaf does -- so this costs about the same number of
    Python-level recursive calls as the m=0-only solve, just returning an
    (n_max+1,)-vector per leaf instead of a scalar."""
    mid = 0.5 * (p0 + p1)
    length = np.linalg.norm(p1 - p0)
    dist = np.linalg.norm(np.array([rho_f, z_f]) - mid)

    if depth < max_depth and length > refine_ratio * dist:
        pm = 0.5 * (p0 + p1)
        sm = 0.5 * (s0 + s1)
        return _segment_potential_modes(
            rho_f, z_f, p0, pm, s0, sm, n_max, refine_ratio, max_depth, depth + 1
        ) + _segment_potential_modes(
            rho_f, z_f, pm, p1, sm, s1, n_max, refine_ratio, max_depth, depth + 1
        )

    total = np.zeros(n_max + 1)
    for xi, wi in zip(_GAUSS_X, _GAUSS_W):
        t = 0.5 * (xi + 1.0)
        rho_s = p0[0] + t * (p1[0] - p0[0])
        z_s = p0[1] + t * (p1[1] - p0[1])
        sigma_s = s0 + t * (s1 - s0)
        seg_len = length * 0.5 * wi
        Phi = ring_potential_modes(rho_f, z_f, rho_s, z_s, n_max)
        total += sigma_s * Phi * seg_len
    return total


def _segment_quadrature_geometry(profile, n_subdiv):
    """Flat, fixed (non-adaptive) per-segment Gauss quadrature nodes over
    the whole profile -- same sub-panel/Gauss-4 scheme as
    `bem.axisymmetric._fixed_quadrature`, but *without* baking in a
    specific nodal density: the density here (sigma_C/sigma_S) is solved
    fresh per `image_force` call (unlike the one-time axisymmetric solve's
    fixed sigma), so each quadrature point instead records which segment
    it's in and its local interpolation parameter t, letting any (n_max+1,
    n_profile) density array be interpolated onto these nodes with a single
    vectorized gather-and-blend (`_interp_nodal_density`) -- no Python loop
    over segments at call time, which is what makes
    `_field_from_joint_mode_density_batch` a single broadcast instead of
    the O(n_segments) Python loop earlier versions of this module used.

    Returns
    -------
    rho_q, z_q, seg_len_q : arrays, shape (n_quad,)
    seg_index_q : int array, shape (n_quad,)
    t_local_q : array, shape (n_quad,)
    """
    rho_q, z_q, seg_len_q, seg_index_q, t_local_q = [], [], [], [], []
    for j in range(len(profile) - 1):
        p0, p1 = profile[j], profile[j + 1]
        length = np.linalg.norm(p1 - p0)
        for k in range(n_subdiv):
            ta, tb = k / n_subdiv, (k + 1) / n_subdiv
            for xi, wi in zip(_GAUSS_X, _GAUSS_W):
                t = ta + 0.5 * (tb - ta) * (xi + 1.0)
                rho_q.append(p0[0] + t * (p1[0] - p0[0]))
                z_q.append(p0[1] + t * (p1[1] - p0[1]))
                seg_len_q.append((length / n_subdiv) * 0.5 * wi)
                seg_index_q.append(j)
                t_local_q.append(t)
    return (
        np.array(rho_q),
        np.array(z_q),
        np.array(seg_len_q),
        np.array(seg_index_q, dtype=np.int64),
        np.array(t_local_q),
    )


def _interp_nodal_density(sigma, seg_index, t_local):
    """sigma: (..., n_profile) -- any number of leading dims (e.g.
    (n_max+1, n_groups) for the grouped joint solve) -- on the same array
    module as seg_index/t_local. Returns (..., n_quad): sigma linearly
    interpolated from profile nodes onto quadrature points, via
    `_segment_quadrature_geometry`'s (seg_index, t_local) -- a single
    vectorized gather, not a Python loop. Indexes the *last* axis
    explicitly (`...`), not a fixed position, so this works the same way
    regardless of how many leading dims sigma has."""
    s0 = sigma[..., seg_index]
    s1 = sigma[..., seg_index + 1]
    t = t_local.reshape((1,) * (s0.ndim - 1) + t_local.shape)
    return s0 + t * (s1 - s0)


def assemble_mode_operators(profile, n_max, refine_ratio=1.0, max_depth=20):
    """The per-mode single-layer collocation matrices A[m] (V_m[sigma_m]
    evaluated at every profile node from a unit density at every other
    node), m=0..n_max, as one array of shape (n_max+1, n, n). One-time,
    geometry-only cost -- reused for every particle position afterwards
    (see `ImageChargeBEMSolution`)."""
    profile = np.asarray(profile, dtype=float)
    n = len(profile)
    A = np.zeros((n_max + 1, n, n))
    # A profile with a pole (rho=0, e.g. a hemisphere-tip apex) can't
    # literally collocate there for m>=1: Phi_m(rho_f, ...) ~ rho_f^m as
    # rho_f -> 0 (a coordinate singularity in the (rho,a) -> chi map, not
    # a real one -- confirmed numerically to converge cleanly from
    # rho_f=1e-5 down for m up to a few, relative to an O(1) length scale),
    # so evaluate collocation exactly at the pole using a tiny floor
    # instead of literal 0 to avoid the 0/0 in the mode kernel.
    rho_floor = 1e-6 * max(np.max(profile[:, 0]), 1e-300)
    for i in range(n):
        rho_f, z_f = profile[i]
        rho_f = max(rho_f, rho_floor)
        for j in range(n - 1):
            p0, p1 = profile[j], profile[j + 1]
            A[:, i, j] += _segment_potential_modes(rho_f, z_f, p0, p1, 1.0, 0.0, n_max, refine_ratio, max_depth)
            A[:, i, j + 1] += _segment_potential_modes(rho_f, z_f, p0, p1, 0.0, 1.0, n_max, refine_ratio, max_depth)
    return A


class ImageChargeBEMSolution:
    """Precomputed per-mode BEM operators for the image-charge problem on
    a fixed axisymmetric surface (typically the *recessed image* surface,
    e.g. `bem.mesh.hemisphere_tip_image_profile` -- see this project's
    existing `geometry.HemisphericalTip`'s analytic z0 image-plane offset
    for why the image surface, not the real one, is the right source
    surface for this correction).

    Build once via `ImageChargeBEMSolution.solve(profile, n_max)`; call
    `.image_field(position, charge, d_lo, d_hi)` per particle position.
    """

    def __init__(self, profile, A, n_max):
        self.profile = profile
        self.A = A
        self.n_max = n_max
        self._geom = _segment_endpoints(profile)
        self._rho_floor = 1e-6 * max(np.max(profile[:, 0]), 1e-300)
        self._quad_geom_cache = {}

    @classmethod
    def solve(cls, profile, n_max, refine_ratio=1.0, max_depth=20):
        profile = np.asarray(profile, dtype=float)
        A = assemble_mode_operators(profile, n_max, refine_ratio=refine_ratio, max_depth=max_depth)
        return cls(profile, A, n_max)

    def _closest_point(self, rho, z):
        closest, tangent_unit, normal_unit, dist = self._closest_points_batch(
            np.asarray([rho]), np.asarray([z])
        )
        return closest[0], tangent_unit[0], normal_unit[0], float(dist[0])

    def _closest_points_batch(self, rho, z, xp=np):
        """Vectorized closest-point-on-profile search for arrays of (rho, z)
        query points -- shape (N,) in, `(closest, tangent_unit, normal_unit,
        dist)` out, shapes (N, 2), (N, 2), (N, 2), (N,).

        `_closest_point_on_profile` is already xp-generic (it's shared with
        bem.axisymmetric's own near-surface regularization, already used
        there with xp=cupy), but it indexes the *profile's* small geometry
        arrays (p0, p1, tangent, normal -- always built as numpy, since the
        profile itself is tiny) with an *xp*-array index (`best_seg`) it
        computes internally; numpy can't be indexed by a cupy array, so
        those small arrays need converting to xp first when xp is cupy."""
        p0, p1, tangent, normal = (_to_xp(a, xp) for a in self._geom)
        best_dist2, best_seg, best_t, best_normal = _closest_point_on_profile(
            rho, z, p0, p1, tangent, normal, xp
        )
        closest = p0[best_seg] + best_t[:, None] * tangent[best_seg]
        seg_length = xp.linalg.norm(tangent[best_seg], axis=-1, keepdims=True)
        tangent_unit = tangent[best_seg] / seg_length
        dist = xp.sqrt(best_dist2)
        return closest, tangent_unit, best_normal, dist

    def _quadrature_geometry(self, n_subdiv):
        """Cached (numpy; converted to xp per call by the caller, matching
        this project's established pattern -- see e.g.
        AxisymmetricBEMSolution.quadrature -- of not bothering to cache the
        xp-converted copy of a small array) flat quadrature geometry, see
        `_segment_quadrature_geometry`."""
        if n_subdiv not in self._quad_geom_cache:
            self._quad_geom_cache[n_subdiv] = _segment_quadrature_geometry(self.profile, n_subdiv)
        return self._quad_geom_cache[n_subdiv]

    def image_field(self, position, charge, d_lo, d_hi, n_max=None):
        """The image-charge (induced-field) contribution at `position`
        (shape (3,)) due to a point charge `charge` there, excluding the
        real charge's own singular self-field -- i.e. exactly the field
        that should multiply `charge` again to get the image force. See
        module docstring: this is exact (given `n_max`, `d_lo`, `d_hi`)
        in the sense that the *converged* (large n_max) answer doesn't
        depend on d_lo/d_hi at all, only how quickly it converges does.

        Units: every kernel in this module (and in bem.ring_modes/
        bem.toroidal/bem.axisymmetric) uses G = 1/(4*pi*r), i.e. natural
        units with vacuum permittivity set to 1 -- matching the rest of
        the bem package (which never needs an explicit epsilon_0, since
        the static-field solver's boundary data is given directly in
        volts). A real point charge does need it: this method's returned
        E is rescaled by 1/VACUUM_PERMITTIVITY before returning, so it's
        directly SI (V/m, for `charge` in Coulombs) and comparable to
        `forces.py`'s COULOMB_CONSTANT = 1/(4*pi*epsilon_0)-based force
        functions -- confirmed by matching `hemispherical_tip_image_force`
        (the project's exact 3-image analytic solution for this same
        hemisphere-on-plane shape) to a few percent near the pole, where
        this module's smoothed-fillet recessed profile is closest to that
        formula's sharp-ridge idealization (see
        tests/test_bem_image_charge.py).

        Parameters
        ----------
        position : array, shape (3,)
        charge : float
        d_lo, d_hi : float
            Mirror-charge blending distances (see module docstring). Pass
            d_hi <= 0 (or d_lo = d_hi = 0) to disable the mirror-charge
            trick entirely (weight always 0) -- useful for the mode-count
            comparison in tests.
        n_max : int, optional
            Defaults to this solution's full precomputed mode count; pass
            a smaller value to see how accuracy degrades with fewer modes
            (reuses the same precomputed operators, just truncated).

        Returns
        -------
        E : array, shape (3,)
        """
        n_max = self.n_max if n_max is None else n_max
        x0, y0, z0 = position
        rho0 = float(np.hypot(x0, y0))
        phi0 = float(np.arctan2(y0, x0))
        rho0_safe = max(rho0, self._rho_floor)

        closest, tangent_unit, normal_unit, dist = self._closest_point(rho0_safe, z0)
        if d_hi > d_lo:
            w = image_charge_weight(dist, d_lo, d_hi)
        else:
            w = 0.0

        rho_img, z_img, side = mirror_point(rho0_safe, z0, closest, tangent_unit, normal_unit)
        q_img = -charge * w

        profile_rho = np.maximum(self.profile[:, 0], self._rho_floor)
        profile_z = self.profile[:, 1]
        g_real = charge * point_charge_potential_modes(profile_rho, profile_z, rho0_safe, z0, n_max)
        if w > 0.0:
            mode_sign = np.ones(n_max + 1) if side > 0 else (-1.0) ** np.arange(n_max + 1)
            g_img = q_img * point_charge_potential_modes(
                profile_rho, profile_z, rho_img, z_img, n_max
            ) * mode_sign[:, None]
        else:
            g_img = 0.0

        rhs = -(g_real + g_img)  # shape (n_max+1, n)
        A = self.A[: n_max + 1, : len(self.profile), : len(self.profile)]
        # np.linalg.solve's batched-vector-RHS form (a: (...,M,M), b: (...,M))
        # isn't accepted by this numpy version's gufunc (it insists on the
        # (...,M,K) matrix form) -- add/drop a trailing size-1 axis.
        sigma_residual = np.linalg.solve(A, rhs[..., None])[..., 0]  # (n_max+1, n)

        # point_charge_potential_modes (hence g_real/g_img/sigma_residual)
        # is built in the convention that the real charge sits at phi'=0 --
        # evaluate the residual field there too (phi=0, not the particle's
        # true phi0) and rotate the resulting vector back by phi0 at the
        # end, rather than threading phi0 through the mode machinery
        # itself (which only knows relative angles anyway).
        E_residual_rot = self._field_from_mode_density(sigma_residual, rho0_safe, 0.0, z0, n_max)
        cos0, sin0 = np.cos(phi0), np.sin(phi0)
        E_residual = np.array(
            [
                E_residual_rot[0] * cos0 - E_residual_rot[1] * sin0,
                E_residual_rot[0] * sin0 + E_residual_rot[1] * cos0,
                E_residual_rot[2],
            ]
        )

        if w > 0.0:
            phi_img = phi0 if side > 0 else phi0 + np.pi
            E_img = _point_charge_field(rho0_safe, phi0, z0, rho_img, phi_img, z_img, q_img)
        else:
            E_img = np.zeros(3)

        on_axis = rho0 <= self._rho_floor
        if on_axis:
            E_residual = np.array([0.0, 0.0, E_residual[2]])
        return (E_residual + E_img) / VACUUM_PERMITTIVITY

    def image_force(self, position, charge, active, d_lo, d_hi, n_max=None, n_subdiv=4, xp=np):
        """Batched image-charge *force* (matching the calling convention of
        `forces.image_charge_force`/`forces.hemispherical_tip_image_force`,
        for use as a `geometry.Geometry.image_force` implementation), via a
        *joint* solve over every active particle -- unlike a per-particle
        `image_field` loop, this includes the cross-term where one
        particle's presence changes the induced charge (and hence the
        force) felt by another, mediated by the conductor.

        By linear superposition, the true combined induced response is
        exactly the sum of what each particle would induce alone -- so one
        combined right-hand side (every active particle's own excitation,
        plus its own local mirror charge, summed together) fed through the
        *same* per-mode operators used elsewhere gives, in one solve, the
        same total residual density that N separate single-particle solves
        summed together would. Since particles generally sit at different,
        arbitrary azimuths, the right-hand side needs both a cos(m*phi) and
        a sin(m*phi) part per mode (`image_field`'s single-particle path
        avoids that by always rotating its one source to phi=0 first,
        which isn't available with more than one source at once).

        The mirror-charge cross-term itself (every particle's own local
        mirror charge's *direct* field, added back per the module
        docstring's exactness argument) is an ordinary all-pairs Coulomb
        sum over active particles, same O(N^2) cost profile as
        `forces.image_charge_force`'s pairwise treatment -- the residual
        (conductor-mediated) part, by contrast, costs only O(N) field
        evaluations after the one shared solve, since it's a single joint
        density rather than N separate ones.

        For a single particle alone in its own group of one, this reduces
        to exactly `image_field`'s single-particle answer (checked in
        tests/test_bem_image_charge.py) -- the cosine-only rotated-frame
        path there is a valid special case of this one, not a different
        approximation.

        Groups never interact (like every other force in this project --
        see forces._pairwise_force/coulomb_force/image_charge_force, and
        tracker.SpecificParticleTracer's own module docstring): the joint
        solve above is done *per group*, not across the whole batch, so a
        particle in one group is never coupled -- through either the
        shared residual density or the all-pairs mirror-charge sum -- to a
        particle in another group, however physically close they happen to
        be. This requires `position` to carry the group axis explicitly
        (shape (n_groups, n_emit, 3), not just any leading batch shape) --
        checked directly: introducing a second, unrelated group changes a
        first group's own particle's force by zero, not by some small but
        nonzero cross-term (which is what an earlier, buggy version of
        this method that flattened away the group axis before solving
        produced -- a real, if usually modest-sized, correctness bug,
        since it let particles from *different* emission groups spuriously
        polarize the same patch of conductor at once).

        Parameters
        ----------
        position : array, shape (n_groups, n_emit, 3)
        charge : array, shape (n_groups, n_emit)
        active : bool array, shape (n_groups, n_emit)
        d_lo, d_hi : float
            Mirror-charge blending distances, shared by every particle.
        n_max : int, optional
            Defaults to this solution's full precomputed mode count.
        n_subdiv : int, optional
            Fixed-quadrature resolution for the field-reconstruction step
            (see `_field_from_joint_mode_density_batch`); doesn't affect
            the operator assembly (always adaptive, see `solve`).
        xp : module, optional
            numpy or cupy. `position`/`charge`/`active` may already be on
            either module (or plain Python/numpy, which is always safe to
            hand to either); everything from here on runs on `xp`,
            including the per-mode linear solve (`xp.linalg.solve`) --
            unlike `image_field` (kept numpy-only, a validated reference/
            special case, see module docstring), this is the path meant to
            actually run on GPU for a simulation with many particles.

        Returns
        -------
        force : array (xp), shape (n_groups, n_emit, 3)

        Note on exact on-axis or exact phi=0 query points: a transverse
        force component that's mathematically zero by symmetry there (the
        y-component right in the phi=0 plane, or both transverse
        components exactly on the axis) comes back as floating-point-level
        noise rather than a literal 0.0, and that noise is *not* identical
        to what the single-particle `image_field` path produces at the
        same point (checked directly -- the two paths agree to machine
        precision everywhere else). Harmless in practice: the noise is
        many orders of magnitude below the dominant (correctly nonzero)
        component, so it only matters if something compares two
        computations of an expected-exactly-zero quantity to each other
        rather than to the physically dominant scale (exactly the
        "small-scale atol trap" this project already watches for
        elsewhere) -- not if it's just summed into a force alongside
        everything else, as the tracker does.

        Cost, worth knowing before pointing this at a very large run: doing
        this correctly (one independent solve per group) genuinely costs
        more than the buggy flattened version did -- that version's single
        shared right-hand side made the per-mode solve O(n_max * P^3)
        *independent of how many groups were in the batch*, which was
        cheap only because it was quietly doing the wrong physics (see
        above). Solving G independent systems (even bundled into one
        multiple-right-hand-side call, as done here) costs O(n_max * (P^3
        + P^2*G)) instead, and the RHS-assembly intermediate arrays scale
        as O(n_max * G * n_emit * n_profile). For this project's typical
        n_emit (small -- a handful to a few tens of particles actually
        close enough in time/space to be worth coupling at all) this is a
        modest, correctness-mandated cost, not a surprise -- but a run
        with a very large number of groups sharing a step at once (this
        project's own multi-process benchmarking has exercised tens of
        thousands) and a generous n_max could use real memory. Not chunked
        here.
        """
        position_xp = _to_xp(position, xp, dtype=float)
        charge_xp = _to_xp(charge, xp, dtype=float)
        active_xp = _to_xp(active, xp, dtype=bool)

        if position_xp.ndim != 3 or position_xp.shape[-1] != 3:
            raise ValueError(
                "position must have shape (n_groups, n_emit, 3) -- groups must stay "
                f"explicit so they can be solved independently (got shape {position_xp.shape})"
            )
        G, E, _ = position_xp.shape
        n_max = self.n_max if n_max is None else n_max

        # Inactive particles contribute nothing to their group's induced
        # response or to the all-pairs mirror sum (same masking convention
        # as forces._pairwise_force), and get zero force at the end.
        q_eff = xp.where(active_xp, charge_xp, 0.0)  # (G, E)

        x, y, z = position_xp[..., 0], position_xp[..., 1], position_xp[..., 2]
        rho = xp.hypot(x, y)
        phi = xp.arctan2(y, x)
        rho_safe = xp.maximum(rho, self._rho_floor)

        # Closest-point/mirror-point geometry is purely per-particle (it
        # doesn't depend on any other particle or which group it's in), so
        # this part can stay flattened across the whole (G, E) batch.
        closest, tangent_unit, normal_unit, dist = self._closest_points_batch(
            rho_safe.reshape(-1), z.reshape(-1), xp=xp
        )
        if d_hi > d_lo:
            w = image_charge_weight(dist, d_lo, d_hi, xp=xp)
        else:
            w = xp.zeros_like(dist)
        rho_img, z_img, side = mirror_points_batch(
            rho_safe.reshape(-1), z.reshape(-1), closest, tangent_unit, normal_unit, xp=xp
        )
        phi_img = xp.where(side > 0, phi.reshape(-1), phi.reshape(-1) + xp.pi)
        q_img = -q_eff.reshape(-1) * w
        rho_img, z_img, phi_img, q_img = (a.reshape(G, E) for a in (rho_img, z_img, phi_img, q_img))

        profile_rho = _to_xp(np.maximum(self.profile[:, 0], self._rho_floor), xp)  # (P,)
        profile_z = _to_xp(self.profile[:, 1], xp)  # (P,)
        P = len(self.profile)
        m = xp.arange(n_max + 1, dtype=float)

        def _rhs_contribution(source_rho, source_z, source_phi, source_charge):
            # source_* : (G, E). Returns (g_c, g_s), each (n_max+1, G, P) --
            # summed over E (within-group only), keeping G as its own axis
            # so different groups' excitations never mix.
            g = point_charge_potential_modes(
                profile_rho[None, None, :], profile_z[None, None, :],
                source_rho[:, :, None], source_z[:, :, None], n_max, xp=xp,
            )  # (n_max+1, G, E, P)
            cos_phi = xp.cos(m[:, None, None] * source_phi[None, :, :])  # (n_max+1, G, E)
            sin_phi = xp.sin(m[:, None, None] * source_phi[None, :, :])
            weighted = source_charge[None, :, :, None] * g  # (n_max+1, G, E, P)
            g_c = xp.sum(weighted * cos_phi[:, :, :, None], axis=2)  # (n_max+1, G, P)
            g_s = xp.sum(weighted * sin_phi[:, :, :, None], axis=2)
            return g_c, g_s

        g_real_c, g_real_s = _rhs_contribution(rho_safe, z, phi, q_eff)
        g_img_c, g_img_s = _rhs_contribution(rho_img, z_img, phi_img, q_img)
        rhs_c = -(g_real_c + g_img_c)  # (n_max+1, G, P)
        rhs_s = -(g_real_s + g_img_s)

        # Every group shares the SAME operator A (geometry-only) but needs
        # its OWN solve -- rather than broadcasting A across a new G axis
        # (which would mean G copies of an (n_max+1, P, P) array), put
        # (group, cos/sin) into the "many right-hand-sides" trailing axis
        # of one ordinary batched solve, matching xp.linalg.solve's native
        # (..., M, M), (..., M, K) form with no replication of A at all.
        rhs = xp.stack([rhs_c, rhs_s], axis=-1)  # (n_max+1, G, P, 2)
        rhs = xp.moveaxis(rhs, 1, 2).reshape(n_max + 1, P, G * 2)  # (n_max+1, P, G*2)
        A = _to_xp(self.A[: n_max + 1, :P, :P], xp)
        sigma = xp.linalg.solve(A, rhs)  # (n_max+1, P, G*2)
        sigma = xp.moveaxis(sigma.reshape(n_max + 1, P, G, 2), 1, 2)  # (n_max+1, G, P, 2)
        sigma_C, sigma_S = sigma[..., 0], sigma[..., 1]

        E_residual = self._field_from_joint_mode_density_batch(
            sigma_C, sigma_S, rho_safe, phi, z, n_max, xp=xp, n_subdiv=n_subdiv
        )  # (G, E, 3)

        query_xyz = _cylindrical_to_cartesian(rho_safe, phi, z, xp=xp)  # (G, E, 3)
        mirror_xyz = _cylindrical_to_cartesian(rho_img, phi_img, z_img, xp=xp)  # (G, E, 3)
        E_mirror = _point_charge_field_batch_all_pairs(query_xyz, mirror_xyz, q_img, xp=xp)  # (G, E, 3)

        E_total = (E_residual + E_mirror) / VACUUM_PERMITTIVITY
        force = q_eff[..., None] * E_total
        return xp.where(active_xp[..., None], force, 0.0)

    def _field_from_mode_density(self, sigma_m, rho, phi, z, n_max):
        """Field E = -grad[S[sum_m sigma_m*cos(m*phi')]] at (rho, phi, z),
        reconstructing from the per-mode nodal density solved above.
        sigma_m has shape (n_max+1, n_profile)."""
        p0, p1, tangent, normal = self._geom
        n_seg = len(p0)
        e_rho = np.zeros(n_max + 1)
        e_z = np.zeros(n_max + 1)
        e_phi_amp = np.zeros(n_max + 1)  # coefficient of sin(m*phi) in E_phi
        for j in range(n_seg):
            for xi, wi in zip(_GAUSS_X, _GAUSS_W):
                t = 0.5 * (xi + 1.0)
                rho_s = p0[j, 0] + t * (p1[j, 0] - p0[j, 0])
                z_s = p0[j, 1] + t * (p1[j, 1] - p0[j, 1])
                seg_len = np.linalg.norm(tangent[j]) * 0.5 * wi
                s0 = sigma_m[:, j]
                s1 = sigma_m[:, j + 1]
                dens = s0 + t * (s1 - s0)
                Phi, dPhi_drho, dPhi_dz = ring_field_modes(rho, z, rho_s, z_s, n_max)
                weight = dens * seg_len
                e_rho += -weight * dPhi_drho
                e_z += -weight * dPhi_dz
                m = np.arange(n_max + 1)
                e_phi_amp += weight * Phi * m / rho

        m = np.arange(n_max + 1)
        cos_m_phi = np.cos(m * phi)
        sin_m_phi = np.sin(m * phi)
        E_rho = np.sum(e_rho * cos_m_phi)
        E_z = np.sum(e_z * cos_m_phi)
        E_phi = np.sum(e_phi_amp * sin_m_phi)

        Ex = E_rho * np.cos(phi) - E_phi * np.sin(phi)
        Ey = E_rho * np.sin(phi) + E_phi * np.cos(phi)
        return np.array([Ex, Ey, E_z])

    def _field_from_joint_mode_density_batch(self, sigma_C, sigma_S, rho_q, phi_q, z_q, n_max, xp=np, n_subdiv=4):
        """Vectorized generalization of the single-particle field
        reconstruction to (a) many query points at once, grouped (shape
        (G, E) arrays, one independent density per group -- see
        `image_force`'s group-isolation requirement) and (b) a *joint*
        density with both a cos(m*phi') part (sigma_C) and a sin(m*phi')
        part (sigma_S) -- needed once source particles sit at different,
        arbitrary azimuths (see `image_force`'s module-docstring-referenced
        derivation), unlike the single-particle `image_field`'s path,
        which gets away with a cosine-only series by always rotating the
        one source to phi'=0 first.

        A single (n_max+1, G, E, n_quad) broadcast over the fixed
        quadrature from `_quadrature_geometry` -- no Python loop over
        profile segments (earlier versions of this method had one; see
        `_segment_quadrature_geometry`'s docstring) -- so this is both
        faster on CPU and, more importantly, GPU-friendly: one batch of
        array ops instead of one Python-level (hence one round of kernel
        launches) iteration per segment.

        sigma_C, sigma_S : shape (n_max+1, G, n_profile), on xp -- group g's
            query points are only ever evaluated against group g's own
            solved density (never another group's), which is what keeps
            groups from cross-coupling here.
        rho_q, phi_q, z_q : shape (G, E), on xp.

        Returns
        -------
        E : array (xp), shape (G, E, 3)
        """
        rho_quad_np, z_quad_np, seg_len_np, seg_index_np, t_local_np = self._quadrature_geometry(n_subdiv)
        rho_quad = _to_xp(rho_quad_np, xp)
        z_quad = _to_xp(z_quad_np, xp)
        seg_len = _to_xp(seg_len_np, xp)
        seg_index = _to_xp(seg_index_np, xp)
        t_local = _to_xp(t_local_np, xp)

        dens_c = _interp_nodal_density(sigma_C, seg_index, t_local)  # (n_max+1, G, n_quad)
        dens_s = _interp_nodal_density(sigma_S, seg_index, t_local)
        weight_c = dens_c * seg_len[None, None, :]
        weight_s = dens_s * seg_len[None, None, :]

        # (n_max+1, G, E, n_quad): every (group, query-within-group) point
        # against every quadrature node of that *same* group's density.
        Phi, dPhi_drho, dPhi_dz = ring_field_modes(
            rho_q[:, :, None], z_q[:, :, None], rho_quad[None, None, :], z_quad[None, None, :], n_max, xp=xp
        )

        e_rho_c = xp.sum(-weight_c[:, :, None, :] * dPhi_drho, axis=-1)  # (n_max+1, G, E)
        e_rho_s = xp.sum(-weight_s[:, :, None, :] * dPhi_drho, axis=-1)
        e_z_c = xp.sum(-weight_c[:, :, None, :] * dPhi_dz, axis=-1)
        e_z_s = xp.sum(-weight_s[:, :, None, :] * dPhi_dz, axis=-1)

        m = xp.arange(n_max + 1, dtype=float)
        m_over_rho = (m[:, None, None] / rho_q[None, :, :])[..., None]  # (n_max+1, G, E, 1)
        e_phi_amp_c = xp.sum(weight_c[:, :, None, :] * Phi * m_over_rho, axis=-1)
        e_phi_amp_s = xp.sum(weight_s[:, :, None, :] * Phi * m_over_rho, axis=-1)

        cos_m_phi = xp.cos(m[:, None, None] * phi_q[None, :, :])  # (n_max+1, G, E)
        sin_m_phi = xp.sin(m[:, None, None] * phi_q[None, :, :])

        E_rho = xp.sum(e_rho_c * cos_m_phi + e_rho_s * sin_m_phi, axis=0)  # (G, E)
        E_z = xp.sum(e_z_c * cos_m_phi + e_z_s * sin_m_phi, axis=0)
        E_phi = xp.sum(e_phi_amp_c * sin_m_phi - e_phi_amp_s * cos_m_phi, axis=0)

        Ex = E_rho * xp.cos(phi_q) - E_phi * xp.sin(phi_q)
        Ey = E_rho * xp.sin(phi_q) + E_phi * xp.cos(phi_q)
        return xp.stack([Ex, Ey, E_z], axis=-1)


def _point_charge_field(rho, phi, z, rho0, phi0, z0, charge):
    """Ordinary Coulomb field (G = 1/(4*pi*r)) of a point charge at
    (rho0, phi0, z0), evaluated at (rho, phi, z) -- used to add the mirror
    charge's own direct contribution back (see module docstring)."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    zc = z
    x0 = rho0 * np.cos(phi0)
    y0 = rho0 * np.sin(phi0)
    d = np.array([x - x0, y - y0, zc - z0])
    r = np.linalg.norm(d)
    return charge * d / (4.0 * np.pi * r**3)


def _cylindrical_to_cartesian(rho, phi, z, xp=np):
    return xp.stack([rho * xp.cos(phi), rho * xp.sin(phi), z], axis=-1)


def _point_charge_field_batch_all_pairs(query_xyz, source_xyz, source_charge, xp=np):
    """Ordinary Coulomb field (G = 1/(4*pi*r)) at each of E query points in
    each of G groups, due to the sum of the E mirror charges *in that same
    group* -- the within-group-only all-pairs cross term needed when
    adding every particle's own local mirror charge back (see
    `ImageChargeBEMSolution.image_force`'s "cross" step): particle i in
    group g feels the direct field of every active particle's mirror
    charge in group g, including its own (j=i, which reduces to the
    single-particle self-image term) -- never a particle in a different
    group (matching `forces._pairwise_force`'s own "gij,gijc->gic"
    group-preserving einsum, which this mirrors).

    Parameters
    ----------
    query_xyz : array, shape (G, E, 3)
    source_xyz : array, shape (G, E, 3)
    source_charge : array, shape (G, E)

    Returns
    -------
    E : array, shape (G, E, 3)
    """
    diff = query_xyz[:, :, None, :] - source_xyz[:, None, :, :]  # (G, E, E, 3)
    r = xp.linalg.norm(diff, axis=-1)  # (G, E, E)
    r_safe = xp.where(r > 0.0, r, 1.0)
    coeff = xp.where(r > 0.0, source_charge[:, None, :] / (4.0 * xp.pi * r_safe**3), 0.0)
    return xp.einsum("gnm,gnmc->gnc", coeff, diff)
