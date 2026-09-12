"""Data helpers for visualizing this project's shapes: (rho, z) generating
profiles, and (rho, z)-grid evaluations of the static and image-charge
potentials -- for both the closed-form analytic cathode shapes (fields.py)
and their BEM counterparts (bem/fields.py, bem/image_charge.py) alike.

Deliberately data-only, not a plotting library: `static_potential_grid`/
`image_potential_grid` just return `(R, Z, V)` numpy arrays -- no
matplotlib import here at all, no opinions about colormap, contour
levels, or colorbar norm baked in for you to fight. An early version of
this module also shipped ready-made `plot_*` wrappers around these; they
were dropped (see git history) once it became clear that plot styling
(a contour level count, a norm, a colorbar) is exactly the kind of detail
worth owning yourself rather than routing through someone else's
defaults -- the actual bug that prompted dropping them was a colorbar
wasting half its range on a sign of value that didn't occur in the data,
from an over-clever attempt at guessing a good symlog level set. See
examples/bem_plotting_example.py for a plotting recipe built directly on
these two functions (and `plot_profiles` below, which -- unlike the two
grid functions -- has no such judgment calls to get wrong, so it stays a
thin convenience).

`plot_profiles` is the one actual matplotlib helper kept here, since it's
just drawing given (already-computed) line data with no numeric choices
of its own -- requires matplotlib (`pip install -e .[plot]`) only if you
call it; the two grid functions have no such dependency at all.

Static potential vs. image-charge potential
--------------------------------------------
`static_potential_grid` computes the *total* external potential (uniform
gun field + the shape's own perturbation) -- purely axisymmetric for
every geometry in this project (the excitation is a uniform on-axis
field), so one (rho, z) meridian slice is the whole story, independent of
phi.

`image_potential_grid` is a different, genuinely 3D quantity: the
potential induced in the conductor by one specific point charge (see
`bem.image_charge.ImageChargeBEMSolution.image_potential`). Unless the
query slice happens to be the source's own meridian half-plane (query_phi
= 0) or its opposite (query_phi = pi), this is *not* an axisymmetric
picture -- a different query_phi gives a genuinely different-looking
result, which is why this function takes one explicitly rather than
assuming phi doesn't matter the way the static-potential grid can.
"""

import numpy as np


def plot_profiles(ax, real_profile=None, image_profile=None, show_nodes=True, real_label="real surface", image_label="image surface"):
    """Plot one or more (rho, z) generating profiles -- the physical
    conductor boundary and/or its recessed image-charge surface -- on a
    matplotlib Axes, in the meridian half-plane (rho >= 0).

    `real_profile`/`image_profile` : (n, 2) arrays of (rho, z) points, in
    order along the surface. For a BEM geometry these are the actual
    discretization -- e.g. `field.solution.profile` (real) and
    `image_solution.profile` (image), or straight from
    `bem.mesh.hemisphere_tip_real_profile`/`_image_profile`,
    `cylindrical_well_real_profile`/`_image_profile`.

    For an *analytic* geometry there's no solver mesh to show at all --
    the reasonable default used throughout this project's own examples is
    to call those same `bem.mesh` profile functions directly on the
    analytic shape's own parameters (R, z0, ...): they're pure geometry,
    independent of which method actually solves the field on that shape,
    so they still draw the right boundary and image-offset curves. Pass
    `show_nodes=False` in that case (there's no real discretization to
    mark) -- see examples/bem_plotting_example.py.
    """
    if real_profile is not None:
        real_profile = np.asarray(real_profile)
        ax.plot(real_profile[:, 0], real_profile[:, 1], "-", color="tab:blue", lw=1.5, label=real_label)
        if show_nodes:
            ax.plot(real_profile[:, 0], real_profile[:, 1], "o", color="tab:blue", ms=3)
    if image_profile is not None:
        image_profile = np.asarray(image_profile)
        ax.plot(image_profile[:, 0], image_profile[:, 1], "--", color="tab:orange", lw=1.5, label=image_label)
        if show_nodes:
            ax.plot(image_profile[:, 0], image_profile[:, 1], "o", color="tab:orange", ms=3)
    ax.set_xlabel("r [m]")
    ax.set_ylabel("z [m]")
    # Not set_aspect("equal") here: these profiles are usually far wider
    # (out to several R of flat plane) than tall, and forcing equal
    # aspect in a multi-panel figure shrinks the axes box to match,
    # pushing the title away from it (a matplotlib layout quirk, not a
    # data problem) -- set it yourself on `ax` after calling this if you
    # want a physically-undistorted view of one profile on its own.
    ax.axhline(0.0, color="0.7", lw=0.5, zorder=0)
    ax.legend()
    return ax


def _r_z_grid(r_max, z_range, n_r, n_z, r_min=0.0):
    r = np.linspace(r_min, r_max, n_r)
    z = np.linspace(z_range[0], z_range[1], n_z)
    return np.meshgrid(r, z)


def static_potential_grid(field, r_max, z_range, n_r=150, n_z=150):
    """The total static potential Phi(r, z) on a grid in the (r, z)
    meridian plane -- e.g. to feed straight into `ax.contourf`/`pcolormesh`
    with whatever colormap, levels, and colorbar you actually want (see
    examples/bem_plotting_example.py for one worked recipe).

    Parameters
    ----------
    field : object with `.potential(position)`, position shape (..., 3) --
        `fields.GunField`, `fields.HemisphericalTipField`,
        `bem.fields.HemisphericalTipBEMField`,
        `bem.fields.CylindricalWellBEMField` all have one.
    r_max : float
        Grid spans r in [0, r_max].
    z_range : (z_min, z_max)
    n_r, n_z : int
        Grid resolution.

    Returns
    -------
    R, Z, V : ndarrays, shape (n_z, n_r) (numpy.meshgrid's default
        indexing) -- V is the total potential [V] at each (R, Z) point, 0
        inside the conductor (physically correct there, not a masked-out
        placeholder -- a grounded conductor's interior sits at its own
        boundary potential).
    """
    R, Z = _r_z_grid(r_max, z_range, n_r, n_z)
    position = np.stack([R, np.zeros_like(R), Z], axis=-1)
    V = field.potential(position)
    return R, Z, V


def image_potential_grid(
    image_solution, source_r0, source_z0, r_max, z_range, charge=1.0, d_lo=None, d_hi=None,
    query_phi=0.0, n_r=150, n_z=150, r_min=None,
):
    """A single point charge's conductor-induced potential, on a grid in
    the (r, z) meridian plane at a fixed query azimuth -- see
    `bem.image_charge.ImageChargeBEMSolution.image_potential` for the
    physics, and examples/bem_plotting_example.py for a plotting recipe
    (this potential diverges like 1/distance approaching the source
    itself, which is worth a log-scaled colormap rather than a plain
    linear one -- see that recipe for one way to do that without wasting
    half a colorbar on a sign of value the data doesn't have).

    Note this quantity is *not* generally axisymmetric the way
    `static_potential_grid`'s is -- see this module's own docstring.
    `query_phi=0.0` (default) is the source's own meridian half-plane,
    where the induced potential is largest; try `query_phi=pi/2` or `pi`
    to see how it falls off away from the source's own half-plane.

    Parameters
    ----------
    image_solution : bem.image_charge.ImageChargeBEMSolution
    source_r0, source_z0 : float
        Source charge position (r, phi=0, z).
    r_max : float
        Grid spans r in [r_min, r_max] (r_min defaults to a small
        fraction of r_max, avoiding the coordinate-singular axis).
    z_range : (z_min, z_max)
    charge : float, optional
        Source charge [C]. Default 1.0 -- rescale the result yourself
        (linear in charge) rather than re-solving for a different value.
    d_lo, d_hi : float, optional
        Mirror-charge blend distances (see `bem.image_charge`'s module
        docstring). Default 0.1*r_max, 0.5*r_max.
    query_phi : float, optional
        Azimuth of the query slice, radians. Default 0.0.
    n_r, n_z : int, optional
        Grid resolution.
    r_min : float, optional
        Default 1e-3 * r_max (avoids rho=0 exactly, a coordinate
        singularity in the underlying toroidal-harmonic kernels).

    Returns
    -------
    R, Z, V : ndarrays, shape (n_z, n_r) -- V [V] is the conductor's
        induced-potential response to the source charge, at each (R, Z)
        point in the query_phi slice (R, Z are the in-slice cylindrical
        radius and height, not the source's own coordinates). 0 inside
        the conductor (see `image_potential`'s own docstring for why
        that's the physically correct value there, not just a mask).
    """
    if r_min is None:
        r_min = 1e-3 * r_max
    if d_lo is None:
        d_lo = 0.1 * r_max
    if d_hi is None:
        d_hi = 0.5 * r_max

    R, Z = _r_z_grid(r_max, z_range, n_r, n_z, r_min=r_min)
    x = R * np.cos(query_phi)
    y = R * np.sin(query_phi)
    query_position = np.stack([x, y, Z], axis=-1)

    source_position = np.array([source_r0, 0.0, source_z0])
    V = image_solution.image_potential(source_position, charge, query_position, d_lo, d_hi)
    return R, Z, V
