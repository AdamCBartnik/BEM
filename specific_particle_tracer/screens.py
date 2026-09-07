"""Screen-crossing detection.

Each screen is a fixed z-plane. Every time a particle's trajectory crosses
that plane during an accepted integration step, its phase-space
coordinates at the crossing are recorded. The crossing is located with a
cubic Hermite interpolant built from the (position, velocity) already
known at both ends of the step -- no extra force evaluations needed -- and
solved with a fully vectorized bisection, so an arbitrary number of
simultaneous crossings (across many groups/particles/screens) are resolved
in one batch rather than one Python-level root-find at a time.
"""

import numpy as np

_BISECTION_ITERATIONS = 60


class ScreenRecorder:
    """Accumulates crossing events for a single z position."""

    def __init__(self, z, xp=np):
        self.z = float(z)
        self.xp = xp
        self._records = []  # list of dict-of-arrays

    def record(self, t, pos, vel, charge, ids, weight, group_index):
        self._records.append(
            dict(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                vx=vel[:, 0], vy=vel[:, 1], vz=vel[:, 2],
                t=t, charge=charge, id=ids, weight=weight, group_index=group_index,
            )
        )

    def has_records(self):
        return len(self._records) > 0

    def concatenate(self):
        keys = self._records[0].keys()
        return {k: self.xp.concatenate([r[k] for r in self._records]) for k in keys}


def find_crossings_grouped(
    recorders, group_idx, t_old, pos_old, vel_old, t_new, pos_new, vel_new,
    charge_grouped, ids_grouped, weight_grouped, xp=np,
):
    """Check a batch of accepted group-steps for screen crossings.

    Parameters
    ----------
    recorders : list of ScreenRecorder
    group_idx : ndarray, shape (k,)
        Original group index of each accepted step in this batch.
    t_old, t_new : ndarray, shape (k,)
    pos_old, vel_old, pos_new, vel_new : ndarray, shape (k, n_emit, 3)
    charge_grouped, ids_grouped, weight_grouped : ndarray, shape (n_groups, n_emit)
        Full per-group bookkeeping arrays; indexed here by group_idx.
    xp : module, optional
        Array backend (numpy or cupy). Default numpy.
    """
    k, n_emit, _ = pos_old.shape
    dt = t_new - t_old  # (k,)
    dt_full = xp.broadcast_to(dt[:, None], (k, n_emit))
    t_old_full = xp.broadcast_to(t_old[:, None], (k, n_emit))

    charge = charge_grouped[group_idx]
    ids = ids_grouped[group_idx]
    weight = weight_grouped[group_idx]
    group_col = xp.broadcast_to(group_idx[:, None], (k, n_emit))

    for recorder in recorders:
        z0 = pos_old[..., 2] - recorder.z
        z1 = pos_new[..., 2] - recorder.z
        crossed = (z0 == 0.0) | (xp.sign(z0) != xp.sign(z1))
        if not xp.any(crossed):
            continue

        gi, pi = xp.nonzero(crossed)

        p0, v0 = pos_old[gi, pi], vel_old[gi, pi]
        p1, v1 = pos_new[gi, pi], vel_new[gi, pi]
        dtc = dt_full[gi, pi]
        t0c = t_old_full[gi, pi]

        s = _bisect_hermite_z_root(p0[:, 2], v0[:, 2], p1[:, 2], v1[:, 2], dtc, recorder.z, xp)
        pos_c = _hermite(p0, v0, p1, v1, dtc, s)
        vel_c = _hermite_derivative(p0, v0, p1, v1, dtc, s)
        t_c = t0c + s * dtc

        recorder.record(
            t_c, pos_c, vel_c,
            charge[gi, pi], ids[gi, pi], weight[gi, pi], group_col[gi, pi],
        )


def _hermite_basis(s):
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return h00, h10, h01, h11


def _hermite_basis_derivative(s):
    s2 = s * s
    dh00 = 6 * s2 - 6 * s
    dh10 = 3 * s2 - 4 * s + 1
    dh01 = -6 * s2 + 6 * s
    dh11 = 3 * s2 - 2 * s
    return dh00, dh10, dh01, dh11


def _hermite(p0, v0, p1, v1, dt, s):
    h00, h10, h01, h11 = _hermite_basis(s)
    return h00[:, None] * p0 + (h10 * dt)[:, None] * v0 + h01[:, None] * p1 + (h11 * dt)[:, None] * v1


def _hermite_derivative(p0, v0, p1, v1, dt, s):
    dh00, dh10, dh01, dh11 = _hermite_basis_derivative(s)
    return (
        (dh00 / dt)[:, None] * p0 + dh10[:, None] * v0
        + (dh01 / dt)[:, None] * p1 + dh11[:, None] * v1
    )


def _hermite_z(p0z, v0z, p1z, v1z, dt, s):
    h00, h10, h01, h11 = _hermite_basis(s)
    return h00 * p0z + h10 * dt * v0z + h01 * p1z + h11 * dt * v1z


def _bisect_hermite_z_root(p0z, v0z, p1z, v1z, dt, target, xp):
    lo = xp.zeros_like(p0z)
    hi = xp.ones_like(p0z)
    f_lo = _hermite_z(p0z, v0z, p1z, v1z, dt, lo) - target

    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (lo + hi)
        f_mid = _hermite_z(p0z, v0z, p1z, v1z, dt, mid) - target
        same_sign_as_lo = xp.sign(f_mid) == xp.sign(f_lo)
        lo = xp.where(same_sign_as_lo, mid, lo)
        f_lo = xp.where(same_sign_as_lo, f_mid, f_lo)
        hi = xp.where(same_sign_as_lo, hi, mid)

    return 0.5 * (lo + hi)
