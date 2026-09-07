"""Accumulates trajectory snapshots at requested output times -- for
plotting trajectories vs. time, as opposed to reading particle coordinates
off z-screens (screens.py).

batched_rk45.integrate already forces every group to land exactly on each
requested output time (see its `output_times`/`on_output`), so there is no
interpolation here: a TrajectoryRecorder just accumulates the (already
flattened to individual, born-by-then particles) batches handed to it as
groups reach its one output time -- across possibly many different
integration steps, since different groups advance at different rates.
"""

import numpy as np


class TrajectoryRecorder:
    """Accumulates every born particle's phase-space state at one
    requested output time. Structurally the same accumulate/concatenate
    pattern as screens.ScreenRecorder, just keyed by a time instead of a
    z-plane."""

    def __init__(self, t, xp=np):
        self.t = float(t)
        self.xp = xp
        self._records = []  # list of dict-of-arrays

    def record(self, t, pos, vel, charge, ids, weight):
        self._records.append(
            dict(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                vx=vel[:, 0], vy=vel[:, 1], vz=vel[:, 2],
                t=t, charge=charge, id=ids, weight=weight,
            )
        )

    def has_records(self):
        return len(self._records) > 0

    def concatenate(self):
        keys = self._records[0].keys()
        return {k: self.xp.concatenate([r[k] for r in self._records]) for k in keys}


def record_output(recorders, output_index, group_idx, t, pos, vel, t_birth, charge, ids, weight, xp=np):
    """`on_output` callback body shared by the in-process tracker and each
    parallel worker: filter `group_idx`'s particles down to the ones born
    by this output time, flatten, and hand them to
    ``recorders[output_index]``.

    Parameters
    ----------
    recorders : list of TrajectoryRecorder
    output_index : int
    group_idx : ndarray, shape (k,)
        Group indices (local to whatever `t_birth`/`charge`/`ids`/`weight`
        are indexed by) that reached this output time on this call.
    t : ndarray, shape (k,)
    pos, vel : ndarray, shape (k, n_emit, 3)
    t_birth, charge, ids, weight : ndarray, shape (n_groups, n_emit)
        Full per-group bookkeeping arrays; indexed here by group_idx.
    """
    tb = t_birth[group_idx]
    born = tb <= t[:, None]
    if not xp.any(born):
        return

    gi, pi = xp.nonzero(born)
    c = charge[group_idx]
    i = ids[group_idx]
    w = weight[group_idx]

    recorders[output_index].record(
        t[gi], pos[gi, pi], vel[gi, pi], c[gi, pi], i[gi, pi], w[gi, pi],
    )
