"""Multi-process CPU parallelism, splitting work across groups.

Groups never interact with each other (that's the whole basis for the
per-group adaptive integrator in batched_rk45.py), so splitting the set of
groups across worker processes and running each subset independently is a
correct, embarrassingly-parallel decomposition -- no communication needed
between workers while they run, just a merge of their screen-crossing and
trajectory-output results at the end.

This uses multiple *processes* (concurrent.futures.ProcessPoolExecutor),
not threads: the per-step control flow in batched_rk45.integrate is a
Python-level loop with a lot of small operations, which would mostly
serialize on the GIL under threads. Real cores need real processes.

Only for the 'cpu' backend -- 'gpu' already parallelizes across the whole
problem on the GPU itself, and spinning up multiple processes each trying
to drive the same device would fight over it rather than help.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from . import batched_rk45
from . import geometry as geometry_module
from .screens import ScreenRecorder, find_crossings_grouped
from .trajectories import TrajectoryRecorder, record_output


def run_parallel(
    n_workers,
    init_pos, init_vel, t_birth, charge, weight, ids,
    mass, geometry_worker_args, plummer_radius, screens, output_times,
    rtol, atol, t_max,
):
    """Run the tracker's integration split across `n_workers` processes,
    partitioning groups (the leading axis of the grouped arrays) evenly
    across them, and return (screens, trajectories): each a list of merged
    dict-of-arrays (or None where there were no records at all), ready to
    build a ParticleGroup from -- one per entry of `screens` /
    `output_times` respectively.

    `geometry_worker_args` is a Geometry's `worker_args()` tuple (plain
    picklable values) rather than a Geometry object itself, so each worker
    rebuilds its own local, numpy-backed copy via
    `geometry.Geometry.from_worker_args` -- the same round trip used to
    move a Geometry onto whatever backend the tracker is using, here just
    used to get it into a fresh worker process instead.
    """
    n_groups = init_pos.shape[0]
    chunk_group_indices = [c for c in np.array_split(np.arange(n_groups), n_workers) if len(c) > 0]

    jobs = [
        (
            init_pos[idx], init_vel[idx], t_birth[idx], charge[idx], weight[idx], ids[idx],
            mass, geometry_worker_args, plummer_radius, screens, output_times, rtol, atol, t_max,
        )
        for idx in chunk_group_indices
    ]

    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        per_chunk_results = list(pool.map(_run_chunk, jobs))

    chunk_screens = [r[0] for r in per_chunk_results]
    chunk_trajectories = [r[1] for r in per_chunk_results]

    merged_screens = [_merge_records([c[i] for c in chunk_screens]) for i in range(len(screens))]
    n_out = 0 if output_times is None else len(output_times)
    merged_trajectories = [_merge_records([c[i] for c in chunk_trajectories]) for i in range(n_out)]

    return merged_screens, merged_trajectories


def _merge_records(pieces):
    pieces = [p for p in pieces if p is not None]
    if not pieces:
        return None
    keys = pieces[0].keys()
    return {k: np.concatenate([p[k] for p in pieces]) for k in keys}


def _run_chunk(args):
    (
        init_pos, init_vel, t_birth, charge, weight, ids,
        mass, geometry_worker_args, plummer_radius, screens, output_times, rtol, atol, t_max,
    ) = args

    geometry = geometry_module.Geometry.from_worker_args(geometry_worker_args, xp=np)
    accel = geometry_module.make_accel_fn(charge, mass, geometry, plummer_radius, xp=np)

    recorders = [ScreenRecorder(z, xp=np) for z in screens]

    def on_step(group_idx, t_old, pos_old, vel_old, t_new, pos_new, vel_new):
        find_crossings_grouped(
            recorders, group_idx, t_old, pos_old, vel_old, t_new, pos_new, vel_new,
            charge, ids, weight, xp=np,
        )

    trajectory_recorders = None
    on_output = None
    if output_times is not None:
        trajectory_recorders = [TrajectoryRecorder(t, xp=np) for t in output_times]

        def on_output(output_index, group_idx, t, pos, vel):
            record_output(
                trajectory_recorders, output_index, group_idx, t, pos, vel,
                t_birth, charge, ids, weight, xp=np,
            )

    batched_rk45.integrate(
        accel, init_pos, init_vel, t_birth, t_max, rtol, atol,
        on_step=on_step, kill_fn=geometry.kill_mask,
        output_times=None if output_times is None else np.asarray(output_times), on_output=on_output, xp=np,
    )

    screen_results = [r.concatenate() if r.has_records() else None for r in recorders]
    trajectory_results = (
        [] if trajectory_recorders is None
        else [r.concatenate() if r.has_records() else None for r in trajectory_recorders]
    )
    return screen_results, trajectory_results
