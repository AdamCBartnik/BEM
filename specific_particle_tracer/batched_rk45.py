"""A vectorized, adaptive Dormand-Prince RK45 integrator with an
*independent* step size per group.

Groups (e.g. n_emit-particle emission groups) never interact with each
other, so there is no physical reason for one group's step size to be
throttled by another's. A single `scipy.integrate.solve_ivp` call over the
whole flattened multi-group state vector would do exactly that (its
adaptive step is chosen to satisfy every component's error at once), which
gets very wasteful once there are many independent groups with different
dynamical timescales -- e.g. one group with a tight two-electron encounter
forcing a tiny step onto every other, unrelated group.

Instead, every group here carries its own current time and step size, and
groups are advanced together only because that's convenient for
vectorization (the acceleration callback is still one batched numpy call
across all currently-running groups) -- each group's accept/reject
decision and step-size update is entirely its own.

This is deliberately dependency-light (just the Butcher tableau) rather
than reusing `scipy.integrate.RK45`, precisely because scipy's stepper
object owns a single global time/step, which is the thing being avoided
here.
"""

import numpy as np

from .constants import SPEED_OF_LIGHT

# Dormand-Prince 5(4) coefficients (same pair scipy's RK45 uses).
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
]
_B = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
_E = np.array(
    [
        71 / 57600,
        0.0,
        -71 / 16695,
        71 / 1920,
        -17253 / 339200,
        22 / 525,
        -1 / 40,
    ]
)
_N_STAGES = 6
_ERROR_ESTIMATOR_ORDER = 4


def integrate(
    accel_fn,
    pos0,
    vel0,
    t_birth,
    t_max,
    rtol,
    atol,
    on_step=None,
    kill_fn=None,
    output_times=None,
    on_output=None,
    xp=np,
    safety=0.9,
    min_factor=0.2,
    max_factor=10.0,
    initial_step_fraction=1e-6,
    max_steps=10_000_000,
):
    """Integrate every group from its own earliest birth time to t_max, with
    its own adaptive step.

    Parameters
    ----------
    accel_fn : callable(pos, vel, active, group_idx) -> accel
        pos, vel, accel: ndarray (n_running, n_emit, 3). active: ndarray of
        bool (n_running, n_emit). group_idx: ndarray (n_running,) giving
        the original (0..n_groups-1) index of each row, constant across
        all stages of one step attempt -- so that any group-static data
        (e.g. per-particle charge) the caller needs can be looked up
        consistently with the current running subset. Only needs to
        compute real physics for `active` entries; entries that are not
        (yet) active (unborn, or killed -- see `kill_fn`) may return
        anything for them, as they are masked out of the position/velocity
        update regardless.
    pos0, vel0 : ndarray, shape (n_groups, n_emit, 3)
    t_birth : ndarray, shape (n_groups, n_emit)
        Per-particle birth time [s]. A particle is inert (frozen) until t
        reaches this value; since it is never touched before then, it
        simply stays at its given (pos0, vel0) the whole time, so no
        special-cased "instantiation" step is needed -- the ODE's right-
        hand side just has a kink in time at t_birth, which the adaptive
        step size naturally resolves.
    t_max : float
        Common stopping time for every group [s].
    rtol, atol : float
        Convergence tolerance, RMS-combined over each group's own
        (position, velocity) state. `atol` is a floor in metres for the
        position components and in dimensionless beta = v/c for the
        velocity components, so that one scalar is meaningful for both
        (see `_attempt_step`). Non-relativistic here: the normalization is
        by c alone, not by gamma*beta.
    on_step : callable, optional
        Called after every batch of accepted steps as
        ``on_step(group_idx, t_old, pos_old, vel_old, t_new, pos_new, vel_new)``
        where ``group_idx`` gives the original group indices of the
        (possibly smaller) accepted subset, and the rest are ndarrays
        shaped (len(group_idx), n_emit, 3) / (len(group_idx),).
    kill_fn : callable(pos) -> bool array, optional
        A particle that was active going into a step and whose position
        satisfies `kill_fn(pos)` (shape (n_run, n_emit, 3) in, matching
        boolean array out) at the end of that step is permanently killed:
        frozen wherever that step left it, and excluded from all further
        dynamics (including no longer exerting or feeling forces on the
        rest of its group), from then on. This is checked once per
        accepted step (not root-found to the exact re-entry instant), so a
        killed particle's last position can be a little past the boundary
        by an amount set by the step size / tolerance in effect at the
        time. None (default) disables killing entirely.
    output_times : ndarray, optional
        Times [s] (shared by every group, unlike `t_birth`) at which every
        group's step is forced to land exactly -- clipped into the same
        checkpoint mechanism as birth times, rather than interpolated
        after the fact, so a recorded state is exact to the integrator's
        own tolerance rather than to the accuracy of some separate
        interpolant.
    on_output : callable, optional
        Called whenever one or more groups land exactly on one of
        `output_times`, as
        ``on_output(output_index, group_idx, t, pos, vel)`` where
        `output_index` is the (scalar) index into `output_times` just
        reached, `group_idx` gives the original group indices that reached
        it on this call (a group reaches each output time exactly once,
        but different groups can reach the same one on different calls, or
        several at once on the same call), and `t`/`pos`/`vel` are that
        group subset's state there (`t` is redundant with
        `output_times[output_index]` but included for convenience/dtype
        consistency).

    Returns
    -------
    pos, vel : ndarray, shape (n_groups, n_emit, 3)
        Final state of every group at t_max.
    """
    n_groups, n_emit, _ = pos0.shape

    pos = pos0.copy()
    vel = vel0.copy()
    t_death = xp.full((n_groups, n_emit), xp.inf, dtype=pos0.dtype)

    # Each group's own distinct birth times, ascending (a particle that is
    # never touched before its birth stays bit-exact at its given initial
    # condition, so there is no "jump" to insert at these times -- but a
    # step is never allowed to straddle one. Without that, a particle
    # sitting exactly at rest on a field discontinuity (e.g. z=0, the
    # cathode surface) can make an RK stage's position dither by roundoff
    # onto the wrong side of the discontinuity, which no amount of step
    # shrinking can resolve (it isn't actually a smoothness problem) and
    # the adaptive controller can spin forever trying anyway.
    checkpoints = xp.sort(t_birth, axis=1)

    # Each group starts its own clock at its own earliest birth time, not
    # at a shared t=0 -- a fixed t=0 start silently breaks the *relative*
    # birth timing within a group whenever any member has t_birth < 0
    # (e.g. a distgen distribution with mean(t)=0): that member would be
    # "active" from the very first step regardless of how negative its
    # birth time really is, since t=0 already satisfies t >= t_birth,
    # collapsing what should be a real head-start over its later-born
    # groupmates. Groups never interact with each other, so starting each
    # at its own min(t_birth) (rather than the distribution's global
    # minimum) also avoids wasting steps on a group whose own members are
    # all born well after some other group's earliest particle.
    t = checkpoints[:, 0].copy()
    h = (t_max - t) * initial_step_fraction

    # Output times are shared by every group (unlike birth times), so they
    # get their own single sorted 1-D array and their own "how many are
    # behind us" counter per group, folded into the same `target` a step
    # is clipped to rather than merged into `checkpoints` itself (which
    # would conflate them with the per-particle birth times `active` is
    # computed from).
    have_output_times = output_times is not None
    if have_output_times:
        output_times_sorted = xp.sort(output_times)
        n_out = output_times_sorted.shape[0]

    if have_output_times and on_output is not None:
        # An output time coinciding *exactly* with a group's own start time
        # would otherwise never fire for that group: the pointer test in
        # the loop below counts `output_times_sorted <= t` as already
        # behind us, and t starts at that very time. Output times strictly
        # before a group's start need no such handling -- nothing in that
        # group is born yet, so `trajectories.record_output` would record
        # nothing anyway -- but the exact coincidence does lose genuinely
        # born particles (the earliest-born ones), so fire those here.
        last_at_or_before = xp.sum(output_times_sorted[None, :] <= t[:, None], axis=1) - 1
        at_start = (last_at_or_before >= 0) & (
            output_times_sorted[xp.maximum(last_at_or_before, 0)] == t
        )
        if xp.any(at_start):
            at_start_idx = xp.nonzero(at_start)[0]
            ptrs = last_at_or_before[at_start_idx]
            for p in xp.unique(ptrs):
                sel = at_start_idx[ptrs == p]
                on_output(int(p), sel, t[sel], pos[sel], vel[sel])

    running = xp.ones(n_groups, dtype=bool)
    # `idx` doubles as the loop condition below (via idx.shape[0]) instead
    # of a separate `if not xp.any(running)` check: nonzero() already has
    # to know that count internally (a host sync on a GPU backend) to size
    # its result, so a second sync just to ask the same question again
    # would be pure waste.
    idx = xp.nonzero(running)[0]

    for _ in range(max_steps):
        if idx.shape[0] == 0:
            break

        # How many of this group's (sorted) checkpoints are already behind
        # us -- a single vectorized count rather than an incremental,
        # early-breaking Python loop, which on a GPU backend would cost a
        # host sync per increment for no benefit (n_emit is small, so
        # doing the full comparison unconditionally is cheap either way).
        n_reached = xp.sum(checkpoints[idx] <= t[idx][:, None], axis=1)

        cp_idx = xp.minimum(n_reached, n_emit - 1)
        exhausted = n_reached >= n_emit
        next_checkpoint = xp.where(exhausted, t_max, checkpoints[idx, cp_idx])
        target = xp.minimum(next_checkpoint, t_max)

        if have_output_times:
            out_ptr = xp.sum(output_times_sorted[None, :] <= t[idx][:, None], axis=1)
            out_exhausted = out_ptr >= n_out
            out_cp_idx = xp.minimum(out_ptr, n_out - 1)
            next_output = xp.where(out_exhausted, t_max, output_times_sorted[out_cp_idx])
            target = xp.minimum(target, next_output)

        h_r = xp.minimum(h[idx], target - t[idx])

        (
            pos_new, vel_new, t_new, accepted, h_next,
        ) = _attempt_step(
            accel_fn, pos[idx], vel[idx], t[idx], h_r, t_birth[idx], t_death[idx], idx,
            rtol, atol, safety, min_factor, max_factor, xp,
        )

        h[idx] = h_next

        # `idx[accepted]` (boolean-mask indexing) needs to know how many
        # entries survive to size its result, which is itself a host sync
        # on a GPU backend -- same cost whether or not it's wrapped in an
        # `if xp.any(accepted)` first, so there is nothing to gain by
        # skipping that check here (unlike the checkpoint loop above, whose
        # per-iteration syncs were purely incremental bookkeeping this
        # replaces with fixed-shape vector ops).
        if xp.any(accepted):
            acc_idx = idx[accepted]
            t_old_acc = t[acc_idx]
            pos_old_acc = pos[acc_idx]
            vel_old_acc = vel[acc_idx]

            was_active = (t_old_acc[:, None] >= t_birth[acc_idx]) & (t_old_acc[:, None] < t_death[acc_idx])

            pos[acc_idx] = pos_new[accepted]
            vel[acc_idx] = vel_new[accepted]
            t[acc_idx] = t_new[accepted]

            # A group whose step just landed on one of its own birth
            # checkpoints is about to integrate real dynamics for the
            # first time with a step size `h` that has nothing to do with
            # that -- it's whatever the *frozen* phase grew it to via free
            # 10x-per-step growth (zero derivative there, so every such
            # step is trivially accepted). Carrying that stale, often wildly
            # oversized h into the newly-active dynamics causes a huge
            # first-attempt error, several rejected retries, and can even
            # land the eventually-accepted step spuriously inside a kill
            # boundary that the particle's actual (outward) velocity would
            # never have carried it into. Reset to a conservative step so
            # the new dynamics ramps up normally instead.
            tol = 1e-9 * xp.maximum(t_max, 1e-30)
            reached_birth_checkpoint = (~exhausted[accepted]) & (t[acc_idx] >= next_checkpoint[accepted] - tol)
            if xp.any(reached_birth_checkpoint):
                h_reset = (t_max - t[acc_idx]) * initial_step_fraction
                h[acc_idx] = xp.where(reached_birth_checkpoint, h_reset, h[acc_idx])

            if kill_fn is not None:
                newly_killed = was_active & kill_fn(pos[acc_idx])
                if xp.any(newly_killed):
                    t_death[acc_idx] = xp.where(newly_killed, t[acc_idx][:, None], t_death[acc_idx])

            if have_output_times and on_output is not None:
                # `target` (hence `t_new`) was clipped to never exceed
                # `next_output`, so landing at/past it (up to float
                # tolerance) only happens when it was actually the binding
                # constraint -- a birth checkpoint or t_max landing would
                # leave t_new strictly below next_output instead.
                out_ptr_acc = out_ptr[accepted]
                next_output_acc = next_output[accepted]
                still_pending_acc = ~out_exhausted[accepted]
                tol = 1e-9 * xp.maximum(t_max, 1e-30)
                hit = still_pending_acc & (t[acc_idx] >= next_output_acc - tol)
                if xp.any(hit):
                    hit_local = xp.nonzero(hit)[0]
                    ptrs_hit = out_ptr_acc[hit_local]
                    for p in xp.unique(ptrs_hit):
                        sel_local = hit_local[ptrs_hit == p]
                        sel_global = acc_idx[sel_local]
                        on_output(int(p), sel_global, t[sel_global], pos[sel_global], vel[sel_global])

            if on_step is not None:
                on_step(acc_idx, t_old_acc, pos_old_acc, vel_old_acc, t[acc_idx], pos[acc_idx], vel[acc_idx])

            running[acc_idx] = t[acc_idx] < t_max * (1 - 1e-14)

        idx = xp.nonzero(running)[0]
    else:
        raise RuntimeError(f"batched RK45 did not converge within {max_steps} steps")

    return pos, vel


def _attempt_step(accel_fn, pos, vel, t, h, t_birth, t_death, group_idx, rtol, atol, safety, min_factor, max_factor, xp):
    """One Dormand-Prince step attempt for a batch of groups, each with its
    own t and h. Returns candidate (pos_new, vel_new, t_new), an `accepted`
    boolean mask, and the next step size to use for every group (whether
    accepted or not)."""
    n = pos.shape[0]
    k_pos = [None] * (_N_STAGES + 1)
    k_vel = [None] * (_N_STAGES + 1)

    # `active` is evaluated once, at the step's start, and held fixed for
    # every stage (including the endpoint). The caller guarantees a step
    # never straddles a birth-time checkpoint, so this is exact for the
    # entire [t, t+h] interval -- a particle's activation at t+h == its own
    # birth time is handled by the *next* step starting fresh there, not by
    # this one's endpoint evaluation. (Re-deriving `active` per stage from
    # the stage time would make the extra end-of-step evaluation flip on
    # while every earlier stage was still off whenever a step lands exactly
    # on a checkpoint, producing a spurious O(h) error term that no amount
    # of step-shrinking can satisfy.) A killed particle (t >= its t_death,
    # set once by the caller after some earlier step) is simply never
    # active again -- no analogous re-death boundary to worry about, since
    # death only gets set at an already-reached time.
    active = (t[:, None] >= t_birth) & (t[:, None] < t_death)

    for i in range(_N_STAGES):
        # Stage 0's coefficient row is empty (c_0 = 0), so pos_stage/vel_stage
        # would just be an unmodified copy of pos/vel -- skip the copy (and
        # its kernel launch) and use them directly instead.
        pos_stage = pos
        vel_stage = vel
        for j, a_ij in enumerate(_A[i]):
            if a_ij == 0.0:
                continue
            pos_stage = pos_stage + (h * a_ij)[:, None, None] * k_pos[j]
            vel_stage = vel_stage + (h * a_ij)[:, None, None] * k_vel[j]

        k_vel[i] = accel_fn(pos_stage, vel_stage, active, group_idx)
        k_pos[i] = xp.where(active[..., None], vel_stage, 0.0)

    pos_new = pos + h[:, None, None] * sum(_B[i] * k_pos[i] for i in range(_N_STAGES))
    vel_new = vel + h[:, None, None] * sum(_B[i] * k_vel[i] for i in range(_N_STAGES))
    t_new = t + h

    k_vel[_N_STAGES] = accel_fn(pos_new, vel_new, active, group_idx)
    k_pos[_N_STAGES] = xp.where(active[..., None], vel_new, 0.0)

    err_pos = h[:, None, None] * sum(_E[i] * k_pos[i] for i in range(_N_STAGES + 1))
    err_vel = h[:, None, None] * sum(_E[i] * k_vel[i] for i in range(_N_STAGES + 1))

    # The velocity channel's error is measured in units of c (i.e. on beta)
    # rather than in m/s, so that one scalar `atol` means something
    # comparable in both channels. Writing the beta-space criterion
    # |err_vel|/c <= atol + rtol*|vel|/c and multiplying through by c
    # leaves the relative term untouched -- it is already scale-free -- and
    # simply converts atol's units from m/s to dimensionless beta.
    # Without this, a single atol is being compared against metres in one
    # channel and metres-per-second in the other, so its effective
    # tightness differs between them by ~c.
    scale_pos = atol + rtol * xp.maximum(xp.abs(pos), xp.abs(pos_new))
    scale_vel = SPEED_OF_LIGHT * atol + rtol * xp.maximum(xp.abs(vel), xp.abs(vel_new))

    n_components = 2 * pos.shape[1] * 3
    sq = (
        xp.sum((err_pos / scale_pos) ** 2, axis=(1, 2))
        + xp.sum((err_vel / scale_vel) ** 2, axis=(1, 2))
    )
    error_norm = xp.sqrt(sq / n_components)
    error_norm = xp.where(error_norm == 0.0, np.finfo(float).tiny, error_norm)

    accepted = error_norm <= 1.0

    factor = safety * error_norm ** (-1.0 / (_ERROR_ESTIMATOR_ORDER + 1))
    factor = xp.clip(factor, min_factor, max_factor)
    h_next = h * factor

    return pos_new, vel_new, t_new, accepted, h_next
