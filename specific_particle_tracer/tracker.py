"""SpecificParticleTracer: the main simulation driver."""

import os
import warnings

import numpy as np

try:
    from beamphysics import ParticleGroup
except ImportError:  # pragma: no cover
    from pmd_beamphysics import ParticleGroup

from . import batched_rk45
from . import backend as backend_module
from . import geometry as geometry_module
from . import parallel as parallel_module
from .constants import ev_c_to_si_momentum, si_momentum_to_ev_c, ev_to_kg
from .emission import group_indices
from .screens import ScreenRecorder, find_crossings_grouped
from .trajectories import TrajectoryRecorder, record_output

DEFAULT_PLUMMER_RADIUS = 1e-12  # m (1 pm)
DEFAULT_RTOL = 1e-7
# atol is a floor only: for velocities it sits far below what rtol already
# demands, and for positions it keeps a component passing through zero (x or
# y near the axis, z at the cathode plane) from demanding impossible
# precision. 1e-12 m is below every length scale in these problems, so
# there is nothing to gain by loosening it -- and it fails in two opposite
# ways on either side. Too loose (1e-9 and up) silently degrades position
# control to the scale of the geometry itself: for a particle a nanometre
# off the surface, rtol*|pos| is ~1e-16 m, so atol becomes the binding
# tolerance and errors of order the whole feature size are accepted. Too
# tight (roughly 1e-16 and below) stops the integration converging at all:
# a velocity component that starts at exactly zero then has its error
# compared against a bound below the force evaluation's own floating-point
# noise, and since that noise shrinks only like h rather than like h^5, the
# step controller collapses h without ever passing the test. That failure
# looks like a hang rather than a wrong answer.
DEFAULT_ATOL = 1e-12


class SpecificParticleTracer:
    """A lightweight tracker for electrons emitted from a cathode over the
    first few microns of flight, including space-charge repulsion and
    cathode image-charge attraction between particles emitted close
    together in time.

    The equations of motion (external field + Plummer-softened Coulomb
    repulsion + image-charge attraction, non-relativistic) are integrated
    with an adaptive-step Dormand-Prince RK45 scheme, giving every emission
    *group* its own independent step size -- since groups never interact,
    one group's close encounter (which needs small steps) never throttles
    every other group's step size, which matters once there are many
    groups.

    Parameters
    ----------
    initial_particles : ParticleGroup
        Initial phase space, in the openPMD-beamphysics convention. Each
        particle i is created at its own time ``t_i`` (as in GPT) with the
        position/momentum given in ``initial_particles``, then tracked
        forward. Particles are inert (frozen) until t_i.
    n_emit : int
        Particles are emitted (and interact -- both Coulomb repulsion and
        image-charge forces) in groups of this size. ``len(initial_particles)``
        must be evenly divisible by ``n_emit``. Groups are formed from
        particles sorted by ascending ID; particles in different groups
        never interact with each other.
    geometry : geometry.Geometry
        The cathode shape -- e.g. ``geometry.FlatCathode(E_gun=-1e8)`` or
        ``geometry.HemisphericalTip(E_gun=-1e8, R=50e-9)``. Bundles the
        external field, the image-charge force, and the "has this particle
        re-entered the conductor" test for that shape; see geometry.py.
        Build it with the default xp=numpy regardless of `backend` below --
        the tracker moves it onto the right array backend itself.
    screens : list of float
        z positions [m] at which to record particle coordinates every time
        a particle crosses them.
    t_out : list of float, optional
        Times [s] at which to record every (already-born) particle's full
        phase-space state, for plotting trajectories vs. time rather than
        reading them off a z-screen. Every group's adaptive step is forced
        to land exactly on each of these (like a screen crossing, but
        needing no interpolation since the integrator actually stops
        there), so the recorded state is exact to the RK45 tolerance
        itself. A particle not yet born by a given t_out is simply absent
        from that time's output; one already killed stays at its last
        (frozen) position for every later t_out, rather than disappearing.
    plummer_radius : float, optional
        Softening length [m] for the Coulomb interaction. Default 1e-12 m
        (1 pm).
    backend : {'cpu', 'gpu'}, optional
        Array backend. 'cpu' (default) runs on numpy. 'gpu' runs on cupy
        instead (requires a working cupy install) -- both compute in
        float64.
    n_workers : int or None, optional
        Only meaningful for backend='cpu'. Groups never interact, so the
        work splits cleanly across processes: n_workers=1 (default) runs
        everything in this process, as before; n_workers>1 partitions the
        groups evenly across that many worker *processes* (not threads --
        the per-step control flow is Python-level enough that threads
        would mostly just serialize on the GIL) and merges their screen
        crossings at the end; n_workers=None uses os.cpu_count(), which
        counts hyperthreads as full cores -- for CPU-bound floating point
        work like this, the physical core count is usually a better
        choice (hyperthread siblings share the same execution units, so
        they help much less than a real core does; in testing, going
        beyond the physical core count made things slightly worse rather
        than better). Not supported with backend='gpu' (each worker would
        need its own device context and they'd contend for the same GPU
        rather than help each other). Only worth it once per-worker
        compute outweighs process-startup cost -- for the field-only case
        (HemisphericalTip's closed-form field + analytic image force),
        measured directly on an 8-physical-core/16-thread machine:
        n_workers>1 was a net loss at 2000 groups (best case 1.41x at
        4-8 workers; 16 workers *slower* than 1), a moderate win by 10000
        groups (4.77x at 8 workers), and close to linear by 40000 groups
        (5.71x at 8, 7.01x at 16). A CPU-heavier per-group force -- e.g.
        bem.image_charge.ImageChargeBEMSolution.image_force, whose own
        per-group cost is real linear algebra rather than a closed-form
        evaluation -- earns a win at far smaller group counts instead
        (measured: 2.96x/5.39x/7.79x at 2/4/8 workers with just 50 groups
        sharing steps -- see that module's own docstring for why, and for
        the bug this same benchmarking caught: an earlier version let
        different groups' image-charge solves leak into each other). On
        Windows (and anywhere multiprocessing uses 'spawn'), calling code
        with n_workers>1 must sit behind ``if __name__ == "__main__":`` in
        the top-level script, or the worker processes will try to re-run
        that script themselves.
    rtol, atol : float, optional
        Relative and absolute convergence tolerance for the RK45
        integration, RMS-combined over each group's own (position [m],
        velocity) state. `atol` is a floor in metres for position and in
        dimensionless beta = v/c for velocity, so one value means something
        comparable in both channels. Defaults are 1e-7 and 1e-12: measured
        on a cylindrical-well trajectory, the integrator's own error at
        rtol=1e-7 is already ~90x smaller than the shift from changing the
        image-charge mode count, so tightening rtol further buys accuracy
        the force model does not have. Spend it on `image_n_max` and mesh
        resolution instead.
    t_max : float, optional
        Total simulated time [s]. If not given, a default is chosen from
        the single-particle transit time to the farthest screen.
    z_max : float, optional
        A particle that flies past z_max is killed (frozen, excluded from
        further dynamics) exactly like one that falls back into the
        conductor. Nothing in this tracker's field model ever turns off,
        so an escaped particle would otherwise coast all the way to t_max
        for no purpose; killing it once it's past the last z of interest
        (e.g. well beyond your farthest screen) can save real time on
        problems that would otherwise spend many steps on already-done
        particles. None (default) disables this cutoff.
    """

    def __init__(
        self,
        initial_particles,
        n_emit,
        geometry,
        screens,
        t_out=None,
        plummer_radius=DEFAULT_PLUMMER_RADIUS,
        backend="cpu",
        n_workers=1,
        rtol=None,
        atol=None,
        t_max=None,
        z_max=None,
    ):
        if n_workers != 1 and backend != "cpu":
            raise ValueError(
                f"n_workers is only supported for backend='cpu' (got backend={backend!r}, n_workers={n_workers!r})"
            )

        self.initial_particles = initial_particles
        self.n_emit = n_emit
        self.screen_positions = list(screens)
        self.output_times_host = None if t_out is None else np.sort(np.asarray(t_out, dtype=float))
        self.plummer_radius = plummer_radius
        self.z_max = z_max
        self.n_workers = os.cpu_count() if n_workers is None else n_workers

        self.backend_name = backend
        self.xp, self.dtype = backend_module.resolve_backend(backend)
        # Rebuilt via worker_args/from_worker_args regardless of what xp the
        # caller built `geometry` with, so a user never has to think about
        # backends when constructing a Geometry -- just pass backend= here.
        self.geometry = geometry_module.Geometry.from_worker_args(geometry.worker_args(), xp=self.xp)

        self.rtol = rtol if rtol is not None else DEFAULT_RTOL
        self.atol = atol if atol is not None else DEFAULT_ATOL

        self._setup_state()

        self.t_max = t_max if t_max is not None else self._default_t_max()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_state(self):
        pg = self.initial_particles
        idx = group_indices(pg, self.n_emit)  # (n_groups, n_emit), numpy
        self.n_groups, self.n_per_group = idx.shape

        mass_ev = pg.mass
        self.mass = ev_to_kg(mass_ev)  # single-particle rest mass [kg]

        # Scalar species charge-to-mass ratio (e.g. -e/m_e for an
        # electron), for the *external field* term: a macroparticle's
        # acceleration under a field is intrinsic to its species, not its
        # statistical weight. Charge-charge terms (Coulomb, image) instead
        # need each macroparticle's own (weight-scaled) charge and mass --
        # see geometry.make_accel_fn's docstring for why conflating these
        # is a real bug, not a hypothetical one.
        self.charge_to_mass = float(pg.species_charge) / self.mass

        # This tracker models individual real particles, not statistical
        # macroparticles -- Coulomb and image-charge forces here scale
        # with each particle's own weight (see geometry.make_accel_fn),
        # which is only physically meaningful if that weight really is one
        # elementary charge (e.g. a macroparticle's own self-image force
        # would otherwise scale with weight^2, not the weight it should
        # scale with for N real particles' aggregate self-attraction).
        # Rather than silently doing the wrong physics for a distribution
        # built with some other total-charge convention (as real-world
        # generators like distgen default to), every particle's weight is
        # forced to the species' elementary charge, with a warning if that
        # changed anything.
        species_charge_magnitude = abs(pg.species_charge)
        weight = np.asarray(pg.weight)
        # atol=0: these are ~1e-19 C values, and np.allclose's default
        # atol=1e-8 would call any two of them "close" regardless of rtol.
        if not np.allclose(weight, species_charge_magnitude, rtol=1e-9, atol=0):
            warnings.warn(
                "SpecificParticleTracer models individual real particles, not "
                "statistical macroparticles: overriding initial_particles.weight "
                f"(which was not uniformly {species_charge_magnitude:.6e} C, the "
                f"species' elementary charge) so every particle carries exactly "
                "one elementary charge of weight.",
                stacklevel=2,
            )
            weight = np.full_like(weight, species_charge_magnitude)

        charge_sign = np.sign(pg.species_charge)
        charge = charge_sign * weight  # signed, per particle [C]
        particle_mass = self.mass * (weight / species_charge_magnitude)

        x = np.asarray(pg.x)
        y = np.asarray(pg.y)
        z = np.asarray(pg.z)
        vx = ev_c_to_si_momentum(np.asarray(pg.px)) / self.mass
        vy = ev_c_to_si_momentum(np.asarray(pg.py)) / self.mass
        vz = ev_c_to_si_momentum(np.asarray(pg.pz)) / self.mass
        t_birth = np.asarray(pg.t)
        ids = np.asarray(pg.id)

        # Everything stays in this (n_groups, n_per_group, ...) grouped
        # form throughout -- the Coulomb/image sums need it, and it's also
        # exactly the shape the per-group batched integrator works with.
        # Built with numpy above (ParticleGroup is always numpy-backed),
        # then moved onto the selected backend/precision here.
        xp, dtype = self.xp, self.dtype
        self.init_pos = xp.asarray(np.stack([x[idx], y[idx], z[idx]], axis=-1), dtype=dtype)
        self.init_vel = xp.asarray(np.stack([vx[idx], vy[idx], vz[idx]], axis=-1), dtype=dtype)
        self.t_birth = xp.asarray(t_birth[idx], dtype=dtype)
        self.charge = xp.asarray(charge[idx], dtype=dtype)
        self.particle_mass = xp.asarray(particle_mass[idx], dtype=dtype)
        self.weight = xp.asarray(weight[idx], dtype=dtype)
        self.ids = xp.asarray(ids[idx])
        self.species = pg.species

        self.output_times = (
            None if self.output_times_host is None else xp.asarray(self.output_times_host, dtype=dtype)
        )

    def _default_t_max(self):
        a0 = self._field_accel_scale()
        z_screens = np.array(self.screen_positions, dtype=float)
        z_pos = z_screens[z_screens > 0]
        if len(z_pos) > 0 and a0 > 0:
            t_transit = np.sqrt(2 * z_pos.max() / a0)
        else:
            t_transit = 1e-12
        default_t_max = float(self.t_birth.max()) + 3.0 * t_transit
        if self.output_times_host is not None:
            default_t_max = max(default_t_max, float(self.output_times_host.max()))
        return default_t_max

    def _field_accel_scale(self):
        # The species charge-to-mass ratio, not weight-scaled: a
        # macroparticle's acceleration under the external field is
        # intrinsic to its species (see geometry.make_accel_fn).
        return abs(self.geometry.E_gun) * abs(self.charge_to_mass)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def run(self, verbose=False):
        """Run the simulation.

        Returns
        -------
        screens : list of ParticleGroup
            One ParticleGroup per entry in ``screens``, containing every
            crossing of that plane by any particle, in the order they
            were found (grouped by integration batch/worker, not by time).
        trajectories : list of ParticleGroup
            One ParticleGroup per entry in ``t_out`` (empty list if it
            wasn't given), containing every already-born particle's state
            at that time.
        """
        if self.n_workers > 1:
            merged_screens, merged_trajectories = parallel_module.run_parallel(
                self.n_workers,
                self.init_pos, self.init_vel, self.t_birth, self.charge, self.particle_mass, self.weight, self.ids,
                self.charge_to_mass, self.geometry.worker_args(), self.plummer_radius,
                self.screen_positions, self.output_times_host, self.rtol, self.atol, self.t_max, self.z_max,
            )
            if verbose:
                print(f"{self.n_groups} groups split across {self.n_workers} worker processes (backend=cpu)")
            return (
                [self._build_particle_group(rec) for rec in merged_screens],
                [self._build_particle_group(rec) for rec in merged_trajectories],
            )

        xp = self.xp
        accel_fn = geometry_module.make_accel_fn(
            self.charge, self.particle_mass, self.charge_to_mass, self.geometry, self.plummer_radius, xp=xp,
        )

        recorders = [ScreenRecorder(z, xp=xp) for z in self.screen_positions]
        n_step_batches = 0

        def on_step(group_idx, t_old, pos_old, vel_old, t_new, pos_new, vel_new):
            nonlocal n_step_batches
            n_step_batches += 1
            find_crossings_grouped(
                recorders, group_idx, t_old, pos_old, vel_old, t_new, pos_new, vel_new,
                self.charge, self.ids, self.weight, xp=xp,
            )

        trajectory_recorders = None
        on_output = None
        if self.output_times is not None:
            trajectory_recorders = [TrajectoryRecorder(t, xp=xp) for t in self.output_times_host]

            def on_output(output_index, group_idx, t, pos, vel):
                record_output(
                    trajectory_recorders, output_index, group_idx, t, pos, vel,
                    self.t_birth, self.charge, self.ids, self.weight, xp=xp,
                )

        kill_fn = geometry_module.make_kill_fn(self.geometry, self.z_max)
        batched_rk45.integrate(
            accel_fn, self.init_pos, self.init_vel, self.t_birth, self.t_max,
            self.rtol, self.atol, on_step=on_step, kill_fn=kill_fn,
            output_times=self.output_times, on_output=on_output, xp=xp,
        )

        if verbose:
            print(f"{n_step_batches} accepted step-batches across {self.n_groups} groups (backend={self.backend_name})")

        screen_groups = [
            self._build_particle_group(r.concatenate() if r.has_records() else None)
            for r in recorders
        ]
        trajectory_groups = (
            []
            if trajectory_recorders is None
            else [self._build_particle_group(r.concatenate() if r.has_records() else None) for r in trajectory_recorders]
        )
        return screen_groups, trajectory_groups

    def _build_particle_group(self, rec):
        if rec is None:
            data = dict(
                x=np.array([]), y=np.array([]), z=np.array([]),
                px=np.array([]), py=np.array([]), pz=np.array([]),
                t=np.array([]), weight=np.array([]), status=np.array([]),
                id=np.array([], dtype=int), species=self.species,
            )
            return ParticleGroup(data=data)

        vx = backend_module.to_numpy(rec["vx"])
        vy = backend_module.to_numpy(rec["vy"])
        vz = backend_module.to_numpy(rec["vz"])
        px = si_momentum_to_ev_c(vx * self.mass)
        py = si_momentum_to_ev_c(vy * self.mass)
        pz = si_momentum_to_ev_c(vz * self.mass)

        t = backend_module.to_numpy(rec["t"])
        data = dict(
            x=backend_module.to_numpy(rec["x"]),
            y=backend_module.to_numpy(rec["y"]),
            z=backend_module.to_numpy(rec["z"]),
            px=px, py=py, pz=pz,
            t=t,
            weight=backend_module.to_numpy(rec["weight"]),
            status=np.ones_like(t),
            id=backend_module.to_numpy(rec["id"]).astype(int),
            species=self.species,
        )
        return ParticleGroup(data=data)
