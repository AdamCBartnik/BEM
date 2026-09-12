"""Warm GPU timings and optional trajectories; run from the repository root.

python -m examples.benchmark_gpu --output benchmark.json --profile
python -m examples.benchmark_gpu --output benchmark.json --trajectories

Synchronizes each timing, excludes setup/JIT warmup, and reports medians.
Use the same process settings, mesh, and batch sizes for comparisons.
"""
import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import numpy as np


def measure(fn, cp, repeat=7):
    # Warm for a little wall time as well as compiling the first call;
    # otherwise a tiny first batch can mostly measure device wake-up.
    warm_until = time.perf_counter() + .2
    for _ in range(2):
        fn()
    cp.cuda.get_current_stream().synchronize()
    while time.perf_counter() < warm_until:
        fn()
        cp.cuda.get_current_stream().synchronize()
    elapsed = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        cp.cuda.get_current_stream().synchronize()
        elapsed.append(1000 * (time.perf_counter() - start))
    return float(np.median(elapsed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 32, 256])
    parser.add_argument("--trajectory-repeats", type=int, default=3)
    args = parser.parse_args()
    import cupy as cp
    from specific_particle_tracer.bem.geometry import HemisphericalTipBEMGeometry, CylindricalWellBEMGeometry
    from specific_particle_tracer.constants import ELEMENTARY_CHARGE

    R, H = 50e-9, 30e-9
    tip = HemisphericalTipBEMGeometry(-1e8, R, z0=3e-9, image_mirror_symmetric=True, xp=cp)
    well = CylindricalWellBEMGeometry(-1e8, R, H, z0=3e-9,
        field_max_length=2e-9, image_max_length=2e-9,
        rim_fillet_radius=5e-9, bottom_fillet_radius=1.5e-9, xp=cp)
    props = cp.cuda.runtime.getDeviceProperties(0)
    result = {"gpu": props["name"].decode(), "cupy": cp.__version__, "timings_ms": {}}
    samples = {}
    rng = np.random.default_rng(721)
    for name, geometry in [("tip", tip), ("well", well)]:
        for groups in args.groups:
            for emit in [1, 4]:
                rho = R * rng.uniform(.1, .7, (groups, emit))
                phi = rng.uniform(-np.pi, np.pi, (groups, emit))
                z = (np.sqrt(R*R-rho*rho)+.02*R if name == "tip"
                     else np.full_like(rho, -H+.02*R))
                position = cp.asarray(np.stack([rho*np.cos(phi), rho*np.sin(phi), z], axis=-1))
                charge = cp.full((groups, emit), -ELEMENTARY_CHARGE)
                active = cp.ones((groups, emit), dtype=bool)
                for label, fn in [
                    ("field", lambda: geometry.field(position)),
                    ("image", lambda: geometry.image_force(position, charge, active, 1e-12)),
                ]:
                    key = f"{name}/G{groups}/E{emit}/{label}"
                    result["timings_ms"][key] = measure(fn, cp)
                    samples[key] = cp.asnumpy(fn())
                    print(key, round(result["timings_ms"][key], 3), "ms", flush=True)
                if args.profile and groups == args.groups[-1] and emit == 1:
                    profiler = cProfile.Profile()
                    profiler.enable()
                    for _ in range(20):
                        geometry.field(position)
                        geometry.image_force(position, charge, active, 1e-12)
                    cp.cuda.get_current_stream().synchronize()
                    profiler.disable()
                    stream = io.StringIO()
                    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(22)
                    result[f"{name}_profile"] = stream.getvalue()
                    print(stream.getvalue(), flush=True)
        if args.trajectories:
            from beamphysics import ParticleGroup
            from specific_particle_tracer import SpecificParticleTracer
            count = 8
            rho = np.linspace(.1, .7, count)*R
            z = np.sqrt(R*R-rho*rho) if name == "tip" else np.full(count, -H)
            pg = ParticleGroup(data=dict(x=rho, y=np.zeros(count), z=z,
                px=np.zeros(count), py=np.zeros(count), pz=np.full(count, np.sqrt(2*.5*510998.95)),
                t=np.zeros(count), weight=np.full(count, ELEMENTARY_CHARGE),
                status=np.ones(count), id=np.arange(count), species="electron"))
            tracer = SpecificParticleTracer(pg, 1, geometry, screens=[200e-9],
                t_max=250e-15, z_max=210e-9, backend="gpu")
            # Warm all force paths; time run separately from geometry assembly.
            geometry.field(cp.asarray(np.stack([pg.x, pg.y, pg.z], axis=-1)))
            cp.cuda.get_current_stream().synchronize()
            times = []
            for _ in range(args.trajectory_repeats):
                start = time.perf_counter()
                (screen,), _ = tracer.run()
                cp.cuda.get_current_stream().synchronize()
                times.append(1000*(time.perf_counter()-start))
            result["timings_ms"][f"{name}/trajectory_G8"] = float(np.median(times))
            result[f"{name}_trajectory_repeats_ms"] = times
            result[f"{name}_arrived"] = len(screen)
            samples[f"{name}/trajectory"] = np.stack([screen.id, screen.t, screen.x,
                screen.y, screen.z, screen.px, screen.py, screen.pz])
            print(name, "trajectory", result["timings_ms"][f"{name}/trajectory_G8"], "ms", flush=True)
    args.output.write_text(json.dumps(result, indent=2))
    np.savez(args.output.with_suffix(".npz"), **samples)


if __name__ == "__main__":
    main()
