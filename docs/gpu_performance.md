# GPU performance measurements

Measured on 2026-09-12 with an NVIDIA GeForce RTX 5070 Ti, CuPy 14.2.0,
and the installed Miniforge Python, comparing against commit `3fc4d7e`.
These are timings on a Windows desktop, not hardware-independent guarantees.

## Results

Complete trajectory runs use eight independent single-electron groups, a
0.5 eV initial kinetic energy, and a screen at 200 nm. Times are medians of
three runs, excluding construction of the geometry and tracker.

| Geometry | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Hemispherical tip | 5.51 s | 3.54 s | 1.56x |
| Cylindrical well | 8.18 s | 6.31 s | 1.30x |

Individual calls use synchronized wall-clock timing, warmup, and medians of
seven calls. `G` is the number of independent groups; `E` is electrons per
group. These samples are 1 nm outside the tip or above the well bottom.

| Geometry / batch | Field before / after | Image force before / after |
| --- | ---: | ---: |
| Tip, G=1 E=1 | 5.16 / 2.71 ms | 7.16 / 4.51 ms |
| Tip, G=32 E=4 | 5.32 / 2.94 ms | 7.44 / 5.15 ms |
| Tip, G=256 E=4 | 7.37 / 6.77 ms | 26.67 / 21.18 ms |
| Well, G=1 E=1 | 5.46 / 3.08 ms | 6.58 / 4.28 ms |
| Well, G=32 E=4 | 8.31 / 5.44 ms | 6.84 / 6.35 ms |
| Well, G=256 E=4 | 11.52 / 10.80 ms | 24.99 / 19.81 ms |

The larger-batch static-field gains are modest; small differences here are
comparable to desktop timing variability. End-to-end measurements cover
single-electron groups only; multi-electron force calls are measured separately.

## Changes

- Cache fixed profile geometry, quadrature, and contiguous closest-point
  kernel inputs on the selected device. Cache keys distinguish CUDA devices
  and quadrature orders. Particle positions and induced densities are recomputed.
- Fuse elementwise ring-field expressions and elliptic-integral seeds.
- Compute toroidal Q derivatives inside the existing recurrence kernel,
  avoiding a shifted copy of all modes and separate derivative kernels.
- Compute harmonic-field geometry coefficients once per point/quadrature
  pair, then fuse their combination with Q and its derivative across modes.
- Combine cosine/sine densities before quadrature, reducing six field sums
  to three, with the angular derivative's mode/radius factor outside the sum.
- Reuse the tip's already-computed plane-image force. Skip real-particle
  Coulomb force for independent single-electron groups, where it is zero.

Two experiments were rejected. Automatically fusing the reductions made a
representative reduction about five times slower. Fusing mode-independent
geometry together with mode-dependent fields repeated square roots and
divisions for every harmonic, making a large ring-mode batch about twice
as slow. Separating those stages brought that batch from about 2.20 ms to
1.45 ms (256 groups, four electrons, 200 quadrature nodes, modes 0 through 16).

Float64, meshes, quadrature orders, harmonic counts, and integration
tolerances are unchanged. The NumPy path remains available; CPU speedups
were not benchmarked. Solution profiles and solved static densities are
treated as fixed, as in the existing quadrature cache; construct a new
solution to change them.

## Accuracy and reproduction

For the sampled fields and image forces, the largest relative array L2
difference from the baseline was 2.3e-15. All eight electrons reached the
screen for both geometries. Maximum screen-position differences were
6.7e-17 m for the tip and 8.1e-15 m for the well. Relative momentum-array
L2 differences were 1.3e-9 and 7.9e-7, respectively; maximum relative arrival
time differences were 2.1e-10 and 3.3e-8. Adaptive trajectories need not be
bitwise identical after changing floating-point operation order.

The tests also compare CPU/GPU fields and image forces, analytic reference
solutions, toroidal values, and derivatives. Added regressions exercise
cached fields with changing query batches and quadrature orders, strided
GPU inputs, mode zero, both recurrence branches, and the high-mode fallback.
The complete suite passed: 169 tests in 177.73 seconds, including GPU and
multiprocessing coverage.

From the repository root, using an environment with CuPy and the project
dependencies installed:

```powershell
python -m examples.benchmark_gpu --output benchmark.json --trajectories
python -m examples.benchmark_gpu --output profile.json --profile
python -m pytest -q -p no:cacheprovider
```

The benchmark writes timings to JSON and field/force/screen samples to an
adjacent NPZ file. `--trajectory-repeats` controls the default three repeats;
`--groups` changes batch sizes. Keep the same group list for output comparisons
because it affects random-number consumption. For an old-code comparison,
use a separate checkout of the baseline commit and copy the same benchmark
script into it. Run comparisons sequentially so GPU workloads do not compete.

Geometry assembly and cold JIT compilation can still dominate short jobs.
Dynamic near-surface quadrature also requires device/host synchronization.
Python profiling attributes waits to the synchronizing operation, so its
`nonzero` time is not a measurement of that GPU kernel alone. Larger workloads
still materialize arrays over modes, groups, electrons, and quadrature nodes;
these changes do not remove that memory scaling.
