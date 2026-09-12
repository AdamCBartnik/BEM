"""Backend selection: which array library the tracker computes with.

'cpu' (numpy) is the default; 'gpu' runs on cupy instead -- same array API
as numpy, so the rest of the package is written against a generic `xp`
module reference rather than importing numpy directly, and works unchanged
on either backend. Both run in float64.

(A float32 'gpu32' path was tried and removed: it showed no measured speed
advantage over 'cpu' at any problem size tested, and float32's ~1.2e-7
relative precision broke the birth-time checkpoint arithmetic -- `t +
(target - t)` silently fails to reach `target` once the two agree to
within float32 precision, which stalls the adaptive stepper on perfectly
ordinary inputs, not just extreme ones.)
"""

import numpy as np

_BACKENDS = {
    "cpu": "numpy",
    "gpu": "cupy",
}


def resolve_backend(name):
    """Return (xp, dtype) for a backend name in {'cpu', 'gpu'}."""
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {sorted(_BACKENDS)}")

    module_name = _BACKENDS[name]
    if module_name == "numpy":
        xp = np
    else:
        try:
            import cupy as xp
        except ImportError as exc:
            raise ImportError(
                f"backend={name!r} requires the 'cupy' package "
                "(a CUDA GPU and a matching cupy install)."
            ) from exc

    return xp, np.dtype("float64")


def to_numpy(array):
    """Bring an array back to numpy/CPU, regardless of which backend produced it."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def array_cache_key(xp):
    """Keep cached device arrays separate for each CUDA device."""
    return (xp.__name__, None if xp is np else xp.cuda.runtime.getDevice())
