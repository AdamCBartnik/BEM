import numpy as np
import pytest

from specific_particle_tracer import batched_rk45
from specific_particle_tracer.screens import ScreenRecorder, find_crossings_grouped


def _record_step(recorder, t0, t1, z0, z1, v0, v1, xp=np):
    def vec(z):
        return xp.array([[[0., 0., z]]])
    find_crossings_grouped([recorder], xp.array([0]), xp.array([t0]), vec(z0), vec(v0),
        xp.array([t1]), vec(z1), vec(v1), xp.ones((1, 1)), xp.ones((1, 1), dtype=int),
        xp.ones((1, 1)), xp=xp)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_cubic_step_records_all_three_crossings(backend):
    xp = np if backend == "cpu" else pytest.importorskip("cupy")
    # z=(t-.2)(t-.5)(t-.8), with roots known independently of the detector.
    recorder = ScreenRecorder(0., xp=xp)
    _record_step(recorder, 0., 1., -.08, .08, .66, .66, xp=xp)
    result = recorder.concatenate()["t"]
    if xp is not np:
        result = xp.asnumpy(result)
    np.testing.assert_allclose(np.sort(result), [.2, .5, .8], rtol=0., atol=1e-13)


def test_turnaround_crossings_survive_large_accepted_rk_step():
    recorder = ScreenRecorder(.4)
    def accel(pos, vel, active, idx):
        a = np.zeros_like(pos)
        a[..., 2] = -1.
        return a
    def step(idx,t0,p0,v0,t1,p1,v1):
        find_crossings_grouped([recorder],idx,t0,p0,v0,t1,p1,v1,
            np.ones((1,1)),np.ones((1,1),dtype=int),np.ones((1,1)))
    pos = np.zeros((1,1,3)); vel = np.array([[[0.,0.,1.]]])
    batched_rk45.integrate(accel,pos,vel,np.zeros((1,1)),2.,1e-8,1e-12,on_step=step)
    np.testing.assert_allclose(recorder.concatenate()["t"], [1-np.sqrt(.2),1+np.sqrt(.2)], atol=1e-13)


def test_step_boundary_crossing_recorded_once_and_frozen_state_ignored():
    recorder = ScreenRecorder(1.)
    _record_step(recorder, 0., 1., 0., 1., 1., 1.)
    _record_step(recorder, 1., 2., 1., 2., 1., 1.)
    _record_step(recorder, 2., 3., 1., 1., 1., 1.)
    np.testing.assert_array_equal(recorder.concatenate()["t"], [1.])
