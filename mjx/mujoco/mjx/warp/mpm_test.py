# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for MPM called from JAX."""

from absl.testing import absltest
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
import mujoco.mjx.warp as mjxw
import numpy as np

if mjxw.WARP_INSTALLED:
  # pylint: disable=g-import-not-at-top
  from mujoco.mjx.third_party import mujoco_warp as mjw
  from mujoco.mjx.third_party.mujoco_warp._src import mpm as mjw_mpm
  from mujoco.mjx.warp import mpm
  import warp as wp
  # pylint: enable=g-import-not-at-top

_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0.02 0 0.13">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" density="300" contype="2"
            conaffinity="2"/>
    </body>
  </worldbody>
</mujoco>
"""
_ARGS = dict(lower=(-0.3, -0.3, -0.1), upper=(0.3, 0.3, 0.5), dx=0.02)
_NSTEP = 50


class MPMJaxTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    self.mjm = mujoco.MjModel.from_xml_string(_XML)
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)
    self.pos = mpm.sample_box((-0.08, -0.08, 0.0), (0.08, 0.08, 0.1), 0.01)

  def test_matches_mujoco_warp(self):
    """The jitted JAX entry point matches MJWarp's step, given the same bodies.

    Both paths read the kinematics MJWarp computes, so the comparison isolates
    the wrapper from differences between the MJX and MJWarp rigid solvers.
    """
    m = mjw.put_model(self.mjm)
    d = mjw.put_data(self.mjm, self.mjd, nworld=2)
    mm, md = mjw_mpm.make(m, d, self.pos, material=mpm.Material.SAND, **_ARGS)

    mx = mjx.put_model(self.mjm)
    dx = jax.vmap(lambda _: mjx.put_data(self.mjm, self.mjd))(jp.arange(2))
    mmx, state = mpm.make(
        mx, self.pos, nworld=2, material=mpm.Material.SAND, **_ARGS
    )
    self.assertEqual(mmx.nsubstep, mm.nsubstep)
    step = jax.jit(mpm.step, static_argnums=2)

    kin = ('xipos', 'geom_xpos', 'geom_xmat', 'subtree_com', 'cvel')
    for _ in range(_NSTEP):
      mjw.step(m, d)
      dx = dx.replace(**{k: jp.asarray(getattr(d, k).numpy()) for k in kin})
      mjw_mpm.step(m, d, mm, md)
      wp.copy(d.xfrc_applied, md.xfrc)
      state = step(mx, dx, mmx, state)

    np.testing.assert_allclose(state.x, md.x.numpy(), atol=1e-4)
    np.testing.assert_allclose(state.v, md.v.numpy(), atol=1e-3)
    # forces up to ~26 N; CPU (JAX) and GPU (MJWarp) sum in different orders
    np.testing.assert_allclose(state.xfrc, md.xfrc.numpy(), atol=0.05)
    self.assertGreater(np.abs(state.xfrc[:, 1, 2]).min(), 0.0)  # box is pushed

  def test_unbatched_data(self):
    mx = mjx.put_model(self.mjm)
    dx = mjx.forward(mx, mjx.put_data(self.mjm, self.mjd))
    mm, state = mpm.make(mx, self.pos, **_ARGS)
    state = jax.jit(mpm.step, static_argnums=2)(mx, dx, mm, state)
    self.assertEqual(state.xfrc.shape, (1, self.mjm.nbody, 6))
    self.assertTrue(np.isfinite(state.x).all())


if __name__ == '__main__':
  absltest.main()
