# Copyright 2026 The Newton Developers
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

"""Tests for the material point method."""

import os

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

from mujoco.mjx.third_party import mujoco_warp as mjw
from mujoco.mjx.third_party.mujoco_warp._src import mpm

_DX = 0.02
_LOWER = (-0.3, -0.3, -0.1)
_UPPER = (0.3, 0.3, 0.5)
_G = 9.81

_FLOOR = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" contype="{c}" conaffinity="{c}"/>
  </worldbody>
</mujoco>
"""


def _setup(xml, nworld=1, **kwargs):
  mjm = mujoco.MjModel.from_xml_string(xml)
  mjd = mujoco.MjData(mjm)
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd, nworld=nworld)
  mjw.forward(m, d)
  pos = kwargs.pop("pos", mpm.sample_box((-0.05, -0.05, 0.1), (0.05, 0.05, 0.2), 0.5 * _DX))
  mm, md = mpm.make(m, d, pos, _LOWER, _UPPER, _DX, **kwargs)
  return m, d, mm, md


def _run(m, d, mm, md, nstep):
  for _ in range(nstep):
    mpm.step(m, d, mm, md)
  return md.x.numpy(), md.v.numpy()


class MPMTest(parameterized.TestCase):
  def test_free_fall(self):
    """Without collisions the block falls ballistically and keeps its shape."""
    m, d, mm, md = _setup(_FLOOR.format(c=0))
    x0 = md.x.numpy().copy()
    nstep = 50
    x, v = _run(m, d, mm, md, nstep)
    t = nstep * 0.002
    np.testing.assert_allclose(v[..., 2].mean(), -_G * t, rtol=1e-3)
    np.testing.assert_allclose(v[..., :2].mean(axis=(0, 1)), 0.0, atol=1e-5)
    np.testing.assert_allclose(x[..., 2].mean() - x0[..., 2].mean(), -0.5 * _G * t * t, rtol=0.02)
    # uniform motion: no deformation
    np.testing.assert_allclose(md.F.numpy(), np.broadcast_to(np.eye(3), md.F.shape + (3, 3)), atol=1e-3)

  def test_worlds_are_independent(self):
    m, d, mm, md = _setup(_FLOOR.format(c=0), nworld=2)
    v0 = np.zeros(md.v.shape + (3,), dtype=np.float32)
    v0[1, :, 0] = 1.0
    md.v.assign(v0)
    _, v = _run(m, d, mm, md, 10)
    np.testing.assert_allclose(v[0, :, 0].mean(), 0.0, atol=1e-5)
    np.testing.assert_allclose(v[1, :, 0].mean(), 1.0, rtol=1e-3)

  @parameterized.parameters(mpm.Material.ELASTIC, mpm.Material.SAND, mpm.Material.LIQUID)
  def test_rest_on_plane(self, material):
    """A block dropped on the floor neither passes through it nor leaves the domain."""
    m, d, mm, md = _setup(_FLOOR.format(c=1), material=material)
    x, v = _run(m, d, mm, md, 800)
    self.assertFalse(np.isnan(x).any())
    self.assertGreater(x[..., 2].min(), -_DX)
    self.assertTrue(np.all(x > np.array(_LOWER)) and np.all(x < np.array(_UPPER)))
    if material != mpm.Material.LIQUID:
      self.assertLess(np.abs(v).mean(), 0.05)

  def test_sand_piles_liquid_spreads(self):
    pos = mpm.sample_box((-0.04, -0.04, 0.0), (0.04, 0.04, 0.16), 0.5 * _DX)
    heights = {}
    for material in (mpm.Material.SAND, mpm.Material.LIQUID):
      m, d, mm, md = _setup(_FLOOR.format(c=1), pos=pos, material=material)
      x, _ = _run(m, d, mm, md, 300)
      heights[material] = np.percentile(x[..., 2], 95)
    self.assertGreater(heights[mpm.Material.SAND], 2.0 * heights[mpm.Material.LIQUID])

  def test_thin_plate_holds_particles(self):
    """A plate thinner than a grid cell still stops the particles: none get inside or through it."""
    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <geom type="box" pos="0 0 0.05" size="0.15 0.15 0.004"/>
      </worldbody>
    </mujoco>
    """
    m, d, mm, md = _setup(xml, material=mpm.Material.SAND)
    x, _ = _run(m, d, mm, md, 300)
    self.assertGreater(x[..., 2].min(), 0.054 - 1e-4)

  def test_reaction_supports_weight(self):
    """The particles resting on a static box push it down with their weight."""
    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <body name="table" pos="0 0 0.05">
          <geom type="box" size="0.15 0.15 0.05"/>
        </body>
      </worldbody>
    </mujoco>
    """
    m, d, mm, md = _setup(xml, pos=mpm.sample_box((-0.05, -0.05, 0.12), (0.05, 0.05, 0.2), 0.5 * _DX))
    _run(m, d, mm, md, 500)
    weight = mm.nparticle * mm.p_mass * _G
    force = md.xfrc.numpy()[0, 1, :3]
    np.testing.assert_allclose(force[2], -weight, rtol=0.1)
    np.testing.assert_allclose(force[:2], 0.0, atol=0.05 * weight)

  def test_two_way_coupling(self):
    """A free box falling on a resting block is held up by the particles."""
    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body pos="0 0 0.17">
          <freejoint/>
          <geom type="box" size="0.03 0.03 0.03" density="300" contype="2" conaffinity="2"/>
        </body>
      </worldbody>
    </mujoco>
    """
    # elastic block 0 < z < 0.1, box starts just above it; plane and box don't collide
    m, d, mm, md = _setup(xml, pos=mpm.sample_box((-0.08, -0.08, 0.0), (0.08, 0.08, 0.1), 0.5 * _DX), E=1e5)
    for _ in range(400):
      mjw.step(m, d)
      mpm.step(m, d, mm, md)
      wp.copy(d.xfrc_applied, md.xfrc)
    z = d.xpos.numpy()[0, 1, 2]
    self.assertGreater(z, 0.1)  # resting on the block (half size 0.03), not on the floor
    self.assertLess(z, 0.17)


# the mujoco.mpm plugin (plugin/mpm), built against the installed mujoco package
_PLUGIN = os.environ.get("MUJOCO_MPM_PLUGIN")

_PLUGIN_XML = """
<mujoco>
  <option timestep="0.002"/>
  <extension><plugin plugin="mujoco.mpm"><instance name="particles">
    <config key="material" value="{material}"/>
    <config key="dx" value="0.02"/>
    <config key="young" value="1e5"/>
    <config key="lower" value="-0.3 -0.3 -0.1"/>
    <config key="upper" value="0.3 0.3 0.5"/>
    <config key="box" value="{box}"/>
  </instance></plugin></extension>
  <worldbody>
    <geom type="plane" size="1 1 .1"/>
    {world}
  </worldbody>
</mujoco>
"""


@absltest.skipUnless(_PLUGIN, "set MUJOCO_MPM_PLUGIN to the mujoco.mpm plugin library")
class CPluginTest(parameterized.TestCase):
  """The C plugin (CPU MuJoCo) and this solver (MJWarp, MJX) agree, from one XML."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    mujoco.mj_loadPluginLibrary(_PLUGIN)

  def _run(self, xml, nstep, coupled):
    spec = mujoco.MjSpec.from_string(xml)
    mjm_c = spec.compile()
    d_c = mujoco.MjData(mjm_c)
    kwargs = mpm.from_spec(spec)
    mjm = spec.compile()
    self.assertEqual(mjm.nplugin, 0)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd)
    mm, md = mpm.make(m, d, **kwargs)
    for _ in range(nstep):
      mujoco.mj_step(mjm_c, d_c)
      if coupled:
        mjw.step(m, d)
      mpm.step(m, d, mm, md)
      if coupled:
        wp.copy(d.xfrc_applied, md.xfrc)
    n = mm.nparticle
    self.assertEqual(mjm_c.plugin_statenum[0], 25 * n + 6 * mjm_c.nbody)
    return d_c, d, d_c.plugin_state[: 3 * n].reshape(n, 3), md.x.numpy()[0]

  @parameterized.parameters("elastic", "liquid", "sand")
  def test_particles_match(self, material):
    """Particles dropped on a plane and a tilted box: same trajectories, up to float32 rounding."""
    world = '<geom type="box" pos="0.03 0 0.04" size="0.03 0.08 0.04" euler="0 20 30"/>'
    xml = _PLUGIN_XML.format(material=material, box="-0.06 -0.05 0.1 0.06 0.05 0.2", world=world)
    _, _, x_c, x_w = self._run(xml, 300, coupled=False)
    np.testing.assert_allclose(x_c, x_w, atol=1e-3)
    self.assertLess(np.abs(x_c - x_w).mean(), 1e-4)

  def test_coupling_matches(self):
    """A free box sinking into sand: same rigid trajectory."""
    world = """
      <body pos="0.01 0 0.17"><freejoint/>
        <geom type="box" size="0.03 0.03 0.03" density="300" contype="2" conaffinity="2"/>
      </body>"""
    xml = _PLUGIN_XML.format(material="sand", box="-0.08 -0.08 0 0.08 0.08 0.1", world=world)
    d_c, d, _, _ = self._run(xml, 300, coupled=True)
    self.assertLess(d_c.qpos[2], 0.12)  # sank into the sand
    np.testing.assert_allclose(d_c.qpos[:3], d.qpos.numpy()[0, :3], atol=1e-3)


if __name__ == "__main__":
  wp.init()
  absltest.main()
