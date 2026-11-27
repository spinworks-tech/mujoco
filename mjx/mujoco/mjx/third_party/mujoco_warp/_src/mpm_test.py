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


# geoms centred at (0, 0, 0.07): type -> (attributes, signed distance of points in the geom frame)
_GEOMS = {
  "sphere": ('size="0.06"', lambda p: np.linalg.norm(p, axis=-1) - 0.06),
  "capsule": ('size="0.04 0.05"', lambda p: np.linalg.norm(p - _axis(p, 0.05), axis=-1) - 0.04),
  "cylinder": ('size="0.06 0.03"', lambda p: np.maximum(np.hypot(p[..., 0], p[..., 1]) - 0.06, abs(p[..., 2]) - 0.03)),
  # first-order distance, exact at the surface
  "ellipsoid": ('size="0.08 0.05 0.03"', lambda p: (np.linalg.norm(p / (0.08, 0.05, 0.03), axis=-1) - 1) * 0.03),
  "box": ('size="0.06 0.05 0.03"', lambda p: (abs(p) - (0.06, 0.05, 0.03)).max(axis=-1)),
}


def _axis(p, half):
  """Closest points to p on the segment of half length half along z."""
  return np.stack([0 * p[..., 0], 0 * p[..., 0], np.clip(p[..., 2], -half, half)], axis=-1)


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

  @parameterized.parameters((20.0, 45.0))
  def test_friction_angle_sets_pile_height(self, low, high):
    pos = mpm.sample_box((-0.04, -0.04, 0.0), (0.04, 0.04, 0.16), 0.5 * _DX)
    heights = []
    for friction_angle in (low, high):
      m, d, mm, md = _setup(_FLOOR.format(c=1), pos=pos, material=mpm.Material.SAND, friction_angle=friction_angle)
      x, _ = _run(m, d, mm, md, 300)
      heights.append(np.percentile(x[..., 2], 95))
    self.assertGreater(heights[1], 1.2 * heights[0])

  def test_geom_friction_slows_sliding(self):
    """Sand on a 20 degree slope slides freely without geom friction, and much less with it."""
    slide = []
    for friction in (0.01, 1.0):
      xml = f"""
      <mujoco>
        <option timestep="0.002"/>
        <worldbody>
          <geom type="plane" size="1 1 0.1" euler="0 20 0" friction="{friction}"/>
        </worldbody>
      </mujoco>
      """
      pos = mpm.sample_box((-0.05, -0.05, 0.03), (0.05, 0.05, 0.09), 0.5 * _DX)
      m, d, mm, md = _setup(xml, pos=pos, material=mpm.Material.SAND)
      x, _ = _run(m, d, mm, md, 120)
      slide.append(x[..., 0].mean() - pos[:, 0].mean())
    self.assertGreater(slide[0], 2.0 * slide[1])
    self.assertGreater(slide[1], 0.0)

  @parameterized.parameters(*_GEOMS)
  def test_geom_types_hold_particles(self, geom):
    """Sand poured on each geom type flows around it: no particle ends up inside."""
    attr, sdf = _GEOMS[geom]
    xml = f"""
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <geom type="{geom}" pos="0 0 0.07" {attr}/>
      </worldbody>
    </mujoco>
    """
    pos = mpm.sample_box((-0.05, -0.05, 0.14), (0.05, 0.05, 0.24), 0.5 * _DX)
    m, d, mm, md = _setup(xml, pos=pos, material=mpm.Material.SAND)
    x, _ = _run(m, d, mm, md, 300)
    dist = sdf(x[0] - np.array([0.0, 0.0, 0.07]))
    self.assertGreater(dist.min(), -1e-5)
    self.assertGreater((dist < 0.5 * _DX).sum(), 20)  # and the sand does touch it
    self.assertGreater(x[..., 2].min(), -1e-5)

  def test_domain_walls_hold_particles(self):
    """Liquid thrown at the domain walls stays inside."""
    m, d, mm, md = _setup(_FLOOR.format(c=1), material=mpm.Material.LIQUID)
    v0 = np.zeros(md.v.shape + (3,), dtype=np.float32)
    v0[..., 0], v0[..., 1] = 3.0, -2.0
    md.v.assign(v0)
    x, _ = _run(m, d, mm, md, 300)
    self.assertTrue(np.isfinite(x).all())
    self.assertGreater(x[..., 0].max(), 0.2)  # reached the wall
    self.assertTrue(np.all(x > np.array(_LOWER) + _DX) and np.all(x < np.array(_UPPER) - _DX))

  def test_particles_outside_domain_raise(self):
    with self.assertRaisesRegex(ValueError, "cells inside"):
      _setup(_FLOOR.format(c=1), pos=mpm.sample_box((0.2, 0.0, 0.1), (0.3, 0.1, 0.2), 0.5 * _DX))

  def test_from_spec_needs_one_instance(self):
    with self.assertRaisesRegex(ValueError, "found 0"):
      mpm.from_spec(mujoco.MjSpec.from_string(_FLOOR.format(c=1)))

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

  @parameterized.parameters(1000.0, 1600.0)
  def test_reaction_supports_weight(self, rho):
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
    m, d, mm, md = _setup(xml, pos=mpm.sample_box((-0.05, -0.05, 0.12), (0.05, 0.05, 0.2), 0.5 * _DX), rho=rho)
    _run(m, d, mm, md, 500)
    weight = rho * 0.1 * 0.1 * 0.08 * _G
    force = md.xfrc.numpy()[0, 1, :3]
    np.testing.assert_allclose(force[2], -weight, rtol=0.1)
    np.testing.assert_allclose(force[:2], 0.0, atol=0.05 * weight)

  @parameterized.parameters(2, 3)
  def test_p_vol_sets_mass(self, per_cell):
    """Sampling a block more densely, with p_vol to match, doesn't change its weight."""
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
    spacing = _DX / per_cell
    pos = mpm.sample_box((-0.05, -0.05, 0.12), (0.05, 0.05, 0.2), spacing)
    m, d, mm, md = _setup(xml, pos=pos, p_vol=spacing**3)
    _run(m, d, mm, md, 500)
    weight = 1000.0 * 0.1 * 0.1 * 0.08 * _G
    np.testing.assert_allclose(mm.nparticle * mm.p_mass * _G, weight, rtol=1e-5)
    np.testing.assert_allclose(md.xfrc.numpy()[0, 1, 2], -weight, rtol=0.1)

  def test_two_way_coupling(self):
    """A free box falling on sand sinks in, and is then held up by the particles."""
    xml = """
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body pos="0 0 0.17">
          <freejoint/>
          <geom type="box" size="0.03 0.03 0.03" density="1000" contype="2" conaffinity="2"/>
        </body>
      </worldbody>
    </mujoco>
    """
    # sand 0 < z < 0.1, box starts just above it; plane and box don't collide
    pos = mpm.sample_box((-0.08, -0.08, 0.0), (0.08, 0.08, 0.1), 0.5 * _DX)
    m, d, mm, md = _setup(xml, pos=pos, material=mpm.Material.SAND)
    for _ in range(400):
      mjw.step(m, d)
      mpm.step(m, d, mm, md)
      wp.copy(d.xfrc_applied, md.xfrc)
    z = d.xpos.numpy()[0, 1, 2]
    self.assertGreater(z, 0.06)  # resting in the sand (half size 0.03), not on the floor
    self.assertLess(z, 0.14)
    self.assertLess(np.abs(d.qvel.numpy()[0, :3]).max(), 0.05)


# the mujoco.mpm plugin (plugin/mpm), built against the installed mujoco package
_PLUGIN = os.environ.get("MUJOCO_MPM_PLUGIN")

_PLUGIN_DEFAULTS = dict(dx=0.02, timestep=0.002, config="")

_PLUGIN_XML = """
<mujoco>
  <option timestep="{timestep}"/>
  <extension><plugin plugin="mujoco.mpm"><instance name="particles">
    <config key="material" value="{material}"/>
    <config key="dx" value="{dx}"/>
    <config key="young" value="1e5"/>
    <config key="lower" value="-0.3 -0.3 -0.1"/>
    <config key="upper" value="0.3 0.3 0.5"/>
    <config key="box" value="{box}"/>
    {config}
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
    xml = _PLUGIN_XML.format(material=material, box="-0.06 -0.05 0.1 0.06 0.05 0.2", world=world, **_PLUGIN_DEFAULTS)
    _, _, x_c, x_w = self._run(xml, 300, coupled=False)
    np.testing.assert_allclose(x_c, x_w, atol=1e-3)
    self.assertLess(np.abs(x_c - x_w).mean(), 1e-4)

  @parameterized.parameters("sphere", "capsule", "cylinder", "ellipsoid")
  def test_geom_types_match(self, geom):
    """Sand poured on the other geom types, tilted."""
    world = f'<geom type="{geom}" pos="0 0 0.07" euler="30 20 0" friction="0.4" {_GEOMS[geom][0]}/>'
    xml = _PLUGIN_XML.format(material="sand", box="-0.05 -0.05 0.14 0.05 0.05 0.24", world=world, **_PLUGIN_DEFAULTS)
    _, _, x_c, x_w = self._run(xml, 300, coupled=False)
    np.testing.assert_allclose(x_c, x_w, atol=2e-3)
    self.assertLess(np.abs(x_c - x_w).mean(), 1e-4)

  @parameterized.parameters("elastic", "liquid", "sand")
  def test_options_match(self, material):
    """Every config key away from its default, with another grid spacing and timestep."""
    config = """
      <config key="poisson" value="0.3"/>
      <config key="density" value="1600"/>
      <config key="friction_angle" value="30"/>
      <config key="nsubstep" value="6"/>
      <config key="nthread" value="3"/>"""
    world = '<geom type="box" pos="0.03 0 0.04" size="0.03 0.08 0.04" euler="0 20 30"/>'
    xml = _PLUGIN_XML.format(
      material=material, box="-0.05 -0.05 0.1 0.05 0.05 0.2", world=world, dx=0.025, timestep=0.003, config=config
    )
    _, _, x_c, x_w = self._run(xml, 200, coupled=False)
    np.testing.assert_allclose(x_c, x_w, atol=1e-3)
    self.assertLess(np.abs(x_c - x_w).mean(), 1e-4)

  def test_coupling_matches(self):
    """A free box sinking into sand: same rigid trajectory."""
    world = """
      <body pos="0.01 0 0.17"><freejoint/>
        <geom type="box" size="0.03 0.03 0.03" density="300" contype="2" conaffinity="2"/>
      </body>"""
    xml = _PLUGIN_XML.format(material="sand", box="-0.08 -0.08 0 0.08 0.08 0.1", world=world, **_PLUGIN_DEFAULTS)
    d_c, d, _, _ = self._run(xml, 300, coupled=True)
    self.assertLess(d_c.qpos[2], 0.12)  # sank into the sand
    np.testing.assert_allclose(d_c.qpos[:3], d.qpos.numpy()[0, :3], atol=1e-3)


if __name__ == "__main__":
  wp.init()
  absltest.main()
