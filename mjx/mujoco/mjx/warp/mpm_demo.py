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
"""MPM demo: a motor-driven paddle plows a sand tray and a ball drops in.

  MUJOCO_GL=egl python mpm_demo.py --gif mpm.gif   # headless recording
  python mpm_demo.py --live                        # interactive viewer
  python mpm_demo.py --material liquid --gif water.gif
"""

import argparse
import time

import mujoco
import mujoco.viewer
from mujoco.mjx.third_party import mujoco_warp as mjw
from mujoco.mjx.third_party.mujoco_warp._src import mpm
import numpy as np
import warp as wp

_XML = """
<mujoco>
  <option timestep="0.002"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".2 .3 .4"
             rgb2=".1 .2 .3" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom type="plane" size="1 1 .1" material="grid"/>
    <geom type="box" pos=" .24 0 .05" size=".03 .27 .05" rgba=".5 .5 .55 .3"/>
    <geom type="box" pos="-.24 0 .05" size=".03 .27 .05" rgba=".5 .5 .55 .3"/>
    <geom type="box" pos="0  .24 .05" size=".21 .03 .05" rgba=".5 .5 .55 .3"/>
    <geom type="box" pos="0 -.24 .05" size=".21 .03 .05" rgba=".5 .5 .55 .3"/>
    <body name="paddle" pos="0 0 .05">
      <joint name="hinge" axis="0 0 1" armature=".05" damping=".1"/>
      <geom type="box" pos=".1 0 0" size=".1 .015 .04" rgba=".8 .2 .2 1"
            contype="2" conaffinity="0"/>
      <geom type="cylinder" size=".015 .06" rgba=".3 .3 .3 1"
            contype="2" conaffinity="0"/>
    </body>
    <body name="ball" pos="-.1 .1 .35">
      <freejoint/>
      <!-- conaffinity bit 2 lets the paddle (contype 2) hit the ball -->
      <geom type="sphere" size=".04" density="3000" rgba=".2 .4 .9 1"
            conaffinity="3"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="hinge" kv="20" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

_COLOR = {"sand": (0.86, 0.7, 0.42), "liquid": (0.25, 0.55, 0.95), "elastic": (0.4, 0.8, 0.4)}


def _add_particles(scn: mujoco.MjvScene, x: np.ndarray, v: np.ndarray, base_rgb):
  """Appends one small sphere per particle to the scene, brighter when moving fast."""
  speed = np.clip(np.linalg.norm(v, axis=1) / 1.0, 0.0, 1.0)
  eye = np.eye(3).flatten()
  size = np.array([0.006, 0.0, 0.0])
  rgb = np.asarray(base_rgb)
  for p, s in zip(x, speed):
    if scn.ngeom >= scn.maxgeom:
      break
    rgba = np.append(rgb + (1.0 - rgb) * 0.6 * s, 1.0)
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, size, p, eye, rgba)
    scn.ngeom += 1


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--gif", default="mpm.gif", help="output GIF (ignored with --live)")
  parser.add_argument("--live", action="store_true", help="open the interactive viewer instead")
  parser.add_argument("--material", choices=list(_COLOR), default="sand")
  parser.add_argument("--seconds", type=float, default=3.0)
  parser.add_argument("--speed", type=float, default=3.0, help="paddle speed (rad/s)")
  args = parser.parse_args()

  mjm = mujoco.MjModel.from_xml_string(_XML)
  mjd = mujoco.MjData(mjm)
  mjd.ctrl[0] = args.speed
  mujoco.mj_forward(mjm, mjd)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)

  dx = 0.025
  pos = mpm.sample_box((-0.2, -0.2, 0.0), (0.2, 0.2, 0.06), 0.5 * dx)
  material = mpm.Material[args.material.upper()]
  mm, md = mpm.make(m, d, pos, (-0.35, -0.35, -0.1), (0.35, 0.35, 0.5), dx, material=material, rho=1600.0)
  print(f"{mm.nparticle} particles, {mm.nsubstep} MPM substeps per step, device {wp.get_device()}")

  def step():
    mjw.step(m, d)
    mpm.step(m, d, mm, md)
    wp.copy(d.xfrc_applied, md.xfrc)

  def sync():
    mjd.qpos[:] = d.qpos.numpy()[0]
    mujoco.mj_kinematics(mjm, mjd)
    return md.x.numpy()[0], md.v.numpy()[0]

  nstep = int(args.seconds / mjm.opt.timestep)
  rgb = _COLOR[args.material]

  if args.live:
    with mujoco.viewer.launch_passive(mjm, mjd) as viewer:
      while viewer.is_running():
        t0 = time.time()
        for _ in range(10):
          step()
        x, v = sync()
        with viewer.lock():
          viewer.user_scn.ngeom = 0
          _add_particles(viewer.user_scn, x, v, rgb)
        viewer.sync()
        time.sleep(max(0.0, 10 * mjm.opt.timestep - (time.time() - t0)))
    return

  import imageio  # pylint: disable=g-import-not-at-top

  cam = mujoco.MjvCamera()
  cam.lookat[:] = (0.0, 0.0, 0.03)
  cam.distance, cam.elevation, cam.azimuth = 0.95, -35.0, 135.0
  frames = []
  t0 = time.time()
  with mujoco.Renderer(mjm, 480, 640, max_geom=mm.nparticle + 100) as renderer:
    for i in range(nstep):
      step()
      if i % 12 == 0:  # ~42 fps of simulated time
        x, v = sync()
        renderer.update_scene(mjd, camera=cam)
        _add_particles(renderer.scene, x, v, rgb)
        frames.append(renderer.render())
  wp.synchronize()
  print(f"simulated {args.seconds:.1f}s in {time.time() - t0:.1f}s wall (incl. rendering)")
  imageio.mimsave(args.gif, frames, duration=1000 / 25, loop=0)
  print(f"wrote {args.gif} ({len(frames)} frames)")


if __name__ == "__main__":
  main()
