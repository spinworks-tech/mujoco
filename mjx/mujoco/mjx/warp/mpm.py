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
"""MPM particles for MJX: the MuJoCo Warp kernels, called from JAX.

Works with either MJX backend, since it reads only kinematics (geom poses,
body velocities) from mjx.Data. Usage, under jax.jit:

  mm, state = mpm.make(m, pos, lower, upper, dx, nworld=B, material=mpm.Material.SAND)
  d = jax.vmap(mjx.step, in_axes=(None, 0))(m, d)
  state = mpm.step(m, d, mm, state)
  d = d.replace(xfrc_applied=state.xfrc)  # two-way coupling
"""

import functools
from typing import NamedTuple

import jax
from jax import numpy as jp
import numpy as np
import warp as wp

from mujoco.mjx._src import types
from mujoco.mjx.third_party.mujoco_warp._src import mpm
from mujoco.mjx.third_party.mujoco_warp._src.mpm import Material  # pylint: disable=unused-import
from mujoco.mjx.third_party.mujoco_warp._src.mpm import MPMModel
from mujoco.mjx.third_party.mujoco_warp._src.mpm import sample_box  # pylint: disable=unused-import


class State(NamedTuple):
  """Per-world particle state (see mujoco_warp mpm.MPMData).

  xfrc is the reaction of the particles on each body, laid out like
  Data.xfrc_applied.
  """

  x: jax.Array  # (nworld, nparticle, 3)
  v: jax.Array  # (nworld, nparticle, 3)
  C: jax.Array  # (nworld, nparticle, 3, 3)
  F: jax.Array  # (nworld, nparticle, 3, 3)
  Jp: jax.Array  # (nworld, nparticle)
  xfrc: jax.Array  # (nworld, nbody, 6)


def make(
    m: types.Model, pos, lower, upper, dx: float, nworld: int = 1, **kwargs
) -> tuple[MPMModel, State]:
  """Creates the MPM model and state; kwargs as mujoco_warp mpm.make_model."""
  mm = mpm.make_model(
      float(np.max(m.opt.timestep)), pos, lower, upper, dx, **kwargs
  )
  s = {k: jp.asarray(a) for k, a in mpm.initial_state(pos, nworld).items()}
  return mm, State(**s, xfrc=jp.zeros((nworld, m.nbody, 6), jp.float32))


_vec3 = wp.array2d(dtype=wp.vec3)


@functools.cache
def _callable(mm: MPMModel):
  """JAX callable running mpm.step_arrays; one per MPMModel (scalars are static)."""

  def f(
      opt_timestep: wp.array(dtype=float),
      opt_gravity: wp.array(dtype=wp.vec3),
      body_rootid: wp.array(dtype=int),
      geom_type: wp.array(dtype=int),
      geom_contype: wp.array(dtype=int),
      geom_conaffinity: wp.array(dtype=int),
      geom_bodyid: wp.array(dtype=int),
      geom_size: _vec3,
      geom_friction: _vec3,
      xipos: _vec3,
      geom_xpos: _vec3,
      geom_xmat: wp.array2d(dtype=wp.mat33),
      subtree_com: _vec3,
      cvel: wp.array2d(dtype=wp.spatial_vector),
      x: _vec3,
      v: _vec3,
      C: wp.array2d(dtype=wp.mat33),
      F: wp.array2d(dtype=wp.mat33),
      Jp: wp.array2d(dtype=float),
      xfrc: wp.array2d(dtype=wp.spatial_vector),
      grid_mv: wp.array4d(dtype=wp.vec3),
      grid_m: wp.array4d(dtype=float),
  ):
    mpm.step_arrays(
        mm, opt_timestep, opt_gravity, body_rootid, geom_type, geom_contype,
        geom_conaffinity, geom_bodyid, geom_size, geom_friction, xipos,
        geom_xpos, geom_xmat, subtree_com, cvel, x, v, C, F, Jp, xfrc,
        grid_mv, grid_m,
    )  # fmt: skip

  # ponytail: no jax.vmap support; worlds are batched explicitly on axis 0.
  return wp.jax_callable(
      f,
      num_outputs=8,
      in_out_argnames=('x', 'v', 'C', 'F', 'Jp'),
      vmap_method=None,
  )


def step(m: types.Model, d: types.Data, mm: MPMModel, state: State) -> State:
  """Advances the particles by one timestep against the geom poses in d.

  d is batched with a leading world axis matching state (call it outside
  jax.vmap), or unbatched for a single world.
  """
  kin = (d.xipos, d.geom_xpos, d.geom_xmat, d.subtree_com, d.cvel)
  if d.geom_xpos.ndim == 2:
    kin = tuple(a[None] for a in kin)
  nworld = state.x.shape[0]
  if kin[1].shape[0] != nworld:
    raise ValueError(f'd has {kin[1].shape[0]} worlds, state has {nworld}.')

  i32 = lambda a: jp.asarray(a, jp.int32)
  f32 = lambda a, *shape: jp.reshape(jp.asarray(a, jp.float32), shape)
  grid = (nworld, *mm.grid_res)
  x, v, C, F, Jp, xfrc, _, _ = _callable(mm)(
      f32(m.opt.timestep, -1),
      f32(m.opt.gravity, -1, 3),
      i32(m.body_rootid),
      i32(m.geom_type),
      i32(m.geom_contype),
      i32(m.geom_conaffinity),
      i32(m.geom_bodyid),
      f32(m.geom_size, -1, m.ngeom, 3),
      f32(m.geom_friction, -1, m.ngeom, 3),
      *kin,
      *state[:5],
      output_dims={'xfrc': (nworld, m.nbody), 'grid_mv': grid, 'grid_m': grid},
  )
  return State(x, v, C, F, Jp, xfrc)
