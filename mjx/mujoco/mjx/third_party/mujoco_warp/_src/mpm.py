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

"""Material point method (MLS-MPM) coupled to MuJoCo Warp rigid bodies.

Ported from Genesis' MPMSolver (genesis/engine/solvers/mpm_solver.py and
genesis/engine/materials/MPM). Particles live on a fixed, dense grid per world,
so every array has a leading world dimension and batches like the rest of
MJWarp.

Usage:
  mm, md = mpm.make(m, d, pos, lower, upper, dx, material=mpm.Material.SAND)
  for _ in range(n):
    mjw.step(m, d)
    mpm.step(m, d, mm, md)
    wp.copy(d.xfrc_applied, md.xfrc)  # two-way coupling (or add to your own)

Geoms collide with the particles when contype or conaffinity is nonzero.
Supported geom types: plane, sphere, capsule, ellipsoid, cylinder, box.
"""

import dataclasses
import enum
import math
from typing import Tuple

import numpy as np
import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src import collision_sdf
from mujoco.mjx.third_party.mujoco_warp._src.types import Data
from mujoco.mjx.third_party.mujoco_warp._src.types import GeomType
from mujoco.mjx.third_party.mujoco_warp._src.types import Model

_BOUND = 3  # grid cells next to the domain walls that stop outward motion
_CONTACT_TOL = 1.0e-4  # distance to a geom, in cells, that counts as on its surface


class Material(enum.IntEnum):
  """MPM material.

  ELASTIC: fixed-corotated elastic solid
  LIQUID: weakly compressible inviscid liquid
  SAND: Drucker-Prager elastoplastic granular media
  """

  ELASTIC = 0
  LIQUID = 1
  SAND = 2


_LIQUID = wp.constant(int(Material.LIQUID))
_SAND = wp.constant(int(Material.SAND))


@dataclasses.dataclass(frozen=True, eq=False)
class MPMModel:
  """Constant MPM parameters, shared by all worlds.

  ponytail: one material per MPMModel; a per-particle material id if scenes mix them.
  """

  material: int
  nparticle: int
  nsubstep: int
  grid_res: wp.vec3i
  lower: wp.vec3
  dx: float
  p_mass: float
  p_vol: float
  mu: float
  lam: float
  sand_alpha: float


@dataclasses.dataclass
class MPMData:
  """Per-world MPM state.

  x, v: particle position and velocity              (nworld, nparticle, 3)
  C: particle affine velocity field                 (nworld, nparticle, 3, 3)
  F: particle deformation gradient                  (nworld, nparticle, 3, 3)
  Jp: sand plastic volume correction                (nworld, nparticle)
  grid_mv: node momentum, then node velocity        (nworld, gx, gy, gz, 3)
  grid_m: node mass                                 (nworld, gx, gy, gz)
  xfrc: reaction of the particles on each body,     (nworld, nbody, 6)
        laid out like Data.xfrc_applied (force, torque at xipos), averaged over the step
  """

  x: wp.array2d[wp.vec3]
  v: wp.array2d[wp.vec3]
  C: wp.array2d[wp.mat33]
  F: wp.array2d[wp.mat33]
  Jp: wp.array2d[float]
  grid_mv: wp.array4d[wp.vec3]
  grid_m: wp.array4d[float]
  xfrc: wp.array2d[wp.spatial_vector]


def sample_box(lo, hi, spacing: float) -> np.ndarray:
  """Particle positions on a regular lattice filling the box [lo, hi]."""
  axes = [np.arange(a + 0.5 * spacing, b, spacing) for a, b in zip(lo, hi)]
  return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)


def from_spec(spec, name: str | None = None) -> dict:
  """Reads a mujoco.mpm plugin instance (plugin/mpm, the CPU MuJoCo side) from spec.

  Returns the keyword arguments of make() / make_model() (pos, lower, upper, dx, material, ...),
  so one XML drives both. Removes the instance from spec: compile the spec afterwards, so that
  MJX doesn't carry the plugin's particle state in every world. Parsing the XML needs the plugin
  library loaded (mujoco.mj_loadPluginLibrary).
  """
  found = [p for p in spec.plugins if p.plugin_name == "mujoco.mpm" and name in (None, p.name)]
  if len(found) != 1:
    raise ValueError(f"expected one mujoco.mpm plugin instance{f' named {name}' if name else ''}, found {len(found)}.")
  cfg = dict(found[0].config)
  # elements may refer to the instance (sand_tray.xml does, so that copies of the spec keep it)
  for element in spec.bodies + spec.geoms + spec.meshes + spec.actuators + spec.sensors:
    if element.plugin.active and element.plugin.name == found[0].name:
      element.plugin.active = False
      element.plugin.name = ""
  spec.delete(found[0])

  def num(key):
    return np.array(cfg[key].split(), dtype=float)

  dx = float(cfg["dx"])
  box = num("box")
  kwargs = dict(
    pos=sample_box(box[:3], box[3:], 0.5 * dx),
    lower=num("lower"),
    upper=num("upper"),
    dx=dx,
    material=Material[cfg.get("material", "elastic").upper()],
  )
  for key, arg in (("young", "E"), ("poisson", "nu"), ("density", "rho"), ("friction_angle", "friction_angle")):
    if key in cfg:
      kwargs[arg] = float(cfg[key])
  if "nsubstep" in cfg:
    kwargs["nsubstep"] = int(cfg["nsubstep"])
  return kwargs


def make_model(
  timestep: float,
  pos: np.ndarray,
  lower,
  upper,
  dx: float,
  material: Material = Material.ELASTIC,
  E: float = 3e5,
  nu: float = 0.2,
  rho: float = 1000.0,
  friction_angle: float = 45.0,
  p_vol: float | None = None,
  nsubstep: int | None = None,
) -> MPMModel:
  """Creates the MPM parameters.

  Args:
    timestep: MuJoCo timestep (the largest one, if worlds differ); sets the substep count.
    pos: initial particle positions (nparticle, 3).
    lower, upper: corners of the simulation domain; particles stay inside.
    dx: grid spacing.
    material: particle material.
    E, nu: Young's modulus and Poisson ratio (for LIQUID, E sets the bulk stiffness).
    rho: density.
    friction_angle: SAND friction angle in degrees.
    p_vol: volume per particle, defaults to (dx/2)^3 (8 particles per cell, see sample_box).
    nsubstep: MPM substeps per MuJoCo step, defaults to the elastic-wave CFL limit.
  """
  pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
  lower = np.asarray(lower, dtype=np.float32)
  upper = np.asarray(upper, dtype=np.float32)
  if not np.all((pos > lower + _BOUND * dx) & (pos < upper - _BOUND * dx)):
    raise ValueError(f"particles must lie at least {_BOUND} cells inside [lower, upper].")
  res = np.ceil((upper - lower) / dx).astype(int) + 1

  mu = E / (2.0 * (1.0 + nu))
  lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
  if material == Material.LIQUID:
    mu = 0.0
  p_vol = p_vol or (0.5 * dx) ** 3
  if nsubstep is None:
    wave_speed = math.sqrt((lam + 2.0 * mu) / rho)
    nsubstep = max(1, math.ceil(timestep / (0.3 * dx / wave_speed)))
  sin_phi = math.sin(math.radians(friction_angle))

  return MPMModel(
    material=int(material),
    nparticle=pos.shape[0],
    nsubstep=nsubstep,
    grid_res=wp.vec3i(*res.tolist()),
    lower=wp.vec3(*lower.tolist()),
    dx=float(dx),
    p_mass=float(rho * p_vol),
    p_vol=float(p_vol),
    mu=float(mu),
    lam=float(lam),
    sand_alpha=math.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi),
  )


def initial_state(pos: np.ndarray, nworld: int) -> dict[str, np.ndarray]:
  """Particle state at rest, replicated in every world: x, v, C, F, Jp."""
  pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
  shape = (nworld, pos.shape[0])
  return {
    "x": np.broadcast_to(pos, shape + (3,)).copy(),
    "v": np.zeros(shape + (3,), np.float32),
    "C": np.zeros(shape + (3, 3), np.float32),
    "F": np.broadcast_to(np.eye(3, dtype=np.float32), shape + (3, 3)).copy(),
    "Jp": np.zeros(shape, np.float32),
  }


def make(m: Model, d: Data, pos: np.ndarray, lower, upper, dx: float, **kwargs) -> Tuple[MPMModel, MPMData]:
  """Creates the MPM model and state for MJWarp, replicating the particles `pos` in every world.

  Takes the arguments of make_model, the timestep coming from m.
  """
  mm = make_model(float(m.opt.timestep.numpy().max()), pos, lower, upper, dx, **kwargs)
  s = initial_state(pos, d.nworld)
  grid = (d.nworld, *mm.grid_res)
  md = MPMData(
    x=wp.array(s["x"], dtype=wp.vec3),
    v=wp.array(s["v"], dtype=wp.vec3),
    C=wp.array(s["C"], dtype=wp.mat33),
    F=wp.array(s["F"], dtype=wp.mat33),
    Jp=wp.array(s["Jp"], dtype=float),
    grid_mv=wp.zeros(grid, dtype=wp.vec3),
    grid_m=wp.zeros(grid, dtype=float),
    xfrc=wp.zeros((d.nworld, d.xfrc_applied.shape[1]), dtype=wp.spatial_vector),
  )
  return mm, md


@wp.func
def _stencil(x: wp.vec3, lower: wp.vec3, inv_dx: float, res: wp.vec3i):
  """Quadratic B-spline stencil: base node, offset in cells, weights (row: node, column: axis)."""
  g = (x - lower) * inv_dx
  base = wp.vec3i(
    wp.clamp(int(wp.floor(g[0] - 0.5)), 0, res[0] - 3),
    wp.clamp(int(wp.floor(g[1] - 0.5)), 0, res[1] - 3),
    wp.clamp(int(wp.floor(g[2] - 0.5)), 0, res[2] - 3),
  )
  fx = g - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
  w0 = wp.cw_mul(wp.vec3(1.5) - fx, wp.vec3(1.5) - fx) * 0.5
  w1 = wp.vec3(0.75) - wp.cw_mul(fx - wp.vec3(1.0), fx - wp.vec3(1.0))
  w2 = wp.cw_mul(fx - wp.vec3(0.5), fx - wp.vec3(0.5)) * 0.5
  return base, fx, wp.matrix_from_rows(w0, w1, w2)


@wp.func
def _stress(material: int, F: wp.mat33, mu: float, lam: float) -> wp.mat33:
  """Kirchhoff stress times F^T, as Genesis' MPM materials update_stress."""
  U = wp.mat33()
  sig = wp.vec3()
  V = wp.mat33()
  wp.svd3(F, U, sig, V)
  if material == _SAND:
    # Hencky strain energy
    e = wp.vec3(wp.log(wp.max(sig[0], 1e-4)), wp.log(wp.max(sig[1], 1e-4)), wp.log(wp.max(sig[2], 1e-4)))
    tr = e[0] + e[1] + e[2]
    center = wp.diag(
      wp.vec3(
        (2.0 * mu * e[0] + lam * tr) / wp.max(sig[0], 1e-4),
        (2.0 * mu * e[1] + lam * tr) / wp.max(sig[1], 1e-4),
        (2.0 * mu * e[2] + lam * tr) / wp.max(sig[2], 1e-4),
      )
    )
    return U @ center @ wp.transpose(V) @ wp.transpose(F)
  J = sig[0] * sig[1] * sig[2]
  return 2.0 * mu * (F - U @ wp.transpose(V)) @ wp.transpose(F) + wp.identity(3, dtype=float) * (lam * J * (J - 1.0))


@wp.kernel
def _p2g(
  # Model:
  opt_timestep: wp.array[float],
  # MPM:
  material: int,
  nsubstep: int,
  res: wp.vec3i,
  lower: wp.vec3,
  dx: float,
  p_mass: float,
  p_vol: float,
  mu: float,
  lam: float,
  x_in: wp.array2d[wp.vec3],
  v_in: wp.array2d[wp.vec3],
  C_in: wp.array2d[wp.mat33],
  F_in: wp.array2d[wp.mat33],
  # Out:
  grid_mv_out: wp.array4d[wp.vec3],
  grid_m_out: wp.array4d[float],
):
  worldid, pid = wp.tid()
  dt = opt_timestep[worldid % opt_timestep.shape[0]] / float(nsubstep)
  inv_dx = 1.0 / dx
  base, fx, w = _stencil(x_in[worldid, pid], lower, inv_dx, res)
  stress = _stress(material, F_in[worldid, pid], mu, lam)
  affine = (-dt * p_vol * 4.0 * inv_dx * inv_dx) * stress + p_mass * C_in[worldid, pid]
  mv = p_mass * v_in[worldid, pid]
  for i in range(3):
    for j in range(3):
      for k in range(3):
        weight = w[i, 0] * w[j, 1] * w[k, 2]
        dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
        wp.atomic_add(grid_mv_out, worldid, base[0] + i, base[1] + j, base[2] + k, weight * (mv + affine @ dpos))
        wp.atomic_add(grid_m_out, worldid, base[0] + i, base[1] + j, base[2] + k, weight * p_mass)


@wp.func
def _geom_sdf(geom_type: int, p: wp.vec3, size: wp.vec3):
  """Signed distance and outward normal of a geom at p, in the geom frame."""
  if geom_type == GeomType.PLANE:
    return p[2], wp.vec3(0.0, 0.0, 1.0)
  if geom_type == GeomType.SPHERE:
    return collision_sdf.sphere(p, size), collision_sdf.grad_sphere(p)
  if geom_type == GeomType.BOX:
    return collision_sdf.box(p, size), collision_sdf.grad_box(p, size)
  if geom_type == GeomType.CAPSULE:
    return collision_sdf.capsule(p, size), collision_sdf.grad_capsule(p, size)
  if geom_type == GeomType.CYLINDER:
    return collision_sdf.cylinder(p, size), collision_sdf.grad_cylinder(p, size)
  if geom_type == GeomType.ELLIPSOID:
    return collision_sdf.ellipsoid(p, size), collision_sdf.grad_ellipsoid(p, size)
  # ponytail: mesh, hfield and sdf geoms don't collide with particles; add via mesh BVH queries.
  return 1.0e10, wp.vec3(0.0)


@wp.func
def _collide(
  # Model:
  ngeom: int,
  body_rootid: wp.array[int],
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # In:
  worldid: int,
  pos: wp.vec3,
  v: wp.vec3,
  mass: float,
  step_dt: float,
  tol: float,
  project: bool,
  # Out:
  xfrc_out: wp.array2d[wp.spatial_vector],
):
  """Contact of a point (grid node or particle) with the geoms.

  Stops the inward velocity relative to each geom it is inside of, with
  Coulomb friction, adds the reaction to xfrc_out, and if project, moves pos out
  of the geoms it is inside. Returns the new pos and v.

  The point is in contact when its distance to the geom is below tol. Points often lie exactly
  on a surface, where rounding would otherwise decide: grid nodes (a floor at a multiple of dx)
  and the particles projected onto it.
  """
  for geomid in range(ngeom):
    if geom_contype[geomid] == 0 and geom_conaffinity[geomid] == 0:
      continue
    rot = geom_xmat_in[worldid, geomid]
    local = wp.transpose(rot) @ (pos - geom_xpos_in[worldid, geomid])
    phi, n_local = _geom_sdf(geom_type[geomid], local, geom_size[worldid % geom_size.shape[0], geomid])
    if phi >= tol:
      continue
    n = wp.normalize(rot @ n_local)
    if project:
      pos -= phi * n

    # velocity of the geom at the point; cvel is (rot, lin) about the root subtree com
    bodyid = geom_bodyid[geomid]
    cvel = cvel_in[worldid, bodyid]
    offset = pos - subtree_com_in[worldid, body_rootid[bodyid]]
    v_rigid = wp.spatial_bottom(cvel) + wp.cross(wp.spatial_top(cvel), offset)

    # stop the inward motion, Coulomb friction on the tangential motion
    rel = v - v_rigid
    vn = wp.dot(rel, n)
    if vn >= 0.0:
      continue
    vt = rel - vn * n
    vt_norm = wp.length(vt)
    friction = geom_friction[worldid % geom_friction.shape[0], geomid][0]
    if vt_norm > 1.0e-10:
      vt *= wp.max(0.0, 1.0 + friction * vn / vt_norm)
    v_new = v_rigid + vt

    # reaction on the body, averaged over the MuJoCo step
    f = -mass * (v_new - v) / step_dt
    torque = wp.cross(pos - xipos_in[worldid, bodyid], f)
    wp.atomic_add(xfrc_out, worldid, bodyid, wp.spatial_vector(f, torque))
    v = v_new
  return pos, v


@wp.kernel
def _grid_update(
  # Model:
  ngeom: int,
  opt_timestep: wp.array[float],
  opt_gravity: wp.array[wp.vec3],
  body_rootid: wp.array[int],
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # MPM:
  nsubstep: int,
  res: wp.vec3i,
  lower: wp.vec3,
  dx: float,
  grid_m_in: wp.array4d[float],
  # In/Out:
  grid_mv: wp.array4d[wp.vec3],
  xfrc_out: wp.array2d[wp.spatial_vector],
):
  worldid, i, j, k = wp.tid()
  mass = grid_m_in[worldid, i, j, k]
  if mass <= 0.0:
    return
  step_dt = opt_timestep[worldid % opt_timestep.shape[0]]
  v = grid_mv[worldid, i, j, k] / mass + (step_dt / float(nsubstep)) * opt_gravity[worldid % opt_gravity.shape[0]]
  pos = lower + dx * wp.vec3(float(i), float(j), float(k))
  pos, v = _collide(
    ngeom, body_rootid, geom_type, geom_contype, geom_conaffinity, geom_bodyid, geom_size, geom_friction,
    xipos_in, geom_xpos_in, geom_xmat_in, subtree_com_in, cvel_in,
    worldid, pos, v, mass, step_dt, _CONTACT_TOL * dx, False, xfrc_out,
  )  # fmt: skip

  # domain walls
  idx = wp.vec3i(i, j, k)
  for a in range(3):
    if (idx[a] < _BOUND and v[a] < 0.0) or (idx[a] >= res[a] - _BOUND and v[a] > 0.0):
      v[a] = 0.0
  grid_mv[worldid, i, j, k] = v


@wp.kernel
def _g2p(
  # Model:
  ngeom: int,
  opt_timestep: wp.array[float],
  body_rootid: wp.array[int],
  geom_type: wp.array[int],
  geom_contype: wp.array[int],
  geom_conaffinity: wp.array[int],
  geom_bodyid: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  geom_friction: wp.array2d[wp.vec3],
  # Data in:
  xipos_in: wp.array2d[wp.vec3],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  subtree_com_in: wp.array2d[wp.vec3],
  cvel_in: wp.array2d[wp.spatial_vector],
  # MPM:
  material: int,
  nsubstep: int,
  res: wp.vec3i,
  lower: wp.vec3,
  dx: float,
  p_mass: float,
  mu: float,
  lam: float,
  sand_alpha: float,
  grid_v_in: wp.array4d[wp.vec3],
  # In/Out:
  x: wp.array2d[wp.vec3],
  v_out: wp.array2d[wp.vec3],
  C_out: wp.array2d[wp.mat33],
  F: wp.array2d[wp.mat33],
  Jp: wp.array2d[float],
  xfrc_out: wp.array2d[wp.spatial_vector],
):
  worldid, pid = wp.tid()
  step_dt = opt_timestep[worldid % opt_timestep.shape[0]]
  dt = step_dt / float(nsubstep)
  inv_dx = 1.0 / dx
  base, fx, w = _stencil(x[worldid, pid], lower, inv_dx, res)
  v = wp.vec3(0.0)
  C = wp.mat33(0.0)
  for i in range(3):
    for j in range(3):
      for k in range(3):
        weight = w[i, 0] * w[j, 1] * w[k, 2]
        dpos = (wp.vec3(float(i), float(j), float(k)) - fx) * dx
        gv = grid_v_in[worldid, base[0] + i, base[1] + j, base[2] + k]
        v += weight * gv
        C += (4.0 * inv_dx * inv_dx * weight) * wp.outer(gv, dpos)

  # the grid collision only acts on nodes inside the geoms, and the stencil mixes in the
  # velocity of nodes outside: push particles that still end up inside back to the surface
  xp, v = _collide(
    ngeom, body_rootid, geom_type, geom_contype, geom_conaffinity, geom_bodyid, geom_size, geom_friction,
    xipos_in, geom_xpos_in, geom_xmat_in, subtree_com_in, cvel_in,
    worldid, x[worldid, pid] + dt * v, v, p_mass, step_dt, _CONTACT_TOL * dx, True, xfrc_out,
  )  # fmt: skip
  x[worldid, pid] = xp
  v_out[worldid, pid] = v
  C_out[worldid, pid] = C

  Fp = (wp.identity(3, dtype=float) + dt * C) @ F[worldid, pid]
  if material == _LIQUID:
    # liquids keep only their volume change
    Fp = wp.identity(3, dtype=float) * wp.pow(wp.max(wp.determinant(Fp), 1.0e-6), 1.0 / 3.0)
  elif material == _SAND:
    # Drucker-Prager return mapping (Klar et al. 2016), as Genesis' Sand._sand_projection
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(Fp, U, sig, V)
    e = wp.vec3(wp.log(wp.max(wp.abs(sig[0]), 1e-4)), wp.log(wp.max(wp.abs(sig[1]), 1e-4)), wp.log(wp.max(wp.abs(sig[2]), 1e-4)))
    tr = e[0] + e[1] + e[2] + Jp[worldid, pid]
    e_hat = e - wp.vec3(tr / 3.0)
    e_hat_norm = wp.length(e_hat) + 1.0e-10
    s = wp.vec3(1.0)
    if tr >= 0.0:
      Jp[worldid, pid] = tr  # expansion: the grains separate, no stress
    else:
      Jp[worldid, pid] = 0.0
      dgamma = wp.max(0.0, e_hat_norm + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * tr * sand_alpha)
      s = wp.vec3(
        wp.exp(e[0] - dgamma / e_hat_norm * e_hat[0]),
        wp.exp(e[1] - dgamma / e_hat_norm * e_hat[1]),
        wp.exp(e[2] - dgamma / e_hat_norm * e_hat[2]),
      )
    Fp = U @ wp.diag(s) @ wp.transpose(V)
  F[worldid, pid] = Fp


def step(m: Model, d: Data, mm: MPMModel, md: MPMData):
  """Advances the particles by one MuJoCo timestep against the current geom poses.

  Fills md.xfrc with the reaction of the particles on the bodies.
  """
  step_arrays(
    mm,
    m.opt.timestep,
    m.opt.gravity,
    m.body_rootid,
    m.geom_type,
    m.geom_contype,
    m.geom_conaffinity,
    m.geom_bodyid,
    m.geom_size,
    m.geom_friction,
    d.xipos,
    d.geom_xpos,
    d.geom_xmat,
    d.subtree_com,
    d.cvel,
    md.x,
    md.v,
    md.C,
    md.F,
    md.Jp,
    md.xfrc,
    md.grid_mv,
    md.grid_m,
  )


def step_arrays(
  mm: MPMModel,
  opt_timestep,
  opt_gravity,
  body_rootid,
  geom_type,
  geom_contype,
  geom_conaffinity,
  geom_bodyid,
  geom_size,
  geom_friction,
  xipos,
  geom_xpos,
  geom_xmat,
  subtree_com,
  cvel,
  x,
  v,
  C,
  F,
  Jp,
  xfrc,
  grid_mv,
  grid_m,
):
  """step on plain Warp arrays, shared by MJWarp and the MJX (JAX) wrapper."""
  xfrc.zero_()
  nworld = x.shape[0]
  for _ in range(mm.nsubstep):
    grid_mv.zero_()
    grid_m.zero_()
    wp.launch(
      _p2g,
      dim=(nworld, mm.nparticle),
      inputs=[opt_timestep, mm.material, mm.nsubstep, mm.grid_res, mm.lower, mm.dx, mm.p_mass, mm.p_vol, mm.mu, mm.lam, x, v, C, F],
      outputs=[grid_mv, grid_m],
    )
    wp.launch(
      _grid_update,
      dim=(nworld, *mm.grid_res),
      inputs=[
        geom_type.shape[0],
        opt_timestep,
        opt_gravity,
        body_rootid,
        geom_type,
        geom_contype,
        geom_conaffinity,
        geom_bodyid,
        geom_size,
        geom_friction,
        xipos,
        geom_xpos,
        geom_xmat,
        subtree_com,
        cvel,
        mm.nsubstep,
        mm.grid_res,
        mm.lower,
        mm.dx,
        grid_m,
      ],
      outputs=[grid_mv, xfrc],
    )
    wp.launch(
      _g2p,
      dim=(nworld, mm.nparticle),
      inputs=[
        geom_type.shape[0],
        opt_timestep,
        body_rootid,
        geom_type,
        geom_contype,
        geom_conaffinity,
        geom_bodyid,
        geom_size,
        geom_friction,
        xipos,
        geom_xpos,
        geom_xmat,
        subtree_com,
        cvel,
        mm.material,
        mm.nsubstep,
        mm.grid_res,
        mm.lower,
        mm.dx,
        mm.p_mass,
        mm.mu,
        mm.lam,
        mm.sand_alpha,
        grid_mv,
      ],
      outputs=[x, v, C, F, Jp, xfrc],
    )
