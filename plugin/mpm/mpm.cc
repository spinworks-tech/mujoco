// Copyright 2026 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mpm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

// A port of mjx/mujoco/mjx/third_party/mujoco_warp/_src/mpm.py; keep the two in sync.

namespace mujoco::plugin::mpm {
namespace {

constexpr int kBound = 3;  // grid cells next to the domain walls that stop outward motion

constexpr const char* kAttributes[] = {
    "material", "dx", "lower", "upper", "box",
    "young", "poisson", "density", "friction_angle", "nsubstep", "nthread",
};

//---------------------------------- small 3d linear algebra (row-major) -------------------------

void MatVec(mjtNum res[3], const mjtNum M[9], const mjtNum v[3]) {
  for (int i = 0; i < 3; i++) res[i] = M[3*i] * v[0] + M[3*i+1] * v[1] + M[3*i+2] * v[2];
}

void MatTVec(mjtNum res[3], const mjtNum M[9], const mjtNum v[3]) {
  for (int i = 0; i < 3; i++) res[i] = M[i] * v[0] + M[3+i] * v[1] + M[6+i] * v[2];
}

void MatMat(mjtNum res[9], const mjtNum A[9], const mjtNum B[9]) {
  mjtNum tmp[9];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tmp[3*i+j] = A[3*i] * B[j] + A[3*i+1] * B[3+j] + A[3*i+2] * B[6+j];
    }
  }
  std::memcpy(res, tmp, sizeof(tmp));
}

// res = A * B^T
void MatMatT(mjtNum res[9], const mjtNum A[9], const mjtNum B[9]) {
  mjtNum tmp[9];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      tmp[3*i+j] = A[3*i] * B[3*j] + A[3*i+1] * B[3*j+1] + A[3*i+2] * B[3*j+2];
    }
  }
  std::memcpy(res, tmp, sizeof(tmp));
}

mjtNum Dot(const mjtNum a[3], const mjtNum b[3]) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

mjtNum Norm(const mjtNum a[3]) { return std::sqrt(Dot(a, a)); }

void Cross(mjtNum res[3], const mjtNum a[3], const mjtNum b[3]) {
  mjtNum tmp[3] = {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
  std::memcpy(res, tmp, sizeof(tmp));
}

// as wp.normalize: zero stays zero
void Normalize(mjtNum a[3]) {
  mjtNum n = Norm(a);
  if (n > 0) for (int i = 0; i < 3; i++) a[i] /= n;
  else a[0] = a[1] = a[2] = 0;
}

mjtNum Sign(mjtNum x) { return x < 0 ? -1 : 1; }  // as wp.sign

// F = U diag(sig) V^T with U, V rotations; a reflection shows up as a negative sig[2]
// (the convention of wp.svd3).
void Svd3(const mjtNum F[9], mjtNum U[9], mjtNum sig[3], mjtNum V[9]) {
  mjtNum A[9], eigval[3], eigvec[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      A[3*i+j] = F[i] * F[j] + F[3+i] * F[3+j] + F[6+i] * F[6+j];
    }
  }

  // eigenvectors of F^T F by cyclic Jacobi rotations (a few times faster than mju_eig3)
  mjtNum scale = A[0] * A[0] + A[4] * A[4] + A[8] * A[8];
  for (int sweep = 0; sweep < 10; sweep++) {
    if (A[1] * A[1] + A[2] * A[2] + A[5] * A[5] <= 1e-28 * scale) break;
    for (int pq = 0; pq < 3; pq++) {
      int p = pq == 2 ? 1 : 0, q = pq == 0 ? 1 : 2;
      mjtNum apq = A[3*p+q];
      if (apq == 0) continue;
      mjtNum theta = (A[3*q+q] - A[3*p+p]) / (2 * apq);
      mjtNum t = Sign(theta) / (std::abs(theta) + std::sqrt(theta * theta + 1));
      mjtNum c = 1 / std::sqrt(t * t + 1), s = t * c;
      for (int k = 0; k < 3; k++) {  // A = A J
        mjtNum akp = A[3*k+p], akq = A[3*k+q];
        A[3*k+p] = c * akp - s * akq;
        A[3*k+q] = s * akp + c * akq;
      }
      for (int k = 0; k < 3; k++) {  // A = J^T A
        mjtNum apk = A[3*p+k], aqk = A[3*q+k];
        A[3*p+k] = c * apk - s * aqk;
        A[3*q+k] = s * apk + c * aqk;
      }
      for (int k = 0; k < 3; k++) {  // eigvec = eigvec J
        mjtNum vkp = eigvec[3*k+p], vkq = eigvec[3*k+q];
        eigvec[3*k+p] = c * vkp - s * vkq;
        eigvec[3*k+q] = s * vkp + c * vkq;
      }
    }
  }
  for (int i = 0; i < 3; i++) eigval[i] = A[4*i];

  // columns of V by decreasing eigenvalue, keeping det(V) = 1
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int a, int b) { return eigval[a] > eigval[b]; });
  int inversions = (order[0] > order[1]) + (order[0] > order[2]) + (order[1] > order[2]);
  for (int c = 0; c < 3; c++) {
    for (int r = 0; r < 3; r++) V[3*r+c] = eigvec[3*r+order[c]];
  }
  if (inversions % 2) for (int r = 0; r < 3; r++) V[3*r+2] = -V[3*r+2];

  mjtNum Fv[3][3], u[3][3];
  for (int c = 0; c < 3; c++) {
    mjtNum vc[3] = {V[c], V[3+c], V[6+c]};
    MatVec(Fv[c], F, vc);
  }
  sig[0] = Norm(Fv[0]);
  if (sig[0] > mjMINVAL) {
    for (int i = 0; i < 3; i++) u[0][i] = Fv[0][i] / sig[0];
  } else {
    u[0][0] = 1, u[0][1] = 0, u[0][2] = 0;
  }
  mjtNum proj = Dot(u[0], Fv[1]);
  for (int i = 0; i < 3; i++) u[1][i] = Fv[1][i] - proj * u[0][i];
  if (Norm(u[1]) > mjMINVAL) {
    Normalize(u[1]);
  } else {
    // any direction orthogonal to u0
    mjtNum axis[3] = {0, 0, 0};
    axis[std::abs(u[0][0]) < 0.5 ? 0 : (std::abs(u[0][1]) < 0.5 ? 1 : 2)] = 1;
    Cross(u[1], u[0], axis);
    Normalize(u[1]);
  }
  sig[1] = Dot(u[1], Fv[1]);
  Cross(u[2], u[0], u[1]);
  sig[2] = Dot(u[2], Fv[2]);
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) U[3*r+c] = u[c][r];
  }
}

// U diag(s) V^T
void FromSvd(mjtNum res[9], const mjtNum U[9], const mjtNum s[3], const mjtNum V[9]) {
  mjtNum Us[9];
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) Us[3*r+c] = U[3*r+c] * s[c];
  }
  MatMatT(res, Us, V);
}

//---------------------------------- geom signed distance (as collision_sdf.py) ------------------

mjtNum SdfBox(const mjtNum p[3], const mjtNum size[3], mjtNum grad[3]) {
  mjtNum a[3], b[3];
  for (int i = 0; i < 3; i++) a[i] = std::abs(p[i]) - size[i];
  mjtNum amax = std::max({a[0], a[1], a[2]});
  if (amax < 0) {
    // inside: radial field
    for (int i = 0; i < 3; i++) b[i] = -size[i] / a[i];
    Normalize(b);
    for (int i = 0; i < 3; i++) b[i] *= Sign(p[i]);
    mjtNum t = 1e30;
    for (int i = 0; i < 3; i++) t = std::min(t, -a[i] / std::abs(b[i]));
    std::memcpy(grad, b, sizeof(b));
    return -t * Norm(b);
  }
  for (int i = 0; i < 3; i++) b[i] = std::max(a[i], mjtNum(0));
  mjtNum c = Norm(b);
  for (int i = 0; i < 3; i++) grad[i] = a[i] <= 0 ? 0 : b[i] / c * (p[i] / std::abs(p[i]));
  return c + std::min(amax, mjtNum(0));
}

mjtNum SdfCylinder(const mjtNum p[3], const mjtNum size[3], mjtNum grad[3]) {
  mjtNum radial = std::sqrt(p[0]*p[0] + p[1]*p[1]);
  mjtNum dx = radial - size[0];
  mjtNum dy = std::abs(p[2]) - size[1];
  mjtNum dist = std::min(std::max(dx, dy), mjtNum(0)) +
                std::sqrt(std::max(dx, mjtNum(0)) * std::max(dx, mjtNum(0)) +
                          std::max(dy, mjtNum(0)) * std::max(dy, mjtNum(0)));
  mjtNum u[3] = {0, 0, 0};
  if (radial > mjMINVAL) u[0] = p[0] / radial, u[1] = p[1] / radial;
  mjtNum w[3] = {0, 0, Sign(p[2])};
  if (dx > 0 && dy > 0) {
    mjtNum len = std::sqrt(dx*dx + dy*dy);
    for (int i = 0; i < 3; i++) grad[i] = len > mjMINVAL ? u[i] * dx / len + w[i] * dy / len : 0;
  } else if (dx > 0 || (dy <= 0 && dx > dy)) {
    std::memcpy(grad, u, sizeof(u));
  } else {
    std::memcpy(grad, w, sizeof(w));
  }
  return dist;
}

mjtNum SdfEllipsoid(const mjtNum p[3], const mjtNum size[3], mjtNum grad[3]) {
  mjtNum a[3], b[3];
  for (int i = 0; i < 3; i++) a[i] = p[i] / size[i], b[i] = a[i] / size[i];
  mjtNum k0 = Norm(a), k1 = Norm(b);
  mjtNum dist = k0 * (k0 - 1) / (k1 != 0 ? k1 : 1e-12);
  mjtNum df_dk0 = (2 * k0 - 1) / k1;
  mjtNum df_dk1 = k0 * (k0 - 1) / (k1 * k1);
  for (int i = 0; i < 3; i++) {
    grad[i] = b[i] / k0 * df_dk0 - b[i] / k1 / (size[i] * size[i]) * df_dk1;
  }
  Normalize(grad);
  return dist;
}

// signed distance and outward normal of a geom at p, in the geom frame
mjtNum GeomSdf(int type, const mjtNum p[3], const mjtNum size[3], mjtNum grad[3]) {
  switch (type) {
    case mjGEOM_PLANE:
      grad[0] = 0, grad[1] = 0, grad[2] = 1;
      return p[2];
    case mjGEOM_SPHERE: {
      mjtNum c = Norm(p);
      for (int i = 0; i < 3; i++) grad[i] = c > 1e-9 ? p[i] / c : 0;
      return c - size[0];
    }
    case mjGEOM_CAPSULE: {
      mjtNum diff[3] = {p[0], p[1], p[2] - std::clamp(p[2], -size[1], size[1])};
      mjtNum c = Norm(diff);
      for (int i = 0; i < 3; i++) grad[i] = c > mjMINVAL ? diff[i] / c : 0;
      return c - size[0];
    }
    case mjGEOM_BOX:
      return SdfBox(p, size, grad);
    case mjGEOM_CYLINDER:
      return SdfCylinder(p, size, grad);
    case mjGEOM_ELLIPSOID:
      return SdfEllipsoid(p, size, grad);
    default:
      // ponytail: mesh, hfield and sdf geoms don't collide with particles; add via mjc_distance.
      return 1e10;
  }
}

// Contact of a point (grid node or particle) with the geoms. Stops the inward velocity relative
// to each geom it is inside of, with Coulomb friction, adds the reaction to xfrc, and if project,
// moves pos out of the geoms. As _collide in mpm.py.
void Collide(const mjModel* m, const mjData* d, mjtNum pos[3], mjtNum v[3], mjtNum mass,
             bool project, mjtNum* xfrc) {
  for (int g = 0; g < m->ngeom; g++) {
    if (m->geom_contype[g] == 0 && m->geom_conaffinity[g] == 0) continue;
    const mjtNum* rot = d->geom_xmat + 9*g;
    mjtNum rel_pos[3], local[3], n_local[3], n[3];
    for (int i = 0; i < 3; i++) rel_pos[i] = pos[i] - d->geom_xpos[3*g+i];
    MatTVec(local, rot, rel_pos);
    mjtNum phi = GeomSdf(m->geom_type[g], local, m->geom_size + 3*g, n_local);
    if (phi >= 0) continue;
    MatVec(n, rot, n_local);
    Normalize(n);
    if (project) for (int i = 0; i < 3; i++) pos[i] -= phi * n[i];

    // velocity of the geom at the point; cvel is (rot, lin) about the root subtree com
    int body = m->geom_bodyid[g];
    const mjtNum* cvel = d->cvel + 6*body;
    const mjtNum* com = d->subtree_com + 3*m->body_rootid[body];
    mjtNum offset[3] = {pos[0] - com[0], pos[1] - com[1], pos[2] - com[2]};
    mjtNum v_rigid[3];
    Cross(v_rigid, cvel, offset);
    for (int i = 0; i < 3; i++) v_rigid[i] += cvel[3+i];

    // stop the inward motion, Coulomb friction on the tangential motion
    mjtNum rel[3] = {v[0] - v_rigid[0], v[1] - v_rigid[1], v[2] - v_rigid[2]};
    mjtNum vn = Dot(rel, n);
    if (vn >= 0) continue;
    mjtNum vt[3] = {rel[0] - vn*n[0], rel[1] - vn*n[1], rel[2] - vn*n[2]};
    mjtNum vt_norm = Norm(vt);
    if (vt_norm > 1e-10) {
      mjtNum scale = std::max(mjtNum(0), 1 + m->geom_friction[3*g] * vn / vt_norm);
      for (int i = 0; i < 3; i++) vt[i] *= scale;
    }

    // reaction on the body, averaged over the MuJoCo step
    mjtNum f[3], arm[3], torque[3];
    for (int i = 0; i < 3; i++) {
      mjtNum v_new = v_rigid[i] + vt[i];
      f[i] = -mass * (v_new - v[i]) / m->opt.timestep;
      arm[i] = pos[i] - d->xipos[3*body+i];
      v[i] = v_new;
    }
    Cross(torque, arm, f);
    for (int i = 0; i < 3; i++) xfrc[6*body+i] += f[i], xfrc[6*body+3+i] += torque[i];
  }
}

// Quadratic B-spline stencil: base node, offset in cells, weights w[node][axis].
void Stencil(const MpmConfig& c, const mjtNum x[3], int base[3], mjtNum fx[3], mjtNum w[3][3]) {
  for (int a = 0; a < 3; a++) {
    mjtNum g = (x[a] - c.lower[a]) / c.dx;
    base[a] = std::clamp(static_cast<int>(std::floor(g - 0.5)), 0, c.res[a] - 3);
    fx[a] = g - base[a];
    w[0][a] = 0.5 * (1.5 - fx[a]) * (1.5 - fx[a]);
    w[1][a] = 0.75 - (fx[a] - 1) * (fx[a] - 1);
    w[2][a] = 0.5 * (fx[a] - 0.5) * (fx[a] - 0.5);
  }
}

mjtNum Det(const mjtNum F[9]) {
  return F[0] * (F[4]*F[8] - F[5]*F[7]) - F[1] * (F[3]*F[8] - F[5]*F[6]) +
         F[2] * (F[3]*F[7] - F[4]*F[6]);
}

// Kirchhoff stress times F^T, as Genesis' MPM materials update_stress, given F = U diag(sig) V^T.
void Stress(const MpmConfig& c, const mjtNum F[9], const mjtNum U[9], const mjtNum sig[3],
            const mjtNum V[9], mjtNum res[9]) {
  if (c.material == kSand) {
    // Hencky strain energy
    mjtNum e[3], s[3];
    for (int i = 0; i < 3; i++) e[i] = std::log(std::max(sig[i], 1e-4));
    mjtNum tr = e[0] + e[1] + e[2];
    for (int i = 0; i < 3; i++) s[i] = (2 * c.mu * e[i] + c.lam * tr) / std::max(sig[i], 1e-4);
    FromSvd(res, U, s, V);
    MatMatT(res, res, F);
    return;
  }
  mjtNum J = sig[0] * sig[1] * sig[2];
  mjtNum R[9], diff[9];
  MatMatT(R, U, V);
  for (int i = 0; i < 9; i++) diff[i] = F[i] - R[i];
  MatMatT(res, diff, F);
  for (int i = 0; i < 9; i++) res[i] *= 2 * c.mu;
  for (int i = 0; i < 3; i++) res[4*i] += c.lam * J * (J - 1);
}

// Runs fn(begin, end, thread) over [0, n) in nthread static chunks.
// ponytail: spawns threads per call (~50us for 4); a persistent pool if small scenes need it.
template <typename Fn>
void ParallelFor(int n, int nthread, Fn fn) {
  nthread = std::max(1, std::min(nthread, n));
  std::vector<std::thread> threads;
  for (int t = 1; t < nthread; t++) {
    threads.emplace_back(fn, static_cast<int>(int64_t(n) * t / nthread),
                         static_cast<int>(int64_t(n) * (t + 1) / nthread), t);
  }
  fn(0, static_cast<int>(int64_t(n) / nthread), 0);
  for (std::thread& thread : threads) thread.join();
}

std::vector<mjtNum> ReadNumbers(const mjModel* m, int instance, const char* key, int n,
                                bool required) {
  const char* value = mj_getPluginConfig(m, instance, key);
  std::vector<mjtNum> out;
  if (value && value[0]) {
    char* end = const_cast<char*>(value);
    while (true) {
      char* start = end;
      mjtNum x = std::strtod(start, &end);
      if (end == start) break;
      out.push_back(x);
    }
    while (*end == ' ' || *end == '\t' || *end == '\n') end++;
    if (*end) mju_error("mujoco.mpm: '%s' must be numbers, got '%s'", key, value);
  }
  if (out.empty() && !required) return out;
  if (static_cast<int>(out.size()) != n) {
    mju_error("mujoco.mpm: '%s' needs %d numbers", key, n);
  }
  return out;
}

mjtNum ReadNumber(const mjModel* m, int instance, const char* key, mjtNum fallback) {
  std::vector<mjtNum> v = ReadNumbers(m, instance, key, 1, false);
  return v.empty() ? fallback : v[0];
}

}  // namespace

MpmConfig MpmConfig::FromModel(const mjModel* m, int instance) {
  MpmConfig c;
  const char* material_attr = mj_getPluginConfig(m, instance, "material");
  std::string material = material_attr ? material_attr : "";
  if (material.empty() || material == "elastic") c.material = kElastic;
  else if (material == "liquid") c.material = kLiquid;
  else if (material == "sand") c.material = kSand;
  else mju_error("mujoco.mpm: material must be elastic, liquid or sand, got '%s'", material.c_str());

  c.dx = ReadNumbers(m, instance, "dx", 1, true)[0];
  if (c.dx <= 0) mju_error("mujoco.mpm: dx must be positive");
  std::vector<mjtNum> lower = ReadNumbers(m, instance, "lower", 3, true);
  std::vector<mjtNum> upper = ReadNumbers(m, instance, "upper", 3, true);
  std::vector<mjtNum> box = ReadNumbers(m, instance, "box", 6, true);
  mjtNum young = ReadNumber(m, instance, "young", 3e5);
  mjtNum poisson = ReadNumber(m, instance, "poisson", 0.2);
  mjtNum density = ReadNumber(m, instance, "density", 1000);
  mjtNum friction_angle = ReadNumber(m, instance, "friction_angle", 45);
  int nsubstep = static_cast<int>(ReadNumber(m, instance, "nsubstep", 0));

  // grid resolution, in float32 like mpm.make_model so both sides get the same grid
  for (int a = 0; a < 3; a++) {
    c.lower[a] = static_cast<float>(lower[a]);
    float extent = static_cast<float>(upper[a]) - static_cast<float>(lower[a]);
    c.res[a] = static_cast<int>(std::ceil(extent / static_cast<float>(c.dx))) + 1;
  }

  // particles on a lattice of spacing dx/2 filling box, as mpm.sample_box
  mjtNum spacing = 0.5 * c.dx;
  int n[3];
  for (int a = 0; a < 3; a++) {
    n[a] = std::max(0, static_cast<int>(std::ceil((box[a+3] - box[a] - 0.5*spacing) / spacing)));
  }
  for (int i = 0; i < n[0]; i++) {
    for (int j = 0; j < n[1]; j++) {
      for (int k = 0; k < n[2]; k++) {
        int idx[3] = {i, j, k};
        for (int a = 0; a < 3; a++) {
          mjtNum p = box[a] + 0.5 * spacing + idx[a] * spacing;
          if (p <= lower[a] + kBound * c.dx || p >= upper[a] - kBound * c.dx) {
            mju_error("mujoco.mpm: box must lie at least %d cells inside [lower, upper]", kBound);
          }
          c.pos.push_back(p);
        }
      }
    }
  }
  c.nparticle = n[0] * n[1] * n[2];
  if (!c.nparticle) mju_error("mujoco.mpm: box holds no particles");

  c.mu = young / (2 * (1 + poisson));
  c.lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson));
  if (c.material == kLiquid) c.mu = 0;
  c.p_vol = spacing * spacing * spacing;
  c.p_mass = density * c.p_vol;
  if (nsubstep <= 0) {
    mjtNum wave_speed = std::sqrt((c.lam + 2 * c.mu) / density);
    nsubstep = std::max(1, static_cast<int>(std::ceil(m->opt.timestep / (0.3 * c.dx / wave_speed))));
  }
  c.nsubstep = nsubstep;
  c.nthread = static_cast<int>(ReadNumber(m, instance, "nthread", 0));
  if (c.nthread <= 0) c.nthread = std::max(1u, std::thread::hardware_concurrency());
  mjtNum sin_phi = std::sin(friction_angle * mjPI / 180);
  c.sand_alpha = std::sqrt(2.0 / 3.0) * 2 * sin_phi / (3 - sin_phi);
  return c;
}

int Mpm::StateSize(const mjModel* m, int instance) {
  return 25 * MpmConfig::FromModel(m, instance).nparticle + 6 * m->nbody;
}

void Mpm::Reset(const mjModel* m, mjtNum* state) const {
  int N = config_.nparticle;
  std::memcpy(state, config_.pos.data(), 3 * N * sizeof(mjtNum));
  std::fill(state + 3*N, state + 25*N + 6*m->nbody, 0);
  mjtNum* F = state + 15*N;
  for (int p = 0; p < N; p++) F[9*p] = F[9*p+4] = F[9*p+8] = 1;
}

void Mpm::Advance(const mjModel* m, mjData* d, mjtNum* state) {
  const MpmConfig& c = config_;
  int N = c.nparticle;
  mjtNum* x = state;
  mjtNum* v = state + 3*N;
  mjtNum* Cs = state + 6*N;
  mjtNum* Fs = state + 15*N;
  mjtNum* Jp = state + 24*N;
  mjtNum* xfrc = state + 25*N;
  int nnode = c.res[0] * c.res[1] * c.res[2];
  int nthread = std::max(1, std::min(c.nthread, N / 256));
  grid_mv_.resize(3 * nnode);
  grid_m_.resize(nnode);
  affine_.resize(9 * N);
  svd_.resize(21 * N);
  slab_.resize(N);
  slab_order_.resize(N);
  xfrc_thread_.assign(nthread, std::vector<mjtNum>(6 * m->nbody));

  mjtNum dt = m->opt.timestep / c.nsubstep;
  mjtNum inv_dx = 1 / c.dx;
  auto node = [&](const int base[3], int i, int j, int k) {
    return ((base[0] + i) * c.res[1] + base[1] + j) * c.res[2] + base[2] + k;
  };

  for (int substep = 0; substep < c.nsubstep; substep++) {
    // stress and affine momentum of each particle
    ParallelFor(N, nthread, [&](int begin, int end, int) {
      for (int p = begin; p < end; p++) {
        mjtNum stress[9];
        mjtNum* F = Fs + 9*p;
        if (c.material == kLiquid) {
          mjtNum J = Det(F);
          std::fill(stress, stress + 9, 0);
          stress[0] = stress[4] = stress[8] = c.lam * J * (J - 1);
        } else {
          // sand reuses the SVD of its plastic projection in the previous substep
          mjtNum* usv = svd_.data() + 21*p;
          if (c.material != kSand || substep == 0) Svd3(F, usv, usv + 9, usv + 12);
          Stress(c, F, usv, usv + 9, usv + 12, stress);
        }
        for (int i = 0; i < 9; i++) {
          affine_[9*p+i] = -dt * c.p_vol * 4 * inv_dx * inv_dx * stress[i] + c.p_mass * Cs[9*p+i];
        }
      }
    });

    // particles to grid; a stencil spans 3 x-slabs, so slabs of the same color (base x mod 3)
    // don't share nodes and scatter in parallel
    std::fill(grid_mv_.begin(), grid_mv_.end(), 0);
    std::fill(grid_m_.begin(), grid_m_.end(), 0);
    slab_start_.assign(c.res[0] + 1, 0);
    for (int p = 0; p < N; p++) {
      slab_[p] = std::clamp(static_cast<int>(std::floor((x[3*p] - c.lower[0]) * inv_dx - 0.5)), 0,
                            c.res[0] - 3);
      slab_start_[slab_[p] + 1]++;
    }
    for (int i = 0; i < c.res[0]; i++) slab_start_[i + 1] += slab_start_[i];
    {
      std::vector<int> fill(slab_start_.begin(), slab_start_.end() - 1);
      for (int p = 0; p < N; p++) slab_order_[fill[slab_[p]]++] = p;
    }
    for (int color = 0; color < 3; color++) {
      int nslab = (c.res[0] - color + 2) / 3;
      ParallelFor(nslab, nthread, [&](int begin, int end, int) {
        for (int slab = 3 * begin + color; slab < 3 * end + color && slab < c.res[0]; slab += 3) {
          for (int o = slab_start_[slab]; o < slab_start_[slab + 1]; o++) {
            int p = slab_order_[o];
            int base[3];
            mjtNum fx[3], w[3][3];
            Stencil(c, x + 3*p, base, fx, w);
            const mjtNum* affine = affine_.data() + 9*p;
            for (int i = 0; i < 3; i++) {
              for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                  mjtNum weight = w[i][0] * w[j][1] * w[k][2];
                  mjtNum dpos[3] = {(i - fx[0]) * c.dx, (j - fx[1]) * c.dx, (k - fx[2]) * c.dx};
                  mjtNum adp[3];
                  MatVec(adp, affine, dpos);
                  int id = node(base, i, j, k);
                  for (int a = 0; a < 3; a++) {
                    grid_mv_[3*id+a] += weight * (c.p_mass * v[3*p+a] + adp[a]);
                  }
                  grid_m_[id] += weight * c.p_mass;
                }
              }
            }
          }
        }
      });
    }

    // grid velocity update: gravity, geoms, domain walls
    ParallelFor(c.res[0], nthread, [&](int begin, int end, int thread) {
      for (int i = begin; i < end; i++) {
        for (int j = 0; j < c.res[1]; j++) {
          for (int k = 0; k < c.res[2]; k++) {
            int id = (i * c.res[1] + j) * c.res[2] + k;
            mjtNum mass = grid_m_[id];
            if (mass <= 0) continue;
            mjtNum* gv = grid_mv_.data() + 3*id;
            for (int a = 0; a < 3; a++) gv[a] = gv[a] / mass + dt * m->opt.gravity[a];
            mjtNum pos[3] = {c.lower[0] + c.dx * i, c.lower[1] + c.dx * j, c.lower[2] + c.dx * k};
            Collide(m, d, pos, gv, mass, /*project=*/false, xfrc_thread_[thread].data());
            int idx[3] = {i, j, k};
            for (int a = 0; a < 3; a++) {
              if ((idx[a] < kBound && gv[a] < 0) || (idx[a] >= c.res[a] - kBound && gv[a] > 0)) {
                gv[a] = 0;
              }
            }
          }
        }
      }
    });

    // grid to particles
    ParallelFor(N, nthread, [&](int begin, int end, int thread) {
      for (int p = begin; p < end; p++) {
        int base[3];
        mjtNum fx[3], w[3][3], vp[3] = {0, 0, 0}, C[9] = {0};
        Stencil(c, x + 3*p, base, fx, w);
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
              mjtNum weight = w[i][0] * w[j][1] * w[k][2];
              mjtNum dpos[3] = {(i - fx[0]) * c.dx, (j - fx[1]) * c.dx, (k - fx[2]) * c.dx};
              const mjtNum* gv = grid_mv_.data() + 3*node(base, i, j, k);
              for (int a = 0; a < 3; a++) {
                vp[a] += weight * gv[a];
                for (int b = 0; b < 3; b++) C[3*a+b] += 4 * inv_dx * inv_dx * weight * gv[a] * dpos[b];
              }
            }
          }
        }

        // the grid collision only acts on nodes inside the geoms, and the stencil mixes in the
        // velocity of nodes outside: push particles that still end up inside back to the surface
        for (int a = 0; a < 3; a++) x[3*p+a] += dt * vp[a];
        Collide(m, d, x + 3*p, vp, c.p_mass, /*project=*/true, xfrc_thread_[thread].data());
        std::memcpy(v + 3*p, vp, sizeof(vp));
        std::memcpy(Cs + 9*p, C, sizeof(C));

        // deformation gradient and plasticity
        mjtNum Fp[9], I_dtC[9];
        for (int i = 0; i < 9; i++) I_dtC[i] = dt * C[i];
        for (int i = 0; i < 3; i++) I_dtC[4*i] += 1;
        MatMat(Fp, I_dtC, Fs + 9*p);
        if (c.material == kLiquid) {
          // liquids keep only their volume change
          mjtNum s = std::pow(std::max(Det(Fp), 1e-6), 1.0 / 3.0);
          std::fill(Fp, Fp + 9, 0);
          Fp[0] = Fp[4] = Fp[8] = s;
        } else if (c.material == kSand) {
          // Drucker-Prager return mapping (Klar et al. 2016), as Genesis' Sand._sand_projection
          mjtNum* U = svd_.data() + 21*p;
          mjtNum* s = U + 9;
          mjtNum* V = U + 12;
          mjtNum sig[3], e[3], e_hat[3];
          Svd3(Fp, U, sig, V);
          for (int i = 0; i < 3; i++) e[i] = std::log(std::max(std::abs(sig[i]), 1e-4));
          mjtNum tr = e[0] + e[1] + e[2] + Jp[p];
          for (int i = 0; i < 3; i++) e_hat[i] = e[i] - tr / 3;
          mjtNum e_hat_norm = Norm(e_hat) + 1e-10;
          s[0] = s[1] = s[2] = 1;
          if (tr >= 0) {
            Jp[p] = tr;  // expansion: the grains separate, no stress
          } else {
            Jp[p] = 0;
            mjtNum dgamma = std::max(mjtNum(0), e_hat_norm + (3 * c.lam + 2 * c.mu) / (2 * c.mu) *
                                                                 tr * c.sand_alpha);
            for (int i = 0; i < 3; i++) s[i] = std::exp(e[i] - dgamma / e_hat_norm * e_hat[i]);
          }
          FromSvd(Fp, U, s, V);  // and U, s, V stay as the SVD of the new F
        }
        std::memcpy(Fs + 9*p, Fp, sizeof(Fp));
      }
    });
  }

  // the reaction, summed over threads in a fixed order
  std::fill(xfrc, xfrc + 6*m->nbody, 0);
  for (const std::vector<mjtNum>& part : xfrc_thread_) {
    for (int i = 0; i < 6*m->nbody; i++) xfrc[i] += part[i];
  }
}

void Mpm::Compute(const mjModel* m, mjData* d, const mjtNum* state) const {
  // the reaction of the particles from the last advance, like xfrc_applied
  const mjtNum* xfrc = state + 25 * config_.nparticle;
  for (int b = 1; b < m->nbody; b++) {
    const mjtNum* f = xfrc + 6*b;
    if (f[0] || f[1] || f[2] || f[3] || f[4] || f[5]) {
      mj_applyFT(m, d, f, f + 3, d->xipos + 3*b, b, d->qfrc_passive);
    }
  }
}

void Mpm::Visualize(const mjModel* m, const mjtNum* state, mjvScene* scn) const {
  static constexpr float kRgba[3][4] = {
      {0.4, 0.8, 0.4, 1}, {0.25, 0.55, 0.95, 1}, {0.86, 0.7, 0.42, 1}};
  mjtNum size[3] = {0.25 * config_.dx, 0, 0};
  for (int p = 0; p < config_.nparticle && scn->ngeom < scn->maxgeom; p++) {
    mjvGeom* geom = scn->geoms + scn->ngeom++;
    mjv_initGeom(geom, mjGEOM_SPHERE, size, state + 3*p, nullptr, kRgba[config_.material]);
    geom->category = mjCAT_DECOR;
  }
}

void Mpm::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.mpm";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;
  plugin.nattribute = sizeof(kAttributes) / sizeof(kAttributes[0]);
  plugin.attributes = kAttributes;
  plugin.nstate = Mpm::StateSize;

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    d->plugin_data[instance] =
        reinterpret_cast<uintptr_t>(new Mpm(MpmConfig::FromModel(m, instance)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Mpm*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data, int instance) {
    reinterpret_cast<Mpm*>(plugin_data)->Reset(m, plugin_state);
  };
  plugin.compute = +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
    reinterpret_cast<Mpm*>(d->plugin_data[instance])
        ->Compute(m, d, d->plugin_state + m->plugin_stateadr[instance]);
  };
  plugin.advance = +[](const mjModel* m, mjData* d, int instance) {
    reinterpret_cast<Mpm*>(d->plugin_data[instance])
        ->Advance(m, d, d->plugin_state + m->plugin_stateadr[instance]);
  };
  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt, mjvScene* scn,
                         int instance) {
    reinterpret_cast<Mpm*>(d->plugin_data[instance])
        ->Visualize(m, d->plugin_state + m->plugin_stateadr[instance], scn);
  };
  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::mpm
