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

#ifndef MUJOCO_PLUGIN_MPM_MPM_H_
#define MUJOCO_PLUGIN_MPM_MPM_H_

#include <array>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::plugin::mpm {

enum Material { kElastic = 0, kLiquid = 1, kSand = 2 };

// Constant parameters of one instance, read from the plugin config.
struct MpmConfig {
  Material material = kElastic;
  int nparticle = 0;
  int nsubstep = 1;
  int nthread = 1;  // worker threads, default: all cores
  std::array<int, 3> res = {0, 0, 0};
  std::array<mjtNum, 3> lower = {0, 0, 0};
  mjtNum dx = 0;
  mjtNum p_mass = 0;
  mjtNum p_vol = 0;
  mjtNum mu = 0;
  mjtNum lam = 0;
  mjtNum sand_alpha = 0;
  std::vector<mjtNum> pos;  // initial particle positions (nparticle x 3)

  // Reads and validates the config of a plugin instance; calls mju_error if
  // invalid.
  static MpmConfig FromModel(const mjModel* m, int instance);
};

// Material point method (MLS-MPM) particles coupled to the rigid bodies.
//
// The same algorithm as mujoco_warp's mpm module (the MJX/GPU path), so a scene
// trained there behaves the same here. Particles advance in the `advance`
// callback against the step's geom poses; their reaction on the bodies is
// applied as a passive force on the next step.
//
// plugin_state layout (nparticle = N), all mjtNum:
//   x[3N] v[3N] C[9N] F[9N] Jp[N] xfrc[6*nbody]
// xfrc is laid out like mjData.xfrc_applied (force, torque at xipos).
class Mpm {
 public:
  static void RegisterPlugin();

  explicit Mpm(MpmConfig config) : config_(std::move(config)) {}

  static int StateSize(const mjModel* m, int instance);
  void Reset(const mjModel* m, mjtNum* state) const;
  void Advance(const mjModel* m, mjData* d, mjtNum* state);
  void Compute(const mjModel* m, mjData* d, const mjtNum* state) const;
  void Visualize(const mjModel* m, const mjtNum* state, mjvScene* scn) const;

 private:
  MpmConfig config_;
  std::vector<mjtNum> grid_mv_;  // node momentum, then velocity
  std::vector<mjtNum> grid_m_;   // node mass
  std::vector<mjtNum> affine_;   // per particle: stress and APIC momentum (9)
  std::vector<mjtNum>
      svd_;  // per particle: SVD of F (U 9, sig 3, V 9), sand reuses it
  std::vector<int> slab_, slab_order_,
      slab_start_;  // particles bucketed by stencil base x
  std::vector<std::vector<mjtNum>> xfrc_thread_;  // reaction, per thread
};

}  // namespace mujoco::plugin::mpm

#endif  // MUJOCO_PLUGIN_MPM_MPM_H_
