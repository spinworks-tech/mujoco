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

// Tests for the mujoco.mpm plugin. Parity with the MuJoCo Warp solver is tested from Python,
// in mjx/mujoco/mjx/warp/mpm_test.py.

#include <algorithm>
#include <cmath>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_replace.h"
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using MpmTest = MujocoTest;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

constexpr mjtNum kDx = 0.02;
constexpr mjtNum kG = 9.81;

// a 10 cm cube of particles above the origin, in a [-0.3, 0.3]^2 x [-0.1, 0.5] domain
constexpr char kXml[] = R"(
<mujoco>
  <option timestep="0.002"/>
  <extension>
    <plugin plugin="mujoco.mpm">
      <instance name="particles">
        <config key="material" value="MATERIAL"/>
        <config key="dx" value="0.02"/>
        <config key="lower" value="-0.3 -0.3 -0.1"/>
        <config key="upper" value="0.3 0.3 0.5"/>
        <config key="box" value="BOX"/>
        EXTRA
      </instance>
    </plugin>
  </extension>
  <worldbody>
    WORLD
  </worldbody>
</mujoco>
)";

constexpr char kFloor[] = R"(<geom type="plane" size="1 1 0.1"/>)";

std::string Xml(const std::string& world, const std::string& material = "elastic",
                const std::string& box = "-0.05 -0.05 0.1 0.05 0.05 0.2",
                const std::string& extra = "") {
  return absl::StrReplaceAll(kXml, {{"MATERIAL", material}, {"BOX", box}, {"EXTRA", extra},
                                    {"WORLD", world}});
}

struct Particles {
  int n;
  const mjtNum* state;
  const mjtNum* x(int p) const { return state + 3*p; }
  const mjtNum* v(int p) const { return state + 3*n + 3*p; }
  const mjtNum* F(int p) const { return state + 15*n + 9*p; }
  const mjtNum* xfrc(int body) const { return state + 25*n + 6*body; }
};

Particles GetParticles(const mjModel* m, const mjData* d) {
  int n = (m->plugin_statenum[0] - 6 * m->nbody) / 25;
  return {n, d->plugin_state + m->plugin_stateadr[0]};
}

void Step(const mjModel* m, mjData* d, int nstep) {
  for (int i = 0; i < nstep; i++) mj_step(m, d);
}

TEST_F(MpmTest, FreeFall) {
  // without colliding geoms the block falls ballistically and keeps its shape
  std::string world = R"(<geom type="plane" size="1 1 0.1" contype="0" conaffinity="0"/>)";
  char error[1024];
  MjModelPtr m = LoadModelFromString(Xml(world), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  Particles p = GetParticles(m.get(), d.get());
  EXPECT_EQ(p.n, 1000);
  mjtNum z0 = 0;
  for (int i = 0; i < p.n; i++) z0 += p.x(i)[2] / p.n;

  int nstep = 50;
  Step(m.get(), d.get(), nstep);
  mjtNum t = nstep * m->opt.timestep, z = 0, vz = 0, vxy = 0, dF = 0;
  for (int i = 0; i < p.n; i++) {
    z += p.x(i)[2] / p.n;
    vz += p.v(i)[2] / p.n;
    vxy = std::max({vxy, std::abs(p.v(i)[0]), std::abs(p.v(i)[1])});
    for (int j = 0; j < 9; j++) dF = std::max(dF, std::abs(p.F(i)[j] - (j % 4 == 0)));
  }
  EXPECT_NEAR(vz, -kG * t, 1e-3 * kG * t);
  EXPECT_NEAR(z - z0, -0.5 * kG * t * t, 0.02 * 0.5 * kG * t * t);
  EXPECT_LT(vxy, 1e-8);
  EXPECT_LT(dF, 1e-6);
}

TEST_F(MpmTest, RestOnPlane) {
  // a block dropped on the floor neither passes through it nor leaves the domain
  for (const char* material : {"elastic", "sand", "liquid"}) {
    char error[1024];
    MjModelPtr m = LoadModelFromString(Xml(kFloor, material), error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    Step(m.get(), d.get(), 800);
    Particles p = GetParticles(m.get(), d.get());
    mjtNum zmin = 1, vmean = 0;
    for (int i = 0; i < p.n; i++) {
      for (int a = 0; a < 3; a++) {
        ASSERT_TRUE(std::isfinite(p.x(i)[a])) << material;
        vmean += std::abs(p.v(i)[a]) / (3 * p.n);
      }
      zmin = std::min(zmin, p.x(i)[2]);
      EXPECT_LT(std::abs(p.x(i)[0]), 0.3);
      EXPECT_LT(std::abs(p.x(i)[1]), 0.3);
    }
    EXPECT_GT(zmin, -kDx) << material;
    if (std::string(material) != "liquid") EXPECT_LT(vmean, 0.05) << material;
  }
}

TEST_F(MpmTest, SandPilesLiquidSpreads) {
  mjtNum height[2];
  const char* materials[2] = {"sand", "liquid"};
  for (int i = 0; i < 2; i++) {
    char error[1024];
    MjModelPtr m = LoadModelFromString(Xml(kFloor, materials[i], "-0.04 -0.04 0 0.04 0.04 0.16"),
                                       error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    Step(m.get(), d.get(), 300);
    Particles p = GetParticles(m.get(), d.get());
    std::vector<mjtNum> z;
    for (int j = 0; j < p.n; j++) z.push_back(p.x(j)[2]);
    std::sort(z.begin(), z.end());
    height[i] = z[static_cast<int>(0.95 * (p.n - 1))];
  }
  EXPECT_GT(height[0], 2 * height[1]);
}

TEST_F(MpmTest, ThinPlateHoldsParticles) {
  // a plate thinner than a grid cell still stops the particles
  std::string world = R"(<geom type="box" pos="0 0 0.05" size="0.15 0.15 0.004"/>)";
  char error[1024];
  MjModelPtr m = LoadModelFromString(Xml(world, "sand"), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  Step(m.get(), d.get(), 300);
  Particles p = GetParticles(m.get(), d.get());
  for (int i = 0; i < p.n; i++) EXPECT_GT(p.x(i)[2], 0.054 - 1e-4);
}

TEST_F(MpmTest, ReactionSupportsWeight) {
  // particles resting on a static box push it down with their weight
  std::string world = R"(<body name="table" pos="0 0 0.05"><geom type="box" size="0.15 0.15 0.05"/></body>)";
  char error[1024];
  MjModelPtr m = LoadModelFromString(Xml(world, "elastic", "-0.05 -0.05 0.12 0.05 0.05 0.2"),
                                     error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  Step(m.get(), d.get(), 500);
  Particles p = GetParticles(m.get(), d.get());
  mjtNum weight = p.n * 1000 * std::pow(0.5 * kDx, 3) * kG;
  const mjtNum* f = p.xfrc(1);
  EXPECT_NEAR(f[2], -weight, 0.1 * weight);
  EXPECT_NEAR(f[0], 0, 0.05 * weight);
  EXPECT_NEAR(f[1], 0, 0.05 * weight);
}

TEST_F(MpmTest, TwoWayCoupling) {
  // a free box falling on a resting elastic block is held up by the particles
  std::string world = R"(
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.17">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" density="300" contype="2" conaffinity="2"/>
    </body>)";
  char error[1024];
  MjModelPtr m = LoadModelFromString(
      Xml(world, "elastic", "-0.08 -0.08 0 0.08 0.08 0.1", R"(<config key="young" value="1e5"/>)"),
      error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  Step(m.get(), d.get(), 400);
  EXPECT_GT(d->xpos[5], 0.1);  // resting on the block (half size 0.03), not on the floor
  EXPECT_LT(d->xpos[5], 0.17);
}

TEST_F(MpmTest, ResetAndCopy) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(Xml(kFloor, "sand"), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  std::vector<mjtNum> initial(d->plugin_state, d->plugin_state + m->npluginstate);
  Step(m.get(), d.get(), 100);

  // a copy continues exactly like the original
  MjDataPtr copy(mj_copyData(nullptr, m.get(), d.get()));
  Step(m.get(), d.get(), 20);
  Step(m.get(), copy.get(), 20);
  for (int i = 0; i < m->npluginstate; i++) {
    ASSERT_EQ(d->plugin_state[i], copy->plugin_state[i]) << i;
  }

  mj_resetData(m.get(), d.get());
  for (int i = 0; i < m->npluginstate; i++) ASSERT_EQ(d->plugin_state[i], initial[i]) << i;
}

TEST_F(MpmTest, ThreadCountDoesNotChangeResult) {
  mjtNum zmean[2];
  const char* nthread[2] = {"1", "4"};
  for (int i = 0; i < 2; i++) {
    char error[1024];
    std::string extra = std::string(R"(<config key="nthread" value=")") + nthread[i] + R"("/>)";
    MjModelPtr m = LoadModelFromString(Xml(kFloor, "sand", "-0.05 -0.05 0.1 0.05 0.05 0.2", extra),
                                       error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    Step(m.get(), d.get(), 100);
    Particles p = GetParticles(m.get(), d.get());
    zmean[i] = 0;
    for (int j = 0; j < p.n; j++) zmean[i] += p.x(j)[2] / p.n;
  }
  EXPECT_NEAR(zmean[0], zmean[1], 1e-9);  // only the summation order differs
}

TEST_F(MpmTest, InvalidConfig) {
  char error[1024];
  EXPECT_THAT(LoadModelFromString(Xml(kFloor, "mud"), error, sizeof(error)), IsNull());
  EXPECT_THAT(error, HasSubstr("material must be elastic, liquid or sand"));
  EXPECT_THAT(LoadModelFromString(Xml(kFloor, "sand", "-0.05 -0.05 0.4 0.05 0.05 0.6"), error,
                                  sizeof(error)),
              IsNull());
  EXPECT_THAT(error, HasSubstr("inside [lower, upper]"));
  EXPECT_THAT(LoadModelFromString(Xml(kFloor, "sand", "0 0 0.1"), error, sizeof(error)), IsNull());
  EXPECT_THAT(error, HasSubstr("'box' needs 6 numbers"));
}

}  // namespace
}  // namespace mujoco
