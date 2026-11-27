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

// Tests for the mujoco.mpm plugin. Parity with the MuJoCo Warp solver is tested
// from Python, in mjx/mujoco/mjx/warp/mpm_test.py.

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <utility>
#include <vector>

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

// a 10 cm cube of particles above the origin, in a [-0.3, 0.3]^2 x [-0.1, 0.5]
// domain
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

std::string Xml(const std::string& world,
                const std::string& material = "elastic",
                const std::string& box = "-0.05 -0.05 0.1 0.05 0.05 0.2",
                const std::string& extra = "") {
  return absl::StrReplaceAll(kXml, {{"MATERIAL", material},
                                    {"BOX", box},
                                    {"EXTRA", extra},
                                    {"WORLD", world}});
}

struct Particles {
  int n;
  const mjtNum* state;
  const mjtNum* x(int p) const { return state + 3 * p; }
  const mjtNum* v(int p) const { return state + 3 * n + 3 * p; }
  const mjtNum* F(int p) const { return state + 15 * n + 9 * p; }
  const mjtNum* xfrc(int body) const { return state + 25 * n + 6 * body; }
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
  std::string world =
      R"(<geom type="plane" size="1 1 0.1" contype="0" conaffinity="0"/>)";
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
    for (int j = 0; j < 9; j++)
      dF = std::max(dF, std::abs(p.F(i)[j] - (j % 4 == 0)));
  }
  EXPECT_NEAR(vz, -kG * t, 1e-3 * kG * t);
  EXPECT_NEAR(z - z0, -0.5 * kG * t * t, 0.02 * 0.5 * kG * t * t);
  EXPECT_LT(vxy, MjTol(1e-8, 1e-4));
  EXPECT_LT(dF, MjTol(1e-6, 1e-4));
}

TEST_F(MpmTest, RestOnPlane) {
  // a block dropped on the floor neither passes through it nor leaves the
  // domain
  for (const char* material : {"elastic", "sand", "liquid"}) {
    char error[1024];
    MjModelPtr m =
        LoadModelFromString(Xml(kFloor, material), error, sizeof(error));
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
    MjModelPtr m = LoadModelFromString(
        Xml(kFloor, materials[i], "-0.04 -0.04 0 0.04 0.04 0.16"), error,
        sizeof(error));
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
  std::string world =
      R"(<geom type="box" pos="0 0 0.05" size="0.15 0.15 0.004"/>)";
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
  std::string world =
      R"(<body name="table" pos="0 0 0.05"><geom type="box" size="0.15 0.15 0.05"/></body>)";
  char error[1024];
  for (const char* density : {"1000", "1600"}) {
    std::string extra =
        std::string(R"(<config key="density" value=")") + density + R"("/>)";
    MjModelPtr m = LoadModelFromString(
        Xml(world, "elastic", "-0.05 -0.05 0.12 0.05 0.05 0.2", extra), error,
        sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    Step(m.get(), d.get(), 500);
    Particles p = GetParticles(m.get(), d.get());
    mjtNum weight = std::stod(density) * 0.1 * 0.1 * 0.08 * kG;
    const mjtNum* f = p.xfrc(1);
    EXPECT_NEAR(f[2], -weight, 0.1 * weight) << density;
    EXPECT_NEAR(f[0], 0, 0.05 * weight) << density;
    EXPECT_NEAR(f[1], 0, 0.05 * weight) << density;
  }
}

TEST_F(MpmTest, TwoWayCoupling) {
  // a free box falling on sand sinks in, and is then held up by the particles
  std::string world = R"(
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.17">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.03" density="1000" contype="2" conaffinity="2"/>
    </body>)";
  char error[1024];
  MjModelPtr m = LoadModelFromString(
      Xml(world, "sand", "-0.08 -0.08 0 0.08 0.08 0.1"), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  Step(m.get(), d.get(), 400);
  EXPECT_GT(d->xpos[5],
            0.06);  // resting in the sand (half size 0.03), not on the floor
  EXPECT_LT(d->xpos[5], 0.14);
  for (int i = 0; i < 3; i++) EXPECT_LT(std::abs(d->qvel[i]), 0.05);
}

TEST_F(MpmTest, FrictionAngleSetsPileHeight) {
  mjtNum height[2];
  const char* angle[2] = {"20", "45"};
  for (int i = 0; i < 2; i++) {
    char error[1024];
    std::string extra = std::string(R"(<config key="friction_angle" value=")") +
                        angle[i] + R"("/>)";
    MjModelPtr m = LoadModelFromString(
        Xml(kFloor, "sand", "-0.04 -0.04 0 0.04 0.04 0.16", extra), error,
        sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    Step(m.get(), d.get(), 300);
    Particles p = GetParticles(m.get(), d.get());
    std::vector<mjtNum> z;
    for (int j = 0; j < p.n; j++) z.push_back(p.x(j)[2]);
    std::sort(z.begin(), z.end());
    height[i] = z[static_cast<int>(0.95 * (p.n - 1))];
  }
  EXPECT_GT(height[1], 1.2 * height[0]);
}

TEST_F(MpmTest, GeomFrictionSlowsSliding) {
  // sand on a 20 degree slope slides freely without geom friction, and much
  // less with it
  mjtNum slide[2];
  const char* friction[2] = {"0.01", "1"};
  for (int i = 0; i < 2; i++) {
    std::string world =
        std::string(
            R"(<geom type="plane" size="1 1 0.1" euler="0 20 0" friction=")") +
        friction[i] + R"("/>)";
    char error[1024];
    MjModelPtr m = LoadModelFromString(
        Xml(world, "sand", "-0.05 -0.05 0.03 0.05 0.05 0.09"), error,
        sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    Step(m.get(), d.get(), 120);
    Particles p = GetParticles(m.get(), d.get());
    slide[i] = 0;
    for (int j = 0; j < p.n; j++)
      slide[i] += p.x(j)[0] / p.n;  // starts centred on x = 0
  }
  EXPECT_GT(slide[0], 2 * slide[1]);
  EXPECT_GT(slide[1], 0);
}

TEST_F(MpmTest, GeomTypesHoldParticles) {
  // sand poured on each geom type flows around it: no particle ends up inside
  using Sdf = std::function<mjtNum(const mjtNum*)>;  // distance of a point in
                                                     // the geom frame
  auto norm = [](mjtNum x, mjtNum y, mjtNum z) {
    return std::sqrt(x * x + y * y + z * z);
  };
  std::vector<std::pair<std::string, Sdf>> geoms = {
      {R"(type="sphere" size="0.06")",
       [&](const mjtNum* p) { return norm(p[0], p[1], p[2]) - 0.06; }},
      {R"(type="capsule" size="0.04 0.05")",
       [&](const mjtNum* p) {
         return norm(p[0], p[1],
                     p[2] - std::clamp(p[2], mjtNum(-0.05), mjtNum(0.05))) -
                0.04;
       }},
      {R"(type="cylinder" size="0.06 0.03")",
       [&](const mjtNum* p) {
         return std::max(norm(p[0], p[1], 0) - 0.06, std::abs(p[2]) - 0.03);
       }},
      // first-order distance, exact at the surface
      {R"(type="ellipsoid" size="0.08 0.05 0.03")",
       [&](const mjtNum* p) {
         return (norm(p[0] / 0.08, p[1] / 0.05, p[2] / 0.03) - 1) * 0.03;
       }},
      {R"(type="box" size="0.06 0.05 0.03")",
       [&](const mjtNum* p) {
         return std::max({std::abs(p[0]) - 0.06, std::abs(p[1]) - 0.05,
                          std::abs(p[2]) - 0.03});
       }},
  };
  for (const auto& [geom, sdf] : geoms) {
    std::string world =
        std::string(kFloor) + "<geom pos=\"0 0 0.07\" " + geom + "/>";
    char error[1024];
    MjModelPtr m = LoadModelFromString(
        Xml(world, "sand", "-0.05 -0.05 0.14 0.05 0.05 0.24"), error,
        sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    Step(m.get(), d.get(), 300);
    Particles p = GetParticles(m.get(), d.get());
    int ntouch = 0;
    for (int i = 0; i < p.n; i++) {
      mjtNum local[3] = {p.x(i)[0], p.x(i)[1], p.x(i)[2] - 0.07};
      mjtNum dist = sdf(local);
      ASSERT_GT(dist, -1e-5) << geom;
      ASSERT_GT(p.x(i)[2], -1e-5) << geom;
      ntouch += dist < 0.5 * kDx;
    }
    EXPECT_GT(ntouch, 20) << geom;  // and the sand does touch it
  }
}

TEST_F(MpmTest, DomainWallsHoldParticles) {
  // liquid thrown at the domain walls stays inside
  char error[1024];
  MjModelPtr m =
      LoadModelFromString(Xml(kFloor, "liquid"), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  Particles p = GetParticles(m.get(), d.get());
  mjtNum* v = d->plugin_state + m->plugin_stateadr[0] + 3 * p.n;
  for (int i = 0; i < p.n; i++) v[3 * i] = 3, v[3 * i + 1] = -2;
  Step(m.get(), d.get(), 300);
  mjtNum xmax = -1;
  for (int i = 0; i < p.n; i++) {
    xmax = std::max(xmax, p.x(i)[0]);
    for (int a = 0; a < 3; a++) {
      ASSERT_TRUE(std::isfinite(p.x(i)[a]));
      ASSERT_GT(p.x(i)[a], (a < 2 ? -0.3 : -0.1) + kDx);
      ASSERT_LT(p.x(i)[a], (a < 2 ? 0.3 : 0.5) - kDx);
    }
  }
  EXPECT_GT(xmax, 0.2);  // reached the wall
}

TEST_F(MpmTest, TwoInstances) {
  // two instances in one model each advance like the same instance on its own
  char error[1024];
  std::string one = Xml(kFloor, "sand", "-0.15 -0.05 0.1 -0.05 0.05 0.2");
  std::string second = R"(
      <instance name="liquid">
        <config key="material" value="liquid"/>
        <config key="dx" value="0.025"/>
        <config key="lower" value="-0.3 -0.3 -0.1"/>
        <config key="upper" value="0.3 0.3 0.5"/>
        <config key="box" value="0.05 -0.05 0.1 0.15 0.05 0.2"/>
      </instance>
    </plugin>)";
  MjModelPtr m1 = LoadModelFromString(one, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjModelPtr m2 = LoadModelFromString(
      absl::StrReplaceAll(one, {{"</plugin>", second}}), error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  ASSERT_EQ(m2->nplugin, 2);
  MjDataPtr d1 = MakeData(m1);
  MjDataPtr d2 = MakeData(m2);
  Step(m1.get(), d1.get(), 100);
  Step(m2.get(), d2.get(), 100);
  for (int i = 0; i < m1->npluginstate; i++) {
    ASSERT_EQ(d1->plugin_state[i], d2->plugin_state[m2->plugin_stateadr[0] + i])
        << i;
  }
  // the liquid fell too
  int n = (m2->plugin_statenum[1] - 6 * m2->nbody) / 25;
  EXPECT_EQ(n, 512);
  const mjtNum* x = d2->plugin_state + m2->plugin_stateadr[1];
  for (int i = 0; i < n; i++) EXPECT_LT(x[3 * i + 2], 0.1);
}

TEST_F(MpmTest, ResetAndCopy) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(Xml(kFloor, "sand"), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  std::vector<mjtNum> initial(d->plugin_state,
                              d->plugin_state + m->npluginstate);
  Step(m.get(), d.get(), 100);

  // a copy continues exactly like the original
  MjDataPtr copy(mj_copyData(nullptr, m.get(), d.get()));
  Step(m.get(), d.get(), 20);
  Step(m.get(), copy.get(), 20);
  for (int i = 0; i < m->npluginstate; i++) {
    ASSERT_EQ(d->plugin_state[i], copy->plugin_state[i]) << i;
  }

  mj_resetData(m.get(), d.get());
  for (int i = 0; i < m->npluginstate; i++)
    ASSERT_EQ(d->plugin_state[i], initial[i]) << i;
}

TEST_F(MpmTest, ThreadCountDoesNotChangeResult) {
  mjtNum zmean[2];
  const char* nthread[2] = {"1", "4"};
  for (int i = 0; i < 2; i++) {
    char error[1024];
    std::string extra =
        std::string(R"(<config key="nthread" value=")") + nthread[i] + R"("/>)";
    MjModelPtr m = LoadModelFromString(
        Xml(kFloor, "sand", "-0.05 -0.05 0.1 0.05 0.05 0.2", extra), error,
        sizeof(error));
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
  EXPECT_THAT(LoadModelFromString(Xml(kFloor, "mud"), error, sizeof(error)),
              IsNull());
  EXPECT_THAT(error, HasSubstr("material must be elastic, liquid or sand"));
  EXPECT_THAT(
      LoadModelFromString(Xml(kFloor, "sand", "-0.05 -0.05 0.4 0.05 0.05 0.6"),
                          error, sizeof(error)),
      IsNull());
  EXPECT_THAT(error, HasSubstr("inside [lower, upper]"));
  EXPECT_THAT(
      LoadModelFromString(Xml(kFloor, "sand", "0 0 0.1"), error, sizeof(error)),
      IsNull());
  EXPECT_THAT(error, HasSubstr("'box' needs 6 numbers"));
}

}  // namespace
}  // namespace mujoco
