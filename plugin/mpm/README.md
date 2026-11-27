# MPM particles (`mujoco.mpm`)

Sand, liquid and elastic solids simulated with the material point method
(MLS-MPM, ported from Genesis) and coupled two-way with the rigid bodies.

This plugin runs on the CPU, in standard MuJoCo and `simulate`. The same solver
runs on the GPU in MuJoCo Warp and MJX
(`mjx/mujoco/mjx/third_party/mujoco_warp/_src/mpm.py`), for parallel training.
Both read the same XML configuration and produce matching trajectories, up to
float32 rounding (tested in `mpm_test.py` there). In a static scene the
particles agree to a fraction of a millimetre after 300 steps. In a stirred
one, such as `sand_tray.xml`, rounding differences grow as in any chaotic
system: after 3 s the sand surfaces differ by about 2 mm on average.

## Configuration

```xml
<extension>
  <plugin plugin="mujoco.mpm">
    <instance name="sand">
      <config key="material" value="sand"/>
      <config key="dx" value="0.025"/>
      <config key="lower" value="-0.35 -0.35 -0.1"/>
      <config key="upper" value="0.35 0.35 0.5"/>
      <config key="box" value="-0.2 -0.2 0 0.2 0.2 0.06"/>
    </instance>
  </plugin>
</extension>
<worldbody>
  <plugin instance="sand"/>
  ...
</worldbody>
```

The `<plugin instance="sand"/>` in the world body is optional for simulation,
but keeps the instance in copies of the spec: `mj_copySpec` (`MjSpec.copy()`
in Python) drops plugin instances which no element refers to.

| key | default | meaning |
|---|---|---|
| `material` | `elastic` | `elastic` (fixed-corotated), `liquid` or `sand` (Drucker-Prager) |
| `dx` | required | grid spacing (m) |
| `lower`, `upper` | required | corners of the simulation domain; particles stay inside |
| `box` | required | `x0 y0 z0 x1 y1 z1`, filled with particles at spacing `dx/2` |
| `young`, `poisson` | `3e5`, `0.2` | elasticity; for liquids, `young` sets the bulk stiffness |
| `density` | `1000` | kg/m³ |
| `friction_angle` | `45` | sand friction angle (degrees) |
| `nsubstep` | stable count | MPM substeps per MuJoCo step |
| `nthread` | all cores | worker threads |

The `box` must lie at least 3 cells inside the domain. Particles collide with
plane, sphere, capsule, cylinder, ellipsoid and box geoms whose `contype` or
`conaffinity` is nonzero, with friction from `geom_friction`. Geoms thinner
than `dx` also stop particles. See `model/plugin/mpm/sand_tray.xml`.

## State

The particles live in `mjData.plugin_state`, so `mj_resetData`, `mj_copyData`
and `mj_getState(mjSTATE_PLUGIN)` cover them. For an instance with N
particles, at `plugin_stateadr[instance]`, all `mjtNum`:

```
x[3N] v[3N] C[9N] F[9N] Jp[N] xfrc[6*nbody]
```

`x` holds the positions, and `xfrc` the reaction of the particles on each body
(force, then torque at `xipos`). Particles advance at the end of each step,
against that step's geom poses. Their reaction is applied as a passive force on
the next step, as in MuJoCo Warp, where it goes through `xfrc_applied`.

## Using it

- **MuJoCo, `simulate`:** build the `mpm` target and put the library in the
  plugin directory. `simulate` draws the particles.
- **Python, MJX training:** build the plugin against the installed `mujoco`
  package and load it with `mujoco.mj_loadPluginLibrary`. The command below
  builds it on Linux:

  ```sh
  P=$(python -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")
  g++ -std=c++20 -O2 -shared -fPIC -I$P/include plugin/mpm/mpm.cc plugin/mpm/register.cc \
      -L$P -l:libmujoco.so.$(python -c "import mujoco; print(mujoco.__version__)") -o libmpm.so
  ```

  Then read the config into the GPU solver. `from_spec` removes the instance,
  so MJX doesn't allocate the plugin state in every world:

  ```python
  spec = mujoco.MjSpec.from_xml_path("sand_tray.xml")
  kwargs = mpm.from_spec(spec)
  mjm = spec.compile()
  mm, md = mpm.make(m, d, **kwargs)
  ```

## Performance

The CPU cost is mostly one or two 3x3 SVDs per particle per substep. In
`sand_tray.xml` (5,120 sand particles, 4 substeps per step) on a 4-core
i5-7400, a step takes 28 ms on one thread and 11.5 ms on four. That is 0.17x
real time. To run faster, use a coarser `dx` (8x fewer particles per doubling)
or a softer material (fewer substeps).

## Limitations

- Mesh, heightfield and SDF geoms don't collide with particles.
- Each instance has a single material.
- Particles come to rest up to about half a cell above a surface, as usual for
  MPM: the grid stops them at the nodes around it.
- Rigid bodies and particles exchange forces once per step, with a one-step
  delay. Sand absorbs the error this introduces, though a body lighter than
  water can rebound higher than it fell before it settles. Elastic material
  doesn't absorb it: a body resting on an elastic block can keep bouncing, and
  a light one on a stiff block gains energy. Use sand or liquid under free
  bodies, or bodies that are heavy relative to the material they press on.

## Tests

`test/plugin/mpm/mpm_test.cc` tests the plugin on its own (`ctest -R MpmTest`).
`mjx/mujoco/mjx/third_party/mujoco_warp/_src/mpm_test.py` tests the GPU solver
and, when `MUJOCO_MPM_PLUGIN` is set to the plugin library built as above,
compares the two on every material, geom type and config key.
