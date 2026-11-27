# MPM particles (`mujoco.mpm`)

Sand, liquid and elastic solids simulated with the material point method
(MLS-MPM, ported from Genesis) and coupled two-way with the rigid bodies.

This plugin runs on the CPU, in standard MuJoCo, `simulate` and the Unity
plugin. The same solver runs on the GPU in MuJoCo Warp and MJX
(`mjx/mujoco/mjx/third_party/mujoco_warp/_src/mpm.py`), for parallel training.
Both read the same XML configuration and produce matching trajectories, up to
float32 rounding (tested in `mpm_test.py` there).

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
```

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
- **Unity:** build the plugin against the same MuJoCo version as the Unity
  package's library, and copy it to `Packages/org.mujoco/Runtime/Plugins/`.
  Importing the XML creates an `MjMpmParticles` component, which draws the
  particles with GPU instancing. Assign it a material that supports instancing
  if you're not on the built-in render pipeline.
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
- Rigid bodies and particles exchange forces once per step, so heavy, fast
  impacts rebound somewhat.
