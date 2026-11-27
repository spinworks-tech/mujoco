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

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Xml;
using UnityEngine;

namespace Mujoco {

// Sand, liquid or elastic particles simulated by the mujoco.mpm engine plugin (plugin/mpm), and
// drawn as instanced spheres. Corresponds to an <instance> of <extension><plugin
// plugin="mujoco.mpm">; the same XML configures the GPU solver used for training (MuJoCo Warp
// mpm.from_spec). Requires the mpm plugin library in Packages/org.mujoco/Runtime/Plugins/.
public class MjMpmParticles : MjComponent {

  public enum MpmMaterial { Elastic, Liquid, Sand }

  [Tooltip("Particle material.")]
  public MpmMaterial Material = MpmMaterial.Sand;

  [Tooltip("Grid spacing (m). Particles are sampled at half of it.")]
  public float Dx = 0.025f;

  [Tooltip("Lower corner of the simulation domain, in MuJoCo coordinates (z up).")]
  public Vector3 Lower = new Vector3(-0.35f, -0.35f, -0.1f);

  [Tooltip("Upper corner of the simulation domain, in MuJoCo coordinates (z up).")]
  public Vector3 Upper = new Vector3(0.35f, 0.35f, 0.5f);

  [Tooltip("Lower corner of the box initially filled with particles, in MuJoCo coordinates.")]
  public Vector3 BoxLower = new Vector3(-0.2f, -0.2f, 0f);

  [Tooltip("Upper corner of the box initially filled with particles, in MuJoCo coordinates.")]
  public Vector3 BoxUpper = new Vector3(0.2f, 0.2f, 0.06f);

  [Tooltip("Young's modulus (Pa); for liquids, the bulk stiffness.")]
  public float Young = 3e5f;

  [Tooltip("Poisson ratio.")]
  public float Poisson = 0.2f;

  [Tooltip("Density (kg/m^3).")]
  public float Density = 1000f;

  [Tooltip("Sand friction angle (degrees).")]
  public float FrictionAngle = 45f;

  [Tooltip("MPM substeps per MuJoCo step; 0 picks the stable count.")]
  public int Substeps = 0;

  [Tooltip("Worker threads; 0 uses all cores.")]
  public int Threads = 0;

  [Tooltip("Mesh drawn for each particle; a sphere if empty.")]
  public Mesh ParticleMesh;

  [Tooltip("Material of the particles; must support GPU instancing. Standard shader if empty.")]
  public UnityEngine.Material ParticleMaterial;

  public override MujocoLib.mjtObj ObjectType => MujocoLib.mjtObj.mjOBJ_PLUGIN;

  public int ParticleCount { get; private set; }

  private const int _batchSize = 1023;  // DrawMeshInstanced limit
  private int _stateAdr;
  private readonly List<Matrix4x4[]> _batches = new List<Matrix4x4[]>();

  private static readonly Dictionary<string, MpmMaterial> _materials =
      new Dictionary<string, MpmMaterial> {
        { "elastic", MpmMaterial.Elastic },
        { "liquid", MpmMaterial.Liquid },
        { "sand", MpmMaterial.Sand },
      };

  protected override void OnParseMjcf(XmlElement mjcf) {
    var config = mjcf.SelectNodes("config").OfType<XmlElement>()
        .ToDictionary(c => c.GetAttribute("key"), c => c.GetAttribute("value"));
    float Number(string key, float fallback) =>
        config.TryGetValue(key, out var value)
            ? float.Parse(value, CultureInfo.InvariantCulture) : fallback;
    float[] Numbers(string key) =>
        config[key].Split((char[])null, StringSplitOptions.RemoveEmptyEntries)
            .Select(v => float.Parse(v, CultureInfo.InvariantCulture)).ToArray();

    if (config.TryGetValue("material", out var material)) {
      Material = _materials[material];
    }
    Dx = Number("dx", Dx);
    var lower = Numbers("lower");
    var upper = Numbers("upper");
    var box = Numbers("box");
    Lower = new Vector3(lower[0], lower[1], lower[2]);
    Upper = new Vector3(upper[0], upper[1], upper[2]);
    BoxLower = new Vector3(box[0], box[1], box[2]);
    BoxUpper = new Vector3(box[3], box[4], box[5]);
    Young = Number("young", Young);
    Poisson = Number("poisson", Poisson);
    Density = Number("density", Density);
    FrictionAngle = Number("friction_angle", FrictionAngle);
    Substeps = (int)Number("nsubstep", Substeps);
    Threads = (int)Number("nthread", Threads);
  }

  // The <instance> element; MjScene wraps it in <extension><plugin plugin="mujoco.mpm">.
  protected override XmlElement OnGenerateMjcf(XmlDocument doc) {
    var mjcf = doc.CreateElement("instance");
    void Config(string key, string value) {
      var config = (XmlElement)mjcf.AppendChild(doc.CreateElement("config"));
      config.SetAttribute("key", key);
      config.SetAttribute("value", value);
    }
    string Vec(Vector3 v) => MjEngineTool.MakeLocaleInvariant($"{v.x} {v.y} {v.z}");
    Config("material", Material.ToString().ToLowerInvariant());
    Config("dx", MjEngineTool.MakeLocaleInvariant($"{Dx}"));
    Config("lower", Vec(Lower));
    Config("upper", Vec(Upper));
    Config("box", $"{Vec(BoxLower)} {Vec(BoxUpper)}");
    Config("young", MjEngineTool.MakeLocaleInvariant($"{Young}"));
    Config("poisson", MjEngineTool.MakeLocaleInvariant($"{Poisson}"));
    Config("density", MjEngineTool.MakeLocaleInvariant($"{Density}"));
    Config("friction_angle", MjEngineTool.MakeLocaleInvariant($"{FrictionAngle}"));
    Config("nsubstep", MjEngineTool.MakeLocaleInvariant($"{Substeps}"));
    Config("nthread", MjEngineTool.MakeLocaleInvariant($"{Threads}"));
    return mjcf;
  }

  protected override unsafe void OnBindToRuntime(
      MujocoLib.mjModel_* model, MujocoLib.mjData_* data) {
    // plugin_state: positions x[3N] first, then v, C, F, Jp (22N) and the reaction (6 nbody)
    _stateAdr = model->plugin_stateadr[MujocoId];
    ParticleCount = (model->plugin_statenum[MujocoId] - 6 * (int)model->nbody) / 25;
    _batches.Clear();
    for (int start = 0; start < ParticleCount; start += _batchSize) {
      _batches.Add(new Matrix4x4[Math.Min(_batchSize, ParticleCount - start)]);
    }
    if (ParticleMesh == null) {
      var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
      ParticleMesh = sphere.GetComponent<MeshFilter>().sharedMesh;
      DestroyImmediate(sphere);
    }
    if (ParticleMaterial == null) {
      ParticleMaterial = new UnityEngine.Material(Shader.Find("Standard"));
      ParticleMaterial.color = Material == MpmMaterial.Sand ? new Color(0.86f, 0.7f, 0.42f)
          : Material == MpmMaterial.Liquid ? new Color(0.25f, 0.55f, 0.95f)
          : new Color(0.4f, 0.8f, 0.4f);
    }
    ParticleMaterial.enableInstancing = true;
    OnSyncState(data);
  }

  public override unsafe void OnSyncState(MujocoLib.mjData_* data) {
    var x = data->plugin_state + _stateAdr;
    var scale = Vector3.one * (0.5f * Dx);
    for (int p = 0; p < ParticleCount; p++) {
      _batches[p / _batchSize][p % _batchSize] = Matrix4x4.TRS(
          MjEngineTool.UnityVector3(x + 3 * p), Quaternion.identity, scale);
    }
  }

  protected void LateUpdate() {
    foreach (var batch in _batches) {
      Graphics.DrawMeshInstanced(ParticleMesh, 0, ParticleMaterial, batch);
    }
  }

  // Positions of the particles, in Unity coordinates.
  public Vector3[] GetPositions() {
    return _batches.SelectMany(batch => batch).Select(m => (Vector3)m.GetColumn(3)).ToArray();
  }
}
}
