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

using System.Xml;
using NUnit.Framework;
using UnityEngine;

namespace Mujoco {
[TestFixture]
public class MjMpmParticlesTests {
  private MjMpmParticles _particles;
  private XmlDocument _doc;

  [SetUp]
  public void SetUp() {
    _particles = new GameObject("particles").AddComponent<MjMpmParticles>();
    _doc = new XmlDocument();
  }

  [TearDown]
  public void TearDown() {
    UnityEngine.Object.DestroyImmediate(_particles.gameObject);
  }

  [Test]
  public void ParseAndGenerateRoundTrip() {
    _doc.LoadXml(
        "<instance name='sand'>" +
        "  <config key='material' value='liquid'/>" +
        "  <config key='dx' value='0.02'/>" +
        "  <config key='lower' value='-0.3 -0.3 -0.1'/>" +
        "  <config key='upper' value='0.3 0.3 0.5'/>" +
        "  <config key='box' value='-0.05 -0.05 0.1 0.05 0.05 0.2'/>" +
        "  <config key='density' value='1600'/>" +
        "</instance>");
    _particles.ParseMjcf(_doc.GetElementsByTagName("instance")[0] as XmlElement);
    Assert.That(_particles.Material, Is.EqualTo(MjMpmParticles.MpmMaterial.Liquid));
    Assert.That(_particles.Dx, Is.EqualTo(0.02f));
    Assert.That(_particles.BoxUpper, Is.EqualTo(new Vector3(0.05f, 0.05f, 0.2f)));
    Assert.That(_particles.Density, Is.EqualTo(1600f));
    Assert.That(_particles.Young, Is.EqualTo(3e5f));  // default

    var mjcf = _particles.GenerateMjcf("sand", _doc);
    Assert.That(mjcf.Name, Is.EqualTo("instance"));
    Assert.That(mjcf.GetAttribute("name"), Is.EqualTo("sand"));
    Assert.That(mjcf.SelectSingleNode("config[@key='material']/@value").Value,
                Is.EqualTo("liquid"));
    Assert.That(mjcf.SelectSingleNode("config[@key='box']/@value").Value,
                Is.EqualTo("-0.05 -0.05 0.1 0.05 0.05 0.2"));
    Assert.That(mjcf.SelectSingleNode("config[@key='dx']/@value").Value, Is.EqualTo("0.02"));
  }
}
}
