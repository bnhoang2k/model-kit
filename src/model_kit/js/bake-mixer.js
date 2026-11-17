// The kit's light mixer: sums the layers `model-kit bake` wrote, live, in three.js.
//
//   const bake = await loadBake('bake/bake.json', renderer);
//   const gltf = await new GLTFLoader().loadAsync('bake/<layout>.glb');
//   applyBake(gltf.scene, bake);
//   setLights(bake, { window: { color: [1, 0.9, 0.7], strength: 2 }, lamp: { color: [1, 1, 1], strength: 0 } });
//
// A layer is one switchable group of lights, baked at white and strength 1. `setLights` keys its
// values by LAYER name (a layer a value does not name stays white at strength 1) and redraws
// every light map as the sum, over the layers, of picture x colour x strength x UNITS.
//
// Units. `strength` is BLENDER's strength for the layer's lights, not three.js's intensity: a
// sun's irradiance in W/m^2, a point's or spot's power in W, a hemisphere's world strength. A
// texel holds the irradiance of its layer at strength 1, divided by pi (`bake.json` "units":
// "irradiance/pi"). A page that holds three.js intensities converts each with
// `strengthFromThree(kind, intensity)`, by the factors in STRENGTH_PER_THREE; the kind of each
// light is in `bake.json` (`layers.<layer>.baked_lights.<light>.kind`). Passing a three.js
// intensity as `strength` lights a baked bulb 4 pi (12.6) times too dim and a baked sky pi times
// too bright.
//
// `full` mode drops live direct, ambient and hemisphere light on still meshes. It does NOT drop
// image-based light (`scene.environment` / `material.envMap`): a page that uses an environment
// map in `full` mode gets it on top of the bake.
//
// Import map: the page maps the bare specifier `three` to its copy of three.js; compare.html
// shows how. Pictures are held top row first, as the .glb's v runs (v = 0 is the top row).

import * as THREE from 'three';
import { EXRLoader } from './vendor/loaders/EXRLoader.js';

// The factor that turns a baked texel into three.js `lightMap` irradiance. Measured in
// test/blender/test_bake_units.py: a white plane under a Blender sun of strength 1 bakes to texel
// 0.318359 (half float), a DirectionalLight of intensity 1 lights the same plane to 0.318310, and
// 3.1411 as UNITS makes the light map match it. That is pi to 0.02%: three.js multiplies a
// lightMap texel by the Lambert 1/pi itself, and the baked texel already holds that 1/pi.
export const UNITS = Math.PI;

// The Blender strength of one unit of three.js intensity, per light kind (linear, no tone map):
// a layer of Blender strength S lights a surface as a three.js light of intensity S divided by
// this. Sun: a DirectionalLight of intensity S, measured (test/blender/test_bake_units.py).
// Point: Blender's S watts are S / (4 pi) candela, the physical watt-to-candela factor of a light
// that sends its power evenly all round; `--compare` matched a live slab under it to 1%. Spot:
// Blender's spot keeps the point's intensity inside its cone, so the same factor; derived, not
// measured. Hemisphere: the world half of strength S is a HemisphereLight of pi x S (a uniform
// sky of radiance S gives irradiance pi x S); derived, not measured. An area light has no factor:
// three.js's RectAreaLight takes luminance, and nothing here measures the match.
export const STRENGTH_PER_THREE = Object.freeze({
  sun: 1,
  point: 4 * Math.PI,
  spot: 4 * Math.PI,
  hemisphere: 1 / Math.PI,
});

/** The Blender `strength` that lights a layer as a three.js light of `kind` at `intensity`. */
export function strengthFromThree(kind, intensity) {
  const factor = STRENGTH_PER_THREE[kind];
  if (factor === undefined) {
    throw new Error(`bake-mixer: no three.js factor for a light of kind "${kind}" (kinds: ${Object.keys(STRENGTH_PER_THREE).join(', ')})`);
  }
  return intensity * factor;
}

const VERTEX = `
varying vec2 vUv;
void main() {
  vUv = position.xy * 0.5 + 0.5;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;

/** The layer picture as a nearest-filtered float texture, row 0 = the picture's top row. */
function pictureTexture(picture) {
  const { width, height, data } = picture;
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, THREE.FloatType);
  texture.flipY = false;
  texture.minFilter = texture.magFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/** The sum pass: one fragment shader that adds every layer picture scaled by its own uniform. */
function sumPass(names, textures) {
  const uniforms = {};
  let body = 'vec3 sum = vec3(0.0);\n';
  names.forEach((name, i) => {
    uniforms[`t${i}`] = { value: textures[name] };
    uniforms[`k${i}`] = { value: new THREE.Vector3() };
    body += `sum += texture2D(t${i}, vUv).rgb * k${i};\n`;
  });
  const decl = names.map((_, i) => `uniform sampler2D t${i}; uniform vec3 k${i};`).join('\n');
  const material = new THREE.ShaderMaterial({
    uniforms,
    vertexShader: VERTEX,
    fragmentShader: `${decl}\nvarying vec2 vUv;\nvoid main() {\n${body}gl_FragColor = vec4(sum, 1.0);\n}`,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  const scene = new THREE.Scene();
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));
  return { scene, camera: new THREE.Camera(), material };
}

/** A light map target: float where the GPU can draw and filter it, else half float. */
function makeTarget(renderer, size) {
  const ext = (name) => renderer.extensions.has(name);
  const full = ext('EXT_color_buffer_float') && ext('OES_texture_float_linear');
  if (!full && !ext('EXT_color_buffer_half_float') && !ext('EXT_color_buffer_float')) {
    throw new Error('bake-mixer: this GPU cannot draw to a float or half-float texture, so the light map would stay black');
  }
  const target = new THREE.WebGLRenderTarget(size, size, {
    type: full ? THREE.FloatType : THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    generateMipmaps: false,
    depthBuffer: false,
  });
  target.texture.flipY = false;
  return target;
}

/**
 * A Bake from `bake.json`'s content and the layer pictures `{layer: {width, height, data}}`
 * (data: RGBA float32, top row first). `loadBake` calls this; a test can call it with pictures
 * it made.
 */
export function makeBake(info, pictures, renderer) {
  const names = Object.keys(info.layers);
  const textures = Object.fromEntries(names.map((n) => [n, pictureTexture(pictures[n])]));
  return {
    info, renderer, names, textures, size: info.size,
    pass: sumPass(names, textures),
    targets: new Map(), // UV channel -> render target; one per channel in use
    values: {},
  };
}

/** Fetch `bake.json` at `url` and every layer picture beside it. */
export async function loadBake(url, renderer) {
  const base = new URL(url, document.baseURI);
  const info = await (await fetch(base)).json();
  const exr = new EXRLoader().setDataType(THREE.FloatType);
  const pictures = {};
  for (const [name, layer] of Object.entries(info.layers)) {
    const texture = await exr.loadAsync(new URL(layer.picture, base).href);
    const { width, height, data } = texture.image;
    texture.dispose();
    const flipped = new Float32Array(data.length); // the loader gives the bottom row first
    const row = width * 4;
    for (let y = 0; y < height; y++) {
      flipped.set(data.subarray((height - 1 - y) * row, (height - y) * row), y * row);
    }
    pictures[name] = { width, height, data: flipped };
  }
  return makeBake(info, pictures, renderer);
}

/** The loaded .glb of the baked layout (`<layout>.glb` beside `bake.json`). */
export async function loadBakeScene(url, bake) {
  const { GLTFLoader } = await import('./vendor/loaders/GLTFLoader.js');
  const glb = new URL(`${bake.info.layout}.glb`, new URL(url, document.baseURI));
  return (await new GLTFLoader().loadAsync(glb.href)).scene;
}

/** The bake's name for a mesh: its own name or an ancestor's (a glTF mesh of several parts is a
 * group), read from `userData.name` first, as GLTFLoader strips `/` from `name`. */
function bakeName(object, known) {
  for (let o = object; o; o = o.parent) {
    for (const name of [o.userData && o.userData.name, o.name]) {
      if (name && known.has(name)) return name;
    }
  }
  return null;
}

// In `full` mode a still mesh takes no live light: after the light loop, direct light and the
// ambient and hemisphere irradiance are cleared, so only the light map (added next) is left.
function dropLiveLight(material) {
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      '#include <lights_fragment_begin>',
      `#include <lights_fragment_begin>
      reflectedLight.directDiffuse = vec3(0.0);
      reflectedLight.directSpecular = vec3(0.0);
      #if defined( RE_IndirectDiffuse )
      irradiance = vec3(0.0);
      #endif`);
  };
  material.customProgramCacheKey = () => 'bake-still';
}

/**
 * Give every still mesh under `root` a light map on the UV channel `bake.json` records, and in
 * `full` mode switch live light off on it. A part the bake lists under `unseen`, and any mesh
 * it does not name (a live piece), is left alone.
 *
 * Returns how many of the bake's parts (`uv_channel`'s names) got a light map. Fewer than all
 * means some baked part was not found under `root` by name, so it draws with no bake.
 */
export function applyBake(root, bake) {
  const { uv_channel: channels, mode } = bake.info;
  const known = new Set(Object.keys(channels));
  for (const name of bake.info.unseen || []) known.add(name);
  const applied = new Set();
  root.traverse((object) => {
    if (!object.isMesh) return;
    const name = bakeName(object, known);
    if (name === null || !(name in channels)) return;
    applied.add(name);
    const channel = channels[name];
    if (!bake.targets.has(channel)) {
      const target = makeTarget(bake.renderer, bake.size);
      target.texture.channel = channel;
      bake.targets.set(channel, target);
    }
    const target = bake.targets.get(channel);
    const clone = (m) => {
      const own = m.clone();
      own.lightMap = target.texture;
      own.lightMapIntensity = 1;
      if (mode === 'full') dropLiveLight(own);
      own.needsUpdate = true;
      return own;
    };
    object.material = Array.isArray(object.material) ? object.material.map(clone) : clone(object.material);
  });
  setLights(bake, bake.values);
  return applied.size;
}

/**
 * Redraw every light map. `values` maps a layer name to `{color: [r, g, b] (linear), strength}`;
 * a layer it does not name is drawn white at strength 1; strength 0 draws the layer as zero.
 */
export function setLights(bake, values) {
  for (const key of Object.keys(values)) {
    if (!bake.names.includes(key)) {
      throw new Error(`bake-mixer: "${key}" is no layer of this bake (layers: ${bake.names.join(', ')})`);
    }
  }
  bake.values = values;
  const { renderer, pass } = bake;
  bake.names.forEach((name, i) => {
    const { color = [1, 1, 1], strength = 1 } = values[name] || {};
    pass.material.uniforms[`k${i}`].value.set(
      color[0] * strength * UNITS, color[1] * strength * UNITS, color[2] * strength * UNITS);
  });
  const held = { target: renderer.getRenderTarget(), autoClear: renderer.autoClear };
  renderer.autoClear = true;
  for (const target of bake.targets.values()) {
    renderer.setRenderTarget(target);
    renderer.render(pass.scene, pass.camera);
  }
  renderer.setRenderTarget(held.target);
  renderer.autoClear = held.autoClear;
}

/** The pixels of the light map on `channel` as RGBA float32, row 0 = the bottom row of the
 * target = the picture's top row. For tests and for looking. */
export function readTarget(bake, channel) {
  const target = bake.targets.get(channel);
  const { size, renderer } = bake;
  if (target.texture.type === THREE.FloatType) {
    const out = new Float32Array(size * size * 4);
    renderer.readRenderTargetPixels(target, 0, 0, size, size, out);
    return out;
  }
  const raw = new Uint16Array(size * size * 4);
  renderer.readRenderTargetPixels(target, 0, 0, size, size, raw);
  return Float32Array.from(raw, (h) => THREE.DataUtils.fromHalfFloat(h));
}
