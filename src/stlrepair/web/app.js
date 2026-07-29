// Relative paths on purpose: bare specifiers would need an import map, and a
// browser without import-map support fails to resolve them silently, leaving
// the page looking alive but with nothing wired up.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { STLLoader } from './vendor/STLLoader.js';

// Ordered worst-first so the most serious problem is always at the top, matching
// the CLI. Keep in step with CHECK_ORDER in report.py.
const CHECKS = [
  ['non_manifold_edges', 'Non-manifold edges'],
  ['self_intersections', 'Self-intersections'],
  ['naked_edges', 'Naked edges'],
  ['non_planar_holes', 'Non-planar holes'],
  ['planar_holes', 'Planar holes'],
  ['inverted_normals', 'Inverted normals'],
  ['degenerate_faces', 'Degenerate faces'],
  ['duplicate_faces', 'Duplicate faces'],
  ['disjoint_shells', 'Disjoint shells'],
];

// Disjoint shells are informational: a model may legitimately be several bodies.
const INFORMATIONAL = new Set(['disjoint_shells']);

const $ = (id) => document.getElementById(id);
const loader = new STLLoader();

let currentFile = null;
let busy = false;

// --------------------------------------------------------------------------
// Viewer: two scenes, one shared camera, so the views can never drift apart.
// --------------------------------------------------------------------------

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);
camera.position.set(0, 0, 5);

const views = ['before', 'after'].map((key) => {
  const canvas = $(`canvas-${key}`);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(key === 'before' ? 0x12151b : 0x11161c);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));

  const key1 = new THREE.DirectionalLight(0xffffff, 1.6);
  key1.position.set(1, 1.4, 1.1);
  const key2 = new THREE.DirectionalLight(0x88aaff, 0.6);
  key2.position.set(-1.2, -0.6, -0.9);
  scene.add(key1, key2);

  return { key, canvas, renderer, scene, mesh: null, edges: null, group: new THREE.Group() };
});
views.forEach((v) => v.scene.add(v.group));

const controls = new OrbitControls(camera, $('overlay'));
controls.enableDamping = true;
controls.dampingFactor = 0.08;

/* The viewports are grid items, so they resize when the panel or the window
   does and a window resize event covers only one of those. A ResizeObserver
   catches both, and the per-frame check backs it up. */
function syncSize() {
  for (const view of views) {
    const { clientWidth: w, clientHeight: h } = view.canvas.parentElement;
    if (w === 0 || h === 0) continue;
    if (view.canvas.width !== w * devicePixelRatio || view.canvas.height !== h * devicePixelRatio) {
      view.renderer.setSize(w, h, false);
    }
    if (camera.aspect !== w / h) {
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }
}

function tick() {
  requestAnimationFrame(tick);
  syncSize();
  controls.update();
  for (const view of views) view.renderer.render(view.scene, camera);
}

function surfaceMaterial() {
  return new THREE.MeshStandardMaterial({
    color: 0xc4ccd8,
    metalness: 0.1,
    roughness: 0.62,
    flatShading: true,
    side: THREE.DoubleSide,
    wireframe: $('wireframe').checked,
  });
}

function setGeometry(view, geometry) {
  view.group.clear();
  view.mesh = null;
  view.edges = null;
  if (!geometry) return;

  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  view.mesh = new THREE.Mesh(geometry, surfaceMaterial());
  view.group.add(view.mesh);
}

/** Draw the boundary edges the server found, so holes are visible at a glance. */
function setNakedEdges(view, base64) {
  if (view.edges) {
    view.group.remove(view.edges);
    view.edges = null;
  }
  if (!base64) return;

  const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
  const positions = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  view.edges = new THREE.LineSegments(
    geometry,
    new THREE.LineBasicMaterial({ color: 0xff5252, depthTest: false }),
  );
  view.edges.renderOrder = 2;
  view.edges.visible = $('highlight').checked;
  view.group.add(view.edges);
}

/** Frame whichever model we have, and centre both groups on the same origin. */
function frameAll() {
  const box = new THREE.Box3();
  let found = false;
  for (const view of views) {
    if (!view.mesh) continue;
    box.union(new THREE.Box3().setFromObject(view.mesh));
    found = true;
  }
  if (!found) return;

  const centre = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-4);

  for (const view of views) view.group.position.copy(centre).multiplyScalar(-1);

  const distance = radius / Math.sin((camera.fov * Math.PI) / 360);
  camera.near = Math.max(distance / 1000, 1e-4);
  camera.far = distance * 100;
  camera.position.set(distance * 0.7, distance * 0.55, distance * 0.7);
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
}

// --------------------------------------------------------------------------
// Report
// --------------------------------------------------------------------------

function cell(value, informational) {
  const td = document.createElement('td');
  if (value === null || value === undefined) {
    td.textContent = 'skipped';
    td.className = 'v-note';
    return td;
  }
  td.textContent = value;
  td.className = value === 0 ? 'v-ok' : informational ? 'v-note' : 'v-bad';
  return td;
}

function renderReport(data) {
  const body = $('checks');
  body.replaceChildren();

  for (const [field, label] of CHECKS) {
    const tr = document.createElement('tr');
    // Rows that were clean to begin with recede, so the eye lands on the ones
    // that actually mattered.
    if (!data.before[field]) tr.className = 'quiet';
    const name = document.createElement('td');
    name.textContent = label;
    tr.append(name, cell(data.before[field], INFORMATIONAL.has(field)),
      cell(data.after[field], INFORMATIONAL.has(field)));
    body.append(tr);
  }

  const fmt = (n) => n.toLocaleString();
  const signed = (n) => (n > 0 ? `+${fmt(n)}` : fmt(n));
  const rows = [
    ['Tier used', data.tier],
    ['Time', `${Math.round(data.elapsed_ms)} ms`],
    ['Vertices', `${fmt(data.before.vertex_count)} → ${fmt(data.after.vertex_count)} (${signed(data.vertex_delta)})`],
    ['Triangles', `${fmt(data.before.triangle_count)} → ${fmt(data.after.triangle_count)} (${signed(data.triangle_delta)})`],
    ['Watertight', data.after.is_watertight ? 'yes' : 'no'],
    ['Volume', data.after.is_watertight ? data.after.volume.toFixed(3) : 'n/a'],
  ];

  const summary = $('summary');
  summary.replaceChildren();
  for (const [term, value] of rows) {
    const dt = document.createElement('dt');
    dt.textContent = term;
    const dd = document.createElement('dd');
    dd.textContent = value;
    summary.append(dt, dd);
  }

  const steps = $('steps');
  steps.replaceChildren();
  for (const step of data.steps) {
    const li = document.createElement('li');
    li.textContent = step;
    steps.append(li);
  }

  $('download').href = `${data.download}?attach=1`;
  $('report').hidden = false;
}

// --------------------------------------------------------------------------
// Flow
// --------------------------------------------------------------------------

function setStatus(message, isError = false) {
  const el = $('status');
  el.textContent = message;
  el.classList.toggle('err', isError);
}

async function loadOriginal(file) {
  const buffer = await file.arrayBuffer();
  setGeometry(views[0], loader.parse(buffer));
  setGeometry(views[1], null);
  setNakedEdges(views[0], '');
  $('tag-before').textContent = '';
  $('tag-after').textContent = '';
  frameAll();
}

function acceptFile(file) {
  if (!/\.stl$/i.test(file.name)) {
    setStatus(`${file.name} is not an STL file.`, true);
    return;
  }
  currentFile = file;
  $('fileinfo').textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  $('report').hidden = true;
  $('run').disabled = false;
  setStatus('');
  loadOriginal(file).catch((err) => setStatus(`Could not display: ${err.message}`, true));
}

async function runRepair() {
  if (!currentFile || busy) return;
  busy = true;
  $('run').disabled = true;
  setStatus('Repairing…');

  const body = new FormData();
  body.append('file', currentFile);
  body.append('mode', $('mode').value);
  body.append('min_shell_fraction', $('min_shell_fraction').value);
  body.append('voxel_resolution', $('voxel_resolution').value);
  for (const id of ['fill_holes', 'resolve_non_manifold', 'check_self_intersections', 'ascii']) {
    body.append(id, $(id).checked ? '1' : '0');
  }

  try {
    const response = await fetch('/api/repair', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `server returned ${response.status}`);

    setNakedEdges(views[0], data.naked_edges_b64);

    const stl = await (await fetch(data.download)).arrayBuffer();
    setGeometry(views[1], loader.parse(stl));

    $('tag-before').textContent = `${data.before.triangle_count.toLocaleString()} tris`;
    $('tag-after').textContent = `${data.after.triangle_count.toLocaleString()} tris`;

    renderReport(data);
    frameAll();
    setStatus(data.after.is_clean
      ? `Repaired via the ${data.tier} tier.`
      : `Improved via the ${data.tier} tier, but problems remain.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    busy = false;
    $('run').disabled = false;
  }
}

// --------------------------------------------------------------------------
// Wiring
// --------------------------------------------------------------------------

// The label element handles opening the picker natively, so no click wiring is
// needed here. Resetting the value first means re-picking the same file still
// fires a change event.
$('file').addEventListener('click', (e) => { e.target.value = ''; });
$('file').addEventListener('change', (e) => e.target.files[0] && acceptFile(e.target.files[0]));
$('run').addEventListener('click', runRepair);
$('recentre').addEventListener('click', frameAll);

$('wireframe').addEventListener('change', (e) => {
  for (const view of views) {
    if (view.mesh) view.mesh.material.wireframe = e.target.checked;
  }
});

$('highlight').addEventListener('change', (e) => {
  for (const view of views) {
    if (view.edges) view.edges.visible = e.target.checked;
  }
});

/* The entire window accepts the drop. A small target in the corner meant most
   drops landed on the page, got swallowed by the navigation guard, and looked
   like nothing had happened. */
const veil = $('veil');
let dragDepth = 0;

const draggingFiles = (e) =>
  Array.from(e.dataTransfer?.types || []).includes('Files');

addEventListener('dragenter', (e) => {
  if (!draggingFiles(e)) return;
  e.preventDefault();
  // dragenter fires again for every child element, so nesting is counted
  // rather than toggled, otherwise the veil flickers as the cursor moves.
  dragDepth += 1;
  veil.hidden = false;
});

addEventListener('dragover', (e) => {
  if (draggingFiles(e)) e.preventDefault();
});

addEventListener('dragleave', (e) => {
  if (!draggingFiles(e)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) veil.hidden = true;
});

addEventListener('drop', (e) => {
  // Always prevent the default, or a stray drop navigates away from the app.
  e.preventDefault();
  dragDepth = 0;
  veil.hidden = true;
  const file = e.dataTransfer?.files[0];
  if (file) acceptFile(file);
});

function redraw() {
  syncSize();
  for (const view of views) view.renderer.render(view.scene, camera);
}

const sizeObserver = new ResizeObserver(redraw);
for (const view of views) sizeObserver.observe(view.canvas.parentElement);
addEventListener('resize', redraw);

syncSize();
tick();

// Tells the boot watchdog in index.html that wiring completed. If anything
// above throws, this never runs and the page says so rather than going quiet.
window.__stlrepairReady = true;
