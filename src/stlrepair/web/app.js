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

let queue = [];
let running = false;
let stopRequested = false;

// --------------------------------------------------------------------------
// Viewer: two scenes, one shared camera, so the views can never drift apart.
// --------------------------------------------------------------------------

/* The 3D preview is a convenience, not the product. WebGL can be unavailable
   for reasons that have nothing to do with this app -- hardware acceleration
   switched off, a locked-down or sandboxed browser, a remote session, a VM
   without a GPU. Repairing a mesh needs none of it, so a failure here disables
   the preview and leaves diagnosis, repair and download working. */
let camera = null;
let controls = null;
let views = [];

function buildView(key) {
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

  const group = new THREE.Group();
  scene.add(group);
  return { key, canvas, renderer, scene, mesh: null, edges: null, group };
}

function startViewer() {
  try {
    camera = new THREE.PerspectiveCamera(45, 1, 0.01, 10000);
    camera.position.set(0, 0, 5);

    views = ['before', 'after'].map(buildView);

    controls = new OrbitControls(camera, $('overlay'));
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const sizeObserver = new ResizeObserver(redraw);
    for (const view of views) sizeObserver.observe(view.canvas.parentElement);
    addEventListener('resize', redraw);

    syncSize();
    tick();
    return true;
  } catch (err) {
    views = [];
    camera = null;
    controls = null;
    showViewerUnavailable(err);
    return false;
  }
}

function showViewerUnavailable(err) {
  const viewer = document.querySelector('.viewer');
  if (viewer) viewer.classList.add('no-webgl');

  const box = $('nowebgl');
  if (box) {
    box.hidden = false;
    box.querySelector('[data-reason]').textContent = String(err && err.message || err);
  }
}

const viewerReady = startViewer();

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
  if (!views.length) return;
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

/* Buffers live in GPU memory and are not garbage collected with the objects
   that referenced them. Working through a batch without disposing them grows
   memory with every file. */
function disposeGroup(group) {
  group.traverse((obj) => {
    obj.geometry?.dispose();
    const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
    for (const material of materials) material?.dispose();
  });
  group.clear();
}

function setGeometry(view, geometry) {
  if (!view) return;
  disposeGroup(view.group);
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
  if (!view) return;
  if (view.edges) {
    view.group.remove(view.edges);
    view.edges.geometry.dispose();
    view.edges.material.dispose();
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
  if (!views.length) return;
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

/* Above this many files the per-file preview is skipped while the batch runs
   and only the last result is shown, so a long queue does not spend its time
   parsing geometry twice per file just to redraw it a moment later. */
const PREVIEW_LIMIT = 5;

function buildValidatedDownloadUrl(downloadPath) {
  try {
    // Minimal path validation
    if (downloadPath.includes('/../') || /\/%2e%2e\//i.test(downloadPath)) {
      throw new Error('Invalid path');
    }
    
    const url = new URL(downloadPath, window.location.origin);
    
    // Ensure same origin
    if (url.origin !== window.location.origin) {
      throw new Error('Invalid host');
    }
    
    return url.href;
  } catch {
    throw new Error('Invalid URL');
  }
}

async function showPreview(item) {
  if (!viewerReady || !item.data) return;
  const data = item.data;

  const original = await item.file.arrayBuffer();
  setGeometry(views[0], loader.parse(original));
  setNakedEdges(views[0], data.naked_edges_b64);

  const repaired = await (await fetch(buildValidatedDownloadUrl(data.download))).arrayBuffer();
  setGeometry(views[1], loader.parse(repaired));

  $('tag-before').textContent = `${data.before.triangle_count.toLocaleString()} tris`;
  $('tag-after').textContent = `${data.after.triangle_count.toLocaleString()} tris`;
  frameAll();
}

function clearPreview() {
  if (!viewerReady) return;
  setGeometry(views[0], null);
  setGeometry(views[1], null);
  $('tag-before').textContent = '';
  $('tag-after').textContent = '';
}

function acceptFiles(fileList) {
  const incoming = Array.from(fileList || []);
  const stls = incoming.filter((f) => /\.stl$/i.test(f.name));
  const rejected = incoming.length - stls.length;

  if (!stls.length) {
    setStatus(
      incoming.length ? 'None of those are STL files.' : 'No files received.',
      true,
    );
    return;
  }

  for (const file of stls) {
    queue.push({ file, name: file.name, size: file.size, status: 'queued', note: '' });
  }

  $('report').hidden = true;
  setStatus(rejected ? `Ignored ${rejected} non-STL file${rejected > 1 ? 's' : ''}.` : '');
  renderQueue();
  updateRunButton();

  // A single file still gets an immediate preview of the original.
  if (queue.length === 1 && viewerReady) {
    stls[0].arrayBuffer()
      .then((buf) => { setGeometry(views[0], loader.parse(buf)); frameAll(); })
      .catch((err) => setStatus(`Could not display: ${err.message}`, true));
  }
}

async function repairOne(file) {
  const body = new FormData();
  body.append('file', file);
  body.append('mode', $('mode').value);
  body.append('min_shell_fraction', $('min_shell_fraction').value);
  body.append('voxel_resolution', $('voxel_resolution').value);
  for (const id of ['fill_holes', 'resolve_non_manifold', 'check_self_intersections', 'ascii']) {
    body.append(id, $(id).checked ? '1' : '0');
  }

  const response = await fetch('/api/repair', { method: 'POST', body });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `server returned ${response.status}`);
  return data;
}

/* Strictly one file at a time. Each repair is CPU and memory hungry, so firing
   the whole queue at once would swamp the machine for no gain: the server
   serialises them anyway. */
/** Files still to do: never started, or set aside by a stop. */
function resumable() {
  return queue.filter((item) => item.status === 'queued' || item.status === 'skipped');
}

function updateRunButton() {
  const left = resumable().length;
  const button = $('run');
  button.disabled = running || left === 0;
  const partial = left > 0 && left < queue.length;
  button.textContent = partial ? `Repair ${left} remaining` : 'Repair';
}

async function runBatch() {
  if (running) return;
  const pending = resumable();
  if (!pending.length) return;

  // A stopped batch is picked up where it left off rather than stranded.
  for (const item of pending) item.status = 'queued';

  running = true;
  stopRequested = false;
  $('run').disabled = true;
  $('stop').hidden = false;
  $('stop').disabled = false;   // a previous stop must not leave it dead
  $('bundle').hidden = true;

  const previewEachFile = pending.length <= PREVIEW_LIMIT;
  if (!previewEachFile) clearPreview();

  let finished = 0;
  let last = null;

  for (const item of queue) {
    if (item.status !== 'queued') continue;

    if (stopRequested) {
      item.status = 'skipped';
      item.note = 'stopped';
      renderQueue();
      continue;
    }

    item.status = 'active';
    renderQueue();
    setStatus(`Repairing ${item.name} — ${finished + 1} of ${pending.length}…`);

    try {
      const data = await repairOne(item.file);
      item.data = data;
      item.status = data.after.is_clean ? 'done' : 'partial';
      item.note = data.after.is_clean
        ? `${data.tier} · ${Math.round(data.elapsed_ms)} ms`
        : 'problems remain';
      last = item;

      renderReport(data);
      if (previewEachFile) await showPreview(item);
    } catch (err) {
      item.status = 'failed';
      item.note = err.message;
    }

    finished += 1;
    renderQueue();
  }

  // For a long queue the preview was skipped; show the final result now.
  if (!previewEachFile && last) {
    try { await showPreview(last); } catch { /* preview is optional */ }
  }

  running = false;
  $('stop').hidden = true;
  updateRunButton();
  summarise();
}

function summarise() {
  const count = (status) => queue.filter((item) => item.status === status).length;
  const ok = count('done');
  const partial = count('partial');
  const failed = count('failed');
  const skipped = count('skipped');

  const parts = [];
  if (ok) parts.push(`${ok} repaired`);
  if (partial) parts.push(`${partial} still imperfect`);
  if (failed) parts.push(`${failed} failed`);
  if (skipped) parts.push(`${skipped} skipped`);
  setStatus(parts.length ? parts.join(' · ') : 'Nothing to do.', failed > 0);

  const tokens = queue
    .filter((item) => item.data)
    .map((item) => item.data.download.split('/').pop());

  const bundle = $('bundle');
  bundle.hidden = tokens.length < 2;
  bundle.href = `/api/bundle?tokens=${tokens.join(',')}`;

  // The report is worth having for a single file too, not just a batch.
  const links = $('reportlinks');
  links.hidden = tokens.length === 0;
  $('reportcsv').href = `/api/report?tokens=${tokens.join(',')}`;
  $('reportjson').href = `/api/report?format=json&tokens=${tokens.join(',')}`;

  if (!$('batchbox').hidden) renderBatchTable();
}

/* Every check for every processed file. The queue answers "did it work"; this
   answers "what was actually wrong with each one", which is what a bulk job
   needs to review afterwards. */
function renderBatchTable() {
  const done = queue.filter((item) => item.data);
  const table = $('batchtable');
  table.replaceChildren();
  if (!done.length) return;

  const head = document.createElement('tr');
  for (const label of ['file', 'tier', 'triangles', ...CHECKS.map(([, l]) => l)]) {
    const th = document.createElement('th');
    th.textContent = label;
    head.append(th);
  }
  table.append(head);

  for (const item of done) {
    const tr = document.createElement('tr');
    const data = item.data;

    const name = document.createElement('td');
    name.textContent = item.name;
    const tier = document.createElement('td');
    tier.textContent = data.tier;
    const tris = document.createElement('td');
    tris.textContent = `${data.before.triangle_count.toLocaleString()} → ` +
      `${data.after.triangle_count.toLocaleString()}`;
    tr.append(name, tier, tris);

    for (const [field] of CHECKS) {
      const td = document.createElement('td');
      const was = data.before[field];
      const now = data.after[field];

      if (was === null || now === null) {
        td.textContent = 'n/c';           // the check could not be run
        td.className = 'note';
      } else if (!was && !now) {
        td.textContent = '0';
        td.className = 'zero';
      } else if (!now) {
        td.textContent = `${was} → 0`;
        td.className = 'fixed';
      } else {
        td.textContent = `${was} → ${now}`;
        td.className = INFORMATIONAL.has(field) ? 'note' : 'bad';
      }
      tr.append(td);
    }
    table.append(tr);
  }
}

function renderQueue() {
  const box = $('queuebox');
  box.hidden = queue.length === 0;
  if (!queue.length) {
    $('fileinfo').textContent = '';
    return;
  }

  const totalMb = queue.reduce((sum, item) => sum + item.size, 0) / 1024 / 1024;
  $('fileinfo').textContent =
    `${queue.length} file${queue.length > 1 ? 's' : ''} · ${totalMb.toFixed(2)} MB`;

  const done = queue.filter((i) => ['done', 'partial', 'failed', 'skipped'].includes(i.status));
  $('queuecount').textContent = `${done.length} of ${queue.length}`;

  const list = $('queue');
  list.replaceChildren();

  const marks = { queued: '·', active: '', done: '✓', partial: '!', failed: '✗', skipped: '–' };

  queue.forEach((item) => {
    const li = document.createElement('li');
    li.className = item.status;

    const mark = document.createElement('span');
    mark.className = 'qmark';
    mark.textContent = marks[item.status] ?? '·';

    const name = document.createElement('span');
    name.className = 'qname';
    name.textContent = item.name;
    name.title = item.name;

    const note = document.createElement('span');
    note.className = 'qnote';

    if (item.data) {
      const link = document.createElement('a');
      link.href = `${item.data.download}?attach=1`;
      link.download = '';
      link.textContent = 'download';
      note.append(link);
      // Clicking the row brings its report and preview back.
      name.style.cursor = 'pointer';
      name.addEventListener('click', () => {
        renderReport(item.data);
        showPreview(item).catch(() => {});
      });
    } else {
      note.textContent = item.note || item.status;
    }

    li.append(mark, name, note);
    list.append(li);
  });
}

// --------------------------------------------------------------------------
// Wiring
// --------------------------------------------------------------------------

// The label element handles opening the picker natively, so no click wiring is
// needed here. Resetting the value first means re-picking the same file still
// fires a change event.
$('file').addEventListener('click', (e) => { e.target.value = ''; });
$('file').addEventListener('change', (e) => acceptFiles(e.target.files));
$('run').addEventListener('click', runBatch);
$('recentre').addEventListener('click', frameAll);

$('stop').addEventListener('click', () => {
  // A no-op once the batch has finished, or it would overwrite the summary.
  if (!running) return;
  stopRequested = true;
  $('stop').disabled = true;
  setStatus('Stopping after the current file…');
});

$('showtable').addEventListener('click', () => {
  $('batchbox').hidden = false;
  renderBatchTable();
});
$('hidetable').addEventListener('click', () => { $('batchbox').hidden = true; });

$('clearqueue').addEventListener('click', () => {
  if (running) return;
  queue = [];
  clearPreview();
  $('report').hidden = true;
  $('bundle').hidden = true;
  $('reportlinks').hidden = true;
  $('batchbox').hidden = true;
  setStatus('');
  renderQueue();
  updateRunButton();
});

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
  const files = e.dataTransfer?.files;
  if (files?.length) acceptFiles(files);
});

// Hoisted, so startViewer can install this as its resize handler above.
function redraw() {
  if (!views.length) return;
  syncSize();
  for (const view of views) view.renderer.render(view.scene, camera);
}

// Tells the boot watchdog in index.html that wiring completed. If anything
// above throws, this never runs and the page says so rather than going quiet.
window.__stlrepairReady = true;
