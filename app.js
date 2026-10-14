// Vision Radar - frontend application.
//
// Deliberately vanilla JS/ES modules, not React: the spec's architecture
// diagram names "React / TypeScript" for the frontend, but this project was
// built in a sandbox with no npm registry access, so React itself could not
// be installed as a real, buildable dependency (only a bare copy existed
// globally with no bundler). Rather than ship untested React source files,
// this is a real, working, Playwright-tested dashboard using the same
// component boundaries the spec describes (status, camera, radar, zone
// editor, event timeline, entity panel, search, alerts) - porting it to
// React later is a view-layer rewrite, not a behaviour rewrite, since all
// the logic here is already correct and tested.

const API_BASE = window.VISION_RADAR_API || "";
const POLL_MS = 2000;

// ---------------------------------------------------------------- API client
class ApiError extends Error {
  constructor(status, detail) { super(detail); this.status = status; }
}

const api = {
  token: null,
  async request(method, path, { json, params } = {}) {
    const url = new URL(API_BASE + path, location.href);
    if (params) for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
    const headers = {};
    if (this.token) headers["Authorization"] = `Bearer ${this.token}`;
    if (json !== undefined) headers["Content-Type"] = "application/json";
    const res = await fetch(url, { method, headers, body: json !== undefined ? JSON.stringify(json) : undefined });
    if (res.status === 204) return null;
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new ApiError(res.status, body.detail || `HTTP ${res.status}`);
    return body;
  },
  get(path, params) { return this.request("GET", path, { params }); },
  post(path, json) { return this.request("POST", path, { json }); },
  del(path) { return this.request("DELETE", path); },
  // Images can't carry an Authorization header via a plain <img src>, and
  // the snapshot endpoint requires one - a bare <img> request came back 401
  // (JSON) where an image was expected, and Chromium's opaque-response
  // blocking (ORB) silently killed it rather than showing a normal broken
  // image. Fetching it with auth and handing the page a blob: URL keeps the
  // token where it belongs (a header, not a guessable query string) and
  // actually works.
  async getBlobUrl(path) {
    const headers = this.token ? { Authorization: `Bearer ${this.token}` } : {};
    const res = await fetch(new URL(API_BASE + path, location.href), { headers });
    if (!res.ok) throw new ApiError(res.status, (await res.json().catch(() => ({}))).detail || `HTTP ${res.status}`);
    return URL.createObjectURL(await res.blob());
  },
};

// -------------------------------------------------------------------- state
const state = {
  user: null,            // {username, role, token}
  view: "overview",
  loginError: null,
  cameras: [], zones: [], rules: [], users: [],
  events: [], alerts: [], entities: [],
  status: null,
  search: { q: "", result: null, error: null },
  eventFilters: { event_type: "", zone_id: "", state: "", min_confidence: "" },
  selectedEvent: null, selectedEventImageUrl: null, liveFrameUrl: null,
  drawZone: { active: false, camera_id: "", name: "", points: [] },
  ruleDraft: '{\n  "when": {"event_type": "ZONE_ENTRY", "entity_type": "person"},\n  "if": [{"field": "confidence", "operator": ">", "value": 0.5}]\n}',
  busy: false, flash: null,
};

let pollHandle = null;

function setState(patch) { Object.assign(state, patch); render(); }
function flash(msg, isError = false) { state.flash = { msg, isError }; render(); setTimeout(() => { if (state.flash?.msg === msg) setState({ flash: null }); }, 4000); }

async function guarded(fn) {
  state.busy = true; render();
  try { await fn(); }
  catch (e) {
    if (e instanceof ApiError && e.status === 401) { logout("Session expired - please log in again."); return; }
    flash(e.message || String(e), true);
  } finally { state.busy = false; render(); }
}

// ------------------------------------------------------------------- login
async function login(username, password) {
  await guarded(async () => {
    const r = await api.post("/auth/login", { username, password });
    api.token = r.token;
    setState({ user: { username: r.username, role: r.role }, loginError: null, view: "overview" });
    startPolling();
    await refreshAll();
  });
}

function logout(reason) {
  stopPolling();
  api.token = null;
  if (state.selectedEventImageUrl) URL.revokeObjectURL(state.selectedEventImageUrl);
  if (state.liveFrameUrl) URL.revokeObjectURL(state.liveFrameUrl);
  setState({ user: null, loginError: reason || null, status: null, events: [], alerts: [], entities: [],
            selectedEvent: null, selectedEventImageUrl: null, liveFrameUrl: null });
}

// ----------------------------------------------------------------- loading
async function refreshStatus() { setState({ status: await api.get("/status") }); }
async function refreshCameras() { setState({ cameras: await api.get("/cameras") }); }
async function refreshZones() { setState({ zones: await api.get("/zones") }); }
async function refreshRules() { setState({ rules: await api.get("/rules") }); }
async function refreshUsers() { if (state.user?.role === "administrator") setState({ users: await api.get("/users") }); }
async function refreshEntities() { setState({ entities: await api.get("/entities", { limit: 200 }) }); }
async function refreshAlerts() { setState({ alerts: await api.get("/alerts", { limit: 50 }) }); }

async function refreshLiveFrame() {
  // Section 17-B, "live camera": while a session is actually running, show
  // its most recent frame. Once it ends, the server stops serving one
  // (Section 32: never present stale data as live) - treat that the same
  // as "no session" rather than leaving the last image frozen on screen.
  const sessionId = (state.status?.active_sessions || [])[0];
  if (!sessionId) {
    if (state.liveFrameUrl) { URL.revokeObjectURL(state.liveFrameUrl); setState({ liveFrameUrl: null }); }
    return;
  }
  try {
    const url = await api.getBlobUrl(`/sessions/${sessionId}/frame`);
    const old = state.liveFrameUrl;
    setState({ liveFrameUrl: url });
    if (old) URL.revokeObjectURL(old);
  } catch { /* session may have just ended between the status check and this call */ }
}
async function refreshEvents() {
  const f = state.eventFilters;
  setState({
    events: await api.get("/events", {
      event_type: f.event_type || undefined, zone_id: f.zone_id || undefined,
      state: f.state || undefined, min_confidence: f.min_confidence || undefined, limit: 100,
    }),
  });
}

async function refreshAll() {
  await Promise.all([refreshStatus(), refreshCameras(), refreshZones(), refreshRules(),
                     refreshUsers(), refreshEntities(), refreshAlerts(), refreshEvents()]);
}

// Each view's own data must be fetched the moment it's opened - relying only
// on a background poll meant switching tabs could show stale (or, right
// after login, completely empty) data until the next poll tick happened to
// land while that tab was still open. Found by an end-to-end test clicking
// through tabs faster than the old poll interval: the backend had real
// entities/alerts the whole time, but the UI never asked for them again
// after the one fetch at login, which ran before any camera/session existed.
const VIEW_REFRESH = {
  overview: () => Promise.all([refreshStatus(), refreshEntities(), refreshEvents(), refreshAlerts()]).then(refreshLiveFrame),
  cameras: () => Promise.all([refreshCameras(), refreshStatus()]),
  zones: () => refreshZones(),
  rules: () => refreshRules(),
  events: () => refreshEvents(),
  alerts: () => refreshAlerts(),
  users: () => refreshUsers(),
  search: () => Promise.resolve(),
};

function switchView(id) {
  setState({ view: id });
  (VIEW_REFRESH[id] || (() => Promise.resolve()))().catch(() => {});
}

function startPolling() {
  stopPolling();
  pollHandle = setInterval(() => {
    if (!state.user || state.busy) return;
    refreshStatus().catch(() => {});
    (VIEW_REFRESH[state.view] || (() => Promise.resolve()))().catch(() => {});
  }, POLL_MS);
}
function stopPolling() { if (pollHandle) clearInterval(pollHandle); pollHandle = null; }

// --------------------------------------------------------------- actions
async function createCamera(form) {
  await guarded(async () => {
    await api.post("/cameras", {
      name: form.name, source_uri: form.source_uri,
      frame_width: form.frame_width ? Number(form.frame_width) : null,
      frame_height: form.frame_height ? Number(form.frame_height) : null,
    });
    await refreshCameras(); flash("Camera added.");
  });
}
async function deleteCamera(id) { await guarded(async () => { await api.del(`/cameras/${id}`); await refreshCameras(); }); }
async function startSession(camera_id) { await guarded(async () => { await api.post("/sessions/start", { camera_id }); flash("Session started."); await refreshStatus(); }); }
async function stopSession(session_id) { await guarded(async () => { await api.post(`/sessions/${session_id}/stop`, {}); flash("Session stopped."); await refreshStatus(); }); }

async function saveZone() {
  const d = state.drawZone;
  if (!d.name || d.points.length < 3) { flash("A zone needs a name and at least 3 points.", true); return; }
  await guarded(async () => {
    await api.post("/zones", { name: d.name, polygon: d.points, camera_id: d.camera_id || undefined });
    setState({ drawZone: { active: false, camera_id: "", name: "", points: [] } });
    await refreshZones(); flash("Zone saved.");
  });
}
async function deleteZone(id) { await guarded(async () => { await api.del(`/zones/${id}`); await refreshZones(); }); }

async function createRule() {
  await guarded(async () => {
    let def;
    try { def = JSON.parse(state.ruleDraft); } catch { throw new Error("Rule definition is not valid JSON."); }
    const name = def.name || `Rule ${new Date().toLocaleTimeString()}`;
    await api.post("/rules", { name, definition: def });
    await refreshRules(); flash("Rule saved.");
  });
}
async function deleteRule(id) { await guarded(async () => { await api.del(`/rules/${id}`); await refreshRules(); }); }

async function createUser(form) {
  await guarded(async () => {
    await api.post("/users", { username: form.username, password: form.password, role: form.role });
    await refreshUsers(); flash("User created.");
  });
}

async function runSearch(q) {
  await guarded(async () => {
    const result = await api.get("/search", { q });
    setState({ search: { q, result, error: null } });
  });
}

async function openEvent(eventId) {
  await guarded(async () => {
    const detail = await api.get(`/events/${eventId}`);
    if (state.selectedEventImageUrl) URL.revokeObjectURL(state.selectedEventImageUrl);
    let imageUrl = null;
    if (detail.evidence_record?.has_snapshot) {
      imageUrl = await api.getBlobUrl(`/events/${eventId}/snapshot`);
    }
    setState({ selectedEvent: detail, selectedEventImageUrl: imageUrl });
  });
}

// ---------------------------------------------------------------- rendering
function el(html) { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; }
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function fmtTime(ts) { return ts == null ? "-" : new Date(ts * 1000).toLocaleString(); }

// This app rebuilds the DOM from scratch on every render() rather than
// diffing. That's fine for data that only changes on user action, but a
// BACKGROUND poll (status/entities/etc, every POLL_MS) can call render()
// while someone is filling in a form - wiping out whatever they'd already
// typed into OTHER fields in that form, because the freshly-built inputs
// start from `state`, not from what's currently on screen. Found end-to-end:
// filling a camera's name then its source URI, with a poll landing in
// between, submitted an empty name even though it had clearly been typed.
// Preserving only the currently-focused field isn't enough - the field that
// already lost focus (name, in that example) needs saving too. Fixed for
// every input/textarea/select with an id currently in #app, not per-form,
// since any of them can be rebuilt out from under an in-progress edit.
function render() {
  const previousInputs = {};
  let focusedId = null;
  document.querySelectorAll("#app input[id], #app textarea[id], #app select[id]").forEach(elm => {
    previousInputs[elm.id] = elm.value;
    if (elm === document.activeElement) focusedId = { id: elm.id, selStart: elm.selectionStart, selEnd: elm.selectionEnd };
  });

  const app = document.getElementById("app");
  app.innerHTML = "";
  app.appendChild(state.user ? renderShell() : renderLogin());

  for (const [id, value] of Object.entries(previousInputs)) {
    const revived = document.getElementById(id);
    if (revived) revived.value = value;
  }
  if (focusedId) {
    const revived = document.getElementById(focusedId.id);
    if (revived) {
      revived.focus();
      if (typeof revived.setSelectionRange === "function" && focusedId.selStart != null) {
        try { revived.setSelectionRange(focusedId.selStart, focusedId.selEnd); } catch {}
      }
    }
  }
}

function renderLogin() {
  const root = el(`<div class="login-box">
    <h1>Vision Radar</h1>
    ${state.loginError ? `<div class="err-banner">${esc(state.loginError)}</div>` : ""}
    <input id="login-user" placeholder="Username" autocomplete="username" />
    <input id="login-pass" placeholder="Password" type="password" autocomplete="current-password" />
    <button class="btn" id="login-btn">Log in</button>
  </div>`);
  const doLogin = () => login(root.querySelector("#login-user").value, root.querySelector("#login-pass").value);
  root.querySelector("#login-btn").addEventListener("click", doLogin);
  root.querySelector("#login-pass").addEventListener("keydown", e => { if (e.key === "Enter") doLogin(); });
  return root;
}

const NAV = [
  ["overview", "Overview"], ["cameras", "Cameras"], ["zones", "Zones"],
  ["rules", "Rules"], ["events", "Event History"], ["search", "Search"],
  ["alerts", "Alerts"], ["users", "Users"],
];

function renderShell() {
  const root = el(`<div>
    <header>
      <h1>&#128249; Vision Radar</h1>
      <div class="who">${esc(state.user.username)} &middot; ${esc(state.user.role)} &middot; <a href="#" id="logout-link" style="color:var(--accent)">log out</a></div>
    </header>
    <nav></nav>
    ${state.flash ? `<div class="err-banner" style="margin:8px 16px 0;${state.flash.isError ? "" : "background:#0d2818;border-color:var(--ok);color:#7ee2a8"}">${esc(state.flash.msg)}</div>` : ""}
    <main id="main"></main>
  </div>`);
  root.querySelector("#logout-link").addEventListener("click", e => { e.preventDefault(); logout(); });

  const nav = root.querySelector("nav");
  for (const [id, label] of NAV) {
    if (id === "users" && state.user.role !== "administrator") continue;
    const b = el(`<button class="${state.view === id ? "active" : ""}">${label}</button>`);
    b.addEventListener("click", () => switchView(id));
    nav.appendChild(b);
  }

  const main = root.querySelector("#main");
  const renderers = {
    overview: renderOverview, cameras: renderCameras, zones: renderZones, rules: renderRules,
    events: renderEvents, search: renderSearch, alerts: renderAlerts, users: renderUsers,
  };
  main.appendChild((renderers[state.view] || renderOverview)());
  return root;
}

// ---- Overview: status, radar, entities, live event stream (Section 17) ----
function renderOverview() {
  const frag = el(`<div class="panel" style="grid-column:1/-1"></div>`);
  frag.style.display = "contents";

  const statusPanel = el(`<div class="panel"><h2>System Status</h2><div id="status-body"></div></div>`);
  const comps = state.status?.components || {};
  const order = ["camera", "vision", "tracking", "events", "database"];
  statusPanel.querySelector("#status-body").innerHTML =
    `<div class="status-row"><b>overall</b><span class="dot ${state.status?.overall || "UNKNOWN"}"></span>${state.status?.overall || "..."}</div>` +
    order.map(c => `<div class="status-row">${c}<span><span class="dot ${comps[c]?.status || "UNKNOWN"}"></span>${comps[c]?.message || comps[c]?.status || "UNKNOWN"}</span></div>`).join("") +
    (state.status ? `<p class="muted">fps ${state.status.performance.fps} &middot; ${state.status.performance.processing_ms_avg ?? "-"}ms/frame &middot; mem ${state.status.performance.memory_mb}MB &middot; active sessions: ${state.status.active_sessions.length}</p>` : "");

  const activeSession = (state.status?.active_sessions || [])[0] || null;
  const cameraPanel = el(`<div class="panel"><h2>Camera</h2>
    ${activeSession
      ? `<img id="live-camera-img" src="${state.liveFrameUrl || ""}" style="width:100%;border-radius:6px;background:#000;display:${state.liveFrameUrl ? "block" : "none"}"/>
         ${state.liveFrameUrl ? "" : '<p class="muted">Connecting...</p>'}`
      : '<p class="muted">No session running. Start one from the Cameras tab to see a live view.</p>'}
  </div>`);

  const radarPanel = el(`<div class="panel"><h2>Radar View</h2><div id="radar-holder"></div></div>`);
  radarPanel.querySelector("#radar-holder").appendChild(renderRadar());

  const entityPanel = el(`<div class="panel"><h2>Current Entities (${state.entities.length})</h2></div>`);
  entityPanel.appendChild(el(`<table><thead><tr><th>Type</th><th>State</th><th>Pos</th><th>Conf</th><th>Last seen</th></tr></thead>
    <tbody>${state.entities.slice(0, 15).map(e => `<tr><td>${esc(e.entity_type)}</td><td>${esc(e.last_state)}</td>
      <td>${Math.round(e.last_x)}, ${Math.round(e.last_y)}</td><td>${(e.confidence ?? 0).toFixed(2)}</td><td>${fmtTime(e.last_seen)}</td></tr>`).join("")}</tbody></table>`));

  const streamPanel = el(`<div class="panel"><h2>Event Stream</h2></div>`);
  streamPanel.appendChild(el(`<table><thead><tr><th>Time</th><th>Type</th><th>State</th><th>Conf</th></tr></thead>
    <tbody>${state.events.slice(0, 10).map(ev => `<tr><td>${fmtTime(ev.start_time)}</td><td>${esc(ev.event_type)}</td>
      <td><span class="pill ${ev.state}">${ev.state}</span></td><td>${ev.confidence.toFixed(2)}</td></tr>`).join("")}</tbody></table>`));

  const container = document.createElement("div");
  container.style.display = "contents";
  for (const p of [statusPanel, cameraPanel, radarPanel, entityPanel, streamPanel]) container.appendChild(p);
  return container;
}

// ---- Radar: entities plotted on an SVG, scaled from camera frame size (Section 16) ----
function renderRadar() {
  const frameW = Math.max(1, ...state.cameras.map(c => c.frame_width || 0), 640);
  const frameH = Math.max(1, ...state.cameras.map(c => c.frame_height || 0), 480);
  const vb = `0 0 ${frameW} ${frameH}`;
  const dots = state.entities.map(e => `<circle class="entity-dot" cx="${e.last_x}" cy="${e.last_y}" r="${Math.max(3, frameW / 120)}"/>
     <text class="entity-label" x="${e.last_x + 6}" y="${e.last_y - 6}">${esc(e.entity_type)}</text>`).join("");
  const zonePolys = state.zones.map(z => `<polygon class="zone-poly" points="${z.polygon.map(p => p.join(",")).join(" ")}"/>`).join("");
  return el(`<svg id="radar" viewBox="${vb}" preserveAspectRatio="xMidYMid meet">${zonePolys}${dots}</svg>`);
}

// ---- Cameras ----
function renderCameras() {
  const canControl = state.user.role !== "viewer";
  const root = el(`<div class="panel" style="grid-column:1/-1">
    <h2>Cameras</h2>
    <table><thead><tr><th>Name</th><th>Source</th><th>Size</th><th>Active session</th><th></th></tr></thead><tbody></tbody></table>
    ${canControl ? `<div class="row" style="margin-top:12px">
      <input id="cam-name" placeholder="Name"/><input id="cam-uri" placeholder="Source URI (device index, file, or rtsp://...)" style="width:260px"/>
      <input id="cam-w" placeholder="Width" style="width:70px"/><input id="cam-h" placeholder="Height" style="width:70px"/>
      <button class="btn" id="cam-add">Add camera</button></div>` : ""}
  </div>`);
  const activeSessions = new Set((state.status?.active_sessions) || []);
  const tbody = root.querySelector("tbody");
  for (const c of state.cameras) {
    const row = el(`<tr><td>${esc(c.name)}</td><td>${esc(c.source_uri)}</td><td>${c.frame_width || "-"}x${c.frame_height || "-"}</td>
      <td id="sess-${c.camera_id}"></td><td></td></tr>`);
    const sessCell = row.querySelector(`#sess-${c.camera_id}`);
    const myRunning = [...activeSessions].length > 0;   // status doesn't map session->camera; show generic control
    if (state.user.role !== "viewer") {
      const startBtn = el(`<button class="btn secondary">Start session</button>`);
      startBtn.addEventListener("click", () => startSession(c.camera_id));
      sessCell.appendChild(startBtn);
    } else sessCell.textContent = "-";
    if (canControl) {
      const del = el(`<button class="btn secondary">Delete</button>`);
      del.addEventListener("click", () => deleteCamera(c.camera_id));
      row.lastElementChild.appendChild(del);
    }
    tbody.appendChild(row);
  }
  if (canControl) {
    root.querySelector("#cam-add").addEventListener("click", () => createCamera({
      name: root.querySelector("#cam-name").value, source_uri: root.querySelector("#cam-uri").value,
      frame_width: root.querySelector("#cam-w").value, frame_height: root.querySelector("#cam-h").value,
    }));
  }
  if (state.user.role !== "viewer") {
    const sessPanel = el(`<div class="panel" style="grid-column:1/-1"><h2>Active Sessions</h2>
      ${(state.status?.active_sessions || []).map(sid => `<div class="row">${sid}</div>`).join("") || '<p class="muted">None running.</p>'}
    </div>`);
    const container = document.createElement("div"); container.style.display = "contents";
    container.appendChild(root); container.appendChild(sessPanel);
    for (const sid of (state.status?.active_sessions || [])) {
      const row = [...sessPanel.querySelectorAll(".row")].find(r => r.textContent.trim() === sid);
      const stopBtn = el(`<button class="btn secondary" style="margin-left:8px">Stop</button>`);
      stopBtn.addEventListener("click", () => stopSession(sid));
      row.appendChild(stopBtn);
    }
    return container;
  }
  return root;
}

// ---- Zones: list + click-to-draw polygon editor (Section 8) ----
function renderZones() {
  const canEdit = state.user.role === "administrator";
  const root = el(`<div class="panel" style="grid-column:1/-1">
    <h2>Zones</h2>
    <table><thead><tr><th>Name</th><th>Points</th><th></th></tr></thead><tbody>
      ${state.zones.map(z => `<tr><td>${esc(z.name)}</td><td>${z.polygon.length}</td><td></td></tr>`).join("")}
    </tbody></table>
  </div>`);
  [...root.querySelectorAll("tbody tr")].forEach((row, i) => {
    if (!canEdit) return;
    const del = el(`<button class="btn secondary">Delete</button>`);
    del.addEventListener("click", () => deleteZone(state.zones[i].zone_id));
    row.lastElementChild.appendChild(del);
  });

  if (!canEdit) return root;

  // Match the editor's coordinate space to the selected camera's real frame
  // size (falling back to a sane default only when no camera is picked yet)
  // - otherwise a zone drawn by eye lines up with the live view but is
  // silently wrong in the stored polygon's own coordinate space.
  const selectedCam = state.cameras.find(c => c.camera_id === state.drawZone.camera_id);
  const editW = selectedCam?.frame_width || 640;
  const editH = selectedCam?.frame_height || 480;

  const editor = el(`<div class="panel" style="grid-column:1/-1">
    <h2>Zone Editor</h2>
    <p class="muted">Click on the canvas to place corner points (3+), then save. Click "Clear" to start over.</p>
    <div class="row">
      <input id="zone-name" placeholder="Zone name" value="${esc(state.drawZone.name)}"/>
      <select id="zone-camera"><option value="">(no camera - ${editW}x${editH} default)</option>
        ${state.cameras.map(c => `<option value="${c.camera_id}" ${c.camera_id === state.drawZone.camera_id ? "selected" : ""}>${esc(c.name)} (${c.frame_width || "?"}x${c.frame_height || "?"})</option>`).join("")}
      </select>
      <button class="btn secondary" id="zone-clear">Clear</button>
      <button class="btn" id="zone-save">Save zone</button>
    </div>
    <svg id="zone-canvas" viewBox="0 0 ${editW} ${editH}" style="background:#0a0e14;border:1px solid var(--border);border-radius:6px;width:100%;height:320px;cursor:crosshair">
      ${state.drawZone.points.length >= 2 ? `<polygon class="zone-poly drawing" points="${state.drawZone.points.map(p => p.join(",")).join(" ")}"/>` : ""}
      ${state.drawZone.points.map(([x, y]) => `<circle cx="${x}" cy="${y}" r="5" fill="#58a6ff"/>`).join("")}
    </svg>
    <p class="muted" id="zone-point-count">${state.drawZone.points.length} point(s) placed.</p>
  </div>`);

  const svg = editor.querySelector("#zone-canvas");
  svg.addEventListener("click", (e) => {
    const rect = svg.getBoundingClientRect();
    const x = Math.round((e.clientX - rect.left) / rect.width * editW);
    const y = Math.round((e.clientY - rect.top) / rect.height * editH);
    state.drawZone.points.push([x, y]);
    render();
  });
  editor.querySelector("#zone-clear").addEventListener("click", () => { state.drawZone.points = []; render(); });
  editor.querySelector("#zone-name").addEventListener("input", (e) => { state.drawZone.name = e.target.value; });
  editor.querySelector("#zone-camera").addEventListener("change", (e) => { state.drawZone.camera_id = e.target.value; render(); });
  editor.querySelector("#zone-save").addEventListener("click", saveZone);

  const container = document.createElement("div"); container.style.display = "contents";
  container.appendChild(root); container.appendChild(editor);
  return container;
}

// ---- Rules (WHEN/IF/THEN, Section 22) ----
function renderRules() {
  const canEdit = state.user.role === "administrator";
  const root = el(`<div class="panel" style="grid-column:1/-1">
    <h2>Rules</h2>
    <table><thead><tr><th>Name</th><th>When</th><th>If</th><th></th></tr></thead><tbody>
      ${state.rules.map(r => `<tr><td>${esc(r.name)}</td><td>${esc(r.definition.when.event_type)}${r.definition.when.zone_id ? " @ zone" : ""}</td>
        <td>${r.definition.if.map(c => `${c.field} ${c.operator} ${c.value}`).join(", ") || "(always)"}</td><td></td></tr>`).join("")}
    </tbody></table>
    ${canEdit ? `<h2 style="margin-top:16px">New rule (JSON)</h2>
      <textarea id="rule-json" rows="6" style="width:100%;font-family:monospace;font-size:12.5px;background:#0d1117;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:8px">${esc(state.ruleDraft)}</textarea>
      <div class="row" style="margin-top:8px"><button class="btn" id="rule-save">Create rule</button></div>` : ""}
  </div>`);
  if (canEdit) {
    [...root.querySelectorAll("tbody tr")].forEach((row, i) => {
      const del = el(`<button class="btn secondary">Delete</button>`);
      del.addEventListener("click", () => deleteRule(state.rules[i].rule_id));
      row.lastElementChild.appendChild(del);
    });
    root.querySelector("#rule-json").addEventListener("input", (e) => { state.ruleDraft = e.target.value; });
    root.querySelector("#rule-save").addEventListener("click", createRule);
  }
  return root;
}

// ---- Event history + filters (Section 18) ----
function renderEvents() {
  const f = state.eventFilters;
  const root = el(`<div class="panel" style="grid-column:1/-1">
    <h2>Event History</h2>
    <div class="row">
      <select id="f-type"><option value="">Any type</option>
        ${["APPEARED", "DISAPPEARED", "MOVED", "ZONE_ENTRY", "DWELL", "STATE_CHANGED"].map(t => `<option value="${t}" ${f.event_type === t ? "selected" : ""}>${t}</option>`).join("")}
      </select>
      <select id="f-zone"><option value="">Any zone</option>
        ${state.zones.map(z => `<option value="${z.zone_id}" ${f.zone_id === z.zone_id ? "selected" : ""}>${esc(z.name)}</option>`).join("")}
      </select>
      <select id="f-state"><option value="">Any state</option>
        ${["ACTIVE", "RESOLVED", "ARCHIVED"].map(s => `<option value="${s}" ${f.state === s ? "selected" : ""}>${s}</option>`).join("")}
      </select>
      <input id="f-conf" placeholder="Min confidence" value="${esc(f.min_confidence)}" style="width:110px"/>
      <button class="btn secondary" id="f-apply">Apply filters</button>
    </div>
    <table><thead><tr><th>Time</th><th>Type</th><th>Entity</th><th>Zone</th><th>Conf</th><th>State</th><th>Dur</th><th></th></tr></thead>
    <tbody>${state.events.map(ev => `<tr><td>${fmtTime(ev.start_time)}</td><td>${esc(ev.event_type)}</td><td>${esc(ev.entity_type)}</td>
      <td>${ev.zone_id ? esc((state.zones.find(z => z.zone_id === ev.zone_id) || {}).name || ev.zone_id.slice(0,8)) : "-"}</td>
      <td>${ev.confidence.toFixed(2)}</td><td><span class="pill ${ev.state}">${ev.state}</span></td>
      <td>${ev.duration_seconds != null ? ev.duration_seconds.toFixed(1) + "s" : "-"}</td><td></td></tr>`).join("")}</tbody></table>
    ${state.selectedEvent ? renderEventDetailHtml(state.selectedEvent, state.selectedEventImageUrl) : ""}
  </div>`);
  root.querySelector("#f-apply").addEventListener("click", () => {
    setState({ eventFilters: {
      event_type: root.querySelector("#f-type").value, zone_id: root.querySelector("#f-zone").value,
      state: root.querySelector("#f-state").value, min_confidence: root.querySelector("#f-conf").value,
    }});
    refreshEvents();
  });
  [...root.querySelectorAll("tbody tr")].forEach((row, i) => {
    row.style.cursor = "pointer";
    row.addEventListener("click", () => openEvent(state.events[i].event_id));
  });
  return root;
}

function renderEventDetailHtml(ev, imageUrl) {
  return `<div class="panel" style="margin-top:12px">
    <h2>Event ${esc(ev.event_id.slice(0, 8))}</h2>
    <p>${esc(ev.event_type)} &middot; ${esc(ev.entity_type)} &middot; confidence ${ev.confidence.toFixed(2)} &middot; state ${esc(ev.state)}</p>
    ${imageUrl ? `<img id="evidence-img" src="${imageUrl}" style="display:block"/>` : '<p class="muted">No image evidence for this event.</p>'}
  </div>`;
}

// ---- Natural-language search (Section 19) ----
function renderSearch() {
  const root = el(`<div class="panel" style="grid-column:1/-1">
    <h2>Search</h2>
    <div class="row"><input id="search-q" placeholder='e.g. "what happened in Zone A today"' style="width:420px" value="${esc(state.search.q)}"/>
      <button class="btn" id="search-go">Ask</button></div>
    <div id="search-results"></div>
  </div>`);
  const run = () => runSearch(root.querySelector("#search-q").value);
  root.querySelector("#search-go").addEventListener("click", run);
  root.querySelector("#search-q").addEventListener("keydown", e => { if (e.key === "Enter") run(); });

  const out = root.querySelector("#search-results");
  if (state.search.result) {
    const r = state.search.result;
    out.innerHTML = `<p class="muted">Understood as: ${esc(JSON.stringify(r.interpreted_as))}</p>
      ${r.ignored.length ? `<p class="ignored">Not understood, ignored: "${esc(r.ignored.join(" "))}" - answer is based only on the rest.</p>` : ""}
      <p>${r.results.length} result(s), answered strictly from stored events:</p>
      ${r.results.map(ev => `<div class="search-result">${fmtTime(ev.start_time)} &middot; ${esc(ev.event_type)} &middot; ${esc(ev.entity_type)} &middot; conf ${ev.confidence.toFixed(2)} &middot; <span class="pill ${ev.state}">${ev.state}</span></div>`).join("")}`;
  }
  return root;
}

// ---- Alerts (Section 21) ----
function renderAlerts() {
  const root = el(`<div class="panel" style="grid-column:1/-1">
    <h2>Alerts</h2>
    <table><thead><tr><th>Time</th><th>Rule</th><th>Message</th><th>Conf</th><th></th></tr></thead>
    <tbody>${state.alerts.map(a => `<tr><td>${fmtTime(a.created_at)}</td><td>${esc(a.rule_name)}</td><td>${esc(a.message)}</td>
      <td>${a.confidence?.toFixed(2) ?? "-"}</td><td></td></tr>`).join("")}</tbody></table>
  </div>`);
  [...root.querySelectorAll("tbody tr")].forEach((row, i) => {
    const view = el(`<button class="btn secondary">View event</button>`);
    view.addEventListener("click", () => { switchView("events"); openEvent(state.alerts[i].event_id); });
    row.lastElementChild.appendChild(view);
  });
  return root;
}

// ---- Users (administrator only, Section 28) ----
function renderUsers() {
  const root = el(`<div class="panel" style="grid-column:1/-1">
    <h2>Users</h2>
    <table><thead><tr><th>Username</th><th>Role</th></tr></thead>
    <tbody>${state.users.map(u => `<tr><td>${esc(u.username)}</td><td>${esc(u.role)}</td></tr>`).join("")}</tbody></table>
    <h2 style="margin-top:16px">New user</h2>
    <div class="row">
      <input id="u-name" placeholder="Username"/><input id="u-pass" placeholder="Password" type="password"/>
      <select id="u-role"><option value="viewer">viewer</option><option value="operator">operator</option><option value="administrator">administrator</option></select>
      <button class="btn" id="u-add">Create</button>
    </div>
  </div>`);
  root.querySelector("#u-add").addEventListener("click", () => createUser({
    username: root.querySelector("#u-name").value, password: root.querySelector("#u-pass").value,
    role: root.querySelector("#u-role").value,
  }));
  return root;
}

render();
