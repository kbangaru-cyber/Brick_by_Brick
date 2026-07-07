/**
 * BrickAgent Web UI — app.js
 *
 * Two-phase flow:
 *   Phase 1: POST /plan  → SSE → planner CoT + analysis + plan → [AWAITING_APPROVAL]
 *   Phase 2: POST /execute → SSE → executor physics + PLACE commands → [DONE]
 *
 * Endpoints:
 *   http://localhost:8081  — Rhino bridge
 *   http://localhost:8080  — PSC server
 */

const RHINO_URL  = "http://localhost:8081";
const SERVER_URL = "http://localhost:8080";
const DEBUG_STREAM = new URLSearchParams(window.location.search).get("debug") === "1";

const BRICK_RE    = /^\s*(?:PLACE[\s:]*)?(\d+\s*x\s*\d+)[\s@:]*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)/i;
const STAGE5_FIELD_RE = /^\s*(ROLE|REASON|PHYSICS|STATUS)\s*:\s*(.*)$/i;
const LAYER_RE    = /---\s*Layer\s+(\d+)/i;
const PHYSICS_RE  = /^\[PHYSICS\]/i;
const PART_NAME_RE = /name="([^"]+)"/;
const SUMMARY_RE  = /<summary>(.*?)<\/summary>/i;
const CHAT_OPEN_RE = /^<chat\b[^>]*\bbuild="(true|false)"/i;
const BUILD_PHRASE_RE = /\b(build\s+it|build\s+now|go\s+ahead|proceed|make\s+it|start\s+building|start\s+the\s+build|let'?s\s+build|ok\s+build|go\s+build)\b/i;
// Intentionally strict: these phrases mean "end negotiation, review the
// build". Must NOT match routine affirmations like "looks good", "we're
// good", "good enough" — those are part of back-and-forth refinement.
const DONE_PHRASE_RE  = /\b(that'?s\s+enough|i'?m\s+done|we'?re\s+done|that'?s\s+it|stop\s+building|that'?s\s+all|i\s+am\s+done|end\s+(the\s+)?build|finalize\s+build|review\s+(the\s+)?build)\b/i;

function normalizeStreamText(text) {
  return String(text)
    .replace(/[\u200B-\u200D\uFEFF]/g, "")
    .replace(/\u00A0/g, " ")
    .replace(/\u2212/g, "-")
    .trim();
}

function parseBrickLine(text) {
  const cleaned = normalizeStreamText(text);

  const m = BRICK_RE.exec(cleaned);
  if (!m) return null;

  return {
    text: cleaned,
    dims: m[1].replace(/\s+/g, "").toLowerCase(),
    x: +m[2],
    y: +m[3],
    z: +m[4],
  };
}

// ─── state ───────────────────────────────────────────────────────
const state = {
  phase:       "idle",   // idle | chatting | building | awaiting_accept
  abortCtrl:   null,
  brickCount:  0,
  parts:       [],
  partIdx:     -1,
  history:     [],       // legacy history (still used by /plan fallback)
  chatHistory: [],       // [{role:"user"|"assistant", content}] — sent to /chat
  assistantBuf: [],      // in-progress assistant reply text during stream
  chatBuildTurn: false,  // true if current /chat response is a build turn
  chatReviewTurn: false, // true if current /chat response is a critical review
  lastSummary: "",
  lastPrompt:  "",
  buildPrompt: "",  // synthesized abstract prompt from planner → executor
  planText:    "",
  rhinoBricks: [],
  lastViews:   {},
  streamPhase: "idle",   // idle | chat | thinking | plan | build | review | exec_prompt
  mode:        "stage3", // "builder" | "stage3" | "stage5"
  fastEnabled: false,    // legacy; unused in unified /chat path
  waitingForUser: false,
  currentWorkArea: null,
  serverCaps:   { fast: false },
  userImages:   [],   // [{name, b64}] — user-uploaded reference images
  // Per-brick reasoning capture. The executor stream emits PLACE lines
  // followed by physics / "Supported by" / warning text that explains why
  // *that* brick was placed. We open a "slot" when a PLACE arrives, push
  // subsequent text lines into it, and post /set_reason once placeBrick
  // resolves with a guid.
  currentReasonSlot: null,  // {lines:[], brick:{dims,x,y,z,stability}} — brick is deferred until reasoning lands
  pickedSeq:   0,
  pickedGuid:  null,
  execPartHasOutput: false,  // true once a brick/layer has been emitted in the
                             // current <part> — prose before that is the
                             // executor's reasoning/plan and we show it.
  execThinkHeaderShown: false, // "── Executor reasoning (CoT) ──" printed?
  execPlanHeaderShown:  false, // "── Executor plan ──" printed?
  _buildingStatusEl:    null,  // DOM span of current breathing status line
  reviewHeaderShown:    false, // "── Review ──" printed for this stream?
};

// ─── DOM refs ────────────────────────────────────────────────────
const chatEl      = document.getElementById("chat");
const cotEl       = document.getElementById("cot");
const buildEl     = document.getElementById("build");
const brickBody   = document.getElementById("brick-body");
const brickStream = document.getElementById("brick-stream");
const brickEmpty  = document.getElementById("brick-empty");
const input        = document.getElementById("prompt-input");
const btnBuild     = document.getElementById("btn-build");
const btnStop      = document.getElementById("btn-stop");
const btnClear     = document.getElementById("btn-clear");
const btnApprove   = document.getElementById("btn-approve");
const btnDirect    = document.getElementById("btn-direct");
const dotRhino     = document.getElementById("dot-rhino");
const dotServer    = document.getElementById("dot-server");
const lblRhino     = document.getElementById("lbl-rhino");
const lblServer    = document.getElementById("lbl-server");
const turnCount    = document.getElementById("turn-count");
const progress     = document.getElementById("progress");
const partDots     = document.getElementById("part-dots");
const progressTxt  = document.getElementById("progress-text");
const brickCount   = document.getElementById("brick-count");
const buildScrubber      = document.getElementById("build-scrubber");
const buildScrubberRange = document.getElementById("build-scrubber-range");
const buildScrubberLabel = document.getElementById("build-scrubber-label");
const btnModeBuilder = document.getElementById("btn-mode-builder");
const btnModeStage3  = document.getElementById("btn-mode-stage3");
const btnModeStage5  = document.getElementById("btn-mode-stage5");
const btnModeFast = document.getElementById("btn-mode-fast");
const waitingBanner  = document.getElementById("waiting-banner");
const imgUpload    = document.getElementById("img-upload");
const btnImg       = document.getElementById("btn-img");
const imgPreview   = document.getElementById("img-preview");
const brickDetail  = document.getElementById("brick-detail-content");
const bdTitle      = document.getElementById("bd-title");
const bdMeta       = document.getElementById("bd-meta");
const bdReason     = document.getElementById("bd-reason");
const bdSupporters = document.getElementById("bd-supporters");
const bdClose      = document.getElementById("bd-close");

// ─── logging ─────────────────────────────────────────────────────
function _appendTo(target, text, cls) {
  const span = document.createElement("span");
  span.className = cls;
  span.textContent = text + "\n";
  target.appendChild(span);
  target.scrollTop = target.scrollHeight;
  return span;
}

function appendChat(text, cls = "normal")  { return _appendTo(chatEl,  text, cls); }
function appendCot(text, cls = "normal")   { return _appendTo(cotEl,   text, cls); }
function appendBuild(text, cls = "normal") { return _appendTo(buildEl, text, cls); }

// Sub-field labels that appear INSIDE the LAYER STRATEGY section. These
// must not be promoted to a top-level section header — they render as a
// small non-bold tag above their sub-bullet content so the block keeps
// its "REASON" / "PHYSICS" / "STATUS" / "ROLE" structure but doesn't
// visually compete with the z=N layer line above it.
const _COT_SUBLABELS = new Set(["REASON", "PHYSICS", "ROLE", "STATUS"]);

// Format an executor CoT/plan line for the yellow panel: uppercase section
// labels (INTERPRETATION:, LAYER STRATEGY:, etc.) render as bold headers and
// every other line becomes one bullet per sentence so long paragraphs are
// readable. Layer rows ("z=0 …") are rendered with a bold "z=N" prefix.
function _emitCotLine(text, _bodyCls) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return;

  // "SECTION LABEL:" optionally followed by inline content on the same line.
  // Requires at least 3 uppercase chars and a colon so single words like "A:"
  // or "OK:" don't hijack prose.
  const labelMatch = /^([A-Z][A-Z0-9 _/()\-]{2,}):\s*(.*)$/.exec(trimmed);
  if (labelMatch) {
    const label = labelMatch[1].trim();
    const rest  = labelMatch[2].trim();
    if (_COT_SUBLABELS.has(label)) {
      // Sub-field like "REASON:" — drop the label and render the content
      // as smaller sub-bullets tucked under the preceding layer row.
      if (rest) _emitCotBullets(rest, /* sub */ true);
      return;
    }
    appendCot("  " + label + ":", "exec-section");
    if (rest) _emitCotBullets(rest);
    return;
  }

  // Already a numbered / dashed item — keep as a single bullet, with
  // special handling for layer rows so "z=N" stands out.
  const bulletMatch = /^(?:\d+\.|[-•])\s+(.*)$/.exec(trimmed);
  if (bulletMatch) {
    _appendCotBullet(bulletMatch[1].trim());
    return;
  }

  _emitCotBullets(trimmed);
}

// Render one bullet. If the bullet starts with a "z=N" layer marker
// (optionally followed by a parenthesised component list), bold that prefix
// so the layer name stands out in the LAYER STRATEGY block. `sub=true`
// emits the bullet in the smaller sub-bullet style.
function _appendCotBullet(text, sub = false) {
  const cls = sub ? "exec-bullet exec-bullet-sub" : "exec-bullet";
  const layerMatch = !sub && /^(z=\d+(?:\s*\([^)]*\))?)(\s*:?\s*)(.*)$/.exec(text);
  if (layerMatch) {
    const span = document.createElement("span");
    span.className = cls;
    span.appendChild(document.createTextNode("    • "));
    const bold = document.createElement("b");
    bold.textContent = layerMatch[1];
    span.appendChild(bold);
    span.appendChild(document.createTextNode(layerMatch[2] + layerMatch[3] + "\n"));
    cotEl.appendChild(span);
    cotEl.scrollTop = cotEl.scrollHeight;
    return;
  }
  appendCot((sub ? "        ◦ " : "    • ") + text, cls);
}

// Split prose into sentences and render each as its own bullet. Sentence
// boundary = ".?!" followed by whitespace and an uppercase / quote start.
// That avoids fragmenting abbreviations ("i.e."), decimals, or coordinates.
function _emitCotBullets(text, sub = false) {
  const sentences = String(text)
    .split(/(?<=[.!?])\s+(?=["'“‘(A-Z])/)
    .map(s => s.trim())
    .filter(Boolean);
  if (sentences.length === 0) return;
  for (const s of sentences) {
    _appendCotBullet(s, sub);
  }
}

// Emit a horizontal rule into the CoT panel — used to separate retry
// attempts so the designer can see where a new attempt begins.
function _appendCotSeparator() {
  const hr = document.createElement("hr");
  hr.className = "cot-separator";
  cotEl.appendChild(hr);
  cotEl.scrollTop = cotEl.scrollHeight;
}

function syncBrickPanelState() {
  if (!brickEmpty || !brickDetail) return;
  const hasLive = !!(brickStream && brickStream.textContent.trim());
  const hasDetail = brickDetail.classList.contains("visible");
  brickEmpty.style.display = hasLive || hasDetail ? "none" : "block";
}

function clearBrickStream() {
  if (brickStream) brickStream.innerHTML = "";
  syncBrickPanelState();
}

// ─── build log scrubber ───────────────────────────────────────────
// Tracks every PLACE line that gets appended to the blue build panel so the
// user can drag the slider below the panel title to jump to any step of the
// construction sequence.
const _buildTimeline = [];

function _resetBuildTimeline() {
  _buildTimeline.length = 0;
  if (buildScrubber) buildScrubber.hidden = true;
  if (buildScrubberRange) {
    buildScrubberRange.max = 0;
    buildScrubberRange.value = 0;
  }
  if (buildScrubberLabel) buildScrubberLabel.textContent = "Step 0 / 0";
}

function _recordPlaceStep(el, brick) {
  if (!el) return;
  const step = _buildTimeline.length + 1;
  el.dataset.step = String(step);
  _buildTimeline.push({ el, step, brick });
  if (buildScrubber) buildScrubber.hidden = false;
  if (buildScrubberRange) {
    buildScrubberRange.max = String(_buildTimeline.length);
    // Auto-follow: while the build is streaming, keep the slider pinned to
    // the latest step so the label reflects progress.
    buildScrubberRange.value = String(_buildTimeline.length);
  }
  if (buildScrubberLabel) {
    buildScrubberLabel.textContent = `Step ${_buildTimeline.length} / ${_buildTimeline.length}`;
  }
}

function _scrubBuildTo(n) {
  if (!_buildTimeline.length) return;
  const idx = Math.max(1, Math.min(_buildTimeline.length, Math.round(Number(n) || 0)));
  for (const entry of _buildTimeline) entry.el.classList.remove("scrubbed-focus");
  const target = _buildTimeline[idx - 1];
  if (!target) return;
  target.el.classList.add("scrubbed-focus");
  target.el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  if (buildScrubberLabel) {
    buildScrubberLabel.textContent = `Step ${idx} / ${_buildTimeline.length}`;
  }
}

if (buildScrubberRange) {
  buildScrubberRange.addEventListener("input", (e) => {
    _scrubBuildTo(e.target.value);
  });
}

function brickStatusClass(status) {
  const value = String(status || "stable").trim().toLowerCase();
  if (/\bweak\b/.test(value)) return "brick-status-weak";
  if (/\bunsupported\b/.test(value)) return "brick-status-unsupported";
  return "brick-status-stable";
}

function normalizePlaceText(brick, rawText) {
  const parsed = rawText ? parseBrickLine(rawText) : null;
  if (parsed) return `PLACE ${parsed.dims} (${parsed.x},${parsed.y},${parsed.z})`;
  return `PLACE ${brick.dims} (${brick.x},${brick.y},${brick.z})`;
}

function classifyBrickReasonLine(text, fallbackStability = "stable") {
  const cleaned = normalizeStreamText(text);
  if (!cleaned) return null;

  const fieldMatch = STAGE5_FIELD_RE.exec(cleaned);
  if (fieldMatch) {
    const label = fieldMatch[1].toUpperCase();
    const value = fieldMatch[2].trim();
    if (label === "ROLE") {
      return { label, text: `ROLE: ${value}`, cls: "brick-role" };
    }
    if (label === "PHYSICS") {
      return { label, text: `PHYSICS: ${value}`, cls: "brick-physics" };
    }
    if (label === "STATUS") {
      const statusText = value || String(fallbackStability || "stable").toUpperCase();
      return {
        label,
        text: `STATUS: ${statusText}`,
        cls: brickStatusClass(statusText),
      };
    }
    return { label, text: `REASON: ${value}`, cls: "brick-reason" };
  }

  if (
    PHYSICS_RE.test(cleaned) ||
    /^supported by:/i.test(cleaned) ||
    /\bstability threshold\b/i.test(cleaned) ||
    /\bCoM\b/i.test(cleaned)
  ) {
    const physicsText = PHYSICS_RE.test(cleaned)
      ? cleaned.replace(PHYSICS_RE, "PHYSICS:").trim()
      : cleaned;
    return { label: "PHYSICS", text: physicsText, cls: "brick-physics" };
  }

  return { label: null, text: cleaned, cls: "brick-generic" };
}

// The green panel is click-driven only — it must display reasoning for the
// brick the designer selects in Rhino, not a streaming log of every brick.
// The per-brick reason is still captured on the slot and sent to the server
// via placeBrick(payload), so it remains available when the user clicks.
function renderBrickReasonBlock(_slot) {
  // Intentional no-op. See renderBrickDetail() for the click-driven view.
}

// Legacy alias — defaults to the build panel. Keep so any un-migrated
// call site still renders somewhere instead of throwing.
function appendLog(text, cls = "normal") { appendBuild(text, cls); }

function appendBlank(target = buildEl) {
  target.appendChild(document.createTextNode("\n"));
}

function appendDebug(text) {
  if (!DEBUG_STREAM) return;
  appendBuild(`[DEBUG] ${text}`, "dim");
}

// The breathing "I have started to build …" banner was removed in favour of
// streaming the actual build log (PLACE + ROLE/REASON/PHYSICS/STATUS) into
// the blue panel. Keep the function names so every existing call site still
// resolves, but drop the side effects.
function showBuildingStatus() {}
function stopBuildingStatus() { state._buildingStatusEl = null; }

function appendChatMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "chat-msg";
  const col = document.createElement("div");
  const label = document.createElement("div");
  label.className = "chat-label";
  label.textContent = "You";
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.textContent = text;
  col.appendChild(label);
  col.appendChild(bubble);
  wrapper.appendChild(col);
  chatEl.appendChild(wrapper);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function appendViewports(views) {
  const entries = Object.entries(views);
  if (!entries.length) return;

  // Separate viewport captures from user reference images
  const vpEntries = entries.filter(([n]) => !n.startsWith("ref_"));
  const refEntries = entries.filter(([n]) => n.startsWith("ref_"));

  // Show viewport captures
  if (vpEntries.length) {
    appendChat("  [4-view capture — what the agent sees]", "sys");
    const ORDER = ["Top","Front","Right","Perspective"];
    const sorted = ORDER
      .filter(n => views[n])
      .map(n => [n, views[n]])
      .concat(vpEntries.filter(([n]) => !ORDER.includes(n)));

    const grid = document.createElement("div");
    grid.className = "viewport-grid";
    for (const [name, b64] of sorted) {
      const cell = document.createElement("div");
      cell.className = "vp-cell";
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${b64}`;
      img.className = "vp-img";
      img.title = name;
      const lbl = document.createElement("div");
      lbl.className = "vp-lbl";
      lbl.textContent = name;
      cell.appendChild(img);
      cell.appendChild(lbl);
      grid.appendChild(cell);
    }
    chatEl.appendChild(grid);
  }

  // Show user reference images
  if (refEntries.length) {
    appendChat(`  [${refEntries.length} reference image(s) attached]`, "sys");
    const grid = document.createElement("div");
    grid.className = "viewport-grid";
    for (const [name, b64] of refEntries) {
      const cell = document.createElement("div");
      cell.className = "vp-cell";
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${b64}`;
      img.className = "vp-img";
      img.title = name;
      const lbl = document.createElement("div");
      lbl.className = "vp-lbl";
      lbl.textContent = name.replace(/^ref_/, "");
      cell.appendChild(img);
      cell.appendChild(lbl);
      grid.appendChild(cell);
    }
    chatEl.appendChild(grid);
  }

  appendBlank(chatEl);
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ─── image upload ────────────────────────────────────────────────
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // strip "data:image/...;base64," prefix
      const b64 = reader.result.split(",")[1];
      resolve(b64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function renderImagePreviews() {
  imgPreview.innerHTML = "";
  if (!state.userImages.length) {
    imgPreview.classList.remove("has-images");
    return;
  }
  imgPreview.classList.add("has-images");
  state.userImages.forEach((img, idx) => {
    const wrap = document.createElement("div");
    wrap.className = "img-thumb-wrap";
    const thumb = document.createElement("img");
    thumb.className = "img-thumb";
    thumb.src = `data:image/png;base64,${img.b64}`;
    thumb.title = img.name;
    const rm = document.createElement("button");
    rm.className = "img-thumb-rm";
    rm.textContent = "\u00d7";
    rm.onclick = () => { state.userImages.splice(idx, 1); renderImagePreviews(); };
    wrap.appendChild(thumb);
    wrap.appendChild(rm);
    imgPreview.appendChild(wrap);
  });
}

async function onImageUpload(e) {
  const files = Array.from(e.target.files || []);
  for (const f of files) {
    const b64 = await fileToBase64(f);
    state.userImages.push({ name: f.name, b64 });
  }
  imgUpload.value = "";
  renderImagePreviews();
}

if (btnImg) btnImg.addEventListener("click", () => imgUpload.click());
if (imgUpload) imgUpload.addEventListener("change", onImageUpload);

function modeLabel(mode) {
  switch (mode) {
    case "builder": return "FAST";
    default: return "INSPECT";
  }
}

function syncFastToggle() {
  if (btnModeFast) {
    btnModeFast.classList.toggle("active", !!state.fastEnabled);
    btnModeFast.disabled = !state.serverCaps.fast || state.phase !== "idle";
  }
}

function setFastEnabled(enabled, quiet = false) {
  if (enabled && !state.serverCaps.fast) {
    appendLog("  Fast mode is not available on this server.", "warn");
    return;
  }

  state.fastEnabled = !!enabled;
  syncFastToggle();

  if (!quiet) {
    appendLog(`  Fast: ${state.fastEnabled ? "ON" : "OFF"}`, "sys");
  }
}

function toggleFastEnabled() {
  setFastEnabled(!state.fastEnabled);
}

function resetCurrentWorkArea() {
  state.currentWorkArea = null;
}

function expandCurrentWorkArea(brick) {
  const [h, w] = brick.dims.split("x").map(Number);
  const bx = {
    minX: brick.x,
    minY: brick.y,
    minZ: brick.z,
    maxX: brick.x + h,
    maxY: brick.y + w,
    maxZ: brick.z + 1,
  };

  if (!state.currentWorkArea) {
    state.currentWorkArea = bx;
    return;
  }

  state.currentWorkArea.minX = Math.min(state.currentWorkArea.minX, bx.minX);
  state.currentWorkArea.minY = Math.min(state.currentWorkArea.minY, bx.minY);
  state.currentWorkArea.minZ = Math.min(state.currentWorkArea.minZ, bx.minZ);
  state.currentWorkArea.maxX = Math.max(state.currentWorkArea.maxX, bx.maxX);
  state.currentWorkArea.maxY = Math.max(state.currentWorkArea.maxY, bx.maxY);
  state.currentWorkArea.maxZ = Math.max(state.currentWorkArea.maxZ, bx.maxZ);
}

function distancePointToBox(pt, box) {
  const dx = Math.max(box.minX - pt.x, 0, pt.x - box.maxX);
  const dy = Math.max(box.minY - pt.y, 0, pt.y - box.maxY);
  const dz = Math.max(box.minZ - pt.z, 0, pt.z - box.maxZ);
  return Math.hypot(dx, dy, dz);
}

function distanceRayToWorkArea(ray, box, samples = 64) {
  if (!ray || !ray.from || !ray.to || !box) return Infinity;
  let best = Infinity;
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    const pt = {
      x: ray.from.x + (ray.to.x - ray.from.x) * t,
      y: ray.from.y + (ray.to.y - ray.from.y) * t,
      z: ray.from.z + (ray.to.z - ray.from.z) * t,
    };
    const d = distancePointToBox(pt, box);
    if (d < best) best = d;
  }
  return best;
}

function rayIntersectsExpandedBox(ray, box, expandBy = 3) {
  if (!ray || !ray.from || !ray.to || !box) return false;

  const b = {
    minX: box.minX - expandBy,
    minY: box.minY - expandBy,
    minZ: box.minZ - expandBy,
    maxX: box.maxX + expandBy,
    maxY: box.maxY + expandBy,
    maxZ: box.maxZ + expandBy,
  };

  let tMin = 0;
  let tMax = 1;
  const p0 = ray.from;
  const p1 = ray.to;

  for (const axis of ["x", "y", "z"]) {
    const d = p1[axis] - p0[axis];
    const min = b[`min${axis.toUpperCase()}`];
    const max = b[`max${axis.toUpperCase()}`];

    if (Math.abs(d) < 1e-9) {
      if (p0[axis] < min || p0[axis] > max) return false;
      continue;
    }

    let t1 = (min - p0[axis]) / d;
    let t2 = (max - p0[axis]) / d;
    if (t1 > t2) [t1, t2] = [t2, t1];
    tMin = Math.max(tMin, t1);
    tMax = Math.min(tMax, t2);
    if (tMin > tMax) return false;
  }

  return true;
}

// ─── progress bar ────────────────────────────────────────────────
function advancePart(name) {
  if (!state.parts.find(p => p.name === name)) {
    state.parts.push({ name, done: false });
    const d = document.createElement("div");
    d.className = "pdot";
    partDots.appendChild(d);
  }
  if (state.partIdx >= 0 && state.partIdx < state.parts.length) {
    state.parts[state.partIdx].done = true;
  }
  state.partIdx = state.parts.findIndex(p => p.name === name);
  renderProgress();
}

function renderProgress() {
  const dots = partDots.querySelectorAll(".pdot");
  state.parts.forEach((p, i) => {
    if (dots[i]) dots[i].className = "pdot" + (p.done ? " done" : i === state.partIdx ? " active" : "");
  });
  const cur = state.partIdx >= 0 ? state.parts[state.partIdx] : null;
  progressTxt.textContent = cur ? `Building: ${cur.name}` : (state.phase === "awaiting_approval" ? "Awaiting approval" : "Idle");
  brickCount.textContent  = `${state.brickCount} bricks`;
}

// ─── status dots ─────────────────────────────────────────────────
async function checkStatus() {
  try {
    const r = await fetch(`${RHINO_URL}/health`, { signal: AbortSignal.timeout(2000) });
    if (r.ok) {
      dotRhino.className = "dot ok";
      const d = await r.json();
      lblRhino.textContent = "Rhino connected";
      // Sync brick count from Rhino when idle (designer may have deleted/added bricks)
      if (state.phase === "idle") {
        state.brickCount = d.bricks;
        brickCount.textContent = `${d.bricks} bricks`;
      }
    } else throw new Error();
  } catch {
    dotRhino.className = "dot err";
    lblRhino.textContent = "Rhino offline";
  }

  try {
    const r = await fetch(`${SERVER_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      dotServer.className = "dot ok";
      const d = await r.json();
      state.serverCaps.fast = !!d.fast_available;
      if (!state.serverCaps.fast && state.fastEnabled) {
        state.fastEnabled = false;
      }
      syncFastToggle();
      lblServer.textContent = "Agents connected";
    } else throw new Error();
  } catch {
    dotServer.className = "dot err";
    lblServer.textContent = "Agents offline";
    state.serverCaps.fast = false;
    state.fastEnabled = false;
    syncFastToggle();
  }
}

// (stability class is now derived from ev.place.stability via _placeClass)

// Parse <part name="...">description</part> blocks from the planner output.
function parsePlanParts(planText) {
  const parts = [];
  const re = /<part\b[^>]*\bname=["']([^"']+)["'][^>]*>([\s\S]*?)<\/part>/gi;
  let m;
  while ((m = re.exec(planText)) !== null) {
    parts.push({ name: m[1].trim(), description: m[2].trim() });
  }
  return parts;
}

// Simplified line processor for the inspect-part SSE — renders each line in
// the inspect-text colour and skips any stray XML wrapper tags.
function processInspectLine(text) {
  const t = text.trim();
  if (!t) return;
  if (t === "<thinking>") { appendChat("  [Inspector reasoning]", "inspect-hdr"); return; }
  if (t === "</thinking>") { appendBlank(chatEl); return; }
  if (/^<\/?(?:think|thinking|plan|part|analysis|build|survey|review)\b/i.test(t)) return;
  appendChat("  " + t, "inspect-text");
}

// Stability class for a brick placement event
function _placeClass(stability) {
  if (stability === "weak") return "place-weak";
  if (stability === "unsupported") return "place-unsupported";
  return "place-stable";
}

// ─── SSE line processor ──────────────────────────────────────────
function processLine(text, ev = null) {
  const t = text.trim();
  if (!t) return;

  // /chat wrapper tags — first tag of every /chat response declares the mode
  const chatOpen = CHAT_OPEN_RE.exec(t);
  if (chatOpen) {
    const isBuild = chatOpen[1].toLowerCase() === "true";
    state.chatBuildTurn = isBuild;
    state.streamPhase = isBuild ? "build" : "chat";
    state.assistantBuf = [];
    // Reset per-turn exec-plan flags so a new build turn shows its own
    // plan header, regardless of what happened on the previous turn.
    state.execPartHasOutput = false;
    state.execThinkHeaderShown = false;
    state.execPlanHeaderShown = false;
    if (!isBuild) {
      appendBlank(chatEl);
      appendChat("  [BrickAgent]", "think-hdr");
    }
    return;
  }
  if (t === "</chat>") {
    state.streamPhase = "idle";
    return;
  }

  // Structured placement events are authoritative. We still defer the actual
  // /place call until the brick's trailing reasoning block finishes so the
  // blue build line, green reasoning block, and Rhino placement stay aligned.
  // EXCEPTION: during a review turn, the planner must not cause any brick
  // placement — if a stray place event slips through, treat it as prose.
  if (ev && ev.place && !state.chatReviewTurn && state.streamPhase !== "review") {
    const brick = {
      dims: String(ev.place.dims || "").trim().toLowerCase(),
      x: Number(ev.place.x),
      y: Number(ev.place.y),
      z: Number(ev.place.z),
      // In Executor mode, all bricks render white regardless of support ratio.
      stability: state.mode === "builder"
        ? "stable"
        : (ev.place.stability || "stable"),
    };
    if (Number.isFinite(brick.x) && Number.isFinite(brick.y) && Number.isFinite(brick.z)) {
      appendDebug(`explicit place ${brick.dims}@(${brick.x},${brick.y},${brick.z}) [${brick.stability}]`);
      state.execPartHasOutput = true;
      flushPendingReason();
      const placeText = normalizePlaceText(brick, ev.brick);
      state.currentReasonSlot = {
        lines: [],
        buildLines: [{ text: placeText, cls: "build-place" }],
        brick,
      };
      renderBrickReasonBlock(state.currentReasonSlot);
      expandCurrentWorkArea(brick);
      state.brickCount++;
      brickCount.textContent = `${state.brickCount} bricks`;
      if (state.streamPhase !== "build") state.streamPhase = "build";
      return;
    }
  }

  // phase-control tags — thinking/analysis/review are kept out of the log
  // entirely. Their content is reasoning that belongs to individual bricks
  // (reachable via Rhino-click brick detail panel) or overall plan context.
  if (t === "<analysis>") { state.streamPhase = "thinking"; return; }
  if (t === "</analysis>") { state.streamPhase = "chat"; return; }
  if (t === "<thinking>" || t === "<think>")  { state.streamPhase = "thinking"; return; }
  if (t === "</thinking>" || t === "</think>") { state.streamPhase = "plan"; return; }
  if (t === "<plan>")      { state.streamPhase = "plan"; return; }
  if (t === "</plan>")     {
    state.streamPhase = "idle";
    return;
  }
  if (t === "<review>")    {
    state.streamPhase = "review";
    stopBuildingStatus();
    return;
  }
  if (t === "</review>")   { state.streamPhase = "idle"; return; }

  if (t.startsWith("<exec_prompt")) {
    state.streamPhase = "exec_prompt";
    appendBlank(cotEl);
    appendCot("  [Executor prompt ▼]", "dim");
    return;
  }
  if (t === "</exec_prompt>") {
    state.streamPhase = "build";  // prompt block ends, build output follows
    return;
  }

  if (t.startsWith("<part")) {
    const nm = PART_NAME_RE.exec(t);
    const name = nm ? nm[1] : "Part";
    state.streamPhase = "build";
    state.execPartHasOutput = false;
    state.execThinkHeaderShown = false;
    state.execPlanHeaderShown = false;
    resetCurrentWorkArea();
    advancePart(name);
    // No main-log output during build — the breathing status line is
    // triggered by </plan> (end of CoT) or by the first brick, not here.
    return;
  }
  if (t === "</part>") { state.streamPhase = "idle"; return; }

  // summary
  const sumMatch = SUMMARY_RE.exec(t);
  if (sumMatch) { state.lastSummary = sumMatch[1].trim(); return; }

  // Capture synthesized build prompt from planner → executor handoff
  const buildPromptMatch = /^\[(?:Build prompt|Planner).*Executor\]\s*"(.+)"$/.exec(t);
  if (buildPromptMatch) {
    const prompt = buildPromptMatch[1].trim();
    state.buildPrompt = prompt;
    appendBlank();
    appendLog("  ── Planner → Executor ──", "plan-hdr");
    appendLog(`  “${prompt}”`, "build-prompt");
    appendBlank();
    return;
  }

  // skip wrapper tags
  if (t === "<build>") {
    state.streamPhase = "build";
    return;
  }
  if (t === "</build>") { state.streamPhase = "idle"; return; }
  if (/^<\/?(survey|review)>$/i.test(t)) return;
  // skip raw XML structural tags in plan section
  if (state.streamPhase === "plan" && /^<\/?(?:part|analysis|plan)\b/i.test(t)) return;

  switch (state.streamPhase) {
    case "exec_prompt":
      // Hide executor prompt block — debug only, not useful in UI.
      return;

    case "chat": {
      // Negotiation turn — plain assistant prose, kept separate from build log
      state.assistantBuf.push(t);
      // Recognise the four required section labels and style them as headers
      const sectionMatch = /^\s*(Current state of Rhino|Inference from image and prompt|What was missing|Draft build prompt)\s*:\s*(.*)$/i.exec(t);
      if (sectionMatch) {
        appendBlank(chatEl);
        appendChat("  " + sectionMatch[1] + ":", "inspect-section");
        const rest = sectionMatch[2].trim();
        if (rest) appendChat("  " + rest, "inspect-text");
        return;
      }
      appendChat("  " + t, "inspect-text");
      return;
    }

    case "thinking": {
      // Executor CoT between <think>…</think> — surface while no bricks yet.
      if (state.chatBuildTurn && !state.execPartHasOutput) {
        if (!state.execThinkHeaderShown) {
          // If this is a retry (CoT panel already has content), draw a
          // horizontal divider between the previous attempt and this one.
          if (cotEl.children.length > 0) _appendCotSeparator();
          appendBlank(cotEl);
          appendCot("  ── Executor reasoning (CoT) ──", "exec-think-hdr");
          state.execThinkHeaderShown = true;
        }
        _emitCotLine(t, "exec-think");
        return;
      }
      if (state.currentReasonSlot) {
        state.currentReasonSlot.lines.push(t);
        renderBrickReasonBlock(state.currentReasonSlot);
      }
      return;
    }
    case "plan": {
      // Executor layer-strategy between <plan>…</plan> — surface while no
      // bricks yet. Distinct header from the CoT block above.
      if (state.chatBuildTurn && !state.execPartHasOutput) {
        if (!state.execPlanHeaderShown) {
          appendBlank(cotEl);
          appendCot("  ── Executor plan ──", "exec-plan-hdr");
          state.execPlanHeaderShown = true;
        }
        _emitCotLine(t, "exec-plan");
        return;
      }
      if (state.currentReasonSlot) {
        state.currentReasonSlot.lines.push(t);
        renderBrickReasonBlock(state.currentReasonSlot);
      }
      return;
    }
    case "review": {
      // Review from the agent: stop the breathing status, show a header
      // once, then stream the review prose into the main log.
      stopBuildingStatus();
      flushPendingReason();
      if (!state.reviewHeaderShown) {
        const line1 = t;
        enqueueUILog(() => {
          appendBlank();
          appendLog("  ── Review ──", "review-hdr");
          appendLog("  " + line1, "review-text");
        });
        state.reviewHeaderShown = true;
      } else {
        const line = t;
        enqueueUILog(() => appendLog("  " + line, "review-text"));
      }
      return;
    }

    case "build": {
      if (/^\[BRIDGE\]\s*REJECTED/i.test(t)) {
        state.execPartHasOutput = true;
        flushPendingReason();
        enqueueUILog(() => appendBuild("  " + t, "reject"));
        return;
      }
      if (LAYER_RE.test(t)) {
        state.execPartHasOutput = true;
        flushPendingReason();
        enqueueUILog(() => {
          appendBlank(buildEl);
          appendBuild("  " + t, "layer-mark");
        });
        return;
      }
      // Fallback brick parsing (for endpoints that don't send ev.place)
      const parsedBrick = parseBrickLine(t);
      if (parsedBrick && Number.isFinite(parsedBrick.x) && Number.isFinite(parsedBrick.y) && Number.isFinite(parsedBrick.z)) {
        const brick = {
          dims: parsedBrick.dims,
          x: parsedBrick.x,
          y: parsedBrick.y,
          z: parsedBrick.z,
          stability: "stable",
        };
        state.execPartHasOutput = true;
        flushPendingReason();
        const placeText = normalizePlaceText(brick, parsedBrick.text);
        state.currentReasonSlot = {
          lines: [],
          buildLines: [{ text: placeText, cls: "build-place" }],
          brick,
        };
        renderBrickReasonBlock(state.currentReasonSlot);
        expandCurrentWorkArea(brick);
        state.brickCount++;
        brickCount.textContent = `${state.brickCount} bricks`;
        return;
      }
      if (state.currentReasonSlot) {
        // Stage-5 blocks emit "  ROLE: ...", "  REASON: ...", "  PHYSICS: ...",
        // "  STATUS: ..." after each PLACE. Capture them as both the brick's
        // reason (green panel) and part of the build log (blue panel).
        state.currentReasonSlot.lines.push(t);
        const fieldMatch = STAGE5_FIELD_RE.exec(t);
        const cls = fieldMatch && fieldMatch[1].toUpperCase() === "STATUS"
          ? _statusCls(fieldMatch[2])
          : "physics";
        state.currentReasonSlot.buildLines.push({ text: t, cls });
        renderBrickReasonBlock(state.currentReasonSlot);
      } else if (!STAGE5_FIELD_RE.test(t) && !PHYSICS_RE.test(t)) {
        enqueueUILog(() => appendBuild("  " + t, "plan"));
      }
      return;
    }

    default:
      // Unknown phase — drop, don't pollute the log.
      return;
  }
}

// ─── continuation helpers ────────────────────────────────────────
function formatBricksForContinuation(bricks) {
  // Group by z-level, format as PLACE lines with layer markers —
  // matches the format the executor model would have generated.
  const byZ = {};
  for (const b of bricks) {
    const z = b.z;
    if (!byZ[z]) byZ[z] = [];
    byZ[z].push(b);
  }
  const zLevels = Object.keys(byZ).map(Number).sort((a, b) => a - b);
  const lines = [];
  for (const z of zLevels) {
    lines.push(`--- Layer ${z}${z === 0 ? " (Ground)" : ""} ---`);
    for (const b of byZ[z]) {
      lines.push(`PLACE ${b.dims} (${b.x},${b.y},${b.z})`);
    }
  }
  return lines.join("\n");
}

async function dispatchContinue() {
  // Resume a paused build: inject existing bricks into the assistant turn
  // and let the executor model continue from that point.
  const [rhinoBricks, views] = await Promise.all([
    fetchRhinoBricks(),
    fetchViewports(),
  ]);
  state.rhinoBricks = rhinoBricks;
  state.brickCount  = rhinoBricks.length;
  brickCount.textContent = `${state.brickCount} bricks`;

  appendViewports(views);

  const existingBricks = formatBricksForContinuation(rhinoBricks);

  _stopRequested       = false;
  state.streamPhase    = "idle";
  state.phase          = "building";
  state.chatBuildTurn  = true;
  state.assistantBuf   = [];
  state._buildingStatusEl = null;
  state.reviewHeaderShown = false;

  setUIBuilding(true);
  progress.classList.add("visible");
  progressTxt.textContent = "Continuing build…";

  // Start activity monitoring before fetch — server may block on lock
  startActivityMonitoring();

  state.abortCtrl = new AbortController();
  let response;
  try {
    response = await fetch(`${SERVER_URL}/continue`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.abortCtrl.signal,
      body: JSON.stringify({
        prompt:          state.buildPrompt || state.lastPrompt,
        existing_bricks: existingBricks,
        brick_state:     rhinoBricks,
        max_new_tokens:  8192,
      }),
    });
  } catch (err) {
    if (err.name !== "AbortError") appendLog(`  ERROR: ${err.message}`, "warn");
    onFinish();
    return;
  }

  let streamCompleted = false;
  await readSSE(
    response,
    processLine,
    () => { streamCompleted = true; },
    () => {},
  );

  if (!streamCompleted) {
    if (!state.waitingForUser) onFinish();
    return;
  }
  if (state.waitingForUser) {
    state.waitingForUser = false;
    waitingBanner.style.display = "none";
    if (_idleTimerId) { clearTimeout(_idleTimerId); _idleTimerId = null; }
  }
  await waitForPlaceQueue();
  stopActivityMonitoring();
  onDone();
}

// ─── Rhino bridge calls ───────────────────────────────────────────
async function placeBrick(data) {
  const task = async () => {
    if (_stopRequested) return null;
    try {
      appendDebug(`POST /place -> ${JSON.stringify(data)}`);
      const r = await fetch(`${RHINO_URL}/place`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!r.ok) {
        appendDebug(`POST /place failed with HTTP ${r.status}`);
        appendLog(`  [Bridge] place failed for ${data.dims} (${data.x},${data.y},${data.z})`, "warn");
        return null;
      }
      const result = await r.json().catch(() => null);
      appendDebug(`POST /place <- ${JSON.stringify(result)}`);
      if (result && result.placed === false) {
        appendLog(`  [Bridge] place rejected for ${data.dims} (${data.x},${data.y},${data.z})`, "warn");
        return null;
      }
      return result && result.guid ? String(result.guid) : null;
    } catch (err) {
      appendDebug(`POST /place error: ${err.message}`);
      appendLog(`  [Bridge] place error for ${data.dims} (${data.x},${data.y},${data.z}): ${err.message}`, "warn");
      return null;
    }
  };

  const queued = _placeQueue.then(task, task);
  _placeQueue = queued.catch(() => null);
  return queued;
}

async function postSetReason(guid, reason) {
  try {
    await fetch(`${RHINO_URL}/set_reason`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ guid, reason }),
    });
  } catch {}
}

async function postHighlightPicked(guid) {
  try {
    const r = await fetch(`${RHINO_URL}/highlight_picked`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ guid }),
    });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

async function postRestoreHighlight() {
  try {
    await fetch(`${RHINO_URL}/restore_highlight`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
  } catch {}
}

async function fetchPickedBrick() {
  try {
    const r = await fetch(`${RHINO_URL}/picked_brick`, { signal: AbortSignal.timeout(1000) });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

// With deferred placement, we no longer need to debounce partial reason
// writes — the reason is baked into /place at flush time. Kept as a no-op
// so existing call sites (if any remain) stay benign.
function scheduleReasonFlush() {}

// Map a Stage-5 STATUS value to a CSS class so "  STATUS: WEAK" etc. glow
// the right warning colour in the blue build panel. STABLE inherits the
// panel colour (no explicit override).
function _statusCls(value) {
  const v = String(value || "").trim().toLowerCase();
  if (v.includes("weak")) return "place-weak";
  if (v.includes("unsupport")) return "place-unsupported";
  return "physics";
}

// Flush the pending brick: place it in Rhino, then emit the matching blue
// PLACE+metadata lines and green reasoning block on the same FIFO queue so
// the UI stays in lockstep with the visible scene. Called when the next
// PLACE arrives OR at the end of the stream.
function flushPendingReason() {
  const slot = state.currentReasonSlot;
  state.currentReasonSlot = null;
  if (!slot) return;
  if (slot._flushTimer) { clearTimeout(slot._flushTimer); slot._flushTimer = null; }
  if (!slot.brick) return;
  const reason = (slot.lines || []).join("\n").trim();
  const payload = { ...slot.brick };
  if (reason) payload.reason = reason;
  // Enqueue the /place POST on _placeQueue, then enqueue the matching log
  // lines and green reasoning block immediately after on the same queue —
  // so the UI can never show a brick before Rhino has actually placed it.
  placeBrick(payload);
  const buildLines = slot.buildLines || [];
  const trackedBrick = slot.brick;
  enqueueUILog(() => {
    for (const bl of buildLines) {
      if (!bl || !bl.text) continue;
      const el = appendBuild("  " + bl.text, bl.cls || "normal");
      if (bl.cls === "build-place") _recordPlaceStep(el, trackedBrick);
    }
    renderBrickReasonBlock(slot);
  });
}

async function waitForPlaceQueue() {
  try {
    await _placeQueue;
  } catch {}
}

async function clearRhino() {
  try { await fetch(`${RHINO_URL}/clear`, { method: "POST" }); } catch {}
}

async function redrawRhino() {
  try {
    await fetch(`${RHINO_URL}/redraw`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zoom: true }),
    });
  } catch {}
}

async function fetchRhinoBricks() {
  try {
    const r = await fetch(`${RHINO_URL}/bricks`, { signal: AbortSignal.timeout(2000) });
    return r.ok ? await r.json() : [];
  } catch { return []; }
}

async function fetchViewports() {
  try {
    const r = await fetch(`${RHINO_URL}/viewport`, { signal: AbortSignal.timeout(5000) });
    return r.ok ? await r.json() : {};
  } catch { return {}; }
}

async function postBeginTurn() {
  try {
    await fetch(`${RHINO_URL}/begin_turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
  } catch {}
}

// ─── brick detail panel ──────────────────────────────────────────
function hideBrickDetail() {
  if (brickDetail) brickDetail.classList.remove("visible");
  // Re-show the live stream of all bricks' reasoning.
  if (brickStream) brickStream.style.display = "";
  syncBrickPanelState();
}

// Render the reason paragraph as a series of line spans so cascade-risk
// lines and weak/unsupported warnings can be coloured independently of the
// default green body text.
function _renderBrickReasonLines(container, rawReason) {
  if (!container) return;
  container.innerHTML = "";
  const text = rawReason && String(rawReason).trim();
  if (!text) {
    container.textContent = "(no reasoning captured for this brick)";
    return;
  }
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  for (const line of lines) {
    const span = document.createElement("div");
    span.className = _brickReasonLineClass(line);
    span.textContent = line;
    container.appendChild(span);
  }
}

function _brickReasonLineClass(line) {
  const l = String(line).toLowerCase();
  if (/\bcascade\s+risk\b/.test(l)) return "bd-cascade";
  if (/\bunsupported\b/.test(l))    return "bd-unsupported";
  if (/(^|\s)⚠|\bweak\b/.test(l))   return "bd-weak";
  return "bd-line";
}

function renderBrickDetail(brick, supporters) {
  if (!brickDetail) return;
  const dims = String(brick.dims || "?").toLowerCase();
  const x = brick.x, y = brick.y, z = brick.z;
  const stab = String(brick.stability || "—").toUpperCase();
  bdTitle.textContent  = `${dims} @ (${x}, ${y}, ${z})`;
  bdMeta.textContent   = `Stability: ${stab}`;
  _renderBrickReasonLines(bdReason, brick.reason);
  if (supporters && supporters.length) {
    bdSupporters.textContent = `Supporters (shown blue): ${supporters.length}`;
    bdSupporters.style.display = "block";
  } else {
    bdSupporters.textContent = z > 0 ? "No bricks support this one." : "Ground-level brick.";
    bdSupporters.style.display = "block";
  }
  brickDetail.classList.add("visible");
  // Hide the streaming log of *every* brick's reason — the green panel
  // must show reasoning for the clicked brick only.
  if (brickStream) brickStream.style.display = "none";
  syncBrickPanelState();
}

async function clearPick() {
  const hadHighlight = !!state.pickedGuid;
  state.pickedGuid = null;
  clearBrickStream();
  hideBrickDetail();
  // Only restore highlight if one was actually applied — Fast (builder) mode
  // never applies a highlight, so the POST would just clobber neighbouring
  // bricks that were never coloured to begin with.
  if (hadHighlight && state.mode !== "builder") {
    await postRestoreHighlight();
  }
}

async function handlePickedGuid(guid) {
  if (!guid) {
    // Empty-space click — restore any active highlight + hide panel.
    await clearPick();
    return;
  }
  if (guid === state.pickedGuid) return;
  state.pickedGuid = guid;
  // Fetch current brick state to look up this brick's reason.
  const bricks = await fetchRhinoBricks();
  const hit = bricks.find(b => String(b.guid || "") === guid);
  if (!hit) {
    await clearPick();
    return;
  }
  // Fast (builder) mode: don't paint current/supporting bricks in Rhino.
  // Still show the reasoning in the green panel — just skip the highlight.
  let supporters = [];
  if (state.mode !== "builder") {
    const result = await postHighlightPicked(guid);
    supporters = (result && result.supporters) || [];
  }
  renderBrickDetail(hit, supporters);
}

async function pollPickedBrick() {
  const data = await fetchPickedBrick();
  if (!data || typeof data.seq !== "number") return;
  if (data.seq === state.pickedSeq) return;
  state.pickedSeq = data.seq;
  if (typeof _analysisOnRhinoBrickPick === "function") {
    _analysisOnRhinoBrickPick(data.guid || null);
  }
  await handlePickedGuid(data.guid || null);
}

// ─── mode toggle ─────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  btnModeBuilder.classList.toggle("active", mode === "builder");
  if (btnModeStage3) btnModeStage3.classList.toggle("active", mode === "stage3");
  if (btnModeStage5) btnModeStage5.classList.toggle("active", mode === "stage5");
  // Direct-to-executor shortcut is only meaningful in Stage 3 / Stage 5 —
  // in Builder mode the prompt already skips the planner.
  if (btnDirect) btnDirect.hidden = (mode === "builder");
}

// ─── activity monitoring (Inspect mode) ──────────────────────────
let _activityHandlers = [];
let _cameraPollId = null;
let _interactionPollId = null;
let _lastInteractionSeq = 0;
let _lastCameraJSON = "";
let _idleTimerId = null;
let _graceTimerId = null;     // grace period: ignore activity for N seconds after build starts
let _graceActive = false;
let _uiLeftMouseHeld = false;
let _placeQueue = Promise.resolve();
let _stopRequested = false;

// Run `fn` on the same FIFO chain as placeBrick so its side effects (log
// lines, etc.) land AFTER every preceding /place POST has completed. Used
// for layer markers and rejections so they never appear above a brick that
// is still in the placement queue.
function enqueueUILog(fn) {
  _placeQueue = _placeQueue.then(fn, fn).catch(() => null);
  return _placeQueue;
}

function addActivityListener(target, type, handler) {
  target.addEventListener(type, handler, true);
  _activityHandlers.push({ target, type, handler });
}

async function fetchCamera() {
  try {
    const r = await fetch(`${RHINO_URL}/camera`, { signal: AbortSignal.timeout(1000) });
    return r.ok ? await r.json() : {};
  } catch { return {}; }
}

async function fetchInteraction() {
  try {
    const r = await fetch(`${RHINO_URL}/interaction`, { signal: AbortSignal.timeout(1000) });
    return r.ok ? await r.json() : {};
  } catch { return {}; }
}

function _onActivity(e) {
  if (state.phase !== "building" && !state.waitingForUser) return;
  // Grace period — ignore activity for a few seconds after build starts
  if (_graceActive) return;
  // Ignore clicks on UI buttons (Stop, Clear, etc.)
  if (e && e.target && e.target.tagName === "BUTTON") return;
  // Only left-click (button 0) pauses — right-click / middle-click are passive (orbit, context menu)
  if (e && e.type === "mousedown" && e.button !== 0) return;
  // Ignore modifier keys alone (Shift/Ctrl/Alt used for camera nav)
  if (e && e.type === "keydown" && ["Shift", "Control", "Alt", "Meta"].includes(e.key)) return;

  if (e && e.type === "mousedown") {
    if (!state.waitingForUser) {
      pauseForUser();
    }
    // Mouse held ⇒ idle timer is suppressed until mouseup. This covers
    // click-drag to scroll a panel, select text, etc. — the agent must
    // stay paused the entire time the button is held, even past 10s.
    if (_idleTimerId) { clearTimeout(_idleTimerId); _idleTimerId = null; }
    _uiLeftMouseHeld = true;
    return;
  }

  if (!state.waitingForUser) {
    pauseForUser();
  }
  resetIdleTimer();
}

function _onActivityMouseUp(e) {
  if (e.button !== 0 || !_uiLeftMouseHeld) return;
  _uiLeftMouseHeld = false;
  // Start the 10s idle countdown only once the designer lets go.
  if (state.waitingForUser) {
    resetIdleTimer();
  }
}

async function pollRhinoInteraction() {
  if (state.mode === "builder") return;
  if (state.phase !== "building" && !state.waitingForUser) return;

  const ev = await fetchInteraction();
  const seq = Number(ev.seq || 0);
  if (!seq || seq <= _lastInteractionSeq) return;
  _lastInteractionSeq = seq;

  if (ev.kind !== "left_click" || !state.currentWorkArea || !ev.grid_ray) return;

  if (!rayIntersectsExpandedBox(ev.grid_ray, state.currentWorkArea, 3)) return;
  const dist = distanceRayToWorkArea(ev.grid_ray, state.currentWorkArea);

  if (!state.waitingForUser) {
    appendBlank(chatEl);
    appendChat(`  [Inspect pause — Rhino click near current work area (${dist.toFixed(2)} units)]`, "sys");
    pauseForUser();
  }
  resetIdleTimer();
}

function startActivityMonitoring() {
  stopActivityMonitoring();

  if (state.mode === "builder") return;

  // Grace period: ignore all activity for 5s after build starts.
  // This prevents accidental pauses from stray clicks/scrolls when the user
  // is just watching output stream.
  _graceActive = true;
  _graceTimerId = setTimeout(() => { _graceActive = false; }, 5000);

  // Left-click + keyboard only — right-click (orbit) is passive, filtered in _onActivity
  addActivityListener(document, "mousedown", _onActivity);
  addActivityListener(document, "keydown", _onActivity);
  addActivityListener(window, "mouseup", _onActivityMouseUp);
}

function stopActivityMonitoring() {
  _activityHandlers.forEach(({ target, type, handler }) => target.removeEventListener(type, handler, true));
  _activityHandlers = [];
  if (_cameraPollId) { clearInterval(_cameraPollId); _cameraPollId = null; }
  if (_interactionPollId) { clearInterval(_interactionPollId); _interactionPollId = null; }
  if (_idleTimerId) { clearTimeout(_idleTimerId); _idleTimerId = null; }
  if (_graceTimerId) { clearTimeout(_graceTimerId); _graceTimerId = null; }
  _graceActive = false;
  _uiLeftMouseHeld = false;
}

function resetIdleTimer() {
  if (_idleTimerId) { clearTimeout(_idleTimerId); _idleTimerId = null; }
  // If the designer is still holding the mouse (scroll-drag, text-select,
  // etc.), don't arm the timer — mouseup will re-arm it.
  if (_uiLeftMouseHeld) return;
  _idleTimerId = setTimeout(onUserIdle, 10000);
}

function pauseForUser() {
  if (state.waitingForUser) return;
  state.waitingForUser = true;

  // Freeze the executor at the next token — the SSE stream stalls but stays
  // open so generation can resume seamlessly from where it left off.
  fetch(`${SERVER_URL}/pause`, { method: "POST", keepalive: true }).catch(() => {});

  waitingBanner.style.display = "flex";
  progressTxt.textContent = "Waiting for designer…";
}

async function onUserIdle() {
  if (!state.waitingForUser || _uiLeftMouseHeld) return;
  state.waitingForUser = false;
  waitingBanner.style.display = "none";

  appendLog("  ── Designer idle 10s — resuming build… ──", "plan-hdr");
  appendBlank();

  if (state.phase === "building") progressTxt.textContent = "Building…";
  else if (state.phase === "reviewing") progressTxt.textContent = "Reviewing build…";
  else progressTxt.textContent = "Negotiating…";

  // Unfreeze the executor — the SSE stream continues automatically from
  // the exact token where it was paused.  Activity monitoring stays active
  // so the user can pause again.
  fetch(`${SERVER_URL}/resume`, { method: "POST", keepalive: true }).catch(() => {});
}

// ─── SSE stream reader ───────────────────────────────────────────
async function readSSE(response, onLine, onDone, onApproval) {
  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let   buf     = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();

      for (const raw of lines) {
        const line = raw.trim();
        if (!line) continue;
        if (line === "data: [DONE]") { onDone(); return; }
        if (line === "data: [AWAITING_APPROVAL]") { onApproval(); return; }
        if (line.startsWith("data: ")) {
          try {
            const ev = JSON.parse(line.slice(6));
            if (ev && typeof ev.brick === "string") {
              appendDebug(`SSE event: place=${!!ev.place} text=${String(ev.brick).slice(0, 160)}`);
              onLine(ev.brick, ev);
            }
          } catch {}
        }
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") appendLog(`  Stream error: ${err.message}`, "warn");
  }
}

// ─── Unified single-agent chat flow ──────────────────────────────
// One endpoint (/chat) for both modes. Three-phase loop:
//
//   1. User sends a prompt → onChatSend() pushes user turn to chatHistory
//      and fires /chat with build_now=true for Fast mode, false for Inspect.
//   2. Server replies — either negotiation text (build_now=false) or a full
//      build stream (build_now=true). On [DONE]:
//        - If negotiation: show the accept/correct panel.
//        - If build: run onDone() like before.
//   3. Designer either clicks "Build It" (onChatAccept) or types a correction
//      into clarif-input and clicks "Send Correction" (onChatCorrection).
//      Both re-enter /chat with the full chatHistory.
async function dispatchChat({ buildNow = false, review = false, skipPlanner = false } = {}) {
  // Capture viewports + scene; images sent to planner in inspect mode.
  const [rhinoBricks, views] = await Promise.all([
    fetchRhinoBricks(),
    fetchViewports(),
  ]);
  state.rhinoBricks = rhinoBricks;
  state.brickCount  = rhinoBricks.length;
  brickCount.textContent = `${state.brickCount} bricks`;

  // Merge user-uploaded reference images into views dict (keyed as
  // "ref_<filename>"). The reference image is STICKY across turns — the
  // planner kept losing the image-derived shape between turns when we
  // cleared after one send, so we keep the bytes in state.userImages
  // until the designer explicitly removes them via the composer pill.
  for (const img of state.userImages) {
    views[`ref_${img.name}`] = img.b64;
  }
  state.lastViews = views;

  // Show viewport captures in the log (includes user ref images if any)
  appendViewports(views);

  // Reset transient stream state
  _stopRequested     = false;
  state.streamPhase  = "idle";
  state.parts        = [];
  state.partIdx      = -1;
  state.lastSummary  = "";
  state.buildPrompt  = "";
  state.assistantBuf = [];
  state.chatBuildTurn = buildNow;
  state._buildingStatusEl = null;
  state.reviewHeaderShown = false;
  state.phase        = buildNow ? "building" : "chatting";
  if (buildNow) { clearBrickStream(); _resetBuildTimeline(); }
  progress.classList.add("visible");
  progressTxt.textContent = buildNow
    ? "Building…"
    : "Negotiating…";

  setUIBuilding(true);
  btnApprove.style.display = "none";
  state.chatReviewTurn = review;
  if (review) {
    state.phase = "reviewing";
    progressTxt.textContent = "Reviewing build…";
  }

  // Mark start of a build turn so the bridge tracks which bricks this turn
  // produces.
  if (buildNow) await postBeginTurn();

  // Start pause-on-interact monitoring BEFORE the fetch — the server
  // may block on _exec_lock for seconds, and typing during that wait
  // should still trigger a pause.
  if (buildNow && state.mode !== "builder") startActivityMonitoring();

  state.abortCtrl = new AbortController();
  let response;
  try {
    response = await fetch(`${SERVER_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.abortCtrl.signal,
      body: JSON.stringify({
        mode:         state.mode,
        messages:     state.chatHistory.slice(-12),
        brick_state:  rhinoBricks,
        views:        views,
        build_now:    buildNow,
        review:       review,
        skip_planner: skipPlanner,
        max_new_tokens: buildNow ? 49152 : 1024,
      }),
    });
  } catch (err) {
    if (err.name !== "AbortError") appendLog(`  ERROR: ${err.message}`, "warn");
    onFinish();
    return;
  }

  let streamCompleted = false;
  await readSSE(
    response,
    processLine,
    () => { streamCompleted = true; },
    () => {}, // /chat has no [AWAITING_APPROVAL]
  );

  if (!streamCompleted) {
    if (!state.waitingForUser) onFinish();
    return;
  }
  if (state.waitingForUser) {
    state.waitingForUser = false;
    waitingBanner.style.display = "none";
    if (_idleTimerId) { clearTimeout(_idleTimerId); _idleTimerId = null; }
  }
  await waitForPlaceQueue();
  stopActivityMonitoring();

  // Commit assistant response to chat history
  const assistantText = state.assistantBuf.join("\n").trim();
  if (assistantText) {
    state.chatHistory.push({ role: "assistant", content: assistantText });
  } else if (state.chatBuildTurn) {
    state.chatHistory.push({
      role: "assistant",
      content: `Built ${state.brickCount} bricks.`,
    });
  }
  state.assistantBuf = [];

  if (state.chatBuildTurn) {
    // Full build completed — onDone shows summary + auto-triggers review
    await onDone();
    return;
  }

  if (state.chatReviewTurn) {
    // Critical review delivered — build is final.
    state.chatReviewTurn = false;
    appendBlank();
    appendLog("  ─ Review complete. ─", "sys");
    appendBlank();
    onFinish();
    return;
  }

  // Negotiation turn completed → show accept/correct panel
  onAwaitingAccept();
}

async function onChatSend() {
  let prompt = input.value.trim();
  // Allow an image-only submission: if there's no typed text but the user
  // has attached a reference image, send a placeholder prompt that tells
  // the planner to describe and build from the image alone.
  if (!prompt && state.userImages.length) {
    prompt = "[reference image attached — please describe it and propose a brick build]";
  }
  if (!prompt) return;

  // During negotiation, Send / Enter always continues the negotiation —
  // it never triggers a build. The build can only be started by clicking
  // the Approve & Build button.
  if (state.phase === "awaiting_accept") {
    input.value = "";
    input.placeholder = "Describe what to build…";
    state.chatHistory.push({ role: "user", content: prompt });
    appendBlank(chatEl);
    appendChatMessage(prompt);
    appendBlank(chatEl);
    btnApprove.style.display = "none";
    state.phase = "idle";
    if (DONE_PHRASE_RE.test(prompt)) {
      await dispatchChat({ review: true });
      return;
    }
    await dispatchChat({ buildNow: false });
    return;
  }

  if (state.phase !== "idle") return;
  input.value = "";

  state.lastPrompt = prompt;
  state.chatHistory.push({ role: "user", content: prompt });
  appendBlank(chatEl);
  appendChatMessage(prompt);
  appendBlank(chatEl);

  // "enough" / "I'm done" → critical review path (Inspect mode only).
  // The planner analyses brick_state + viewports and produces a review
  // instead of handing off to the executor.
  const hasAssistant = state.chatHistory.some(m => m.role === "assistant");
  if (state.mode !== "builder" && hasAssistant && DONE_PHRASE_RE.test(prompt)) {
    await dispatchChat({ review: true });
    return;
  }

  // Executor (builder): build immediately, no negotiation.
  // Fast (builder): auto-build, no negotiation phase.
  // Inspect: first user turn ALWAYS negotiates. Phrase detection is only
  // honored after at least one assistant reply exists, so an opening line
  // like "build a table" cannot skip the negotiation step.
  const phraseBuild = hasAssistant && BUILD_PHRASE_RE.test(prompt);
  const buildNow = state.mode === "builder" || phraseBuild;
  await dispatchChat({ buildNow });
}

// Direct: skip the planner entirely in Stage 3 / Stage 5 and send the
// prompt straight to the stage-LoRA executor (Agent 2).
async function onChatDirect() {
  if (state.mode === "builder") return;             // Fast mode already bypasses planner
  if (state.phase !== "idle" && state.phase !== "awaiting_accept") return;

  const prompt = input.value.trim();
  if (!prompt) return;
  input.value = "";
  input.placeholder = "Describe what to build…";

  state.lastPrompt = prompt;
  state.chatHistory.push({ role: "user", content: prompt });
  appendBlank(chatEl);
  appendChatMessage(prompt);
  appendBlank(chatEl);

  btnApprove.style.display = "none";
  state.phase = "idle";
  await dispatchChat({ buildNow: true, skipPlanner: true });
}

function onAwaitingAccept() {
  state.phase = "awaiting_accept";
  appendBlank(chatEl);
  appendChat(
    "  ─ Type a correction + Enter, or click 'Approve & Build' to proceed ─",
    "plan-hdr",
  );
  appendBlank(chatEl);
  btnApprove.style.display = "inline-block";
  progressTxt.textContent = "Awaiting your acceptance…";
  input.value = "";
  input.placeholder = "Type a correction and press Enter, or click Approve & Build…";
  input.focus();
  setUIBuilding(false);
  input.disabled = false;
}

async function onChatAccept() {
  if (state.phase !== "awaiting_accept") return;
  const note = input.value.trim();
  input.value = "";
  input.placeholder = "Describe what to build…";
  btnApprove.style.display = "none";

  if (note) {
    // Designer typed a note before hitting Approve & Build —
    // treat it as a final user turn before proceeding.
    state.chatHistory.push({ role: "user", content: note });
    appendBlank(chatEl);
    appendChatMessage(note);
    appendBlank(chatEl);
  } else {
    state.chatHistory.push({ role: "user", content: "Build it." });
    appendBlank(chatEl);
    appendChatMessage("Build it.");
    appendBlank(chatEl);
  }
  state.phase = "idle";
  await dispatchChat({ buildNow: true });
}

// ─── Direct modes: skip planner, go straight to a single executor ─────────
async function onBuildDirect({ fast = false } = {}) {
  const prompt = input.value.trim();
  if (!prompt || state.phase !== "idle") return;
  input.value = "";

  const endpoint = fast ? "/execute-fast" : "/execute";
  const modeName = fast ? "Executor + Fast" : "Executor";
  const maxNewTokens = fast ? 8192 : 16384;

  _stopRequested    = false;
  state.phase       = "executing";
  state.brickCount  = 0;
  state.parts       = [];
  state.partIdx     = -1;
  state.planText    = "";
  state.lastSummary = "";
  state.buildPrompt = "";
  state.lastPrompt  = prompt;
  state.streamPhase = "idle";

  setUIBuilding(true);

  appendBlank(chatEl);
  appendChatMessage(prompt);
  appendBlank(chatEl);

  const rhinoBricks = await fetchRhinoBricks();
  state.rhinoBricks = rhinoBricks;

  if (rhinoBricks.length > 0) {
    appendBuild(`  [Bridge] ${rhinoBricks.length} existing bricks on grid (kept)`, "sys");
    state.brickCount = rhinoBricks.length;
  }

  // Bridge tracks per-turn bricks so finalize can translate just this turn's output.
  await postBeginTurn();

  progress.classList.add("visible");
  progressTxt.textContent = fast ? "Building (direct, fast)…" : "Building (direct)…";
  appendBuild(`  [${modeName} — sending prompt directly to ${fast ? "fast prompt-only path" : "executor"}]`, "sys");
  appendBlank(buildEl);

  state.abortCtrl = new AbortController();
  let response;
  try {
    response = await fetch(`${SERVER_URL}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.abortCtrl.signal,
      body: JSON.stringify({
        plan_text:     "",
        prompt,
        brick_state:   rhinoBricks,
        clarification: "",
        history:       state.history.slice(-8),
        max_new_tokens: maxNewTokens,
        fast_mode:     fast,
      }),
    });
  } catch (err) {
    if (err.name !== "AbortError") appendLog(`  ERROR: ${err.message}`, "warn");
    onFinish(); return;
  }

  let streamCompleted = false;
  await readSSE(
    response,
    processLine,
    () => { streamCompleted = true; },
    () => {}
  );
  if (!streamCompleted || state.phase === "idle" || state.waitingForUser) return;
  await waitForPlaceQueue();
  onDone();
}

// ─── Phase 1: Plan ───────────────────────────────────────────────
async function onBuild() {
  const prompt = input.value.trim();
  if (!prompt || state.phase !== "idle") return;
  input.value = "";

  state.phase       = "planning";
  state.brickCount  = 0;
  state.parts       = [];
  state.partIdx     = -1;
  state.planText    = "";
  state.lastSummary = "";
  state.buildPrompt = "";
  state.lastPrompt  = prompt;
  state.streamPhase = "idle";

  setUIBuilding(true);

  appendBlank(chatEl);
  appendChatMessage(prompt);
  appendBlank(chatEl);

  // Capture viewports and current brick state (chat workflow — keep existing bricks)
  const [rhinoBricks, views] = await Promise.all([fetchRhinoBricks(), fetchViewports()]);
  state.rhinoBricks = rhinoBricks;
  state.lastViews   = views;

  appendViewports(views);

  if (rhinoBricks.length > 0) {
    appendBuild(`  [Bridge] ${rhinoBricks.length} existing bricks on grid (kept)`, "sys");
    state.brickCount = rhinoBricks.length;
  }
  if (state.fastEnabled) {
    appendBuild("  [Fast enabled — planner stays the same, executor will use the simplified prompt]", "sys");
  }
  appendBlank(buildEl);

  // Bridge tracks per-turn bricks so finalize can translate just this turn's output.
  await postBeginTurn();

  progress.classList.add("visible");
  progressTxt.textContent = "Planning…";

  state.abortCtrl = new AbortController();
  let response;
  try {
    response = await fetch(`${SERVER_URL}/plan`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.abortCtrl.signal,
      body: JSON.stringify({
        prompt,
        brick_state: rhinoBricks,
        history: state.history.slice(-8),
        views,          // ← send viewport images to the VLM planner
      }),
    });
  } catch (err) {
    if (err.name !== "AbortError") appendLog(`  ERROR: ${err.message}`, "warn");
    onFinish(); return;
  }

  // Collect plan text while streaming
  const planLines = [];
  const origProcessLine = (text, ev) => {
    planLines.push(text);
    processLine(text, ev);
  };

  await readSSE(
    response,
    origProcessLine,
    () => onFinish(),           // [DONE] during plan phase — unexpected but handle gracefully
    () => onAwaitingApproval(planLines.join("\n")),
  );
}

// ─── Approval gate ───────────────────────────────────────────────
function onAwaitingApproval(planText) {
  state.phase    = "awaiting_approval";
  state.planText = planText;

  appendBlank(chatEl);
  appendChat("  ─────────────────────────────────────────────", "dim");
  appendChat("  Plan ready. Review above, then approve or add a clarification.", "plan-hdr");
  appendChat("  ─────────────────────────────────────────────", "dim");
  appendBlank(chatEl);

  btnApprove.style.display = "inline-block";
  progressTxt.textContent = "Awaiting your approval…";
  input.value = "";
  input.placeholder = "Type a clarification and press Enter, or click Approve & Build…";
  input.focus();
}

// ─── Phase 2: Execute (part-by-part with inspection) ─────────────
async function onApprove() {
  if (state.phase !== "awaiting_approval") return;

  const clarification = input.value.trim();
  state.phase       = "executing";
  state.streamPhase = "idle";

  input.value = "";
  input.placeholder = "Describe what to build…";
  btnApprove.style.display = "none";

  if (clarification) {
    appendChat(`  Clarification: ${clarification}`, "plan");
    appendBlank(chatEl);
  }

  const planParts = parsePlanParts(state.planText);

  if (planParts.length === 0) {
    // Fallback: no parseable parts → execute full plan in one shot
    appendBuild("  [No parts found in plan — executing in one pass…]", "plan-hdr");
    appendBlank(buildEl);
    progressTxt.textContent = state.fastEnabled ? "Executing (fast)…" : "Executing…";
    await executeFull(clarification);
    return;
  }

  appendBuild(
    `  [Executing ${planParts.length} parts with planner inspection between each${state.fastEnabled ? " — Fast enabled" : ""}]`,
    "plan-hdr",
  );
  appendBlank(buildEl);

  // Initialise part-progress dots
  state.parts = [];
  partDots.innerHTML = "";
  for (const p of planParts) {
    state.parts.push({ name: p.name, done: false });
    const d = document.createElement("div");
    d.className = "pdot";
    partDots.appendChild(d);
  }
  progress.classList.add("visible");

  for (let i = 0; i < planParts.length; i++) {
    if (state.phase === "idle") break;          // user clicked Stop
    if (state.waitingForUser) break;            // designer paused (Inspect mode)

    const part = planParts[i];
    state.partIdx = i;
    renderProgress();

    // ── Execute this part ──
    appendBlank(buildEl);
    appendBuild(`  ━━━ Part ${i+1}/${planParts.length}: ${part.name} ━━━`, "part-hdr");
    progressTxt.textContent = `Building ${i+1}/${planParts.length}: ${part.name}…`;

    state.streamPhase  = "idle";
    state.abortCtrl    = new AbortController();

    let response;
    try {
      response = await fetch(`${SERVER_URL}/execute-part`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: state.abortCtrl.signal,
        body: JSON.stringify({
          part_name:        part.name,
          part_description: part.description,
          full_plan:        state.planText,
          prompt:           state.lastPrompt,
          brick_state:      state.rhinoBricks,
          part_index:       i + 1,
          total_parts:      planParts.length,
          clarification,
          max_new_tokens:   8192,
          fast_mode:        state.fastEnabled,
        }),
      });
    } catch (err) {
      if (err.name !== "AbortError") appendLog(`  ERROR: ${err.message}`, "warn");
      break;
    }

    startActivityMonitoring();
    await readSSE(response, processLine, () => {}, () => {});
    await waitForPlaceQueue();
    if (!state.waitingForUser) stopActivityMonitoring();
    if (state.phase === "idle" || state.waitingForUser) break;

    state.parts[i].done = true;
    renderProgress();

    // ── Capture state after this part ──
    const [rhinoBricks, views] = await Promise.all([fetchRhinoBricks(), fetchViewports()]);
    state.rhinoBricks = rhinoBricks;
    state.brickCount  = rhinoBricks.length;
    brickCount.textContent = `${state.brickCount} bricks`;
    appendBlank(chatEl);
    appendViewports(views);

    // ── Planner inspection (skip after final part) ──
    if (i < planParts.length - 1) {
      appendChat(`  ┄┄┄ Planner inspecting Part ${i+1} ┄┄┄`, "inspect-hdr");
      progressTxt.textContent = `Inspecting ${i+1}/${planParts.length}…`;

      state.streamPhase = "idle";
      state.abortCtrl   = new AbortController();
      try {
        const inspectResp = await fetch(`${SERVER_URL}/inspect-part`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: state.abortCtrl.signal,
          body: JSON.stringify({
            part_name:    part.name,
            part_index:   i + 1,
            total_parts:  planParts.length,
            brick_state:  rhinoBricks,
            prompt:       state.lastPrompt,
            full_plan:    state.planText,
          }),
        });
        await readSSE(inspectResp, processInspectLine, () => {}, () => {});
      } catch (err) {
        if (err.name !== "AbortError") appendChat(`  Inspection error: ${err.message}`, "warn");
      }
      appendBlank(chatEl);
      if (state.phase === "idle" || state.waitingForUser) break;
    }
  }

  if (state.phase !== "idle" && !state.waitingForUser) onDone();
}

// Fallback: execute the full plan in one shot (used when planner output has
// no parseable <part> blocks).
async function executeFull(clarification) {
  state.abortCtrl = new AbortController();
  let response;
  try {
    response = await fetch(`${SERVER_URL}/execute`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: state.abortCtrl.signal,
      body: JSON.stringify({
        plan_text:     state.planText,
        prompt:        state.lastPrompt,
        brick_state:   state.rhinoBricks,
        clarification,
        history:       state.history.slice(-8),
        max_new_tokens: 16384,
        fast_mode:     state.fastEnabled,
      }),
    });
  } catch (err) {
    if (err.name !== "AbortError") appendLog(`  ERROR: ${err.message}`, "warn");
    onFinish(); return;
  }

  startActivityMonitoring();
  let streamCompleted = false;
  await readSSE(
    response,
    processLine,
    () => { streamCompleted = true; },
    () => {}
  );
  if (!streamCompleted || state.phase === "idle" || state.waitingForUser) return;
  await waitForPlaceQueue();
  onDone();
  if (!state.waitingForUser) stopActivityMonitoring();
}

async function onReplan() {
  if (state.phase !== "awaiting_approval") return;
  btnApprove.style.display = "none";
  state.phase = "idle";
  onFinish();
  appendChat("  Replan: edit your prompt and try again.", "sys");
}

// ─── Done ────────────────────────────────────────────────────────
// The executor (Agent 2) emits its own [Review] block at the end of the
// stream with stability counts, so we no longer auto-trigger the planner
// critical review here — it was duplicate work.
async function onDone() {
  // Flush the final buffered brick onto the queue, then wait for the whole
  // queue to drain (/place POSTs + their log lines) before printing "Done".
  // This guarantees Rhino is fully caught up with the log.
  flushPendingReason();
  await waitForPlaceQueue();
  stopBuildingStatus();
  appendBlank(buildEl);
  appendBuild(`  Done — ${state.brickCount} bricks placed.`, "sys");
  appendBlank(buildEl);

  if (state.lastPrompt) {
    state.history.push({ role: "user",      content: state.lastPrompt });
    state.history.push({ role: "assistant", content: state.lastSummary || `Placed ${state.brickCount} bricks.` });
    if (state.history.length > 20) state.history = state.history.slice(-20);
  }
  turnCount.textContent = `${Math.floor(state.history.length / 2)} turn(s)`;
  redrawRhino();
  onFinish();
}

function onFinish() {
  flushPendingReason();
  stopBuildingStatus();
  state.phase          = "idle";
  state.streamPhase    = "idle";
  state.waitingForUser = false;
  resetCurrentWorkArea();

  waitingBanner.style.display = "none";
  stopActivityMonitoring();

  if (state.partIdx >= 0 && state.partIdx < state.parts.length) {
    state.parts[state.partIdx].done = true;
  }
  renderProgress();

  btnApprove.style.display = "none";
  setUIBuilding(false);
  checkStatus();
}

function setUIBuilding(active) {
  btnBuild.disabled       = active;
  btnBuild.style.display  = active ? "none" : "inline-block";
  btnStop.style.display   = active ? "inline-block" : "none";
  // In Stage 3 / Stage 5 keep the input enabled during builds so typing triggers pause
  input.disabled          = active && state.mode === "builder";
  btnModeBuilder.disabled = active;
  if (btnModeStage3) btnModeStage3.disabled = active;
  if (btnModeStage5) btnModeStage5.disabled = active;
  if (btnModeFast) btnModeFast.disabled = active || !state.serverCaps.fast;
  if (btnDirect) btnDirect.disabled = active;
  if (!active) input.focus();
}

// ─── Stop ────────────────────────────────────────────────────────
function onStop() {
  // Prevent queued brick placements from firing to Rhino
  _stopRequested = true;
  // Signal the server to stop generating FIRST, then abort the stream.
  // keepalive ensures the POST is delivered even if the SSE connection tears down.
  fetch(`${SERVER_URL}/abort`, { method: "POST", keepalive: true }).catch(() => {});
  if (state.abortCtrl) state.abortCtrl.abort();
  btnApprove.style.display = "none";
  appendLog("  Stopped.", "warn");
  onFinish();
}

// ─── Clear ───────────────────────────────────────────────────────
async function onClear() {
  if (state.abortCtrl) state.abortCtrl.abort();
  state.waitingForUser = false;
  resetCurrentWorkArea();
  waitingBanner.style.display = "none";
  stopActivityMonitoring();
  clearBrickStream();
  await clearRhino();
  state.history      = [];
  state.chatHistory  = [];
  state.assistantBuf = [];
  state.userImages   = [];
  state.brickCount   = 0;
  renderImagePreviews();
  state.parts        = [];
  state.partIdx      = -1;
  state.phase        = "idle";
  state.currentReasonSlot = null;
  state.pickedGuid   = null;
  hideBrickDetail();
  turnCount.textContent = "";
  chatEl.innerHTML = "";
  cotEl.innerHTML = "";
  buildEl.innerHTML = "";
  _resetBuildTimeline();
  progress.classList.remove("visible");
  partDots.innerHTML = "";
  btnApprove.style.display = "none";
  setUIBuilding(false);
  appendChat("  Scene + chat cleared.", "sys");
  checkStatus();
}

// ─── events ───────────────────────────────────────────────────────
btnModeBuilder.addEventListener("click", () => setMode("builder"));
if (btnModeStage3) btnModeStage3.addEventListener("click", () => setMode("stage3"));
if (btnModeStage5) btnModeStage5.addEventListener("click", () => setMode("stage5"));
btnBuild.addEventListener("click",   onChatSend);
btnStop.addEventListener("click",    onStop);
btnClear.addEventListener("click",   onClear);
btnApprove.addEventListener("click", onChatAccept);
if (btnDirect) btnDirect.addEventListener("click", onChatDirect);
input.addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  // Shift+Enter inserts a newline in the textarea; plain Enter submits.
  if (e.shiftKey) return;
  e.preventDefault();
  if (state.phase === "idle") { onChatSend(); return; }
  if (state.phase === "awaiting_accept") {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    input.placeholder = "Describe what to build…";
    state.chatHistory.push({ role: "user", content: text });
    appendBlank(chatEl);
    appendChatMessage(text);
    appendBlank(chatEl);
    btnApprove.style.display = "none";
    state.phase = "idle";
    // Enter always continues negotiation. To start building, the designer
    // must click "Approve & Build". Only "done" phrases short-circuit to
    // a critical review here.
    if (DONE_PHRASE_RE.test(text)) {
      dispatchChat({ review: true });
      return;
    }
    dispatchChat({ buildNow: false });
    return;
  }
  // Stage 3 / Stage 5 pause: user typed a correction → abort the current
  // (paused) generation entirely, then re-dispatch through the planner
  // with the new direction.
  if (state.waitingForUser && state.mode !== "builder") {
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    if (_idleTimerId) { clearTimeout(_idleTimerId); _idleTimerId = null; }
    state.waitingForUser = false;
    waitingBanner.style.display = "none";
    stopActivityMonitoring();

    // Abort the paused generation — we're either reviewing or correcting
    fetch(`${SERVER_URL}/abort`, { method: "POST", keepalive: true }).catch(() => {});
    if (state.abortCtrl) state.abortCtrl.abort();

    appendBlank(chatEl);
    appendChatMessage(text);
    appendBlank(chatEl);

    // "enough" / "I'm done" → critical review, not a correction
    if (DONE_PHRASE_RE.test(text)) {
      appendChat("  ── Finishing build — requesting critical review… ──", "plan-hdr");
      state.chatHistory.push({ role: "user", content: text });
      state.phase = "idle";
      dispatchChat({ review: true });
      return;
    }

    appendChat("  ── Restarting with your correction… ──", "plan-hdr");

    state.lastPrompt = text;
    state.chatHistory.push({
      role: "user",
      content: "[CORRECTION] " + text + "\n[SYNTHESIZE BUILD PROMPT]",
    });
    state.phase = "idle";
    dispatchChat({ buildNow: true });
  }
});

// ─── boot ────────────────────────────────────────────────────────
checkStatus();
setInterval(checkStatus, 15000);

// Poll for the brick the designer picked in Rhino (ray-AABB on click).
pollPickedBrick();
setInterval(pollPickedBrick, 800);

if (bdClose) bdClose.addEventListener("click", clearPick);

// ═════════════════════════════════════════════════════════════════
//  ANALYSIS MODE — records mouse clicks + chat-box focus sessions,
//  renders them as bucketed bar charts, and saves each session as a
//  JSONL file under Data_collection/ on the server.
// ═════════════════════════════════════════════════════════════════
const analysis = {
  active: false,
  sessionId: null,
  startedAt: 0,
  // Flat event stream. Each item: {t_ms, stream, kind, ...detail}
  //   stream: "mouse" | "key" | "chat" | "button" | "rhino"
  //   kind:   "click"|"down"|"sample"|"move"|"up"|"focus"|"blur"|"press"|"pick_brick"|"pick_target"
  events: [],
  chatSessions: [],  // summary pairs [{start_ms, end_ms, duration_ms, chars_typed}]
  _chatFocusStart: 0,
  _chatStartValueLen: 0,
  // active-state trackers for hold sampling
  _pointerDown: false,
  _pointerHoldTimer: null,
  _keysDown: null,        // Set
  _keyHoldTimer: null,
  _chatFocused: false,
  _chatFocusTimer: null,
  // handler refs (kept so we can detach on stop)
  _clickHandler: null,
  _pointerDownHandler: null,
  _pointerMoveHandler: null,
  _pointerUpHandler: null,
  _keyDownHandler: null,
  _keyUpHandler: null,
  _focusHandler: null,
  _blurHandler: null,
  _beforeUnloadHandler: null,
  _visibilityHandler: null,
  _autoSaveTimer: null,
};

const AP_HOLD_SAMPLE_MS = 250;   // sampling rate during held mouse/key/chat-focus
const AP_CONNECT_MS = 500;       // draw a line between two events in the same stream if dt < this

async function _analysisSessionId() {
  // Ask the server for the next User-study-N name so the file ends up
  // named consistently under Data_collection/. Fall back to a timestamp
  // if the server is unreachable — we still want recording to start.
  try {
    const res = await fetch(`${SERVER_URL}/analysis_session/new_id`, {
      method: "GET",
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.session_id) return data.session_id;
    }
  } catch (_) {}
  const d = new Date();
  const p = n => String(n).padStart(2, "0");
  return `User-study-fallback-${d.getFullYear()}${p(d.getMonth()+1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function _pushEvent(stream, kind, detail) {
  if (!analysis.active) return;
  analysis.events.push({
    t_ms: Date.now() - analysis.startedAt,
    stream,
    kind,
    ...(detail || {}),
  });
}

// ── mouse ─────────────────────────────────────────────────────────
function _analysisOnPointerDown(e) {
  _pushEvent("mouse", "down", { x: e.clientX, y: e.clientY, button: e.button });
  analysis._pointerDown = true;
  if (!analysis._pointerHoldTimer) {
    analysis._pointerHoldTimer = setInterval(() => {
      if (analysis._pointerDown) _pushEvent("mouse", "sample");
    }, AP_HOLD_SAMPLE_MS);
  }
}

function _analysisOnPointerMove(e) {
  // Only record drags (pointer held down) — idle mouse motion would flood.
  if (!analysis._pointerDown) return;
  _pushEvent("mouse", "move", { x: e.clientX, y: e.clientY });
}

function _analysisOnPointerUp(e) {
  _pushEvent("mouse", "up", { x: e.clientX, y: e.clientY, button: e.button });
  analysis._pointerDown = false;
  if (analysis._pointerHoldTimer) {
    clearInterval(analysis._pointerHoldTimer);
    analysis._pointerHoldTimer = null;
  }
}

function _analysisOnClick(e) {
  const tgt = e.target || {};
  _pushEvent("mouse", "click", {
    x: e.clientX, y: e.clientY,
    targetId: tgt.id || "",
    targetTag: (tgt.tagName || "").toLowerCase(),
  });
  // If the click hit a <button>, also log it as a button-stream press so
  // the input graph shows button activity alongside keys and chat.
  const btn = tgt.closest ? tgt.closest("button") : null;
  if (btn) {
    _pushEvent("button", "press", {
      buttonId: btn.id || "",
      label: (btn.textContent || "").trim().slice(0, 40),
    });
  }
}

// ── keyboard ──────────────────────────────────────────────────────
function _analysisOnKeyDown(e) {
  _pushEvent("key", "down", { key: e.key, repeat: !!e.repeat });
  if (!analysis._keysDown) analysis._keysDown = new Set();
  analysis._keysDown.add(e.key);
  if (!analysis._keyHoldTimer) {
    analysis._keyHoldTimer = setInterval(() => {
      if (analysis._keysDown && analysis._keysDown.size > 0) {
        _pushEvent("key", "sample");
      }
    }, AP_HOLD_SAMPLE_MS);
  }
}

function _analysisOnKeyUp(e) {
  _pushEvent("key", "up", { key: e.key });
  if (analysis._keysDown) analysis._keysDown.delete(e.key);
  if ((!analysis._keysDown || analysis._keysDown.size === 0) && analysis._keyHoldTimer) {
    clearInterval(analysis._keyHoldTimer);
    analysis._keyHoldTimer = null;
  }
}

// ── chat box focus ────────────────────────────────────────────────
function _analysisOnFocus() {
  analysis._chatFocusStart = Date.now();
  analysis._chatStartValueLen = (input.value || "").length;
  analysis._chatFocused = true;
  _pushEvent("chat", "focus");
  if (!analysis._chatFocusTimer) {
    analysis._chatFocusTimer = setInterval(() => {
      if (analysis._chatFocused) _pushEvent("chat", "sample");
    }, AP_HOLD_SAMPLE_MS);
  }
}

function _analysisOnBlur() {
  if (analysis._chatFocusStart) {
    const end = Date.now();
    const duration_ms = end - analysis._chatFocusStart;
    const chars_typed = Math.max(0, (input.value || "").length - analysis._chatStartValueLen);
    analysis.chatSessions.push({
      start_ms: analysis._chatFocusStart - analysis.startedAt,
      end_ms: end - analysis.startedAt,
      duration_ms,
      chars_typed,
    });
    analysis._chatFocusStart = 0;
    analysis._chatStartValueLen = 0;
  }
  analysis._chatFocused = false;
  _pushEvent("chat", "blur");
  if (analysis._chatFocusTimer) {
    clearInterval(analysis._chatFocusTimer);
    analysis._chatFocusTimer = null;
  }
}

// ── rhino-side interaction (picked via bridge polls) ─────────────
// Called by pollPickedBrick when the seq changes — i.e. the designer
// clicked on a brick inside Rhino.
function _analysisOnRhinoBrickPick(guid) {
  _pushEvent("rhino", "pick_brick", { guid: guid || "" });
}

async function startAnalysis() {
  if (analysis.active) return;
  analysis.sessionId = await _analysisSessionId();
  analysis.active = true;
  analysis.startedAt = Date.now();
  analysis.events = [];
  analysis.chatSessions = [];
  analysis._keysDown = new Set();

  analysis._clickHandler        = _analysisOnClick;
  analysis._pointerDownHandler  = _analysisOnPointerDown;
  analysis._pointerMoveHandler  = _analysisOnPointerMove;
  analysis._pointerUpHandler    = _analysisOnPointerUp;
  analysis._keyDownHandler      = _analysisOnKeyDown;
  analysis._keyUpHandler        = _analysisOnKeyUp;
  analysis._focusHandler        = _analysisOnFocus;
  analysis._blurHandler         = _analysisOnBlur;

  analysis._beforeUnloadHandler = () => {
    // Primary persistence path on tab close — sendBeacon survives unload.
    try {
      if (analysis._chatFocusStart) _analysisOnBlur();
      const payload = _analysisPayload();
      const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
      navigator.sendBeacon(`${SERVER_URL}/analysis_session`, blob);
    } catch (_) {}
  };
  analysis._visibilityHandler = () => {
    if (document.visibilityState === "hidden") {
      try {
        if (analysis._chatFocusStart) _analysisOnBlur();
        const payload = _analysisPayload();
        const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
        navigator.sendBeacon(`${SERVER_URL}/analysis_session`, blob);
      } catch (_) {}
    }
  };

  // All listeners attach at the window/document level in the capture
  // phase so we record activity regardless of which DOM subtree currently
  // has focus. (Rhino is a separate process — its native-window events
  // aren't reachable from the browser; we only see Rhino activity when
  // the bridge poll-loops report a new seq, which is hooked separately.)
  window.addEventListener("pointerdown", analysis._pointerDownHandler, true);
  window.addEventListener("pointermove", analysis._pointerMoveHandler, true);
  window.addEventListener("pointerup",   analysis._pointerUpHandler,   true);
  window.addEventListener("click",       analysis._clickHandler,       true);
  window.addEventListener("keydown",     analysis._keyDownHandler,     true);
  window.addEventListener("keyup",       analysis._keyUpHandler,       true);
  input.addEventListener("focus", analysis._focusHandler);
  input.addEventListener("blur",  analysis._blurHandler);

  window.addEventListener("beforeunload", analysis._beforeUnloadHandler);
  window.addEventListener("pagehide",     analysis._beforeUnloadHandler);
  document.addEventListener("visibilitychange", analysis._visibilityHandler);

  // Defensive: autosave every 30s so a crash loses ≤ 30s of data.
  analysis._autoSaveTimer = setInterval(() => {
    saveAnalysisSession({ silent: true }).catch(() => {});
  }, 30000);
}

async function stopAnalysis({ save = true, silent = false } = {}) {
  if (!analysis.active) return;
  analysis.active = false;
  if (analysis._chatFocusStart) _analysisOnBlur();

  window.removeEventListener("pointerdown", analysis._pointerDownHandler, true);
  window.removeEventListener("pointermove", analysis._pointerMoveHandler, true);
  window.removeEventListener("pointerup",   analysis._pointerUpHandler,   true);
  window.removeEventListener("click",       analysis._clickHandler,       true);
  window.removeEventListener("keydown",     analysis._keyDownHandler,     true);
  window.removeEventListener("keyup",       analysis._keyUpHandler,       true);
  input.removeEventListener("focus", analysis._focusHandler);
  input.removeEventListener("blur",  analysis._blurHandler);
  window.removeEventListener("beforeunload", analysis._beforeUnloadHandler);
  window.removeEventListener("pagehide",     analysis._beforeUnloadHandler);
  document.removeEventListener("visibilitychange", analysis._visibilityHandler);

  if (analysis._pointerHoldTimer) { clearInterval(analysis._pointerHoldTimer); analysis._pointerHoldTimer = null; }
  if (analysis._keyHoldTimer)     { clearInterval(analysis._keyHoldTimer);     analysis._keyHoldTimer = null; }
  if (analysis._chatFocusTimer)   { clearInterval(analysis._chatFocusTimer);   analysis._chatFocusTimer = null; }
  if (analysis._autoSaveTimer)    { clearInterval(analysis._autoSaveTimer);    analysis._autoSaveTimer = null; }

  if (save) await saveAnalysisSession({ silent });
}

function _analysisPayload() {
  const byStream = { mouse: 0, key: 0, chat: 0, button: 0, rhino: 0 };
  for (const ev of analysis.events) {
    if (byStream[ev.stream] !== undefined) byStream[ev.stream]++;
  }
  return {
    session_id: analysis.sessionId,
    started_at_iso: new Date(analysis.startedAt).toISOString(),
    ended_at_iso: new Date().toISOString(),
    duration_ms: Date.now() - analysis.startedAt,
    event_count: analysis.events.length,
    events_by_stream: byStream,
    chat_session_count: analysis.chatSessions.length,
    total_chat_duration_ms: analysis.chatSessions.reduce((s, c) => s + c.duration_ms, 0),
    events: analysis.events,
    chat_sessions: analysis.chatSessions,
  };
}

function _payloadToJsonl(payload) {
  const lines = [];
  const header = {
    kind: "session",
    session_id: payload.session_id,
    started_at_iso: payload.started_at_iso,
    ended_at_iso: payload.ended_at_iso,
    duration_ms: payload.duration_ms,
    event_count: payload.event_count,
    events_by_stream: payload.events_by_stream,
    chat_session_count: payload.chat_session_count,
    total_chat_duration_ms: payload.total_chat_duration_ms,
  };
  lines.push(JSON.stringify(header));
  for (const e of (payload.events || [])) lines.push(JSON.stringify({ kind: "event", ...e }));
  for (const c of (payload.chat_sessions || [])) lines.push(JSON.stringify({ kind: "chat_summary", ...c }));
  return lines.join("\n") + "\n";
}

function _downloadAnalysisSession(payload) {
  const jsonl = _payloadToJsonl(payload);
  const blob = new Blob([jsonl], { type: "application/x-ndjson" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${payload.session_id}.jsonl`;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 100);
}

async function saveAnalysisSession({ silent = false } = {}) {
  if (!analysis.sessionId) return;
  const payload = _analysisPayload();
  try {
    const res = await fetch(`${SERVER_URL}/analysis_session`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json().catch(() => ({}));
    if (!silent) {
      appendBlank(chatEl);
      appendChat(`  Analysis saved → ${data.path || "Data_collection/" + analysis.sessionId + ".jsonl"}`, "sys");
    }
  } catch (err) {
    // Autosaves fail quietly — the session is still held in memory and
    // will be re-attempted on the next tick (or captured via sendBeacon
    // on tab close).
    if (silent) return;
    // Manual save clicked while the server is offline → fall back to a
    // local browser download so the session isn't lost.
    try {
      _downloadAnalysisSession(payload);
      appendBlank(chatEl);
      appendChat(`  Server unreachable — downloaded ${analysis.sessionId}.jsonl locally (move it into PSC/Data_collection/).`, "sys");
    } catch (dlErr) {
      appendChat(`  Analysis save failed: ${err.message}`, "warn");
    }
  }
}

// ── graph rendering (vanilla canvas, line graph with dots) ──
//
// Each canvas shows one or more tracks. A track is a set of events with a
// shared Y-row — events plot as dots at their timestamp, and consecutive
// events in the same track are connected by line segments when the gap
// between them is < AP_CONNECT_MS (so a held mouse or held key appears as
// a continuous line, while a lone click shows as an isolated dot).

function _drawTrackGraph(canvas, tracks, xMaxMs) {
  // tracks: [{label, color, events: [{t_ms, ...}]}]
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const padL = 88, padR = 10, padT = 12, padB = 22;
  const innerW = cssW - padL - padR;
  const innerH = cssH - padT - padB;
  const xMax = Math.max(1000, xMaxMs);

  // frame
  ctx.strokeStyle = "#4a4d55";
  ctx.lineWidth = 1;
  ctx.strokeRect(padL, padT, innerW, innerH);

  // track-row layout: evenly spaced rows, with a faint baseline per track
  const nTracks = Math.max(1, tracks.length);
  const rowStep = innerH / (nTracks + 1);
  ctx.font = "10px monospace";
  ctx.textBaseline = "middle";

  const xForT = t => padL + (Math.max(0, Math.min(xMax, t)) / xMax) * innerW;

  for (let i = 0; i < nTracks; i++) {
    const track = tracks[i];
    const y = padT + rowStep * (i + 1);

    // baseline
    ctx.strokeStyle = "#3a3d44";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);
    ctx.beginPath();
    ctx.moveTo(padL, y); ctx.lineTo(padL + innerW, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // track label (left gutter)
    ctx.fillStyle = track.color;
    ctx.textAlign = "right";
    ctx.fillText(track.label, padL - 6, y);

    const evs = track.events;
    if (!evs || evs.length === 0) continue;

    // line segments between consecutive events within AP_CONNECT_MS
    ctx.strokeStyle = track.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let prev = null;
    for (const e of evs) {
      const x = xForT(e.t_ms);
      if (prev !== null && (e.t_ms - prev.t_ms) < AP_CONNECT_MS) {
        ctx.moveTo(xForT(prev.t_ms), y);
        ctx.lineTo(x, y);
      }
      prev = e;
    }
    ctx.stroke();

    // dots on top of the lines
    ctx.fillStyle = track.color;
    for (const e of evs) {
      const x = xForT(e.t_ms);
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // x-axis ticks (5 evenly spaced timestamps)
  ctx.fillStyle = "#b0b4bc";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 5; i++) {
    const tms = (xMax * i) / 5;
    const x = padL + (i / 5) * innerW;
    const sec = (tms / 1000).toFixed(tms < 10000 ? 1 : 0);
    ctx.fillText(`${sec}s`, x, padT + innerH + 4);
  }
}

function renderAnalysisDashboard() {
  const durMs = Date.now() - analysis.startedAt;
  const durSec = Math.round(durMs / 1000);
  document.getElementById("ap-session-id").textContent = analysis.sessionId || "—";
  document.getElementById("ap-started").textContent = analysis.startedAt
    ? new Date(analysis.startedAt).toLocaleTimeString() : "—";
  document.getElementById("ap-duration").textContent = `${durSec}s`;
  document.getElementById("ap-status").textContent = analysis.active ? "RECORDING" : "STOPPED";

  // Summary counters
  let clickN = 0, mouseHoldN = 0, keyN = 0, chatN = 0, buttonN = 0, rhinoN = 0;
  for (const ev of analysis.events) {
    if (ev.stream === "mouse" && ev.kind === "click") clickN++;
    if (ev.stream === "mouse" && (ev.kind === "sample" || ev.kind === "move")) mouseHoldN++;
    if (ev.stream === "key" && (ev.kind === "down" || ev.kind === "sample")) keyN++;
    if (ev.stream === "chat") chatN++;
    if (ev.stream === "button") buttonN++;
    if (ev.stream === "rhino") rhinoN++;
  }
  document.getElementById("ap-clicks").textContent = String(clickN);
  document.getElementById("ap-chat-sessions").textContent = String(analysis.chatSessions.length);
  const totalChatMs = analysis.chatSessions.reduce((s, c) => s + c.duration_ms, 0);
  document.getElementById("ap-chat-duration").textContent = `${(totalChatMs / 1000).toFixed(1)}s`;

  // Graph 1: mouse stream broken into tracks.
  //   - "clicks" row: discrete click events only
  //   - "held"   row: down/move/sample/up events (forms a solid line while dragging)
  //   - "rhino"  row: brick-pick + target-pick events from the bridge polls
  const mouseClicks = analysis.events.filter(e => e.stream === "mouse" && e.kind === "click");
  const mouseHeld   = analysis.events.filter(e => e.stream === "mouse" && e.kind !== "click");
  const rhinoEvents = analysis.events.filter(e => e.stream === "rhino");
  _drawTrackGraph(
    document.getElementById("ap-canvas-clicks"),
    [
      { label: "mouse click", color: "#5aa8ff", events: mouseClicks },
      { label: "mouse held",  color: "#7fb8ff", events: mouseHeld },
      { label: "rhino pick",  color: "#e0a060", events: rhinoEvents },
    ],
    durMs,
  );

  // Graph 2: input streams (keyboard, chat focus, button presses).
  const keyEvents    = analysis.events.filter(e => e.stream === "key");
  const chatEvents   = analysis.events.filter(e => e.stream === "chat");
  const buttonEvents = analysis.events.filter(e => e.stream === "button");
  _drawTrackGraph(
    document.getElementById("ap-canvas-chat"),
    [
      { label: "keyboard",    color: "#9fd08a", events: keyEvents },
      { label: "chat active", color: "#7fb8ff", events: chatEvents },
      { label: "button press",color: "#e0a060", events: buttonEvents },
    ],
    durMs,
  );
}

const btnAnalysis = document.getElementById("btn-analysis");
const analysisModal = document.getElementById("analysis-modal");

// Button just opens the dashboard — recording auto-starts on page load
// and runs until the tab is closed.
let _apLiveTimer = null;
function _startLiveRedraw() {
  if (_apLiveTimer) return;
  _apLiveTimer = setInterval(() => {
    if (!analysisModal.classList.contains("visible")) return;
    renderAnalysisDashboard();
  }, 1000);
}
function _stopLiveRedraw() {
  if (_apLiveTimer) { clearInterval(_apLiveTimer); _apLiveTimer = null; }
}
btnAnalysis.addEventListener("click", () => {
  analysisModal.classList.add("visible");
  renderAnalysisDashboard();
  _startLiveRedraw();
});

document.getElementById("ap-refresh").addEventListener("click", renderAnalysisDashboard);
document.getElementById("ap-save").addEventListener("click", async () => {
  await saveAnalysisSession({ silent: false });
  renderAnalysisDashboard();
});
// Synchronous download — runs entirely within the click gesture, so no
// browser ever blocks it (no awaited fetch in the chain). Use this if
// the server is offline.
const btnApDownload = document.getElementById("ap-download");
if (btnApDownload) {
  btnApDownload.addEventListener("click", () => {
    if (!analysis.sessionId) return;
    const payload = _analysisPayload();
    try {
      _downloadAnalysisSession(payload);
      appendBlank(chatEl);
      appendChat(`  Downloaded ${analysis.sessionId}.jsonl — move it into PSC/Data_collection/.`, "sys");
    } catch (err) {
      appendChat(`  Download failed: ${err.message}`, "warn");
    }
  });
}
// Close only hides the dashboard; recording continues.
document.getElementById("ap-close").addEventListener("click", () => {
  analysisModal.classList.remove("visible");
  _stopLiveRedraw();
});
// Clicking the dim backdrop also just hides the dashboard.
analysisModal.addEventListener("click", (e) => {
  if (e.target === analysisModal) {
    analysisModal.classList.remove("visible");
    _stopLiveRedraw();
  }
});

// Manual start/stop toggle — recording does NOT auto-start. Click once
// to begin recording, click again to stop and download the JSONL file.
const btnRecord = document.getElementById("btn-record");
function _setRecordBtnLabel() {
  if (!btnRecord) return;
  if (analysis.active) {
    btnRecord.textContent = "■ Stop & Download";
    btnRecord.classList.add("recording");
  } else {
    btnRecord.textContent = "● Start Recording";
    btnRecord.classList.remove("recording");
  }
}
if (btnRecord) {
  btnRecord.addEventListener("click", async () => {
    if (!analysis.active) {
      // Start a fresh session.
      try {
        await startAnalysis();
      } catch (err) {
        appendChat(`  Analysis start failed: ${err.message}`, "warn");
      }
      _setRecordBtnLabel();
      return;
    }
    // Stop + download (synchronous download stays inside the click gesture
    // so the browser never blocks it; server save runs in the background).
    const payload = _analysisPayload();
    try {
      _downloadAnalysisSession(payload);
      appendBlank(chatEl);
      appendChat(`  Recording stopped — downloaded ${analysis.sessionId}.jsonl.`, "sys");
    } catch (err) {
      appendChat(`  Download failed: ${err.message}`, "warn");
    }
    stopAnalysis({ save: true, silent: true }).catch(() => {});
    _setRecordBtnLabel();
  });
  _setRecordBtnLabel();
}
