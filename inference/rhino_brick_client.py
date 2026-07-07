"""
BrickAgent Bridge — Rhino 8

Starts a lightweight HTTP server on localhost:8081 that the web UI talks to.
The bridge exposes the current Rhino scene (viewport images, brick list) and
accepts commands to place / clear bricks.

Run from:  Tools → ScriptEditor → Open this file → Run

Endpoints
─────────
  GET  /health      → {"status":"ok","bricks":N}
  GET  /bricks      → [{dims,x,y,z,stability}, …]
  GET  /viewport    → {"Top":"<b64>","Front":"<b64>","Right":"<b64>","Perspective":"<b64>"}
  POST /place       ← {dims,x,y,z}   → places one brick, redraws
  POST /clear       → clears BrickAgent layer, resets registry
  POST /redraw      ← {zoom?:bool}   → rs.Redraw() + optional ZoomExtents

All responses include CORS headers so the local browser page can call them.
The web UI displays the captured viewport images to the human designer; the
PSC server itself is text-only and ignores them.
"""

import base64
import json
import os
import re
import threading
import time

try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
except ImportError:
    from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler  # Py2 fallback

import rhinoscriptsyntax as rs
import Rhino
import Rhino.Display
import Rhino.UI
import scriptcontext as sc

import System
import System.IO
import System.Drawing
import System.Drawing.Imaging

# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════

BRIDGE_PORT  = 8081
# BrickAgent's grid cells are 10 cm cubes. Rhino is running in meters.
CELL_SIZE    = 0.1
LAYER_HEIGHT = 0.1
PARENT_LAYER = "BrickAgent"

# Folder next to this script where viewport PNGs are written.
# Inspect those files if you want to verify what Rhino is capturing.
# ScriptEditor may not define __file__, and os.getcwd() is usually the Rhino
# install dir — neither is reliable. Hardcode the known project folder so the
# log file ends up somewhere the user can actually find it.
_PROJECT_DIR = r"f:\STUDIO.COMPUTE\Brick_agent\brickagent_ui"
try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _SCRIPT_DIR = _PROJECT_DIR
if not os.path.isdir(_SCRIPT_DIR):
    _SCRIPT_DIR = _PROJECT_DIR
CAPTURE_DIR = os.path.join(_SCRIPT_DIR, "capture")
ERROR_LOG_PATH = os.path.join(_PROJECT_DIR, "bridge_errors.log")


def _log_line(label, msg=""):
    """Append one line to bridge_errors.log — used for both errors and
    trace breadcrumbs. Rhino's command line drops background-thread prints,
    so we persist everything interesting to disk."""
    try:
        with open(ERROR_LOG_PATH, "a") as fh:
            fh.write(
                "[{}] {} {}\n".format(
                    time.strftime("%Y-%m-%d %H:%M:%S"), label, msg
                )
            )
    except Exception:
        pass


def _log_error(label, ex):
    import traceback
    try:
        with open(ERROR_LOG_PATH, "a") as fh:
            fh.write("=" * 60 + "\n")
            fh.write("[{}] ERROR {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S"), label))
            fh.write("{}: {}\n".format(type(ex).__name__, ex))
            fh.write(traceback.format_exc() + "\n")
    except Exception:
        pass


# Startup breadcrumb — if you see this line in bridge_errors.log the new
# code has been reloaded into Rhino's ScriptEditor.
_log_line("STARTUP", "rhino_brick_client loaded, log at {}".format(ERROR_LOG_PATH))

BRICK_RE = re.compile(r"(?:PLACE\s+)?(\d+x\d+)\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")

# Names of viewports we want to capture, in display order.
# Rhino's default 4-viewport layout opens Top, Front, Right, Perspective.
# Any of these that are NOT currently open in the document are silently skipped.
_TARGET_VIEWPORTS = ["Top", "Front", "Right", "Perspective"]

# ── shared state ──────────────────────────────────────────────────
# guid string → {dims, x, y, z, stability, reason, guid, cascade_risk}
_brick_registry = {}
_registry_lock  = threading.Lock()
_interaction_state = {
    "seq": 0,
    "kind": "none",
    "ts": 0.0,
    "view": "",
    "grid_ray": None,
}
_interaction_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
#  STABILITY COLOURS
#    stable      → white   (default)
#    unstable    → yellow  (marginal support)
#    unsupported → red     (0% support above ground; always wins)
#    cascade     → orange  (supported brick whose reason flags cascade risk)
# ═══════════════════════════════════════════════════════════════════

STABILITY_COLORS = {
    "stable":      (255, 255, 255),  # white
    "weak":        (220, 180,   0),  # yellow  (server uses "weak")
    "unstable":    (220, 180,   0),  # yellow  (bridge uses "unstable")
    "unsupported": (200,  50,  50),  # red
}
CASCADE_COLOR = (224, 128, 64)  # orange, matches the UI's cascade warning
CASCADE_RISK_RE = re.compile(r"\bcascade\s+risk\b", re.IGNORECASE)


def _has_cascade_risk(reason):
    return bool(CASCADE_RISK_RE.search(str(reason or "")))


def _brick_display_color(stability, reason=""):
    key = str(stability or "").strip().lower()
    # Unsupported bricks must stay red even if their reason text mentions
    # cascade risk; orange is only for bricks that are still supported.
    if key != "unsupported" and _has_cascade_risk(reason):
        return CASCADE_COLOR
    return STABILITY_COLORS.get(key, STABILITY_COLORS["stable"])


def _apply_brick_display_color(guid_str, stability, reason=""):
    try:
        rs.ObjectColor(guid_str, _brick_display_color(stability, reason))
    except Exception as ex:
        _log_error("ObjectColor {}".format(guid_str), ex)


def _compute_stability(dims, x, y, z, registry):
    """Return 'stable', 'unstable', or 'unsupported' for a brick about to be placed."""
    if z == 0:
        return "stable"

    h, w = (int(v) for v in dims.split("x"))
    footprint = set()
    for dx in range(h):
        for dy in range(w):
            footprint.add((x + dx, y + dy))

    supported = set()
    for brick in registry.values():
        if brick["z"] != z - 1:
            continue
        bh, bw = (int(v) for v in brick["dims"].split("x"))
        for dx in range(bh):
            for dy in range(bw):
                cell = (brick["x"] + dx, brick["y"] + dy)
                if cell in footprint:
                    supported.add(cell)

    ratio = len(supported) / len(footprint) if footprint else 0
    if ratio == 0:
        return "unsupported"
    max_dim = max(h, w)
    threshold = 0.75 if max_dim >= 6 else 0.50
    return "stable" if ratio >= threshold else "unstable"


def _pt_to_grid_dict(pt):
    return {
        "x": round(pt.X / CELL_SIZE, 3),
        "y": round(pt.Y / CELL_SIZE, 3),
        "z": round(pt.Z / LAYER_HEIGHT, 3),
    }


def _viewport_point_to_line(vp, client_pt):
    """Best-effort conversion from viewport pixel to world-space frustum line."""
    try:
        line = vp.ClientToWorld(client_pt)
        if line:
            return line
    except Exception:
        pass

    try:
        ok, line = vp.GetFrustumLine(float(client_pt.X), float(client_pt.Y))
        if ok:
            return line
    except Exception:
        pass

    return None


def _set_interaction(kind, view_name, line):
    if line is None:
        return
    with _interaction_lock:
        _interaction_state["seq"] += 1
        _interaction_state["kind"] = kind
        _interaction_state["ts"] = time.time()
        _interaction_state["view"] = view_name or ""
        _interaction_state["grid_ray"] = {
            "from": _pt_to_grid_dict(line.From),
            "to": _pt_to_grid_dict(line.To),
        }


def _get_interaction_state():
    with _interaction_lock:
        data = dict(_interaction_state)
        ray = data.get("grid_ray")
        if ray:
            data["grid_ray"] = {
                "from": dict(ray["from"]),
                "to": dict(ray["to"]),
            }
        return data


# ═══════════════════════════════════════════════════════════════════
#  BRICK PICK + HIGHLIGHT
#    Mouse callback ray-intersects click line against every brick's
#    bounding box → exposes the hit guid via /picked_brick. UI calls
#    /highlight_picked to color the picked brick + its supporters
#    blue (originals cached for /restore_highlight).
# ═══════════════════════════════════════════════════════════════════

_picked_brick = {"seq": 0, "guid": None, "ts": 0.0}
_picked_lock = threading.Lock()
# Last selection guid we observed — used to bump seq only on change,
# so the UI doesn't re-run highlighting every poll.
_last_selected_guid = [None]

# guid_str -> (r,g,b) original object color, captured before an override.
_highlight_cache = {}
_highlight_lock = threading.Lock()

PICKED_COLOR    = ( 80, 200, 120)   # green — the clicked brick
SUPPORTER_COLOR = ( 60, 140, 255)   # blue  — bricks that support it


def _set_picked_brick(guid):
    with _picked_lock:
        _picked_brick["seq"] += 1
        _picked_brick["guid"] = guid  # may be None (empty-space click)
        _picked_brick["ts"] = time.time()


def _get_picked_brick():
    with _picked_lock:
        return dict(_picked_brick)


def _picked_brick_from_selection():
    """Return guid_str of the first selected brick (matching registry), or None.
    Runs on the UI thread — queries Rhino's native selection instead of
    ray-intersecting the click, which plays better with orthographic
    viewports, clipping planes, and occluded bricks."""
    try:
        selected = sc.doc.Objects.GetSelectedObjects(False, False)
    except Exception:
        return None
    if not selected:
        return None
    with _registry_lock:
        reg_keys = set(_brick_registry.keys())
    for obj in selected:
        try:
            guid_str = str(obj.Id)
        except Exception:
            continue
        if guid_str in reg_keys:
            return guid_str
    return None


def _refresh_picked_brick():
    """Query Rhino selection on UI thread, bump seq if guid changed."""
    try:
        guid = _run_on_ui_thread(
            _picked_brick_from_selection,
            timeout=2.0,
            label="picked brick selection",
        )
    except Exception:
        guid = None
    if _last_selected_guid[0] != guid:
        _last_selected_guid[0] = guid
        _set_picked_brick(guid)
    return _get_picked_brick()


def _find_supporters(guid_str):
    """Return guids of bricks at z-1 whose footprint overlaps `guid_str`."""
    with _registry_lock:
        target = _brick_registry.get(guid_str)
        if not target or int(target["z"]) == 0:
            return []
        th, tw = (int(v) for v in target["dims"].split("x"))
        tz = int(target["z"])
        footprint = set()
        for dx in range(th):
            for dy in range(tw):
                footprint.add((target["x"] + dx, target["y"] + dy))
        supporters = []
        for g, b in _brick_registry.items():
            if g == guid_str or int(b["z"]) != tz - 1:
                continue
            bh, bw = (int(v) for v in b["dims"].split("x"))
            hit = False
            for dx in range(bh):
                for dy in range(bw):
                    if (b["x"] + dx, b["y"] + dy) in footprint:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                supporters.append(g)
        return supporters


def _set_object_color(guid_str, rgb):
    """Override a brick's display color. Cache the original once."""
    try:
        prev = rs.ObjectColor(guid_str)
    except Exception:
        prev = None
    with _highlight_lock:
        if guid_str not in _highlight_cache and prev is not None:
            _highlight_cache[guid_str] = (prev.R, prev.G, prev.B)
    try:
        rs.ObjectColor(guid_str, rgb)
    except Exception as ex:
        _log_error("ObjectColor {}".format(guid_str), ex)


def _highlight_supporters_on_ui(guid_str):
    """Color the picked brick green and its supporters blue."""
    _restore_highlight_on_ui()  # clear any prior selection
    supporters = _find_supporters(guid_str)
    _set_object_color(guid_str, PICKED_COLOR)
    for g in supporters:
        _set_object_color(g, SUPPORTER_COLOR)
    try:
        rs.Redraw()
    except Exception:
        pass
    return {"picked": guid_str, "supporters": supporters}


def _restore_highlight_on_ui():
    """Restore cached original colors."""
    with _highlight_lock:
        cache = dict(_highlight_cache)
        _highlight_cache.clear()
    for g, rgb in cache.items():
        try:
            rs.ObjectColor(g, rgb)
        except Exception:
            pass
    if cache:
        try:
            rs.Redraw()
        except Exception:
            pass
    return {"restored": len(cache)}


# ═══════════════════════════════════════════════════════════════════
#  GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════

def _ensure_layer(z):
    if not rs.IsLayer(PARENT_LAYER):
        rs.AddLayer(PARENT_LAYER, color=(139, 90, 43))
    sub = "{}::Z{}".format(PARENT_LAYER, z)
    if not rs.IsLayer(sub):
        rs.AddLayer(sub, color=(139, 90, 43))
    return sub


def _clear_scene():
    if not rs.IsLayer(PARENT_LAYER):
        return 0
    objs = rs.ObjectsByLayer(PARENT_LAYER, True)
    if not objs:
        return 0
    rs.DeleteObjects(objs)
    return len(objs)


def _add_brick(dims, x, y, z, layer_name):
    """Create a Rhino box at (x,y,z) grid cells."""
    h, w = (int(v) for v in dims.split("x"))
    x1, y1, z1 = x * CELL_SIZE, y * CELL_SIZE, z * LAYER_HEIGHT
    x2, y2, z2 = (x + h) * CELL_SIZE, (y + w) * CELL_SIZE, (z + 1) * LAYER_HEIGHT
    corners = [
        (x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1),
        (x1, y1, z2), (x2, y1, z2), (x2, y2, z2), (x1, y2, z2),
    ]
    guid = rs.AddBox(corners)
    if guid:
        rs.ObjectLayer(guid, layer_name)
    return guid


def _run_on_ui_thread(fn, timeout=10.0, label="ui task"):
    """
    Run a callable on Rhino's main UI thread and wait for the result.
    Geometry creation and scene mutation are unreliable from the bridge's
    background HTTP thread.
    """
    holder = {"result": None, "error": None}
    done = threading.Event()

    def runner():
        try:
            holder["result"] = fn()
        except Exception as ex:
            holder["error"] = ex
            print("  [bridge] {} failed on UI thread: {}".format(label, ex))
            _log_error("{} (ui thread)".format(label), ex)
        finally:
            done.set()

    try:
        Rhino.RhinoApp.InvokeOnUiThread(System.Action(runner))
    except Exception as ex:
        print("  [bridge] InvokeOnUiThread dispatch failed for {}: {} — running inline".format(label, ex))
        return fn()

    if not done.wait(timeout=timeout):
        raise RuntimeError("{} timed out after {:.1f}s".format(label, timeout))

    if holder["error"] is not None:
        raise holder["error"]

    return holder["result"]


def _place_brick_on_ui(dims, x, y, z, stability, reason=""):
    color = _brick_display_color(stability, reason)
    layer = _ensure_layer(z)
    guid = _add_brick(dims, x, y, z, layer)
    placed = guid is not None
    if placed:
        rs.ObjectColor(guid, color)
        rs.Redraw()
        return {"placed": True, "guid": str(guid)}
    return {"placed": False, "guid": None}


def _clear_scene_on_ui():
    n = _clear_scene()
    rs.Redraw()
    return n


def _redraw_on_ui(zoom=True):
    rs.Redraw()
    if zoom:
        rs.ZoomExtents()
    return True


# ═══════════════════════════════════════════════════════════════════
#  VIEWPORT CAPTURE
#
#  Strategy:
#    1. Iterate every open viewport in the Rhino document and look up its
#       name (Top, Front, Right, Perspective, …). We do NOT use -SetView
#       to flip the active viewport — that returned stale views whose
#       CaptureToBitmap call always returned None.
#    2. For each target viewport, ZoomExtents and capture via the modern
#       Rhino.Display.ViewCapture API, falling back to the older instance
#       method if needed.
#    3. Save each capture as a PNG to capture/ via .NET's native Bitmap.Save,
#       then read the file back with pure Python file I/O. This bypasses
#       the CPython↔.NET bytes() interop bug entirely.
# ═══════════════════════════════════════════════════════════════════

def _ensure_capture_dir():
    if not os.path.isdir(CAPTURE_DIR):
        os.makedirs(CAPTURE_DIR)


def _save_bitmap_png(bmp, path):
    """Save a System.Drawing.Bitmap to a PNG file via .NET's native file API."""
    bmp.Save(path, System.Drawing.Imaging.ImageFormat.Png)


def _file_to_b64(path):
    """Read a PNG file and return base64-encoded contents (pure Python)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 8 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        head = data[:8].hex() if data else "<empty>"
        print("  [bridge] WARN: {} is not a valid PNG (head={}, len={})".format(
            os.path.basename(path), head, len(data)))
    return base64.b64encode(data).decode("ascii")


def _do_capture_on_ui_thread():
    """
    The actual viewport-capture work. MUST run on the Rhino main UI thread —
    CaptureToBitmap silently returns None when called from a background HTTP
    handler thread.
    """
    result = {}
    size   = System.Drawing.Size(512, 512)

    # Build {viewport_name: view} map for everything currently open
    views_by_name = {}
    for v in sc.doc.Views:
        if v is None:
            continue
        try:
            views_by_name[v.ActiveViewport.Name] = v
        except Exception:
            pass

    if not views_by_name:
        print("  [bridge] no viewports open in this document — cannot capture")
        return result

    print("  [bridge] open viewports: {}".format(sorted(views_by_name.keys())))
    _ensure_capture_dir()

    for name in _TARGET_VIEWPORTS:
        view = views_by_name.get(name)
        if view is None:
            continue
        try:
            view.ActiveViewport.ZoomExtents()
            sc.doc.Views.Redraw()
            bmp = view.CaptureToBitmap(size)
            if bmp is None:
                print("  [bridge] {}: CaptureToBitmap returned None".format(name))
                continue
            path = os.path.join(CAPTURE_DIR, "{}.png".format(name))
            try:
                _save_bitmap_png(bmp, path)
            finally:
                bmp.Dispose()
            kb = os.path.getsize(path) / 1024.0
            print("  [bridge] captured {:11s} → {} ({:.1f} KB)".format(name, path, kb))
            result[name] = _file_to_b64(path)
        except Exception as ex:
            print("  [bridge] {} capture failed: {}".format(name, ex))

    return result


def _capture_all_views():
    """
    Marshal _do_capture_on_ui_thread() onto the Rhino main UI thread and
    block the HTTP handler until it finishes. Rhino's display capture APIs
    only work from the UI thread; calling them from the bridge's background
    thread silently returns None.
    """
    holder = {"result": {}, "error": None}
    done   = threading.Event()

    def runner():
        try:
            holder["result"] = _do_capture_on_ui_thread()
        except Exception as ex:
            holder["error"] = str(ex)
            print("  [bridge] capture raised on UI thread: {}".format(ex))
        finally:
            done.set()

    try:
        Rhino.RhinoApp.InvokeOnUiThread(System.Action(runner))
    except Exception as ex:
        print("  [bridge] InvokeOnUiThread dispatch failed: {} — running inline".format(ex))
        return _do_capture_on_ui_thread()

    if not done.wait(timeout=20.0):
        print("  [bridge] capture timed out waiting for UI thread (20s)")
        return {}

    return holder["result"]


# ═══════════════════════════════════════════════════════════════════
#  CAMERA STATE (for activity detection in Inspect mode)
# ═══════════════════════════════════════════════════════════════════

def _get_camera_state_on_ui():
    """Read camera location/target for each open viewport. Must run on UI thread."""
    result = {}
    for v in sc.doc.Views:
        if v is None:
            continue
        try:
            vp = v.ActiveViewport
            cam = vp.CameraLocation
            tar = vp.CameraTarget
            result[vp.Name] = {
                "cx": round(cam.X, 3), "cy": round(cam.Y, 3), "cz": round(cam.Z, 3),
                "tx": round(tar.X, 3), "ty": round(tar.Y, 3), "tz": round(tar.Z, 3),
            }
        except Exception:
            pass
    return result


def _get_camera_state():
    """Marshal camera-state read onto the Rhino UI thread."""
    holder = {"result": {}}
    done = threading.Event()

    def runner():
        try:
            holder["result"] = _get_camera_state_on_ui()
        except Exception:
            pass
        finally:
            done.set()

    try:
        Rhino.RhinoApp.InvokeOnUiThread(System.Action(runner))
    except Exception:
        return _get_camera_state_on_ui()

    done.wait(timeout=2.0)
    return holder["result"]


# ═══════════════════════════════════════════════════════════════════════════
#  RHINO MOUSE ACTIVITY
#    Capture Rhino left-clicks as world/grid pick rays so the browser can
#    decide whether the user clicked near the agent's current work area.
# ═══════════════════════════════════════════════════════════════════════════

class _BridgeMouseCallback(Rhino.UI.MouseCallback):
    def OnEndMouseDown(self, e):
        try:
            button = getattr(e, "MouseButton", None)
            if button is None:
                button = getattr(e, "Button", None)
            if "Left" not in str(button):
                return

            view = getattr(e, "View", None)
            if view is None:
                return

            client_pt = getattr(e, "ViewportPoint", None)
            if client_pt is None:
                return

            vp = view.ActiveViewport
            line = _viewport_point_to_line(vp, client_pt)
            _set_interaction("left_click", vp.Name, line)
            # Brick picking is done via Rhino's native selection state —
            # see _picked_brick_from_selection, called by /picked_brick.
        except Exception as ex:
            print("  [bridge] mouse callback failed: {}".format(ex))


# ═══════════════════════════════════════════════════════════════════
#  HTTP BRIDGE HANDLER
# ═══════════════════════════════════════════════════════════════════

class BridgeHandler(BaseHTTPRequestHandler):

    # ── CORS + helpers ────────────────────────────────────────────

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header(
            "Vary",
            "Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Access-Control-Request-Private-Network",
        )

    def _json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def log_message(self, fmt, *args):
        pass  # silence default access log

    # ── OPTIONS (CORS pre-flight) ─────────────────────────────────

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    # ── GET ───────────────────────────────────────────────────────

    def do_GET(self):
        if self.path == "/health":
            with _registry_lock:
                count = len(_brick_registry)
            self._json_response({"status": "ok", "bricks": count})

        elif self.path == "/bricks":
            with _registry_lock:
                bricks = list(_brick_registry.values())
            self._json_response(bricks)

        elif self.path == "/camera":
            cam = _get_camera_state()
            self._json_response(cam)

        elif self.path == "/interaction":
            self._json_response(_get_interaction_state())

        elif self.path == "/picked_brick":
            self._json_response(_refresh_picked_brick())

        elif self.path == "/viewport":
            views = _capture_all_views()
            self._json_response(views)

        else:
            self._json_response({"error": "not found"}, 404)

    # ── POST ──────────────────────────────────────────────────────

    def do_POST(self):
        if self.path == "/place":
            data = self._read_body()
            dims = data.get("dims", "1x1")
            x    = int(data.get("x", 0))
            y    = int(data.get("y", 0))
            z    = int(data.get("z", 0))
            reason = str(data.get("reason", "")).strip()

            _log_line("PLACE req", "{} ({},{},{})".format(dims, x, y, z))

            with _registry_lock:
                snapshot = dict(_brick_registry)

            # Prefer server-provided stability (from ev.place.stability) for
            # consistency, fall back to local computation.
            provided = data.get("stability", "").strip().lower()
            if provided in STABILITY_COLORS:
                stability = provided
            else:
                stability = _compute_stability(dims, x, y, z, snapshot)
            _log_line("PLACE stab", "{} -> {}".format(dims, stability))
            try:
                result = _run_on_ui_thread(
                    lambda: _place_brick_on_ui(dims, x, y, z, stability, reason),
                    timeout=10.0,
                    label="place brick {}".format(dims),
                )
                _log_line("PLACE ok", "{} result={}".format(dims, result))
            except Exception as ex:
                print("  [bridge] PLACE {} ({},{},{}) failed: {}".format(dims, x, y, z, ex))
                _log_error(
                    "PLACE {} ({},{},{})".format(dims, x, y, z),
                    ex,
                )
                self._json_response({"placed": False, "stability": stability, "error": str(ex)}, 500)
                return

            placed = bool(result.get("placed"))
            guid = result.get("guid")
            if placed and guid:
                guid_str = str(guid)
                with _registry_lock:
                    _brick_registry[guid_str] = {
                        "dims": dims, "x": x, "y": y, "z": z,
                        "stability": stability,
                        "reason": reason,
                        "guid": guid_str,
                        "cascade_risk": _has_cascade_risk(reason),
                    }
                print("  [bridge] PLACE {} ({},{},{}) -> ok [{}]".format(dims, x, y, z, stability))
            else:
                print("  [bridge] PLACE {} ({},{},{}) -> no object created".format(dims, x, y, z))

            self._json_response({
                "placed": placed,
                "stability": stability,
                "guid": str(guid) if guid else None,
            })

        elif self.path == "/clear":
            with _registry_lock:
                _brick_registry.clear()
            _set_picked_brick(None)
            with _highlight_lock:
                _highlight_cache.clear()
            n = _run_on_ui_thread(_clear_scene_on_ui, timeout=10.0, label="clear scene")
            self._json_response({"cleared": n})

        elif self.path == "/redraw":
            data = self._read_body()
            _run_on_ui_thread(
                lambda: _redraw_on_ui(data.get("zoom", True)),
                timeout=10.0,
                label="redraw scene",
            )
            self._json_response({"ok": True})

        elif self.path == "/set_reason":
            data = self._read_body()
            guid_str = str(data.get("guid", "")).strip()
            reason   = str(data.get("reason", "")).strip()
            if not guid_str:
                self._json_response({"ok": False, "error": "missing guid"}, 400)
                return
            with _registry_lock:
                if guid_str in _brick_registry:
                    _brick_registry[guid_str]["reason"] = reason
                    _brick_registry[guid_str]["cascade_risk"] = _has_cascade_risk(reason)
                    stability = _brick_registry[guid_str].get("stability", "stable")
                else:
                    self._json_response({"ok": False, "error": "guid not found"}, 404)
                    return
            try:
                _run_on_ui_thread(
                    lambda: _apply_brick_display_color(guid_str, stability, reason),
                    timeout=5.0,
                    label="set reason {}".format(guid_str),
                )
            except Exception as ex:
                self._json_response({"ok": False, "error": str(ex)}, 500)
                return
            self._json_response({"ok": True, "cascade_risk": _has_cascade_risk(reason)})

        elif self.path == "/highlight_picked":
            data = self._read_body()
            guid_str = str(data.get("guid", "")).strip()
            if not guid_str:
                self._json_response({"ok": False, "error": "missing guid"}, 400)
                return
            try:
                result = _run_on_ui_thread(
                    lambda: _highlight_supporters_on_ui(guid_str),
                    timeout=5.0, label="highlight picked",
                )
            except Exception as ex:
                self._json_response({"ok": False, "error": str(ex)}, 500)
                return
            self._json_response({"ok": True, **result})

        elif self.path == "/restore_highlight":
            try:
                result = _run_on_ui_thread(
                    _restore_highlight_on_ui,
                    timeout=5.0, label="restore highlight",
                )
            except Exception as ex:
                self._json_response({"ok": False, "error": str(ex)}, 500)
                return
            self._json_response({"ok": True, **result})

        else:
            self._json_response({"error": "not found"}, 404)


# ═══════════════════════════════════════════════════════════════════
#  SERVER STARTUP
# ═══════════════════════════════════════════════════════════════════

def start_bridge():
    server = HTTPServer(("localhost", BRIDGE_PORT), BridgeHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, t


# ── run ───────────────────────────────────────────────────────────
print("")
print("=" * 52)
print("  BrickAgent Bridge")
print("  http://localhost:{}".format(BRIDGE_PORT))
print("")
print("  GET  /health       — ping")
print("  GET  /bricks       — placed brick list")
print("  GET  /camera       — viewport camera state (activity detection)")
print("  GET  /interaction  — latest Rhino left-click pick ray")
print("  GET  /viewport     — 4-view captures (saved to capture/)")
print("  GET  /picked_brick — nearest brick hit by last click (ray-AABB)")
print("  POST /place        — place one brick")
print("  POST /clear        — clear scene")
print("  POST /redraw       — redraw + zoom extents")
print("  POST /set_reason   — attach reasoning text to a brick guid")
print("  POST /highlight_picked — green + blue supporters")
print("  POST /restore_highlight — restore original brick colors")
print("=" * 52)
print("")
print("Open brickagent_ui/index.html in your browser.")
print("Keep this script running — press Stop in ScriptEditor to shut down.")
print("")

_mouse_callback = _BridgeMouseCallback()
_mouse_callback.Enabled = True
_server, _server_thread = start_bridge()
