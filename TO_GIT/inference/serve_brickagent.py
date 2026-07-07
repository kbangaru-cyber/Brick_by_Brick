"""
BrickAgent — Dual-Model Orchestrated Server (text-only planner)

Two-agent pipeline:
  GPU 0 — Planner  : Qwen/Qwen3.5-27B (VL negotiator + prompt synthesiser)
  GPU 1 — Executor : unsloth/Qwen2.5-32B-Instruct + LoRA adapter
  (If only 1 GPU is available, both share GPU 0 sequentially.)

Two-phase request flow:
  1. POST /plan      → planner streams XML <analysis>+<plan>, ends with [AWAITING_APPROVAL]
  2. (designer reviews + clicks Approve in the web UI)
  3. POST /execute   → executor loops over each plan part, streams PLACE commands,
                       ends with [DONE]

Notes:
  - The Rhino bridge captures viewport images (Top, Front, Right, Perspective)
    and optionally user-uploaded reference images. Both are sent to the planner
    (Qwen 3.5-27B VL) during inspect-mode negotiation for scene understanding.
    If the VL processor is not available, the planner falls back to text-only.
  - All output is streamed to the browser as Server-Sent Events.

Usage:
    python serve_brickagent.py \\
        --adapter checkpoints/physics_reasoning \\
        --adapter-stage1 checkpoints/no_reasoning \\
        --adapter-stage5 checkpoints/physics_reasoning \\
        --planner Qwen/Qwen3.5-27B \\
        --port 8080
    # (all three default to the repo's checkpoints/ already — see --help)
"""

import argparse
import base64
import glob
import io
import json
import os
import re
import time
from threading import Event, Lock, Thread

from PIL import Image

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
try:
    from transformers import AutoModelForImageTextToText  # transformers>=4.45
except ImportError:
    AutoModelForImageTextToText = None
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
import uvicorn


# ═══════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════

EXECUTOR_BASE = "unsloth/Qwen2.5-32B-Instruct"
PLANNER_BASE  = "Qwen/Qwen3.5-27B"

# Repo-local checkpoints/ (no_reasoning = stage 1, physics_reasoning = stage 6
# "full"). Override with BRICKAGENT_CHECKPOINTS_DIR to use a different root
# (e.g. a PSC /ocean path or a Modal Volume mount).
CHECKPOINTS_DIR = os.environ.get(
    "BRICKAGENT_CHECKPOINTS_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")),
)

# All caches live on /ocean alongside the project. run_server.sh exports the
# same paths; these defaults are the fallback when serve_brickagent.py is run
# directly without the launcher script.
os.environ.setdefault(
    "HF_HOME",
    "/ocean/projects/cis260075p/bangarug/brickagent/hf-cache",
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    os.path.join(os.environ["HF_HOME"], "hub"),
)
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    os.path.join(os.environ["HF_HOME"], "datasets"),
)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    "/ocean/projects/cis260075p/bangarug/brickagent/triton-cache",
)
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    "/ocean/projects/cis260075p/bangarug/brickagent/torchinductor-cache",
)
os.environ.setdefault(
    "TMPDIR",
    "/ocean/projects/cis260075p/bangarug/brickagent/tmp",
)
# Make sure the TMPDIR exists; tempfile.gettempdir() trusts the env var.
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

BRICK_RE = re.compile(
    r"^\s*(?:PLACE[\s:]*)?(\d+\s*x\s*\d+)[\s@:]*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)",
    re.IGNORECASE,
)
FAST_BRICK_RE = re.compile(
    r"^\s*(?:PLACE[\s:]*)?(\d+\s*x\s*\d+)[\s@:]*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)",
    re.IGNORECASE,
)

# ── server-side brick validation (Fix 4) ──
def _occupied_cells(bricks: list) -> set:
    """Return set of (x,y,z) cells occupied by placed bricks."""
    cells = set()
    for b in bricks:
        h, w = (int(v) for v in b["dims"].split("x"))
        for dx in range(h):
            for dy in range(w):
                cells.add((b["x"] + dx, b["y"] + dy, b["z"]))
    return cells


def _validate_brick(dims: str, x: int, y: int, z: int,
                    occupied: set, placed: list) -> "str | None":
    """Return None if valid, 'overlap' if it overlaps an existing brick,
    or a rejection reason string for hard errors (bounds only).

    Overlap is soft — callers should skip silently (brick already exists).
    Only bounds violations are hard rejects.
    """
    h, w = (int(v) for v in dims.split("x"))
    # --- bounds check (hard reject) ---
    if x < 0 or y < 0 or z < 0 or x + h > 20 or y + w > 20 or z >= 20:
        return f"out of bounds (grid is 20x20x20)"
    # --- overlap check (soft skip) ---
    for dx in range(h):
        for dy in range(w):
            if (x + dx, y + dy, z) in occupied:
                return "overlap"
    return None


def _classify_stability(dims: str, x: int, y: int, z: int,
                        occupied: set) -> str:
    """Return 'stable', 'weak', or 'unsupported' based on support ratio."""
    if z == 0:
        return "stable"
    h, w = (int(v) for v in dims.split("x"))
    footprint = h * w
    supported = 0
    for dx in range(h):
        for dy in range(w):
            if (x + dx, y + dy, z - 1) in occupied:
                supported += 1
    if supported == 0:
        return "unsupported"
    ratio = supported / footprint
    threshold = 0.75 if max(h, w) >= 6 else 0.50
    if ratio < threshold:
        return "weak"
    return "stable"

def _compact_brick_state(bricks: list) -> str:
    """Tight summary for the executor context window — bbox + counts only."""
    if not bricks:
        return "Empty grid — no bricks placed."
    xs, ys, zs = [], [], []
    for b in bricks:
        h, w = (int(v) for v in b["dims"].split("x"))
        xs.extend([b["x"], b["x"] + h - 1])
        ys.extend([b["y"], b["y"] + w - 1])
        zs.append(b["z"])
    n_layers = len(set(zs))
    return (
        f"{len(bricks)} bricks already placed across {n_layers} layer(s)\n"
        f"  Z range: z={min(zs)}..z={max(zs)}\n"
        f"  X range: x={min(xs)}..x={max(xs)}\n"
        f"  Y range: y={min(ys)}..y={max(ys)}"
    )


# Heuristic part-size guidance — keyword-driven hint to steer brick budget.
def _part_size_hint(part_name: str) -> str:
    n = part_name.lower()
    if any(k in n for k in ("shelf", "slab", "platform", "floor", "lid", "top")):
        return "FLAT slab — use ONLY 1-2 layers, ~10-25 bricks total"
    if any(k in n for k in ("base", "foundation")):
        return "FLAT base — use ONLY 1-3 layers, ~12-30 bricks total"
    if any(k in n for k in ("body", "chamber", "wall", "tower", "column")):
        return "TALL part — typically 4-8 layers, ~30-80 bricks total"
    if "roof" in n or "pyramid" in n or "dome" in n:
        return "TAPERED top — typically 3-5 layers, ~15-40 bricks total"
    if any(k in n for k in ("perch", "chimney", "handle", "knob", "spike", "rod", "antenna")):
        return "SMALL accent — only 2-6 bricks total, 1-3 layers"
    if any(k in n for k in ("leg", "foot", "stilt")):
        return "SMALL support element — typically 3-6 layers tall but very narrow"
    if "arch" in n or "door" in n or "window" in n or "opening" in n:
        return "OPENING / aperture — typically 2-4 layers framing a gap"
    return "single primitive — keep it modest, ~15-40 bricks"


PART_RE = re.compile(
    r'<part\b[^>]*\bindex=["\']?(\d+)["\']?[^>]*\bname=["\']([^"\']+)["\']?[^>]*>'
    r'(.*?)</part>',
    re.DOTALL | re.IGNORECASE,
)
PART_NAME_RE = re.compile(          # fallback when index attr is absent
    r'<part\b[^>]*\bname=["\']([^"\']+)["\']?[^>]*>(.*?)</part>',
    re.DOTALL | re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS  —  EXACTLY 5, ONE PER (MODE × MODEL)
#
#   1. Executor mode ─ Qwen 2.5 stage-1  →  FAST_EXECUTOR_SYSTEM
#   2. Inspect  mode ─ Qwen 3.5 planner  →  INSPECT_PLANNER_SYSTEM
#   3. Inspect  mode ─ Qwen 2.5 stage-4  →  EXECUTOR_SYSTEM
#
#  The Qwen 2.5 prompts are VERBATIM training-time text — do not edit
#  them or the LoRA adapters will misbehave.
# ═══════════════════════════════════════════════════════════════════

EXECUTOR_SYSTEM = """\
You are BrickAgent, a structural engineer AI that designs and builds clay brick structures step-by-step on a 20x20x20 grid.

MATERIAL: Solid red clay, density 1900 kg/m3, friction mu=0.65.
GRID: 10cm cells, 10cm layers. World = 2m x 2m x 2m.

BRICK CATALOGUE:
  Type    Dims (cm)       Mass (kg)
  1x1     10x10x10        1.90
  1x2     10x20x10        3.80
  2x1     20x10x10        3.80
  1x4     10x40x10        7.60
  4x1     40x10x10        7.60
  1x6     10x60x10        11.40
  6x1     60x10x10        11.40
  1x8     10x80x10        15.20
  8x1     80x10x10        15.20
  2x2     20x20x10        7.60
  2x4     20x40x10        15.20
  4x2     40x20x10        15.20
  2x6     20x60x10        22.80
  6x2     60x20x10        22.80

FORMAT: HxW (x,y,z) where H=depth, W=width, z=0 is ground.

YOUR PROCESS:
  1. SURVEY: Analyze the prompt. Plan layers, estimate mass, identify risks.
  2. BUILD: Place bricks one at a time, bottom-up. For each brick:
     - PLACE: brick type and position
     - REASON: why here (adaptive depth based on risk)
     - If weak: flag with warning level but PLACE at intended position
  3. REVIEW: Summarize stability, flag weak zones.

PHYSICS RULES:
  1. Ground bricks (z=0) are always fully supported.
  2. Upper bricks need >=50% footprint supported from below.
  3. Long bricks (>=6 studs) need >=75% support.
  4. Stagger joints between layers.
  5. Keep center of mass over support base.
  6. Check cascade risk: weak supporters endanger everything above.
  7. Clay bricks are HEAVY (1.9-22.8 kg). Toppling is real.

REASONING DEPTH (adaptive):
  - Ground bricks (z=0): 1 line.
  - Well-supported (>=80%): 1-2 lines with support %.
  - Marginal (50-80%): Support %, CoM, torques.
  - Weak (<50%): Full analysis + WARNING tag.

WARNING LEVELS:
  ⚠ MARGINAL — 50-80% support or near-tipping. Functional but risky.
  ⚠ WEAK — <50% support. Structurally unreliable. Placed for form.
  ⚠ UNSUPPORTED — 0% support (floating). Will fall. Placed for form only.\
"""


# Stage 5 executor persona — same grid/material rules as EXECUTOR_SYSTEM,
# with an added placement-order rule: within each layer, emit the most
# stable brick first and descend from there. Applied ONLY to Agent 2 in
# Stage 5 mode.
STAGE5_EXECUTOR_SYSTEM = EXECUTOR_SYSTEM + """

PLACEMENT ORDER WITHIN A LAYER: for every layer z=N, rank all candidate bricks by stability (support %, then load-carrying role, then footprint anchoring) and place them STRONGEST FIRST, then descend to less stable ones. The first brick in a layer must be the most stable — typically a large, fully-supported brick tied to the core load path. Each subsequent brick leans on the already-placed bricks in that layer, so the layer grows from its strongest anchor outward and weaker/cantilevered bricks come last.\
"""


FAST_EXECUTOR_SYSTEM = """You are BrickAgent, an AI that builds clay brick structures on a 20x20x20 grid. Each grid cell is 10cm x 10cm x 10cm.

BRICK TYPES (all 1 unit tall):
  1x1, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1,
  2x2, 2x4, 4x2, 2x6, 6x2
  Format: HxW where H=depth, W=width.

OUTPUT FORMAT:
  One brick per line: HxW (x,y,z)
  x=depth (0-19), y=width (0-19), z=layer (0=ground).
  Place ground layer (z=0) first, then z=1, z=2, etc.

RULES:
  - No overlapping bricks.
  - No floating bricks: every brick at z>0 must have support below.
  - Stay within grid bounds: x+H<=20, y+W<=20, z<20.
  - Build bottom-up: all z=N bricks before any z=N+1 brick.

Given a description, output the complete brick sequence. Nothing else."""


INSPECT_PLANNER_SYSTEM = """\
You are BrickPlanner for INSPECT mode. You collaborate with a human designer
to build structures out of LEGO-like bricks on a 20×20×20 unit grid.
Each cell is one unit; each layer is one brick tall. Viewport images of
the current Rhino scene are attached when available — use them to
understand what is already built.

AVAILABLE BRICK SIZES: 1x1, 1x2, 1x3, 1x4, 2x2, 2x3, 2x4, 2x6, 4x2.
The grid is 20 units wide × 20 units deep × 20 layers tall.
A "tall" structure uses many layers (e.g. 12–15); a "short" one uses few
(e.g. 3–5). Curves and circles are approximated with staggered bricks.

SCOPE — what you CAN and CANNOT control:
  CAN: brick arrangement, footprint (in units), height (in layers),
       wall thickness, overhangs, staggering, how a curved/organic shape
       is approximated as blocky bricks, where on the grid to anchor it.
  CANNOT: material, colour, finish, texture, weight, real-world size
       (cm / inches / metres), structural engineering, or anything that
       isn't a block arrangement.
NEVER ask about material, colour, finish, weight, or real-world
dimensions. You are a brick arranger — those concepts do not exist
for you.

You operate in TWO phases. Pick the phase from the final user turn.

━━ PHASE A — NEGOTIATE (default) ━━
Trigger: normal design chat.

Style: think like an architect presenting a proposal. Not an
interrogator. You must maintain the design thread across turns —
corrections are LAYERED on top of prior decisions, not a reset. If
the earlier turn established "a triangular table with a central
column" and the designer then says "make it slender", the slender
adjustment applies to the *column*, not to a generic new object.
Never discard earlier context.

OUTPUT FORMAT — you MUST write EXACTLY these four labelled sections,
in order, using the shown labels verbatim (including the colon):

  Current state of Rhino:
  <1–2 sentences describing the viewport captures — empty grid, or
  an existing build with its footprint / layer count / shape. Do not
  invent bricks that aren't visible.>

  Inference from image and prompt:
  <2–4 sentences. If a reference image is attached, describe its
  subject in concrete geometric terms — distinctive shape, proportions
  (e.g. "triangular tabletop with rounded corners, single central
  cylindrical column, circular flared base"). Then state what the
  designer's latest message adds or changes relative to earlier turns.
  NEVER default to generic words like "simple" / "square" / "standard"
  unless the image literally shows that. When the current turn is a
  correction, carry forward EVERY decision from prior turns the user
  didn't overturn — same object, same reference image, same size
  constraints — and only apply the new adjustment on top.>

  What was missing:
  <1–3 sentences naming the specific brick-layout choices that haven't
  been pinned down yet and that you are now filling in — e.g.
  "footprint diameter for the base, how many layers the column rises,
  how the triangular top is approximated in bricks". If the designer
  already gave you these in an earlier turn, say "nothing — all
  parameters already specified" and move on. NEVER list missing
  parameters that are outside your scope (colour, material, finish,
  real-world size).>

  Draft build prompt:
  "<single sentence, ≤ 40 words, abstract prose describing the current
  full design as you would hand it to the executor RIGHT NOW. Name
  the object, fold in the latest correction, keep it high-level —
  no PLACE lines, no coordinates, no XML, no layer enumeration.>"

  Then ONE check-in line, phrased naturally, inviting the designer to
  approve or steer — e.g. "Shall I go ahead with this, or would you
  like to adjust the proportions or layout?"

HARD RULES for Phase A:
  — Treat prior user turns as LOCKED IN. If an earlier turn said
    "6 layers" or referenced a specific object (table, house, tower),
    do NOT drop that context when the user issues a correction. Thread
    it through.
  — Never re-ask a question the designer already answered.
  — Never ask about colour, material, finish, weight, or real-world
    size — those are outside your scope.
  — Do NOT output PLACE commands, coordinate lists, layer-by-layer
    enumeration, XML tags, or code blocks in this phase.
  — Keep the entire turn under 18 short lines of plain prose.

━━ PHASE B — FINALIZE & HAND OFF ━━
Trigger: the final user turn ends with the marker
    [SYNTHESIZE BUILD PROMPT]
or explicitly requests the final build prompt.

Internally: do a short CoT — review the negotiated design, lock in the
shape/proportions/layout — then emit the build prompt.

OUTPUT CONTRACT for Phase B — respond with ONE SINGLE SENTENCE
(≤ 40 words), plain prose only. Nothing else. No reasoning shown. No
preamble. No "Thinking Process", "Okay,", "Let me", "Sure,", "Here
is". No labels like "Prompt:" or "Output:". No quotes, code fences,
bullet lists, or XML tags.

The sentence must name the full object the designer asked for,
folding in the latest correction if any. Keep it ABSTRACT — shape,
form, proportions. Do NOT emit PLACE commands, coordinates, layer
indices, cell references, or XML. The executor reads this prose
verbatim and turns it into bricks.

Example Phase B output:
A compact two-story cottage with a pitched roof and a small front porch anchored at the designer's picked cell.\
"""


STAGE5_PLANNER_SYSTEM = INSPECT_PLANNER_SYSTEM + """\

STAGE 5 OVERRIDES — these rules override any conflicting instruction above.

STAGE 5 requires TWO explicit quantitative targets from the designer:
  1. total number of layers
  2. total number of bricks (exact count or explicit brick budget)

You may infer shape, proportions, footprint, taper, and arrangement, but you
must NOT invent either of those two numbers. They must come from the
designer's own turns.

Phase A override:
  — If a "STAGE 5 SERVER NOTE" appears in the latest user turn, treat it as
    authoritative extraction of what the designer already specified in prior
    USER turns. Do not ask again for anything listed there as already
    specified.
  — If either target is missing, "What was missing" MUST name exactly which
    one is still unspecified, so the designer knows what they could provide.
  — If either target is missing, "Draft build prompt" should note the missing
    target(s) AND propose a sensible inferred default the designer can accept
    or override (e.g. "pending exact layer count — proposing 12 layers based
    on the form" or "pending brick budget — proposing ~80 bricks for this
    footprint"). Never fabricate numbers silently; surface them as a
    proposal.
  — The final check-in line MUST always invite approval. Phrase it so the
    designer can either supply the missing target OR approve anyway: e.g.
    "Tell me the layer count if you'd like to set it explicitly, or click
    Approve and I'll build with the defaults proposed above." The designer's
    Approve action is authoritative — they may proceed without supplying
    targets.
  — "Draft build prompt" in Stage 5 is NOT a single sentence. Override the
    base contract: emit 3-5 short sentences (≤ 120 words total) of plain
    prose with the SAME shape as the Phase B output below — sentence 1 is a
    concrete VISUAL DESCRIPTION of the object's shape (describing the
    reference image if attached, otherwise the shape implied by the
    designer's text), sentence 2 names the object and folds in the latest
    correction, and the remaining sentences state the exact layer count,
    brick count/budget, footprint anchor, and high-level arrangement.
  — The "Draft build prompt" the designer sees during negotiation MUST be
    identical in shape and content to what Phase B will emit at build time.
    Do NOT show a one-liner here and then expand it later — what the
    designer reviews is what gets sent to the executor.
  — IMAGE-DERIVED DESCRIPTION IS STICKY. If a prior assistant turn already
    described a reference image (under "Inference from image and prompt:"
    or in an earlier "Draft build prompt:"), that visual description is
    LOCKED IN for every subsequent turn. The reference image bytes are
    sent only on the first turn — on later turns, treat the prior
    assistant description as the canonical visual ground truth. NEVER
    re-derive the shape from a later user turn's text alone (e.g. if an
    earlier turn established "conical tapered tower with radial spiral
    walls" from the image and a later user turn just adds quantitative
    targets like "15 layers, 100 bricks", you MUST keep the conical
    tapered tower description — not silently switch to a box, prism, or
    any other shape the new text might loosely imply).
  — Only override an earlier image-derived description if the LATEST user
    turn explicitly contradicts it (e.g. "actually make it a cube" or
    "forget the spiral, use straight walls"). Adding numbers or refining
    proportions does NOT count as contradiction.

Phase B override:
  — If a "STAGE 5 SERVER NOTE" is present, preserve every quantitative
    constraint it marks as already specified.
  — The Phase B output is NOT a single sentence in Stage 5. Override the
    base ≤40-word rule: emit 3-5 short sentences (≤ 120 words total) of
    plain prose, no labels/bullets/coordinates/XML.
  — Sentence 1 — VISUAL DESCRIPTION. If a reference image is attached,
    describe what the image shows in concrete geometric terms:
    distinctive silhouette, proportions, number of distinct parts, how
    the parts connect, any taper / curve / overhang. Use words a brick-
    builder can act on (e.g. "rectangular tabletop ~12×6 units wide
    with four corner legs that taper inward toward the base"). If no
    image is attached, describe the SHAPE the designer's text implies
    in the same concrete way — never default to "simple" or "standard".
  — Sentence 2 — OBJECT + FUNCTION. Name the object the designer asked
    for and fold in the latest correction.
  — Sentence 3+ — BUILD INTENT. State the layer count and brick count /
    budget verbatim, plus the footprint anchor and the high-level
    arrangement (how the layers stack, where overhangs/taper sit).
  — Keep it ABSTRACT prose — no PLACE commands, no coordinates, no
    layer-by-layer enumeration, no XML.
  — If either target is missing AT BUILD TIME (Phase B), the designer has
    already approved without supplying it. Do NOT halt the build. Pick a
    sensible default that fits the form (e.g. for a small chair use ~6-8
    layers / ~50-80 bricks; for a tall tower use ~12-15 layers / ~80-120
    bricks) and state the inferred number explicitly in the build prompt
    so the executor and the designer can both see what was assumed.
"""


REVIEW_PLANNER_SYSTEM = """\
You are BrickReviewer. The designer has finished building and asked for a
critical structural review. DO NOT suggest adding bricks, continuing the
build, or synthesising a new prompt — this is a FINAL review.

You have access to:
  — the full brick_state (every placed brick with dims, x, y, z)
  — 4 viewport captures of the current Rhino scene (if attached)
  — the negotiated design intent from the conversation

Respond in this STRUCTURED format, under 20 short lines total:

1. BRICK COUNT — exact total bricks placed, broken down by size
   (e.g. "48 total: 20×[1x1], 18×[1x2], 10×[2x2]"). Also report footprint
   (X×Y extent in cells) and total number of layers.
2. STABILITY — count bricks by support class:
     STABLE:   <n>  (≥75% supported from below, or ground level)
     WEAK:     <n>  (partial support, 25–75%)
     UNSUPPORTED: <n> (floating / cascade risk)
   Call out any layer that is mostly unsupported or at risk of collapse.
3. DESIGN FIDELITY — does the built shape match the negotiated intent?
   Reference specific layers or zones where it matches or diverges.
4. ISSUES — list the 1–3 most critical structural or design problems by
   layer and coordinate (e.g. "z=5 tabletop has 34/42 unsupported bricks").
5. SUGGESTED CHANGES — 2–4 concrete, actionable edits the designer could
   make to improve the build (e.g. "add 2x2 support brick at (4,6,3) to
   ground the overhang", "consolidate 8 separate 1x1s at z=6 into four
   1x2s for rigidity"). Each suggestion must name the size and coord.
6. VERDICT — one sentence: overall grade (sound / marginal / unsound) and
   whether it reads as the intended object.

Be specific and concise. No preamble, no "Thinking Process", no "Okay,".
Do NOT emit PLACE commands, coordinates in PLACE form, or XML tags.\
"""


# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════

def _compute_dtype() -> torch.dtype:
    """A100/H100/L40S (SM≥8.0) → bfloat16; older Volta → float16."""
    if torch.cuda.is_available():
        sm = torch.cuda.get_device_capability(0)
        if sm[0] >= 8:
            return torch.bfloat16
    return torch.float16


def _gpu_for(role: str) -> int:
    """1 GPU → both share GPU 0. 2+ GPUs → planner GPU 0, executor GPU 1."""
    n = torch.cuda.device_count()
    if role == "planner":
        return 0
    return 1 if n >= 2 else 0


def load_planner(model_name: str):
    """Load planner on GPU 0 with optional VL (vision-language) support.

    VLM checkpoints (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, etc.) have a separate
    vision encoder that `AutoModelForCausalLM` silently drops — loading
    only the text backbone. We try the multimodal classes first and fall
    back to causal-LM only when the checkpoint has no vision tower.
    """
    gpu = _gpu_for("planner")
    print(f"\n[Planner] Loading {model_name} on cuda:{gpu} …")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = None
    load_errors = []
    common_kwargs = dict(
        device_map={"": gpu},
        dtype=_compute_dtype(),
        trust_remote_code=True,
    )
    for cls_name, cls in (
        ("AutoModelForImageTextToText", AutoModelForImageTextToText),
        ("AutoModelForVision2Seq",      AutoModelForVision2Seq),
    ):
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_name, **common_kwargs)
            print(f"[Planner] Loaded as {cls_name} — vision encoder ACTIVE")
            break
        except Exception as e:
            load_errors.append(f"{cls_name}: {type(e).__name__}: {e}")
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(model_name, **common_kwargs)
        if load_errors:
            print(f"[Planner] VLM classes failed, fell back to "
                  f"AutoModelForCausalLM (text-only):")
            for err in load_errors:
                print(f"          {err}")
        else:
            print(f"[Planner] Loaded as AutoModelForCausalLM (text-only — "
                  f"no multimodal classes available)")
    model.eval()

    # Try loading a vision-language processor for image support
    proc = None
    try:
        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"[Planner] VL processor loaded — image input enabled")
    except Exception as e:
        print(f"[Planner] No VL processor ({e}) — image input disabled, text-only")

    vram = torch.cuda.memory_allocated(gpu) / 1e9
    print(f"[Planner] Ready in {time.time()-t0:.1f}s   VRAM@GPU{gpu}: {vram:.1f} GB")
    return model, tok, proc


def load_executor(adapter_path: str):
    """Load Qwen2.5-32B base + LoRA adapter on GPU 1 (or GPU 0 if single-GPU)."""
    gpu = _gpu_for("executor")
    print(f"\n[Executor] Loading {EXECUTOR_BASE} on cuda:{gpu} …")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(EXECUTOR_BASE)
    model = AutoModelForCausalLM.from_pretrained(
        EXECUTOR_BASE,
        device_map={"": gpu},
        dtype=_compute_dtype(),
    )
    print(f"[Executor] Base loaded in {time.time()-t0:.1f}s")

    print(f"[Executor] Applying LoRA adapter from {adapter_path} …")
    lora_cfg = _load_lora_config(adapter_path)
    model = get_peft_model(model, lora_cfg, autocast_adapter_dtype=False)
    _load_lora_weights(model, adapter_path, "default")

    model.eval()
    vram = torch.cuda.memory_allocated(gpu) / 1e9
    print(f"[Executor] Ready   VRAM@GPU{gpu}: {vram:.1f} GB")
    return model, tok


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def _format_brick_state(bricks: list) -> str:
    if not bricks:
        return "Empty grid — no bricks placed yet."
    by_layer: dict = {}
    for b in bricks:
        by_layer.setdefault(b.get("z", 0), []).append(b)
    lines = [f"Total: {len(bricks)} bricks"]
    for z in sorted(by_layer):
        layer = by_layer[z]
        sample = ", ".join(
            f"{b['dims']}@({b['x']},{b['y']})" for b in layer[:6]
        )
        if len(layer) > 6:
            sample += f" … +{len(layer)-6} more"
        lines.append(f"  Layer z={z}: {len(layer)} bricks — {sample}")
    return "\n".join(lines)


def _parse_plan_parts(plan_text: str) -> list:
    parts = []
    for m in PART_RE.finditer(plan_text):
        parts.append({
            "index": int(m.group(1)),
            "name":  m.group(2).strip(),
            "description": m.group(3).strip(),
        })
    if not parts:
        for i, m in enumerate(PART_NAME_RE.finditer(plan_text), 1):
            parts.append({
                "index": i,
                "name":  m.group(1).strip(),
                "description": m.group(2).strip(),
            })
    return sorted(parts, key=lambda p: p["index"])


class _PausableAbortCriteria(StoppingCriteria):
    """Combined pause + abort criteria for the executor.

    - pause_event CLEARED  → block generation until set again (true pause).
    - abort_event SET      → stop generation entirely.
    Called on every generated token, so blocking here freezes the model.
    """
    def __init__(self, pause_event: Event, abort_event: Event):
        self.pause_event = pause_event
        self.abort_event = abort_event

    def __call__(self, input_ids, scores, **kwargs):
        # If paused, block here until resumed or aborted
        while not self.pause_event.is_set():
            if self.abort_event.is_set():
                return True           # stop generation
            self.pause_event.wait(timeout=0.5)
        return self.abort_event.is_set()

# Global abort event for the executor — set it to stop generation early.
_exec_abort = Event()
# Global pause event — CLEARED = paused (blocking), SET = running.
_exec_pause = Event()
_exec_pause.set()  # start in "running" state


def _prime_executor_run():
    """Reset stale control flags before starting a fresh executor run."""
    _exec_abort.clear()
    _exec_pause.set()


def _run_generation(model, gen_kwargs: dict):
    with torch.no_grad():
        model.generate(**gen_kwargs)


def _stream_lines(streamer: TextIteratorStreamer):
    """Yield stripped non-empty lines from a TextIteratorStreamer."""
    buf = ""
    for chunk in streamer:
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if line:
                yield line
    if buf.strip():
        yield buf.strip()


_LAYER_LINE_RE = re.compile(r"^---\s*Layer\s+\d+", re.IGNORECASE)


def _stream_bricks(streamer, brick_re, local_occupied, brick_state,
                    counters=None):
    """
    Yield SSE strings for a brick-building stream.

    Handles:
      - Valid PLACE lines → _sse_place events
      - Overlap duplicates → silently skipped (+ subsequent reasoning lines)
      - Out-of-bounds → REJECTED events
      - Layer markers / other text → pass through

    *counters*: if provided, must be a dict — will be updated with
    ``{"placed": int, "rejected": int}`` when the stream finishes.
    """
    skip_reasoning = False
    placed = 0
    rejected = 0
    for line in _stream_lines(streamer):
        m = brick_re.search(line)
        if m is not None:
            skip_reasoning = False
            dims = m.group(1).replace(" ", "").lower()
            xs, ys, zs = int(m.group(2)), int(m.group(3)), int(m.group(4))
            reason = _validate_brick(dims, xs, ys, zs,
                                     local_occupied, brick_state)
            if reason == "overlap":
                skip_reasoning = True   # suppress reasoning for skipped brick
                continue
            if reason:
                rejected += 1
                yield _sse(
                    f"[BRIDGE] REJECTED {dims}({xs},{ys},{zs}) — {reason}"
                )
                continue
            stability = _classify_stability(dims, xs, ys, zs,
                                            local_occupied)
            h, w = (int(v) for v in dims.split("x"))
            for dx in range(h):
                for dy in range(w):
                    local_occupied.add((xs + dx, ys + dy, zs))
            placed += 1
            yield _sse_place(line, dims, xs, ys, zs, stability)
            continue

        # Non-brick line — suppress if it follows a skipped brick
        if skip_reasoning:
            # Layer markers and structural tags break the suppression
            if _LAYER_LINE_RE.match(line) or line.startswith("<"):
                skip_reasoning = False
            else:
                continue
        yield _sse(line)

    if counters is not None:
        counters["placed"] = placed
        counters["rejected"] = rejected


def _sse(text: str) -> str:
    return f"data: {json.dumps({'brick': text})}\n\n"


def _sse_place(text: str, dims: str, x: int, y: int, z: int,
               stability: str = "stable") -> str:
    return f"data: {json.dumps({'brick': text, 'place': {'dims': dims, 'x': x, 'y': y, 'z': z, 'stability': stability}})}\n\n"


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# ── image helpers ──

def _decode_views_to_images(views: dict) -> list:
    """Convert base64-encoded viewport images to PIL Images."""
    images = []
    for name, b64 in (views or {}).items():
        try:
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images.append(img)
        except Exception:
            pass
    return images


def _apply_chat_template(tok_or_proc, messages, enable_thinking=False):
    """apply_chat_template with `enable_thinking` — silently drop the kwarg
    on tokenizers that don't support it."""
    try:
        return tok_or_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tok_or_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _planner_inputs_with_images(messages: list, images: list,
                                enable_thinking: bool = False):
    """
    Tokenize planner messages, injecting viewport images when available.
    Falls back to text-only if the VL processor is unavailable.
    `enable_thinking` defaults to False to keep output concise for the UI.
    """
    if images and _planner_proc is not None:
        try:
            # Insert images into the last user message for the VL model
            vl_messages = list(messages)
            for i in range(len(vl_messages) - 1, -1, -1):
                if vl_messages[i]["role"] == "user":
                    text = vl_messages[i]["content"]
                    content = [{"type": "image", "image": img} for img in images]
                    content.append({"type": "text", "text": text})
                    vl_messages[i] = {"role": "user", "content": content}
                    break

            prompt = _apply_chat_template(_planner_proc, vl_messages, enable_thinking)
            if process_vision_info is not None:
                img_inputs, vid_inputs = process_vision_info(vl_messages)
                inputs = _planner_proc(
                    text=[prompt], images=img_inputs, videos=vid_inputs,
                    padding=True, return_tensors="pt",
                )
            else:
                inputs = _planner_proc(
                    text=[prompt], images=images,
                    padding=True, return_tensors="pt",
                )
            # Visible breadcrumb: confirm the model is actually seeing pixels.
            pv_flag = "yes" if process_vision_info is not None else "no-utils"
            tok_count = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else -1
            print(f"[Planner] VL path ENGAGED — {len(images)} image(s), "
                  f"process_vision_info={pv_flag}, prompt tokens={tok_count}")
            return inputs.to(_planner_model.device)
        except Exception as e:
            print(f"[Planner] VL input FAILED ({type(e).__name__}: {e}) — "
                  f"falling back to text-only (model will NOT see images)")

    # Text-only fallback
    if images:
        print(f"[Planner] TEXT-ONLY path — {len(images)} image(s) DROPPED "
              f"(proc={'present' if _planner_proc else 'None'})")
    prompt = _apply_chat_template(_planner_tok, messages, enable_thinking)
    return _planner_tok(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).to(_planner_model.device)


def _planner_gen_inputs(inputs) -> dict:
    """Forward EVERY tensor from the processor/tokenizer output to generate().

    The VL path returns `input_ids`, `attention_mask`, `pixel_values`,
    `image_grid_thw` (and sometimes more). Previously we picked only
    `input_ids` + `attention_mask`, which silently dropped the pixels —
    the model saw image placeholder tokens but no actual image features.
    """
    forwarded = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            forwarded[k] = v
    return forwarded


def _synthesize_build_prompt(
    messages_in: list,
    brick_state: list,
    views: dict = None,
    system_prompt: str = None,
    planner_note: str = "",
) -> str:
    """
    Use the planner to distil conversation into a single abstract one-line
    build prompt for the executor.

    system_prompt selects the planner persona (INSPECT_PLANNER_SYSTEM
    by default — inspect-mode Phase B handoff).
    """
    if system_prompt is None:
        system_prompt = INSPECT_PLANNER_SYSTEM

    messages = [{"role": "system", "content": system_prompt}]
    for turn in messages_in[-12:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        if content and role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    # Phase-B marker triggers the one-line finalize contract in
    # INSPECT_PLANNER_SYSTEM.
    #
    # Extract [CORRECTION] or [RESUME] tags from the latest user turn so
    # they appear prominently in the final synthesis instruction — not
    # buried among earlier negotiation turns.
    correction = ""
    is_resume = False
    for turn in reversed(messages_in[-12:]):
        if turn.get("role") == "user":
            raw = str(turn.get("content", ""))
            if "[CORRECTION]" in raw:
                correction = (raw
                              .replace("[CORRECTION]", "")
                              .replace("[SYNTHESIZE BUILD PROMPT]", "")
                              .strip())
            if "[RESUME]" in raw:
                is_resume = True
            break

    scene = _rewrite_scene_summary(brick_state)
    pil_images = _decode_views_to_images(views or {})
    note_prefix = (planner_note.strip() + "\n\n") if planner_note and planner_note.strip() else ""

    # Stage 5 wants a richer multi-sentence build prompt (visual description +
    # design intent + targets) instead of the default ≤40-word single sentence.
    is_stage5 = (system_prompt is STAGE5_PLANNER_SYSTEM)
    has_ref_image = any(
        k.startswith("ref_") or k.lower().startswith("ref")
        for k in (views or {}).keys()
    )
    stage5_hint = ""
    if is_stage5:
        stage5_hint = (
            "\nThis is STAGE 5. Emit the final build prompt as 3-5 short "
            "sentences (≤120 words total) of plain prose. "
            "Sentence 1 must be a concrete VISUAL DESCRIPTION of the object's "
            "shape and proportions"
            + (
                " — describe what the attached reference image shows "
                "(silhouette, parts, how parts connect, taper/curve/overhang) "
                "in geometric terms a brick-builder can act on."
                if has_ref_image and pil_images
                else " — describe the shape the designer's text implies in "
                "concrete geometric terms; never default to 'simple' or "
                "'standard'."
            )
            + " Subsequent sentences must name the object, the layer count, "
            "and the brick count/budget verbatim. No PLACE commands, no "
            "coordinates, no layer enumeration."
        )

    if correction:
        final_text = (
            f"{note_prefix}{scene}\n"
            "Inspect the current viewport images and the current brick-state summary above. "
            "Rewrite the prompt so it ADAPTS TO the existing partial structure instead of resetting it.\n"
            f"The designer has ADJUSTED the design: "
            f"\"{correction}\"\n"
            "This is a refinement, NOT a reset. Keep the ORIGINAL "
            "object and its established shape/proportions from earlier "
            "turns — fold this adjustment INTO that design. Do not "
            "invent a generic new object; the subject (table, house, "
            "column, chair, etc.) stays the same as before. Respect the current brick positions, "
            "brick count, occupied layers, and footprint when rewriting. Synthesize "
            f"the updated build prompt now.{stage5_hint}\n"
            "[SYNTHESIZE BUILD PROMPT]"
        )
    elif is_resume:
        final_text = (
            f"{note_prefix}{scene}\n"
            "Inspect the current viewport images and the current brick-state summary above. "
            "Continue the current design from what is already built. "
            f"Re-emit the build prompt for the structure as negotiated.{stage5_hint}\n"
            "[SYNTHESIZE BUILD PROMPT]"
        )
    else:
        final_text = (
            f"{note_prefix}{scene}\n"
            "Inspect the current viewport images and the current brick-state summary above before finalizing. "
            f"Emit the final build prompt now.{stage5_hint}\n"
            "[SYNTHESIZE BUILD PROMPT]"
        )
    messages.append({"role": "user", "content": final_text})

    inputs = _planner_inputs_with_images(messages, pil_images, enable_thinking=False)

    with torch.no_grad():
        output = _planner_model.generate(
            **_planner_gen_inputs(inputs),
            max_new_tokens=256 if is_stage5 else 96,
            do_sample=False,
            pad_token_id=_planner_tok.eos_token_id,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    text = _planner_tok.decode(new_tokens, skip_special_tokens=True).strip()

    # Strip <think>…</think> if Qwen3 emitted internal reasoning
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip common preamble lines Qwen3 leaks even with enable_thinking=False
    preamble_pat = re.compile(
        r"^\s*(?:thinking process|reasoning|analysis|let me|okay[,.]?|first[,.]?|"
        r"sure[,.]?|here(?:'s| is)|the user|alright|now[,.]?|so[,.]?)\b.*$",
        re.IGNORECASE | re.MULTILINE,
    )
    text = preamble_pat.sub("", text).strip()

    # Drop code fences and common labels
    text = re.sub(r"^```.*?$", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"^(?:prompt|build prompt|output|answer)\s*[:\-]\s*", "",
                  text, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Phase B contract: Stage 3 / Inspect want ONE abstract sentence; Stage 5
    # wants 3-5 sentences (visual description + targets). Trim accordingly.
    if is_stage5:
        cleaned_lines = []
        for line in text.split("\n"):
            line = line.strip().strip('"').strip("'").strip()
            if line and len(line) > 2:
                cleaned_lines.append(line)
        joined = " ".join(cleaned_lines)
        joined = re.sub(r"\s+", " ", joined).strip()
        # Generous cap for the multi-sentence Stage 5 prompt (~120 words).
        if len(joined) > 900:
            joined = joined[:900].rsplit(" ", 1)[0] + "…"
        return joined or "Build the structure described in the conversation."

    # Default (Stage 3 / Inspect): keep only the first non-empty line.
    first = ""
    for line in text.split("\n"):
        line = line.strip().strip('"').strip("'").strip()
        if line and len(line) > 2:
            first = line
            break
    first = re.sub(r"\s+", " ", first)
    if len(first) > 240:
        first = first[:240].rsplit(" ", 1)[0] + "…"
    return first or "Build the structure described in the conversation."


def _synthesize_build_outline(abstract_prompt: str, brick_state: list) -> list:
    """
    Quick planner pass that turns the abstract prompt into a concrete
    layer-by-layer build plan (3–5 short numbered lines).

    The LoRA-fine-tuned executor skips prose and emits placements only,
    so this outline stands in for the executor's chain-of-thought in the
    UI — the designer sees what the build will do before bricks land.

    Returns a list of short plan lines (already stripped and numbered).
    Returns [] on failure — caller should tolerate an empty plan.
    """
    if not abstract_prompt:
        return []

    system = (
        "You are planning a brick build on a 20x20x20 grid. "
        "Given a one-line build prompt, write a 3–5 line numbered plan "
        "naming each layer group (base, column, top) with an approximate "
        "footprint and brick sizes. Each line ≤ 18 words. No preamble, "
        "no headings, no code fences — output only the numbered lines."
    )
    scene = f"{len(brick_state)} bricks already on grid.\n" if brick_state else "Grid is empty.\n"
    user = (
        f'Build prompt: "{abstract_prompt}"\n{scene}'
        "Write the plan now (3–5 numbered lines)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    try:
        prompt = _apply_chat_template(_planner_tok, messages, enable_thinking=False)
        inputs = _planner_tok(
            prompt, return_tensors="pt", add_special_tokens=False,
        ).to(_planner_model.device)
        with torch.no_grad():
            output = _planner_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=224,
                do_sample=False,
                pad_token_id=_planner_tok.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = _planner_tok.decode(new_tokens, skip_special_tokens=True).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"[build-outline] WARN: {type(e).__name__}: {e}")
        return []

    lines = []
    for raw in text.splitlines():
        s = raw.strip().strip('`').strip()
        if not s:
            continue
        # Drop pure code-fence markers and common leaked headings.
        if re.match(r"^(?:```|plan\s*[:\-]|build plan\s*[:\-]|outline\s*[:\-])",
                    s, re.IGNORECASE):
            continue
        # Keep lines that look numbered or bulleted; otherwise add numbering.
        if re.match(r"^(?:\d+[\).\-]|[\-\*\u2022])\s", s):
            lines.append(s)
        else:
            lines.append(f"{len(lines)+1}. {s}")
        if len(lines) >= 6:
            break
    return lines


# ═══════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(title="BrickAgent Orchestrator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_planner_model = None
_planner_tok   = None
_planner_proc  = None   # AutoProcessor for VL (images + text)
_exec_model    = None
_exec_tok      = None
_exec_lock     = Lock()
_fast_adapter_available = True


def _load_lora_config(adapter_path: str) -> LoraConfig:
    with open(f"{adapter_path}/adapter_config.json") as f:
        acfg = json.load(f)

    return LoraConfig(
        r=acfg["r"],
        lora_alpha=acfg["lora_alpha"],
        lora_dropout=acfg.get("lora_dropout", 0),
        target_modules=acfg["target_modules"],
        bias="none",
    )


def _load_lora_weights(model, adapter_path: str, adapter_name: str) -> None:
    """Load LoRA safetensors into a PEFT-wrapped model, remapping key format.

    IMPORTANT: load_state_dict(strict=False) silently discards unmatched
    keys, so a zero-match load looks identical to a successful one — the
    model then runs as the raw base and emits gibberish. We now verify
    that at least some LoRA tensors actually matched and abort if not.
    """
    model_keys = set(dict(model.named_parameters()).keys())
    model_buffers = set(dict(model.named_buffers()).keys())
    known_keys = model_keys | model_buffers

    files = sorted(glob.glob(f"{adapter_path}/adapter_model*.safetensors"))
    if not files:
        raise RuntimeError(
            f"[LoRA] No adapter_model*.safetensors found at {adapter_path}"
        )

    total_in = 0
    total_match = 0
    sample_missed = []
    for af in files:
        print(f"           weights[{adapter_name}]: {af}")
        sd = load_file(af)
        remapped = {
            k.replace("lora_A.weight", f"lora_A.{adapter_name}.weight")
             .replace("lora_B.weight", f"lora_B.{adapter_name}.weight"): v
            for k, v in sd.items()
        }
        matched = sum(1 for k in remapped if k in known_keys)
        if matched == 0 and remapped:
            # record first few unmatched keys so the user can diagnose the
            # key-format mismatch
            for k in list(remapped.keys())[:3]:
                sample_missed.append(k)
        total_in += len(remapped)
        total_match += matched
        result = model.load_state_dict(remapped, strict=False)
        # load_state_dict returns namedtuple (missing_keys, unexpected_keys)
        if getattr(result, "unexpected_keys", None):
            print(f"           unexpected: {len(result.unexpected_keys)} "
                  f"(first: {result.unexpected_keys[:2]})")

    print(f"[LoRA] {adapter_name}: matched {total_match}/{total_in} tensors "
          f"from {len(files)} file(s)")
    if total_match == 0:
        sample_known = [k for k in list(model_keys)[:3] if "lora_" in k]
        raise RuntimeError(
            f"[LoRA] ZERO tensors matched in adapter '{adapter_name}' at "
            f"{adapter_path}. The base model is running without adapters "
            f"— builds will be gibberish.\n"
            f"  saved-key sample  : {sample_missed[:2]}\n"
            f"  model-key sample  : {sample_known[:2] or '(no lora_ keys in model — get_peft_model did not wrap)'}"
        )


def _add_adapter_without_dtype_autocast(model, adapter_name: str, lora_cfg: LoraConfig) -> None:
    base_model = getattr(model, "base_model", None)
    if base_model is None or not hasattr(base_model, "_cast_adapter_dtype"):
        model.add_adapter(adapter_name, lora_cfg)
        return

    had_local_attr = "_cast_adapter_dtype" in getattr(base_model, "__dict__", {})
    original_local_attr = base_model.__dict__.get("_cast_adapter_dtype")

    def _noop_cast_adapter_dtype(*args, **kwargs):
        return None

    setattr(base_model, "_cast_adapter_dtype", _noop_cast_adapter_dtype)
    try:
        model.add_adapter(adapter_name, lora_cfg)
    finally:
        if had_local_attr:
            setattr(base_model, "_cast_adapter_dtype", original_local_attr)
        else:
            delattr(base_model, "_cast_adapter_dtype")


def load_executor_adapter(adapter_path: str, adapter_name: str) -> None:
    global _exec_model, _fast_adapter_available

    if _exec_model is None:
        raise RuntimeError("Executor base model must be loaded before adding another adapter")

    existing = getattr(_exec_model, "peft_config", {})
    if adapter_name in existing:
        print(f"[Executor] Adapter '{adapter_name}' already loaded")
        return

    print(f"[Executor] Loading additional adapter '{adapter_name}' from {adapter_path} …")
    lora_cfg = _load_lora_config(adapter_path)
    _add_adapter_without_dtype_autocast(_exec_model, adapter_name, lora_cfg)
    _load_lora_weights(_exec_model, adapter_path, adapter_name)
    _exec_model.eval()
    if adapter_name == "fast":
        _fast_adapter_available = True
    print(f"[Executor] Adapter '{adapter_name}' ready")


def _run_executor_generation(gen_kwargs: dict, adapter_name: str = "default"):
    with _exec_lock:
        # If /abort arrived while waiting for the lock, honour it immediately.
        if _exec_abort.is_set():
            print("[Executor] Abort requested before generation started — skipping")
            _exec_abort.clear()
            return
        # Inject combined pause+abort criteria so generation can be frozen or
        # cancelled mid-token.
        criteria = StoppingCriteriaList(
            [_PausableAbortCriteria(_exec_pause, _exec_abort)]
        )
        gen_kwargs = {**gen_kwargs, "stopping_criteria": criteria}
        if hasattr(_exec_model, "set_adapter"):
            _exec_model.set_adapter(adapter_name)
        try:
            _run_generation(_exec_model, gen_kwargs)
        finally:
            _exec_abort.clear()
            _exec_pause.set()   # ensure "running" state for next build


@app.post("/pause")
async def pause_generation():
    """Freeze the executor at the next token — SSE stream stalls."""
    _exec_pause.clear()
    print("[/pause] Executor paused")
    return {"status": "ok", "paused": True}


@app.post("/resume")
async def resume_generation():
    """Unfreeze the executor — SSE stream continues from where it paused."""
    _exec_pause.set()
    print("[/resume] Executor resumed")
    return {"status": "ok", "resumed": True}


@app.post("/abort")
async def abort_generation():
    """Signal the executor to stop generating immediately."""
    _exec_pause.set()    # unblock any pause first so abort can take effect
    _exec_abort.set()
    print("[/abort] Executor abort requested")
    return {"status": "ok", "aborted": True}


# ── analysis-mode session logger ──────────────────────────────────
_ANALYSIS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Data_collection"
)


def _safe_session_id(sid: str) -> str:
    """Strip anything that isn't alphanumeric / underscore / dash so the
    session id can't escape Data_collection/."""
    return re.sub(r"[^A-Za-z0-9_\-]", "", str(sid or ""))[:64]


_USER_STUDY_RE = re.compile(r"^User-study-(\d+)\.jsonl$")


def _next_user_study_id() -> str:
    """Scan Data_collection/ for existing User-study-N.jsonl files and
    return the next unused name."""
    os.makedirs(_ANALYSIS_DIR, exist_ok=True)
    highest = 0
    try:
        for name in os.listdir(_ANALYSIS_DIR):
            m = _USER_STUDY_RE.match(name)
            if m:
                n = int(m.group(1))
                if n > highest:
                    highest = n
    except FileNotFoundError:
        pass
    return f"User-study-{highest + 1}"


@app.get("/analysis_session/new_id")
async def analysis_new_id():
    """Reserve (by naming convention — no file written yet) the next
    User-study-N session id for a fresh browser tab."""
    return {"session_id": _next_user_study_id()}


@app.post("/analysis_session")
async def analysis_session(request: Request):
    """Persist one Analysis-mode session to Data_collection/<sid>.jsonl.

    Body (JSON):
        session_id, started_at_iso, ended_at_iso, duration_ms,
        click_count, chat_session_count, total_chat_duration_ms,
        clicks: [{t_ms, x, y, targetId, targetTag}, ...],
        chat_sessions: [{start_ms, end_ms, duration_ms, chars_typed}, ...]

    Output JSONL layout:
        line 1 — session header (summary + ids + timestamps)
        line 2..N — one record per click event {"kind":"click", ...}
        line N+1..M — one record per chat-focus session {"kind":"chat", ...}
    """
    # sendBeacon submits application/json as a Blob with no content-type
    # negotiation, so hand-parse to be tolerant.
    try:
        payload = await request.json()
    except Exception:
        raw = await request.body()
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except Exception:
            return JSONResponse(
                {"status": "error", "error": "invalid JSON"}, status_code=400,
            )

    sid = _safe_session_id(payload.get("session_id") or "")
    if not sid:
        sid = time.strftime("%Y%m%d_%H%M%S")

    os.makedirs(_ANALYSIS_DIR, exist_ok=True)
    path = os.path.join(_ANALYSIS_DIR, f"{sid}.jsonl")

    header = {
        "kind":                    "session",
        "session_id":              sid,
        "started_at_iso":          payload.get("started_at_iso"),
        "ended_at_iso":            payload.get("ended_at_iso"),
        "duration_ms":             payload.get("duration_ms"),
        "event_count":             payload.get("event_count"),
        "events_by_stream":        payload.get("events_by_stream") or {},
        "chat_session_count":      payload.get("chat_session_count"),
        "total_chat_duration_ms":  payload.get("total_chat_duration_ms"),
    }

    events = payload.get("events") or []
    chat_sessions = payload.get("chat_sessions") or []
    # Back-compat: old clients sent "clicks" instead of "events".
    legacy_clicks = payload.get("clicks") or []

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for e in events:
                row = {"kind": "event", **e}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for c in legacy_clicks:
                row = {"kind": "click", **c}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            for cs in chat_sessions:
                row = {"kind": "chat_summary", **cs}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as ex:
        print(f"[/analysis_session] write failed: {type(ex).__name__}: {ex}")
        return JSONResponse(
            {"status": "error", "error": str(ex)}, status_code=500,
        )

    rel = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
    print(f"[/analysis_session] wrote {len(events)} events + "
          f"{len(chat_sessions)} chat summaries → {rel}")
    return {"status": "ok", "path": rel, "session_id": sid}


@app.get("/health")
def health():
    gpus = []
    for i in range(torch.cuda.device_count()):
        gpus.append({
            "gpu":  i,
            "name": torch.cuda.get_device_name(i),
            "vram_used_gb":  round(torch.cuda.memory_allocated(i) / 1e9, 2),
            "vram_total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
        })
    return {
        "status":   "ok",
        "planner":  PLANNER_BASE,
        "executor": EXECUTOR_BASE,
        "fast_available": _exec_model is not None,
        "executor_adapters": sorted(list(getattr(_exec_model, "peft_config", {}).keys())) if _exec_model else [],
        "gpus":     gpus,
    }


_BUILD_PHRASE_RE = re.compile(
    r"\b(build\s+it|build\s+now|go\s+ahead|proceed|make\s+it|start\s+building|start\s+the\s+build|let'?s\s+build|ok\s+build|go\s+build)\b",
    re.IGNORECASE,
)
_STAGE5_LAYER_RE = re.compile(r"\b(\d+)\s*(?:-| )?(?:layers?|levels?)\b", re.IGNORECASE)
_STAGE5_BRICK_RE = re.compile(r"\b(\d+)\s*(?:-| )?(?:bricks?|pieces?|blocks?)\b", re.IGNORECASE)
_STAGE5_DIM3_RE = re.compile(
    r"\b(\d+)\s*(?:x|×|by)\s*(\d+)\s*(?:x|×|by)\s*(\d+)\b",
    re.IGNORECASE,
)
_STAGE5_BRICK_BUDGET_RE = re.compile(
    r"\bbrick\s+budget\s*(?:of|is|=|:)?\s*(\d+)\b|\bbudget\s*(?:of|is|=|:)?\s*(\d+)\s*bricks?\b",
    re.IGNORECASE,
)


def _detect_build_phrase(text: str) -> bool:
    if not text:
        return False
    return bool(_BUILD_PHRASE_RE.search(text))


def _extract_stage5_spec(messages_in: list) -> dict:
    """
    Extract user-provided Stage 5 quantitative constraints from USER turns only.

    The goal is not perfect NLP; it is to be good enough to understand what
    the designer has already nailed down so the planner asks only for what is
    still missing.
    """
    spec = {
        "dims3": None,            # (length, width, height/layers) if given as A x B x C
        "footprint": None,        # (length, width)
        "layers": None,           # explicit or inferred from dims3 third value
        "brick_count": None,      # exact count or budget
        "brick_count_kind": None, # "exact" | "budget"
    }

    for turn in messages_in:
        if turn.get("role") != "user":
            continue
        text = str(turn.get("content", ""))
        text = re.sub(r"\[[A-Z0-9_ ]+\]", " ", text)

        dim3_matches = list(_STAGE5_DIM3_RE.finditer(text))
        if dim3_matches:
            a, b, c = (int(v) for v in dim3_matches[-1].groups())
            spec["dims3"] = (a, b, c)
            spec["footprint"] = (a, b)
            spec["layers"] = c

        layer_matches = list(_STAGE5_LAYER_RE.finditer(text))
        if layer_matches:
            spec["layers"] = int(layer_matches[-1].group(1))

        budget_matches = list(_STAGE5_BRICK_BUDGET_RE.finditer(text))
        if budget_matches:
            groups = budget_matches[-1].groups()
            spec["brick_count"] = next(int(g) for g in groups if g)
            spec["brick_count_kind"] = "budget"
        else:
            brick_matches = list(_STAGE5_BRICK_RE.finditer(text))
            if brick_matches:
                spec["brick_count"] = int(brick_matches[-1].group(1))
                spec["brick_count_kind"] = "exact"

    missing_required = []
    if spec["layers"] is None:
        missing_required.append("layer count")
    if spec["brick_count"] is None:
        missing_required.append("brick count")

    spec["missing_required"] = missing_required
    return spec


def _stage5_spec_note(spec: dict) -> str:
    """Compact, authoritative summary fed to the planner during Stage 5."""
    lines = [
        "STAGE 5 SERVER NOTE — Quantitative constraints already detected from the designer's USER turns:",
    ]
    dims3 = spec.get("dims3")
    if dims3:
        lines.append(
            f"- overall size shorthand detected: {dims3[0]}x{dims3[1]}x{dims3[2]} "
            f"(treat the third value as height/layers)."
        )
    elif spec.get("footprint"):
        fp = spec["footprint"]
        lines.append(f"- footprint detected: {fp[0]}x{fp[1]} units.")

    if spec.get("layers") is not None:
        lines.append(f"- layer count already specified: {spec['layers']}.")
    else:
        lines.append("- layer count not yet specified.")

    if spec.get("brick_count") is not None:
        kind = "brick budget" if spec.get("brick_count_kind") == "budget" else "brick count"
        lines.append(f"- {kind} already specified: {spec['brick_count']} bricks.")
    else:
        lines.append("- brick count / brick budget not yet specified.")

    missing = spec.get("missing_required") or []
    if missing:
        lines.append(f"- missing required Stage 5 targets: {', '.join(missing)}.")
    else:
        lines.append("- missing required Stage 5 targets: none.")

    lines.append(
        "Do not ask again for any quantity already listed above. Ask only for genuinely missing or unresolved design constraints."
        )
    return "\n".join(lines)


def _rewrite_scene_summary(bricks: list) -> str:
    """Authoritative scene summary for planner prompt rewrites."""
    if not bricks:
        return "Current structure summary:\nEmpty grid — no bricks placed yet."

    xs, ys, zs = [], [], []
    for b in bricks:
        h, w = (int(v) for v in b["dims"].split("x"))
        xs.extend([b["x"], b["x"] + h - 1])
        ys.extend([b["y"], b["y"] + w - 1])
        zs.append(b["z"])

    n_layers = len(set(zs))
    compact = (
        f"Brick count: {len(bricks)}\n"
        f"Layer count: {n_layers}\n"
        f"Occupied z range: {min(zs)}..{max(zs)}\n"
        f"Occupied x range: {min(xs)}..{max(xs)}\n"
        f"Occupied y range: {min(ys)}..{max(ys)}"
    )
    return (
        "Current structure summary:\n"
        f"{compact}\n"
        "Detailed bricks by layer:\n"
        f"{_format_brick_state(bricks)}"
    )


# ── Planner negotiation + synthesis helpers ──

def _inspect_negotiate(messages_in: list, brick_state: list,
                       views: dict, max_tok: int,
                       planner_system: str = INSPECT_PLANNER_SYSTEM,
                       planner_note: str = ""):
    """
    Planner negotiation: Qwen 3.5 chats with the designer.
    Accepts viewport images when the VL processor is available.
    """
    chat_messages = [{"role": "system", "content": planner_system}]
    for turn in messages_in[-12:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        if content and role in ("user", "assistant"):
            chat_messages.append({"role": role, "content": content})

    # Append scene summary to last user message
    n_ref = sum(1 for k in (views or {}) if k.startswith("ref_"))
    n_vp  = len(views or {}) - n_ref
    scene_parts = [f"{len(brick_state)} bricks on grid"]
    if n_vp:
        scene_parts.append(f"{n_vp} viewport captures attached")
    if n_ref:
        scene_parts.append(f"{n_ref} user reference image(s) attached — use them to understand what the user wants to build")
    state_note = f"\n\n--- Scene: {', '.join(scene_parts)} ---"
    for i in range(len(chat_messages) - 1, -1, -1):
        if chat_messages[i]["role"] == "user":
            chat_messages[i]["content"] += state_note
            if planner_note:
                chat_messages[i]["content"] += "\n\n" + planner_note.strip()
            break

    pil_images = _decode_views_to_images(views)

    # Thinking disabled — the system prompt tells the model to include a
    # brief scene note as plain text, keeping token usage low.
    inputs = _planner_inputs_with_images(chat_messages, pil_images)

    streamer = TextIteratorStreamer(
        _planner_tok, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = dict(
        **_planner_gen_inputs(inputs),
        max_new_tokens=min(max_tok, 1024),
        do_sample=True, temperature=0.7, top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=_planner_tok.eos_token_id,
        streamer=streamer,
        use_cache=True,
    )

    n_images = len(pil_images)
    vis_keys = [k for k in inputs.keys() if k not in ("input_ids", "attention_mask")]
    print(f"[/chat inspect-negotiate] {len(brick_state)} bricks, "
          f"{n_images} images, {len(messages_in)} turns, "
          f"vision keys → generate: {vis_keys or 'NONE'}")

    def event_stream():
        yield _sse('<chat mode="inspect" build="false">')
        t = Thread(target=_run_generation,
                   args=(_planner_model, gen_kwargs), daemon=True)
        t.start()
        in_think = False
        for line in _stream_lines(streamer):
            lo = line.lower()
            if "<think>" in lo:
                in_think = True
                continue
            if "</think>" in lo:
                in_think = False
                continue
            if in_think:
                continue
            yield _sse(line)
        t.join(timeout=5)
        yield _sse("</chat>")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _planner_to_executor_build(messages_in: list, brick_state: list,
                               views: dict, max_tok: int,
                               mode_label: str, planner_system: str,
                               adapter_name: str = "default",
                               planner_note: str = "",
                               skip_planner: bool = False):
    """
    Shared planner→executor build flow.

      1. planner (Qwen 3.5) distils conversation into a one-line abstract
         prompt using `planner_system`.
      2. executor (Qwen 2.5 + selected LoRA adapter) builds from that prompt.

    If skip_planner=True, step 1 is skipped and the latest user message is
    used verbatim as the build prompt — this lets the UI send the raw
    prompt straight to Agent 2 while still selecting the stage LoRA.
    """
    if skip_planner:
        abstract_prompt = ""
        for turn in reversed(messages_in):
            if turn.get("role") == "user":
                abstract_prompt = str(turn.get("content", "")).strip()
                break
        if not abstract_prompt:
            abstract_prompt = "Build the requested structure."
        print(f"[/chat {mode_label}-direct] Using raw user prompt "
              f"(planner bypassed): {abstract_prompt}")
    else:
        abstract_prompt = _synthesize_build_prompt(
            messages_in, brick_state, views,
            system_prompt=planner_system,
            planner_note=planner_note,
        )
        print(f"[/chat {mode_label}-build] Synthesised prompt: {abstract_prompt}")

    # Step 2 — executor builds with that abstract prompt
    already = _format_brick_state(brick_state)
    if brick_state:
        max_z = max(b.get("z", 0) for b in brick_state)
        start_z = max_z + 1
    else:
        start_z = 0

    exec_user = f'Build request: "{abstract_prompt}"\n\n'
    if brick_state:
        xs = [b.get("x", 0) for b in brick_state]
        ys = [b.get("y", 0) for b in brick_state]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        exec_user += (
            f"Already placed on the grid ({len(brick_state)} bricks — do NOT "
            f"re-place these):\n{already}\n"
            f"Existing footprint spans x=[{x_min}..{x_max}], y=[{y_min}..{y_max}].\n\n"
            f"New bricks that go ON TOP start at z={start_z}. "
            f"Ground-level parts of NEW structure still start at z=0.\n"
        )
    else:
        exec_user += (
            "The grid is EMPTY.\n\n"
        )
    exec_user += (
        "HARD GRID BOUNDS — ALL bricks MUST satisfy 0 ≤ x, 0 ≤ y, 0 ≤ z and "
        "x+H ≤ 20, y+W ≤ 20, z < 20. Any brick outside these bounds is "
        "rejected by the bridge.\n\n"
        "Build the ENTIRE structure in one pass, bottom-up. "
        "Place every brick needed."
    )

    # Stage 5 gets the engineer-mindset executor prompt (explicit load solve);
    # Stage 3 and any other mode keep the standard EXECUTOR_SYSTEM.
    executor_system = STAGE5_EXECUTOR_SYSTEM if mode_label == "stage5" else EXECUTOR_SYSTEM
    exec_messages = [
        {"role": "system", "content": executor_system},
        {"role": "user",   "content": exec_user},
    ]
    exec_text = _exec_tok.apply_chat_template(
        exec_messages, tokenize=False, add_generation_prompt=True,
    )
    exec_inputs = _exec_tok(
        exec_text, return_tensors="pt", add_special_tokens=False,
    ).to(_exec_model.device)

    exec_streamer = TextIteratorStreamer(
        _exec_tok, skip_prompt=True, skip_special_tokens=True,
    )
    exec_gen_kwargs = dict(
        input_ids=exec_inputs["input_ids"],
        attention_mask=exec_inputs["attention_mask"],
        max_new_tokens=max_tok,
        do_sample=True, temperature=0.7, top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=_exec_tok.eos_token_id,
        eos_token_id=[151643, 151645],
        streamer=exec_streamer,
        use_cache=True,
    )

    local_occupied = _occupied_cells(brick_state)

    def event_stream():
        yield _sse(f'<chat mode="{mode_label}" build="true">')
        yield _sse(f'[Build prompt → Executor] "{abstract_prompt}"')
        yield _sse('<part name="Build">')

        _prime_executor_run()
        t = Thread(target=_run_executor_generation,
                   args=(exec_gen_kwargs, adapter_name), daemon=True)
        t.start()

        counts = {}
        yield from _stream_bricks(exec_streamer, BRICK_RE,
                                   local_occupied, brick_state, counts)

        t.join(timeout=5)
        yield _sse("</part>")
        yield _sse(f"<summary>Placed {counts.get('placed', 0)} bricks</summary>")
        yield _sse("</chat>")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _inspect_review(messages_in: list, brick_state: list,
                    views: dict, max_tok: int):
    """
    Critical structural review: the planner analyses the existing brick_state
    and viewport images and returns a structured review. NO executor is run.
    """
    chat_messages = [{"role": "system", "content": REVIEW_PLANNER_SYSTEM}]
    for turn in messages_in[-12:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        if content and role in ("user", "assistant"):
            chat_messages.append({"role": role, "content": content})

    # Inject the brick_state summary so the planner has structured data to
    # reason about, alongside the viewport images.
    brick_summary = _format_brick_state(brick_state)
    n_ref = sum(1 for k in (views or {}) if k.startswith("ref_"))
    n_vp  = len(views or {}) - n_ref
    review_note = (
        f"\n\n--- FINAL BUILD STATE ---\n{brick_summary}\n"
        f"--- Views attached: {n_vp} viewport(s), {n_ref} reference image(s) ---\n"
        "Provide your critical review now, following the structured format."
    )
    # Append as a fresh user turn so the planner treats it as the current ask.
    chat_messages.append({"role": "user", "content": review_note.strip()})

    pil_images = _decode_views_to_images(views)
    inputs = _planner_inputs_with_images(chat_messages, pil_images)

    streamer = TextIteratorStreamer(
        _planner_tok, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = dict(
        **_planner_gen_inputs(inputs),
        max_new_tokens=min(max_tok, 1536),
        do_sample=True, temperature=0.6, top_p=0.9,
        repetition_penalty=1.05,
        pad_token_id=_planner_tok.eos_token_id,
        streamer=streamer,
        use_cache=True,
    )

    vis_keys = [k for k in inputs.keys() if k not in ("input_ids", "attention_mask")]
    print(f"[/chat inspect-review] {len(brick_state)} bricks, "
          f"{len(pil_images)} images, vision keys → generate: {vis_keys or 'NONE'}")

    def event_stream():
        yield _sse('<chat mode="inspect" build="false" review="true">')
        yield _sse("  ── Critical Review ──")
        t = Thread(target=_run_generation,
                   args=(_planner_model, gen_kwargs), daemon=True)
        t.start()
        in_think = False
        for line in _stream_lines(streamer):
            lo = line.lower()
            if "<think>" in lo:
                in_think = True
                continue
            if "</think>" in lo:
                in_think = False
                continue
            if in_think:
                continue
            yield _sse(line)
        t.join(timeout=5)
        yield _sse("</chat>")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _inspect_build(messages_in: list, brick_state: list,
                   views: dict, max_tok: int,
                   mode_label: str = "stage3",
                   adapter_name: str = "default",
                   planner_note: str = "",
                   skip_planner: bool = False):
    """Planner synthesises after negotiation, then the selected LoRA executor builds."""
    planner_system = STAGE5_PLANNER_SYSTEM if mode_label == "stage5" else INSPECT_PLANNER_SYSTEM
    return _planner_to_executor_build(
        messages_in, brick_state, views, max_tok,
        mode_label=mode_label,
        planner_system=planner_system,
        adapter_name=adapter_name,
        planner_note=planner_note,
        skip_planner=skip_planner,
    )


@app.post("/chat")
async def chat(request: Request):
    """
    Unified endpoint for Builder / Inspect / Agentic modes.

    POST body:
      mode            str   — "builder" | "inspect"
      messages        list  — full chat history [{role, content}, ...]
      brick_state     list  — current Rhino scene snapshot
      views           dict  — base64 viewport images (inspect mode → planner)
      build_now       bool  — skip negotiation, emit bricks this turn
      max_new_tokens  int

    Mode matrix:
      builder                        → FAST_EXECUTOR_SYSTEM  (Qwen 2.5 stage-1, direct)
      stage3/stage5 + !build_now     → INSPECT_PLANNER_SYSTEM (Qwen 3.5, negotiate)
      stage3 + build_now             → planner → executor with 'default' LoRA (stage-3)
      stage5 + build_now             → planner → executor with 'stage5'  LoRA (stage-5)

    SSE stream:
      - In negotiation: plain chat text.
      - In build: _sse_place events for each validated PLACE line,
        plus plain text for reasoning/layers/summary.
    """
    body          = await request.json()
    mode          = str(body.get("mode", "builder")).strip().lower()
    messages_in   = body.get("messages", []) or []
    brick_state   = body.get("brick_state", []) or []
    views         = body.get("views", {}) or {}
    build_now     = _as_bool(body.get("build_now", False))
    review        = _as_bool(body.get("review", False))
    skip_planner  = _as_bool(body.get("skip_planner", False))
    max_tok       = min(int(body.get("max_new_tokens", 49152)), 49152)

    # Legacy client compat: "inspect" → "stage3".
    if mode == "inspect":
        mode = "stage3"
    if mode not in ("builder", "stage3", "stage5"):
        mode = "builder"

    # Phrase detection: if the latest user message says "build it" etc,
    # flip to build. This lets Inspect/Builder skip the accept button.
    # BUT: never skip negotiation on the very first user turn — we need at
    # least one assistant reply before phrase-detection is allowed to fire,
    # otherwise inputs like "build a table" would collapse negotiation.
    has_assistant_history = any(t.get("role") == "assistant" for t in messages_in)
    last_user = ""
    for turn in reversed(messages_in):
        if turn.get("role") == "user":
            last_user = str(turn.get("content", ""))
            break
    if (
        not build_now
        and has_assistant_history
        and _detect_build_phrase(last_user)
    ):
        build_now = True

    stage5_spec = None
    stage5_note = ""
    if mode == "stage5":
        stage5_spec = _extract_stage5_spec(messages_in)
        stage5_note = _stage5_spec_note(stage5_spec)
        missing = stage5_spec.get("missing_required") or []
        # Designer-override: if the user explicitly hits Approve / Build it /
        # Direct, build_now arrives True even when layer/brick targets are
        # absent. The planner is allowed to ask for them during negotiation,
        # but the designer's explicit approval wins — proceed with whatever
        # numbers the planner can infer from the conversation.
        if build_now and missing:
            print(f"[/chat stage5] missing designer targets: {', '.join(missing)} — "
                  "designer approved anyway, planner will infer defaults")

    # ── CRITICAL REVIEW: always handled by the planner (Agent 1) regardless
    #    of mode. Fast/Inspect both get the same structured review so the
    #    executor is NEVER re-invoked during the analysis phase. ──
    if review:
        return _inspect_review(messages_in, brick_state, views, max_tok)

    # ── STAGE 3 / STAGE 5 MODE: planner-driven flow (negotiate | synth→build) ──
    if mode in ("stage3", "stage5"):
        # skip_planner forces a direct-to-executor build — the UI's
        # "Direct" button sets this so the prompt bypasses Agent 1 and
        # goes straight to the stage-LoRA executor.
        if skip_planner:
            build_now = True
        print(f"[/chat {mode}] build_now={build_now} skip_planner={skip_planner}")
        if not build_now:
            planner_system = STAGE5_PLANNER_SYSTEM if mode == "stage5" else INSPECT_PLANNER_SYSTEM
            return _inspect_negotiate(
                messages_in, brick_state, views, max_tok,
                planner_system=planner_system,
                planner_note=stage5_note if mode == "stage5" else "",
            )
        adapter_name = "stage5" if mode == "stage5" else "default"
        return _inspect_build(
            messages_in, brick_state, views, max_tok,
            mode_label=mode,
            adapter_name=adapter_name,
            planner_note=stage5_note if mode == "stage5" else "",
            skip_planner=skip_planner,
        )

    # ── EXECUTOR (builder) MODE: prompt goes straight to stage-1 executor ──
    # No negotiation, no planner — the frontend always sends build_now=True.
    system_prompt = FAST_EXECUTOR_SYSTEM
    brick_re = FAST_BRICK_RE
    build_now = True

    # Build chat messages (trim history to last 12 turns for context budget)
    chat_messages = [{"role": "system", "content": system_prompt}]
    for turn in messages_in[-12:]:
        role = turn.get("role", "user")
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if role in ("user", "assistant"):
            chat_messages.append({"role": role, "content": content})

    # On a build turn, append scene context to the last user message.
    if build_now:
        context_note = (
            f"\n\n--- Scene context ---\n{_format_brick_state(brick_state)}\n"
            "Build the complete structure now. Output the full brick sequence."
        )
        # Find the last user message and append; if none, inject one.
        for i in range(len(chat_messages) - 1, -1, -1):
            if chat_messages[i]["role"] == "user":
                chat_messages[i] = {
                    "role": "user",
                    "content": chat_messages[i]["content"] + context_note,
                }
                break
        else:
            chat_messages.append({"role": "user", "content": context_note.strip()})

    # Render prompt
    prompt_text = _exec_tok.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _exec_tok(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    ).to(_exec_model.device)

    # Diagnostic: confirm the prompt actually reaches the stage-1 executor.
    _last_user_msg = next(
        (m["content"] for m in reversed(chat_messages) if m["role"] == "user"),
        "<NO USER MESSAGE>",
    )
    _preview = _last_user_msg.replace("\n", " ")
    if len(_preview) > 200:
        _preview = _preview[:200] + "…"
    print(f"[/chat {mode}] build_now={build_now} "
          f"messages={len(chat_messages)} "
          f"prompt_tokens={inputs['input_ids'].shape[-1]} "
          f"last_user={_preview!r}")

    streamer = TextIteratorStreamer(
        _exec_tok, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tok,
        pad_token_id=_exec_tok.eos_token_id,
        eos_token_id=[151643, 151645],
        streamer=streamer,
        use_cache=True,
    )
    gen_kwargs.update(do_sample=False, repetition_penalty=1.02)

    # Validation state
    local_occupied = _occupied_cells(brick_state)
    local_placed   = list(brick_state)

    def event_stream():
        yield _sse(f'<chat mode="{mode}" build="true">')

        _prime_executor_run()
        # Fast/Builder mode uses the dedicated stage-1 LoRA adapter to match
        # the FAST_EXECUTOR_SYSTEM prompt style.
        t = Thread(
            target=_run_executor_generation,
            args=(gen_kwargs, "stage1"),
            daemon=True,
        )
        t.start()

        counts = {}
        yield from _stream_bricks(streamer, brick_re,
                                   local_occupied, local_placed, counts)

        t.join(timeout=5)
        yield _sse("</chat>")
        if build_now:
            yield _sse(f"<summary>Placed {counts.get('placed', 0)} bricks</summary>")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/continue")
async def continue_build(request: Request):
    """
    Continue a previously interrupted build.

    The existing brick lines are injected into the assistant's turn
    (as if the model already generated them), then the model continues
    generating from that point — no planner synthesis needed.

    POST body:
      prompt            str   — original build prompt
      existing_bricks   str   — PLACE lines the model "already generated"
      brick_state       list  — current Rhino scene [{dims,x,y,z}, ...]
      max_new_tokens    int
    """
    body            = await request.json()
    prompt          = str(body.get("prompt", "")).strip()
    existing_text   = str(body.get("existing_bricks", "")).strip()
    brick_state     = body.get("brick_state", []) or []
    max_tok         = min(int(body.get("max_new_tokens", 49152)), 49152)

    exec_user = f'Build request: "{prompt}"\n\n'
    exec_user += (
        "HARD GRID BOUNDS — ALL bricks MUST satisfy 0 ≤ x, 0 ≤ y, 0 ≤ z and "
        "x+H ≤ 20, y+W ≤ 20, z < 20. Any brick outside these bounds is "
        "rejected by the bridge.\n\n"
        "Build the ENTIRE structure in one pass, bottom-up. "
        "Place every brick needed."
    )

    exec_messages = [
        {"role": "system", "content": EXECUTOR_SYSTEM},
        {"role": "user",   "content": exec_user},
    ]
    exec_text = _exec_tok.apply_chat_template(
        exec_messages, tokenize=False, add_generation_prompt=True,
    )
    # Inject existing bricks as if the model already generated them
    exec_text += existing_text + "\n"

    exec_inputs = _exec_tok(
        exec_text, return_tensors="pt", add_special_tokens=False,
    ).to(_exec_model.device)

    streamer = TextIteratorStreamer(
        _exec_tok, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = dict(
        input_ids=exec_inputs["input_ids"],
        attention_mask=exec_inputs["attention_mask"],
        max_new_tokens=max_tok,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=_exec_tok.eos_token_id,
        eos_token_id=[151643, 151645],
        streamer=streamer,
        use_cache=True,
    )

    local_occupied = _occupied_cells(brick_state)

    print(f"[/continue] prompt={prompt!r}, "
          f"{len(brick_state)} existing bricks, "
          f"{len(existing_text.splitlines())} injected lines")

    def event_stream():
        yield _sse('<chat mode="continue" build="true">')
        yield _sse(f'[Continuing build — {len(brick_state)} bricks already placed]')
        yield _sse('<part name="Continue">')

        _prime_executor_run()
        t = Thread(target=_run_executor_generation,
                   args=(gen_kwargs, "default"), daemon=True)
        t.start()

        counts = {}
        yield from _stream_bricks(streamer, BRICK_RE,
                                   local_occupied, brick_state, counts)

        t.join(timeout=5)
        yield _sse("</part>")
        yield _sse(f"<summary>Continued — placed {counts.get('placed', 0)} new bricks</summary>")
        yield _sse("</chat>")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/plan")
async def plan(request: Request):
    """
    Phase 1 — Planner only (TEXT-only).

    POST body:
      prompt       str
      brick_state  list[{dims,x,y,z}]
      history      list[{role,content}]
      views        dict[str,str]   — accepted but IGNORED (planner is text-only)

    SSE stream: thinking → analysis → plan → [AWAITING_APPROVAL]
    """
    body          = await request.json()
    user_prompt   = body.get("prompt", "").strip()
    brick_state   = body.get("brick_state", [])
    history       = body.get("history", [])
    n_views       = len(body.get("views", {}) or {})

    print(f"\n[/plan] {len(brick_state)} bricks, history={len(history)}, "
          f"views={n_views} (ignored — text-only)")

    user_text = (
        f"Current brick state:\n{_format_brick_state(brick_state)}\n\n"
        f"Build request: {user_prompt}\n\n"
        "Analyse, plan, and await approval."
    )

    messages = [{"role": "system", "content": INSPECT_PLANNER_SYSTEM}]
    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn.get("content", "")})
    messages.append({"role": "user", "content": user_text})

    prompt_text = _planner_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _planner_tok(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    ).to(_planner_model.device)

    streamer = TextIteratorStreamer(
        _planner_tok, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=4096,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=_planner_tok.eos_token_id,
        streamer=streamer,
        use_cache=True,
    )

    def event_stream():
        t = Thread(target=_run_generation, args=(_planner_model, gen_kwargs), daemon=True)
        t.start()
        for line in _stream_lines(streamer):
            lo = line.lower()
            if "<think>" in lo:
                yield _sse("<thinking>")
                after = re.sub(r'(?i)<think>', '', line).strip()
                if after:
                    yield _sse(after)
                continue
            if "</think>" in lo:
                yield _sse("</thinking>")
                continue
            yield _sse(line)
        t.join(timeout=5)
        yield "data: [AWAITING_APPROVAL]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/execute")
async def execute(request: Request):
    """
    Phase 2 — Single-call executor (after designer approval).

    Fixes applied:
      1. start_z computed and injected into prompt
      2. Full plan sent as guidance in ONE prompt (no multi-part loop)
      3. Greedy decoding (do_sample=False)
      4. Server-side PLACE validation (overlap, support, bounds)

    POST body:
      plan_text      str   — full planner output captured by the UI
      prompt         str   — original user prompt
      brick_state    list  — scene state at time of approval
      clarification  str   — optional designer note added at approval
      history        list
      max_new_tokens int
    """
    body          = await request.json()
    plan_text     = body.get("plan_text", "")
    user_prompt   = body.get("prompt", "").strip()
    brick_state   = body.get("brick_state", [])
    clarification = body.get("clarification", "").strip()
    fast_mode     = _as_bool(body.get("fast_mode", False))
    max_tok       = min(body.get("max_new_tokens", 16384), 16384)

    # Fix 1: compute start_z from existing bricks
    if brick_state:
        max_z = max(b.get("z", 0) for b in brick_state)
        # The next free layer is max_z + 1 ONLY if we're stacking.
        # But bricks can be placed at any z; the key constraint is "don't
        # go below what's already there if building on top."
        start_z = max_z + 1
    else:
        start_z = 0

    # Fix 2: extract plan parts as guidance text (not separate calls)
    parts = _parse_plan_parts(plan_text)
    if parts:
        plan_guidance = "\n".join(
            f"  Part {p['index']}: {p['name']} — {p['description']}"
            for p in parts
        )
    else:
        plan_guidance = plan_text  # fallback: raw plan text

    clarif_note = f"\nDesigner note: {clarification}" if clarification else ""
    already = _format_brick_state(brick_state)

    if fast_mode:
        exec_user = f'Build request: "{user_prompt}"{clarif_note}\n\n'
        if plan_guidance.strip():
            exec_user += (
                "Build plan to follow:\n"
                f"{plan_guidance}\n\n"
            )
        if brick_state:
            exec_user += (
                "Existing bricks already on the grid. Output ONLY new bricks to add. "
                "Do not overlap or repeat existing bricks.\n"
                f"{already}\n\n"
                f"If you are building on top of existing work, the next free layer is z={start_z}. "
                "If you are building beside existing work, ground-level z=0 is still allowed.\n\n"
            )
        else:
            exec_user += (
                "The grid is empty. Anchor the footprint near (x=2, y=2, z=0). "
                "Do NOT invent a large origin.\n\n"
            )
        exec_user += (
            "HARD GRID BOUNDS — every brick MUST have 0≤x, 0≤y, 0≤z and "
            "x+H≤20, y+W≤20, z<20. Out-of-bounds bricks are rejected.\n\n"
            "Output ONLY the brick sequence for the full build. Nothing else."
        )
        system_prompt = FAST_EXECUTOR_SYSTEM
        brick_re = FAST_BRICK_RE
        repetition_penalty = 1.02
    else:
        # Single unified prompt with start_z constraint and full plan as guidance
        exec_user = (
            f'Build request: "{user_prompt}"{clarif_note}\n\n'
            f'=== BUILD PLAN (from planner — follow this as your guide) ===\n'
            f'{plan_guidance}\n'
            f'=== END PLAN ===\n\n'
        )

        if brick_state:
            exec_user += (
                f'Already placed on the grid ({len(brick_state)} bricks — do NOT re-place these):\n'
                f'{already}\n\n'
                f'CRITICAL: The existing structure occupies up to z={max_z}. '
                f'New bricks that go ON TOP of existing structure start at z={start_z}. '
                f'Do not place anything that overlaps existing bricks. '
                f'Ground-level parts of NEW structure still start at z=0 if they are beside (not on top of) existing bricks.\n'
            )
        else:
            exec_user += (
                'The grid is EMPTY. Anchor the footprint at (x=2, y=2, z=0). '
                'Do NOT invent a large origin like (36,46,0) — always start '
                'near (0,0,0).\n'
            )

        exec_user += (
            '\nHARD GRID BOUNDS — every brick MUST satisfy 0 ≤ x, 0 ≤ y, '
            '0 ≤ z AND x+H ≤ 20, y+W ≤ 20, z < 20. Bricks outside these '
            'bounds are rejected by the bridge.\n'
            '\nBuild the ENTIRE structure in one pass, bottom-up. '
            'Place every brick needed for all parts of the plan.'
        )
        system_prompt = EXECUTOR_SYSTEM
        brick_re = BRICK_RE
        repetition_penalty = 1.05

    print(f"[/execute] single-call, {len(brick_state)} existing bricks, "
          f"start_z={start_z}, fast={fast_mode}, max_tok={max_tok}")

    exec_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": exec_user},
    ]
    exec_text = _exec_tok.apply_chat_template(
        exec_messages, tokenize=False, add_generation_prompt=True,
    )
    exec_inputs = _exec_tok(
        exec_text, return_tensors="pt", add_special_tokens=False,
    ).to(_exec_model.device)

    exec_streamer = TextIteratorStreamer(
        _exec_tok, skip_prompt=True, skip_special_tokens=True,
    )

    # Fix 3: greedy decoding — no sampling
    exec_gen_kwargs = dict(
        input_ids=exec_inputs["input_ids"],
        attention_mask=exec_inputs["attention_mask"],
        max_new_tokens=max_tok,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        pad_token_id=_exec_tok.eos_token_id,
        eos_token_id=[151643, 151645],
        streamer=exec_streamer,
        use_cache=True,
    )

    local_occupied = _occupied_cells(brick_state)

    def event_stream():
        # Stream the prompt to UI so designer can see what was sent
        yield _sse('<exec_prompt part="Full Build">')
        for pline in exec_user.split('\n'):
            yield _sse(pline)
        yield _sse('</exec_prompt>')

        yield _sse('<part name="Build">')

        _prime_executor_run()
        exec_thread = Thread(
            target=_run_executor_generation,
            args=(exec_gen_kwargs, "default"),
            daemon=True,
        )
        exec_thread.start()

        counts = {}
        yield from _stream_bricks(exec_streamer, brick_re,
                                   local_occupied, brick_state, counts)

        exec_thread.join(timeout=5)
        yield _sse("</part>")

        yield _sse(f"<summary>Built {counts.get('placed', 0)} bricks</summary>")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/execute-part")
async def execute_part(request: Request):
    """
    Execute ONE part of the build plan. The frontend loops over plan parts and
    calls this endpoint per part, with the planner inspecting between calls.

    POST body:
      part_name        str   — name of the part to build (e.g. "Cuboid Base")
      part_description str   — planner's description of this part
      full_plan        str   — entire planner output (for context)
      prompt           str   — original user request
      brick_state      list  — bricks currently on the grid
      part_index       int   — 1-based
      total_parts      int
      clarification    str   — optional designer note from approval gate
      max_new_tokens   int
    """
    body          = await request.json()
    part_name     = body.get("part_name", "Part")
    part_desc     = body.get("part_description", "")
    full_plan     = body.get("full_plan", "")
    user_prompt   = body.get("prompt", "").strip()
    brick_state   = body.get("brick_state", [])
    part_index    = int(body.get("part_index", 1))
    total_parts   = int(body.get("total_parts", 1))
    clarification = body.get("clarification", "").strip()
    fast_mode     = _as_bool(body.get("fast_mode", False))
    max_tok       = min(body.get("max_new_tokens", 8192), 8192)

    if brick_state:
        max_z   = max(b.get("z", 0) for b in brick_state)
        start_z = max_z + 1
    else:
        max_z   = -1
        start_z = 0

    # Pretty plan guidance for context
    parts = _parse_plan_parts(full_plan)
    if parts:
        plan_guidance = "\n".join(
            f"  Part {p['index']}: {p['name']} — {p['description']}" for p in parts
        )
    else:
        plan_guidance = full_plan

    state_summary = _compact_brick_state(brick_state)
    size_hint     = _part_size_hint(part_name)
    clarif_note   = f"\nDesigner note: {clarification}" if clarification else ""

    if fast_mode:
        exec_user = (
            f'Build request: "{user_prompt}"{clarif_note}\n\n'
            f'Build ONLY Part {part_index}/{total_parts} — {part_name}\n'
            f'{part_desc}\n\n'
            f'Overall plan context:\n{plan_guidance}\n\n'
        )

        if brick_state:
            exec_user += (
                "Existing bricks already on the grid. Output ONLY new bricks to add. "
                "Do not overlap or repeat existing bricks.\n"
                f"{state_summary}\n\n"
                f"If this part builds on top of existing work, do not place any brick below z={start_z}. "
                "If this part starts beside existing work, ground-level z=0 is still allowed. "
                "Build only this single part and stop when that part is complete.\n\n"
            )
        else:
            exec_user += (
                "The grid is empty. Anchor the footprint near (x=2, y=2, z=0). "
                "Start at z=0 and do NOT invent a large origin. "
                "Build only this single part and stop when that part is complete.\n\n"
            )

        exec_user += (
            "HARD GRID BOUNDS — every brick MUST have 0≤x, 0≤y, 0≤z and "
            "x+H≤20, y+W≤20, z<20. Out-of-bounds bricks are rejected.\n\n"
        )

        exec_user = exec_user.replace(
            f'Overall plan context:\n{plan_guidance}\n\n',
            f'Overall plan context:\n{plan_guidance}\n\nSHAPE CLASS: {size_hint}\n\n',
        )
        exec_user += "Output ONLY the brick sequence for this single part. Nothing else."
        system_prompt = FAST_EXECUTOR_SYSTEM
        brick_re = FAST_BRICK_RE
        repetition_penalty = 1.02
    else:
        exec_user = (
            f'Build request: "{user_prompt}"{clarif_note}\n\n'
            f'=== FULL BUILD PLAN (context only) ===\n'
            f'{plan_guidance}\n'
            f'=== END PLAN ===\n\n'
            f'YOUR CURRENT TASK: Build Part {part_index}/{total_parts} — {part_name}\n'
            f'{part_desc}\n\n'
            f'SHAPE CLASS: {size_hint}\n\n'
        )

        if brick_state:
            exec_user += (
                f'=== CURRENT GRID STATE ===\n{state_summary}\n=== END STATE ===\n\n'
                f'HARD CONSTRAINTS — VIOLATIONS WILL BE REJECTED:\n'
                f'  1. Layers z=0 through z={max_z} are ALREADY DONE. NEVER place a brick at z<={max_z}.\n'
                f'  2. ALL of your new bricks MUST be at z>={start_z} (next free layer).\n'
                f'  3. NEVER place a brick that overlaps any existing brick (same cell).\n'
                f'  4. Build ONLY this single part — STOP when this part is finished.\n'
                f'  5. Respect the SHAPE CLASS budget above. Do NOT exceed those layer/brick counts.\n'
                f'  6. Do NOT re-survey or re-plan the entire structure. Just place this one part.\n'
                f'  7. GRID BOUNDS: every brick MUST have 0≤x, 0≤y, x+H≤20, y+W≤20, z<20.\n'
            )
        else:
            exec_user += (
                f'HARD CONSTRAINTS — VIOLATIONS WILL BE REJECTED:\n'
                f'  1. The grid is empty. Anchor the footprint at (x=2, y=2, z=0). '
                f'Do NOT invent a large origin like (36,46,0).\n'
                f'  2. Build ONLY this single part — STOP when finished.\n'
                f'  3. Respect the SHAPE CLASS budget above.\n'
                f'  4. GRID BOUNDS: every brick MUST have 0≤x, 0≤y, x+H≤20, y+W≤20, z<20.\n'
            )

        exec_user += (
            '\nBegin placing bricks for this part now, bottom-up. '
            'After the last brick of THIS part, output "COMPLETE: <part name>" and STOP. '
            'The planner will then inspect and tell you to begin the next part.'
        )
        system_prompt = EXECUTOR_SYSTEM
        brick_re = BRICK_RE
        repetition_penalty = 1.05

    print(f"[/execute-part] {part_index}/{total_parts} '{part_name}', "
          f"{len(brick_state)} existing, fast={fast_mode}, max_tok={max_tok}")

    exec_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": exec_user},
    ]
    exec_text = _exec_tok.apply_chat_template(
        exec_messages, tokenize=False, add_generation_prompt=True,
    )
    exec_inputs = _exec_tok(
        exec_text, return_tensors="pt", add_special_tokens=False,
    ).to(_exec_model.device)

    exec_streamer = TextIteratorStreamer(
        _exec_tok, skip_prompt=True, skip_special_tokens=True,
    )
    exec_gen_kwargs = dict(
        input_ids=exec_inputs["input_ids"],
        attention_mask=exec_inputs["attention_mask"],
        max_new_tokens=max_tok,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        pad_token_id=_exec_tok.eos_token_id,
        eos_token_id=[151643, 151645],
        streamer=exec_streamer,
        use_cache=True,
    )

    # Pre-compute occupied cells from existing bricks for overlap rejection.
    occupied = _occupied_cells(brick_state)

    def event_stream():
        local_occupied = set(occupied)  # mutated as new bricks are accepted

        yield _sse(f'<exec_prompt part="{part_name}">')
        for pline in exec_user.split('\n'):
            yield _sse(pline)
        yield _sse('</exec_prompt>')

        yield _sse(f'<part name="{part_name}">')

        _prime_executor_run()
        exec_thread = Thread(
            target=_run_executor_generation,
            args=(exec_gen_kwargs, "default"),
            daemon=True,
        )
        exec_thread.start()

        counts = {}
        yield from _stream_bricks(exec_streamer, brick_re,
                                   local_occupied, brick_state, counts)

        exec_thread.join(timeout=5)
        yield _sse("</part>")
        placed = counts.get("placed", 0)
        yield _sse(
            f"<summary>Part {part_index}/{total_parts} ({part_name}): "
            f"{placed} placed</summary>"
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/inspect-part")
async def inspect_part(request: Request):
    """
    Planner inspects the result of a single executed part and gives a brief
    review. Streams the planner's review text and ends with [DONE].

    POST body:
      part_name    str
      part_index   int
      total_parts  int
      brick_state  list
      prompt       str
      full_plan    str
    """
    body         = await request.json()
    part_name    = body.get("part_name", "Part")
    part_index   = int(body.get("part_index", 1))
    total_parts  = int(body.get("total_parts", 1))
    brick_state  = body.get("brick_state", [])
    user_prompt  = body.get("prompt", "")
    full_plan    = body.get("full_plan", "")

    parts = _parse_plan_parts(full_plan)
    if parts:
        plan_outline = "\n".join(
            f"  Part {p['index']}: {p['name']} — {p['description']}" for p in parts
        )
    else:
        plan_outline = full_plan

    state_summary = _compact_brick_state(brick_state)

    user_text = (
        f'Original request: "{user_prompt}"\n\n'
        f'=== APPROVED PLAN ===\n{plan_outline}\n=== END PLAN ===\n\n'
        f'JUST FINISHED: Part {part_index}/{total_parts} — {part_name}\n\n'
        f'CURRENT GRID STATE:\n{state_summary}\n\n'
        f'Review now. Output EXACTLY four lines: Shape:, Size:, Next:, then '
        f'the decision marker. No reasoning, no preamble. Begin with "Shape:".'
    )

    messages = [
        {"role": "system", "content": INSPECT_PLANNER_SYSTEM},
        {"role": "user",   "content": user_text},
    ]
    prompt_text = _planner_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _planner_tok(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    ).to(_planner_model.device)

    streamer = TextIteratorStreamer(
        _planner_tok, skip_prompt=True, skip_special_tokens=True,
    )
    # Greedy decoding + tight cap → forces the four-line format and stops
    # the model from rambling through a "Thinking Process" preamble.
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=384,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=_planner_tok.eos_token_id,
        streamer=streamer,
        use_cache=True,
    )

    print(f"[/inspect-part] {part_index}/{total_parts} '{part_name}', "
          f"{len(brick_state)} bricks on grid")

    # Drop common preamble / scratchpad markers the planner sometimes emits
    # before the actual review (Qwen3 likes to think out loud even when told
    # not to). We discard everything until we see the first "Shape:" line.
    PREAMBLE_HINT = re.compile(
        r"^\s*(thinking process|let me|first[, ]|i('| a)?m going|i need to|"
        r"analyz(e|ing)|step \d|reasoning|to (review|inspect|assess))",
        re.IGNORECASE,
    )

    def event_stream():
        t = Thread(target=_run_generation, args=(_planner_model, gen_kwargs), daemon=True)
        t.start()
        in_think    = False
        seen_review = False
        for line in _stream_lines(streamer):
            lo = line.lower()
            # Hard skip explicit <think>...</think> tags
            if "<think>" in lo:
                in_think = True
                continue
            if "</think>" in lo:
                in_think = False
                continue
            if in_think:
                continue

            # Until we see the first "Shape:" line, drop preamble noise.
            if not seen_review:
                if line.lstrip().lower().startswith("shape:"):
                    seen_review = True
                elif PREAMBLE_HINT.match(line):
                    continue
                else:
                    # Drop any other line before the structured review begins
                    # (numbered steps, bullets, "OK so the user wants…", etc.)
                    continue

            yield _sse(line)
        t.join(timeout=5)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/execute-fast")
async def execute_fast(request: Request):
    """
    Single-agent fast mode. Reuses the main executor adapter but switches to
    a minimal prompt that asks for only the brick sequence.
    """
    body        = await request.json()
    user_prompt = body.get("prompt", "").strip()
    brick_state = body.get("brick_state", [])
    max_tok     = min(body.get("max_new_tokens", 8192), 8192)

    if not user_prompt:
        return JSONResponse({"error": "prompt is required"}, status_code=400)

    fast_user = user_prompt
    if brick_state:
        fast_user += (
            "\n\nExisting bricks already on the grid. Output ONLY new bricks to add. "
            "Do not overlap or repeat existing bricks.\n"
            f"{_format_brick_state(brick_state)}"
        )

    print(f"[/execute-fast] {len(brick_state)} existing bricks, max_tok={max_tok}")

    exec_messages = [
        {"role": "system", "content": FAST_EXECUTOR_SYSTEM},
        {"role": "user",   "content": fast_user},
    ]
    exec_text = _exec_tok.apply_chat_template(
        exec_messages, tokenize=False, add_generation_prompt=True,
    )
    exec_inputs = _exec_tok(
        exec_text, return_tensors="pt", add_special_tokens=False,
    ).to(_exec_model.device)

    exec_streamer = TextIteratorStreamer(
        _exec_tok, skip_prompt=True, skip_special_tokens=True,
    )
    exec_gen_kwargs = dict(
        input_ids=exec_inputs["input_ids"],
        attention_mask=exec_inputs["attention_mask"],
        max_new_tokens=max_tok,
        do_sample=False,
        repetition_penalty=1.02,
        pad_token_id=_exec_tok.eos_token_id,
        eos_token_id=[151643, 151645],
        streamer=exec_streamer,
        use_cache=True,
    )

    occupied = _occupied_cells(brick_state)

    def event_stream():
        local_occupied = set(occupied)

        yield _sse('<exec_prompt part="Fast Build">')
        for pline in fast_user.split('\n'):
            yield _sse(pline)
        yield _sse('</exec_prompt>')
        yield _sse('<part name="Fast Build">')

        _prime_executor_run()
        # /execute-fast mirrors the /chat Fast path and runs on the stage-1 LoRA.
        exec_thread = Thread(
            target=_run_executor_generation,
            args=(exec_gen_kwargs, "stage1"),
            daemon=True,
        )
        exec_thread.start()

        counts = {}
        yield from _stream_bricks(exec_streamer, FAST_BRICK_RE,
                                   local_occupied, brick_state, counts)

        exec_thread.join(timeout=5)
        yield _sse("</part>")
        placed = counts.get("placed", 0)
        rej = counts.get("rejected", 0)
        yield _sse(f"<summary>Built {placed} bricks in fast mode; rejected {rej}</summary>")
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    global _planner_model, _planner_tok, _planner_proc, _exec_model, _exec_tok

    parser = argparse.ArgumentParser(description="BrickAgent Orchestrator Server")
    parser.add_argument(
        "--adapter",
        default=os.path.join(CHECKPOINTS_DIR, "physics_reasoning"),
        help="Path to the primary executor LoRA adapter (loaded as 'default'). "
             "The original PSC deployment used a separate stage-3 checkpoint here; "
             "this repo only ships no_reasoning/physics_reasoning, so this defaults "
             "to physics_reasoning.",
    )
    parser.add_argument(
        "--adapter-stage1",
        default=os.path.join(CHECKPOINTS_DIR, "no_reasoning"),
        help="Path to the no_reasoning (stage-1) executor LoRA adapter (loaded as 'stage1', used by Fast/Builder mode)",
    )
    parser.add_argument(
        "--adapter-stage5",
        default=os.path.join(CHECKPOINTS_DIR, "physics_reasoning"),
        help="Path to the physics_reasoning (stage-6 full) executor LoRA adapter (loaded as 'stage5')",
    )
    parser.add_argument(
        "--planner",
        default=PLANNER_BASE,
        help="Planner model (HF ID or local path)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    dtype  = _compute_dtype()
    print(f"\n{'='*60}")
    print(f"  BrickAgent Orchestrator")
    print(f"  Available GPUs : {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        sm    = f"SM {props.major}.{props.minor}"
        print(f"    GPU {i}: {props.name}  ({props.total_memory/1e9:.0f} GB, {sm})")
    print(f"  Compute dtype  : {dtype}  "
          f"({'bfloat16 — A100/H100/L40S' if dtype == torch.bfloat16 else 'float16 — V100/older'})")
    if n_gpus < 2:
        print("  GPU sharing    : both models on GPU 0 (sequential pipeline — OK)")
    print(f"{'='*60}\n")

    _planner_model, _planner_tok, _planner_proc = load_planner(args.planner)
    _exec_model,    _exec_tok    = load_executor(args.adapter)
    load_executor_adapter(args.adapter_stage1, "stage1")
    load_executor_adapter(args.adapter_stage5, "stage5")

    print(f"\n{'='*60}")
    print(f"  Serving on http://{args.host}:{args.port}")
    print(f"  GET  /health")
    print(f"  POST /plan          — planner phase (streams to [AWAITING_APPROVAL])")
    print(f"  POST /execute       — executor phase, full plan in one shot")
    print(f"  POST /execute-part  — executor phase, single part (per-part loop)")
    print(f"  POST /execute-fast  — single-agent fast mode using the simplified prompt")
    print(f"  POST /inspect-part  — planner reviews one finished part")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
