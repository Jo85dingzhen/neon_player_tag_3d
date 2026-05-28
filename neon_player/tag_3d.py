"""
AprilTag 3D Coordinate System
==============================
Pupil Labs Neon · Head Pose Tracker

Origin  : room corner at floor level (0, 0, 0)
X axis  : along wall, rightward (positive)
Y axis  : upward from floor (positive)
Z axis  : away from wall into room (positive)
Unit    : cm

Physical measurements (all from room corner to tag 403 bottom-left corner):
  X = 28 inch  = 71.12 cm  (horizontal along wall)
  Y = 2.9 ft   = 88.39 cm  (desk height / vertical)
  Z = 6.4 ft   = 195.07 cm (distance from wall into room)

Requirements:
  pip install pandas numpy

Usage:
  python coordinate_system.py
"""

import json
import numpy as np
import pandas as pd
import webbrowser
import os
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_CSV   = Path(__file__).resolve().parent / "exports" / "000" / "head_pose_tracker_model.csv"
OUTPUT_HTML = "coordinate_report.html"
# ─── PHYSICAL MEASUREMENTS (cm) ──────────────────────────────────────────────

# Scale: derived from desk height (2.9 ft)
# tag 403 bottom-left (vert_3) is at Y=1 unit in model → Y = 20.11 cm on desk
# desk height = 2.9 ft = 88.39 cm → delta = 88.39 - 20.11 = 68.28 cm from model Y=0
SCALE_CM = 20.11   # 1 model unit = 20.11 cm

# Tag 403 bottom-left corner = desk corner
# Distance from room corner (floor) to this point:
X_OFFSET_CM = 28 * 2.54    # 28 inch  = 71.12 cm  along wall
Y_OFFSET_CM = 2.9 * 30.48  # 2.9 ft   = 88.39 cm  desk height
Z_OFFSET_CM = 6.4 * 30.48  # 6.4 ft   = 195.07 cm from wall

# tag 403 vert_3 (bottom-left) in model = (0, 1, 0) units = (0, 20.11, 0) cm
TAG403_BOTTOMLEFT_CM = np.array([0.0, 1.0 * SCALE_CM, 0.0])  # (0, 20.11, 0) cm

# Room corner in model space (cm) — this is our world origin
ORIGIN_CM = TAG403_BOTTOMLEFT_CM - np.array([X_OFFSET_CM, Y_OFFSET_CM, Z_OFFSET_CM])
# = (0 - 71.12,  20.11 - 88.39,  0 - 195.07)
# = (-71.12,     -68.28,          -195.07)  cm

# Anomaly markers to flag (known scan issues)
ANOMALY_IDS = {501, 504, 505, 508}

# Marker location descriptions
LOCATIONS = {
    403: "Desk surface — anchor tag (bottom-left = desk corner)",
    401: "Desk surface, wall side",
    409: "Desk surface, near corner",
    404: "Desk surface, mid",
    405: "Desk surface, far end",
    422: "Desk surface, right side",
    512: "Left-wall area, desk level",
    514: "Left wall surface",
    408: "Left wall surface, far end",
    509: "Left-wall area, far end",
    501: "Left wall (anomaly — Y below floor)",
    504: "Corner floor tag (anomaly — Z flipped)",
    505: "Corner floor tag (anomaly — Z flipped)",
    508: "Corner wall tag (anomaly — Z flipped)",
}

# ─── COORDINATE TRANSFORM ────────────────────────────────────────────────────

def to_world_cm(model_pos_units):
    """
    Convert model-space position (units) to world coordinates (cm).
    World origin = room corner at floor level.

    Args:
        model_pos_units: array-like [x, y, z] in model units

    Returns:
        np.array [x_cm, y_cm, z_cm] where:
            x_cm > 0 = rightward along wall
            y_cm > 0 = upward (0 = floor, ~88 = desk surface)
            z_cm > 0 = into room away from wall
    """
    model_cm = np.array(model_pos_units, dtype=float) * SCALE_CM
    return model_cm - ORIGIN_CM


def to_model_units(world_cm):
    """
    Convert world coordinates (cm) back to model units.

    Args:
        world_cm: array-like [x_cm, y_cm, z_cm]

    Returns:
        np.array [x, y, z] in model units
    """
    model_cm = np.array(world_cm, dtype=float) + ORIGIN_CM
    return model_cm / SCALE_CM


# ─── LOAD & PROCESS MODEL DATA ───────────────────────────────────────────────

def get_center(row):
    return np.array([
        np.mean([row[f"vert_{i}_x"] for i in range(4)]),
        np.mean([row[f"vert_{i}_y"] for i in range(4)]),
        np.mean([row[f"vert_{i}_z"] for i in range(4)]),
    ])

def get_normal(row):
    v = np.array([[row[f"vert_{i}_x"], row[f"vert_{i}_y"], row[f"vert_{i}_z"]]
                  for i in range(4)])
    n = np.cross(v[1] - v[0], v[3] - v[0])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else n

def get_verts_world(row):
    verts = []
    for i in range(4):
        v_units = np.array([row[f"vert_{i}_x"], row[f"vert_{i}_y"], row[f"vert_{i}_z"]])
        verts.append(to_world_cm(v_units).tolist())
    return verts

def classify_surface(normal):
    ax, ay, az = abs(normal[0]), abs(normal[1]), abs(normal[2])
    if ay > 0.7:   return "Horizontal (desk/floor)"
    elif az > 0.7: return "Wall · Z-facing"
    elif ax > 0.7: return "Wall · X-facing"
    else:          return "Tilted"


def load_markers(model_csv):
    df  = pd.read_csv(model_csv)
    rows = {int(r["marker_id"]): r for _, r in df.iterrows()}

    markers = []
    for mid, row in rows.items():
        center_units = get_center(row)
        normal       = get_normal(row)
        world        = to_world_cm(center_units)
        verts_world  = get_verts_world(row)

        bad    = mid in ANOMALY_IDS
        issues = []
        if mid == 501:          issues.append("Y below floor level")
        if mid in {504,505,508}: issues.append("Z sign flipped during scan")

        markers.append({
            "id":       mid,
            "status":   "anomaly" if bad else ("anchor" if mid == 403 else "valid"),
            "location": LOCATIONS.get(mid, ""),
            "surface":  classify_surface(normal),
            "x_cm":     round(float(world[0]), 2),
            "y_cm":     round(float(world[1]), 2),
            "z_cm":     round(float(world[2]), 2),
            "x_ft":     round(float(world[0]) / 30.48, 3),
            "y_ft":     round(float(world[1]) / 30.48, 3),
            "z_ft":     round(float(world[2]) / 30.48, 3),
            "x_in":     round(float(world[0]) / 2.54,  2),
            "y_in":     round(float(world[1]) / 2.54,  2),
            "z_in":     round(float(world[2]) / 2.54,  2),
            "nx":       round(float(normal[0]), 3),
            "ny":       round(float(normal[1]), 3),
            "nz":       round(float(normal[2]), 3),
            "verts":    verts_world,
            "issues":   "; ".join(issues),
        })

    markers.sort(key=lambda m: (
        0 if m["status"] == "anchor" else
        1 if m["status"] == "valid"  else 2,
        m["id"]
    ))
    return markers


# ─── PRINT SUMMARY ───────────────────────────────────────────────────────────

def print_summary(markers):
    print()
    print("=" * 65)
    print("  3D Coordinate System — World Coordinates")
    print("=" * 65)
    print(f"  Origin  : room corner at floor level (0, 0, 0)")
    print(f"  X axis  : rightward along wall")
    print(f"  Y axis  : upward  (floor=0, desk≈{Y_OFFSET_CM:.1f} cm)")
    print(f"  Z axis  : into room away from wall")
    print(f"  Scale   : 1 model unit = {SCALE_CM} cm")
    print()
    print(f"  Anchor  : tag 403 bottom-left = desk corner")
    print(f"    X = {X_OFFSET_CM:.2f} cm  ({X_OFFSET_CM/2.54:.1f} inch)")
    print(f"    Y = {Y_OFFSET_CM:.2f} cm  ({Y_OFFSET_CM/30.48:.2f} ft)")
    print(f"    Z = {Z_OFFSET_CM:.2f} cm  ({Z_OFFSET_CM/30.48:.2f} ft)")
    print()
    print(f"  ORIGIN_CM = {ORIGIN_CM.tolist()}  # cm in model space")
    print()
    print(f"  {'ID':>5}  {'X (cm)':>9}  {'Y (cm)':>9}  {'Z (cm)':>9}  "
          f"{'X (in)':>8}  {'Y (ft)':>7}  {'Z (ft)':>7}  Status")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*9}  "
          f"{'-'*8}  {'-'*7}  {'-'*7}  ------")
    for m in markers:
        flag = "⚓" if m["status"] == "anchor" else \
               "✓" if m["status"] == "valid"  else "✗"
        print(f"  {m['id']:>5}  {m['x_cm']:>+9.2f}  {m['y_cm']:>+9.2f}  {m['z_cm']:>+9.2f}  "
              f"{m['x_in']:>+8.2f}  {m['y_ft']:>+7.3f}  {m['z_ft']:>+7.3f}  "
              f"{flag} {m['status']}"
              + (f"  ← {m['issues']}" if m["issues"] else ""))
    print("=" * 65)
    print()


# ─── GENERATE HTML REPORT ────────────────────────────────────────────────────

def generate_html(markers, output_path):
    markers_js  = json.dumps(markers, indent=2)
    origin_js   = json.dumps(ORIGIN_CM.tolist())
    scale       = SCALE_CM
    x_offset    = X_OFFSET_CM
    y_offset    = Y_OFFSET_CM
    z_offset    = Z_OFFSET_CM

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AprilTag 3D Coordinate System</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root{{
  --bg:#0d0f14;--sur:#141720;--bor:#252830;--txt:#e8e6df;
  --mut:#6b6b75;--acc:#4a8fff;--ok:#3ec88a;--ok2:#1a3d2a;
  --err:#ff4d3a;--err2:#3d1a14;--warn:#ffb84a;--warn2:#3d2e10;
  --mono:'IBM Plex Mono',monospace;--sans:'IBM Plex Sans',sans-serif;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--sans);background:var(--bg);color:var(--txt);font-size:14px;line-height:1.6;display:grid;grid-template-columns:240px 1fr;height:100vh;overflow:hidden}}

/* sidebar */
.sidebar{{background:var(--sur);border-right:1px solid var(--bor);display:flex;flex-direction:column;overflow-y:auto}}
.brand{{padding:18px 16px 14px;border-bottom:1px solid var(--bor)}}
.brand-title{{font-size:13px;font-weight:600}}
.brand-sub{{font-size:10px;color:var(--mut);margin-top:2px;letter-spacing:.04em}}
.nav-label{{padding:14px 16px 6px;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--mut)}}
.nav-item{{padding:8px 16px;font-size:12px;cursor:pointer;color:#aaa;border-left:2px solid transparent;transition:all .15s}}
.nav-item:hover,.nav-item.active{{color:var(--txt);background:rgba(74,143,255,.07);border-left-color:var(--acc)}}
.sidebar-footer{{margin-top:auto;padding:14px 16px;border-top:1px solid var(--bor)}}
.sf-row{{display:flex;justify-content:space-between;font-size:11px;color:var(--mut);margin-bottom:5px}}
.sf-val{{font-family:var(--mono);color:var(--txt)}}

/* main */
.main{{overflow-y:auto;padding:28px 32px 60px}}
.section{{display:none}}.section.active{{display:block}}

/* cards */
.card{{background:var(--sur);border:1px solid var(--bor);border-radius:10px;padding:18px 22px;margin-bottom:14px}}
.card-label{{font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--mut);margin-bottom:12px}}
.page-title{{font-size:22px;font-weight:600;letter-spacing:-.02em;margin-bottom:6px}}
.page-sub{{font-size:13px;color:var(--mut);margin-bottom:24px}}

/* meta grid */
.meta-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}}
.meta-card{{background:var(--sur);border:1px solid var(--bor);border-radius:8px;padding:12px 14px}}
.meta-label{{font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--mut);margin-bottom:3px}}
.meta-val{{font-family:var(--mono);font-size:18px;font-weight:500}}
.meta-unit{{font-size:11px;color:var(--mut)}}

/* axes */
.axes{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:14px}}
.axis-card{{background:var(--sur);border:1px solid var(--bor);border-radius:8px;padding:14px}}
.axis-head{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.axis-dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0}}
.axis-name{{font-family:var(--mono);font-weight:500;font-size:14px}}
.axis-desc{{font-size:12px;color:var(--mut);line-height:1.7}}

/* origin box */
.origin-box{{background:#1a1508;border:1px solid #3d2e10;border-radius:8px;padding:16px 18px;margin-bottom:14px}}
.origin-title{{font-weight:600;font-size:13px;color:var(--warn);margin-bottom:8px}}
.code-block{{font-family:var(--mono);font-size:12px;color:#a0c8ff;background:#0d1020;padding:10px 14px;border-radius:6px;line-height:1.9;overflow-x:auto;margin-top:10px}}

/* 3D viewport */
#vp{{width:100%;height:500px;border-radius:10px;overflow:hidden;background:#090b0f;border:1px solid var(--bor);position:relative;cursor:grab}}
#vp:active{{cursor:grabbing}}
.vp-hint{{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);font-size:11px;color:rgba(255,255,255,.2);pointer-events:none}}
.tt{{position:absolute;top:10px;right:10px;background:rgba(13,15,20,.93);border:1px solid var(--bor);border-radius:8px;padding:11px 14px;font-size:11px;display:none;min-width:190px}}
.tt-id{{font-size:14px;font-family:var(--mono);font-weight:500;color:var(--warn);margin-bottom:7px}}
.tt-row{{color:var(--mut);margin-bottom:2px}}.tt-row span{{color:var(--acc);font-family:var(--mono)}}

/* table */
.table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-family:var(--mono)}}
thead tr{{background:#1c1f28}}
thead th{{padding:9px 12px;text-align:left;font-size:10px;font-weight:500;letter-spacing:.07em;color:var(--mut);white-space:nowrap;border-bottom:1px solid var(--bor)}}
tbody tr{{border-bottom:1px solid var(--bor);transition:background .1s}}
tbody tr:hover{{background:#181b24}}
tbody td{{padding:9px 12px;vertical-align:middle;white-space:nowrap}}
.pill{{display:inline-block;font-size:10px;padding:2px 8px;border-radius:20px;font-family:var(--sans);font-weight:500}}
.pill-ok{{background:var(--ok2);color:var(--ok)}}
.pill-err{{background:var(--err2);color:var(--err)}}
.pill-warn{{background:var(--warn2);color:var(--warn)}}
.cell-err{{color:var(--err)}}
.note-cell{{font-family:var(--sans);font-size:11px;color:var(--mut);max-width:220px;white-space:normal;line-height:1.5}}
.issue-cell{{font-family:var(--sans);font-size:11px;color:var(--err);max-width:180px;white-space:normal}}
</style>
</head>
<body>

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="brand">
    <div class="brand-title">AprilTag 3D Report</div>
    <div class="brand-sub">PUPIL LABS NEON · HEAD POSE TRACKER</div>
  </div>
  <div class="nav-label">Sections</div>
  <div class="nav-item active" onclick="show('overview',this)">Overview</div>
  <div class="nav-item" onclick="show('viewer',this)">3D Viewer</div>
  <div class="nav-item" onclick="show('table',this)">Marker Table</div>
  <div class="nav-item" onclick="show('code',this)">Code Reference</div>
  <div class="sidebar-footer">
    <div class="sf-row"><span>Scale</span><span class="sf-val">{scale} cm/unit</span></div>
    <div class="sf-row"><span>Origin</span><span class="sf-val">Room corner</span></div>
    <div class="sf-row"><span>Anchor tag</span><span class="sf-val">#403 ↙</span></div>
    <div class="sf-row"><span>Valid markers</span><span class="sf-val" style="color:var(--ok)">10 / 14</span></div>
  </div>
</div>

<!-- MAIN -->
<div class="main">

<!-- OVERVIEW -->
<div id="s-overview" class="section active">
  <div class="page-title">Coordinate System Overview</div>
  <div class="page-sub">World origin = room corner at floor level · all units in cm</div>

  <div class="meta-grid">
    <div class="meta-card">
      <div class="meta-label">Anchor tag</div>
      <div class="meta-val">#403 <span class="meta-unit">bottom-left</span></div>
    </div>
    <div class="meta-card">
      <div class="meta-label">Scale</div>
      <div class="meta-val">{scale} <span class="meta-unit">cm / unit</span></div>
    </div>
    <div class="meta-card">
      <div class="meta-label">Valid / Total</div>
      <div class="meta-val" style="color:var(--ok)">10 <span class="meta-unit" style="color:var(--mut)">/ 14 markers</span></div>
    </div>
  </div>

  <div class="origin-box">
    <div class="origin-title">📍 Origin = Room Corner (floor level)</div>
    <div style="font-size:13px;color:#c8a870">
      Tag 403 bottom-left corner = desk corner, measured from room corner:
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:10px;font-size:12px">
      <div><div style="color:var(--mut)">X (along wall)</div><div style="font-family:var(--mono);font-size:15px;color:#ff8888">{x_offset:.2f} cm = 28 inch</div></div>
      <div><div style="color:var(--mut)">Y (desk height)</div><div style="font-family:var(--mono);font-size:15px;color:#88cc88">{y_offset:.2f} cm = 2.9 ft</div></div>
      <div><div style="color:var(--mut)">Z (from wall)</div><div style="font-family:var(--mono);font-size:15px;color:#8899ff">{z_offset:.2f} cm = 6.4 ft</div></div>
    </div>
  </div>

  <div class="axes">
    <div class="axis-card">
      <div class="axis-head"><div class="axis-dot" style="background:#ff4444"></div><div class="axis-name">X Axis</div></div>
      <div class="axis-desc">Rightward along wall.<br>0 = room corner wall<br>+{x_offset:.0f} cm = desk corner</div>
    </div>
    <div class="axis-card">
      <div class="axis-head"><div class="axis-dot" style="background:#44cc77"></div><div class="axis-name">Y Axis</div></div>
      <div class="axis-desc">Upward from floor.<br>0 = floor level<br>+{y_offset:.0f} cm = desk surface</div>
    </div>
    <div class="axis-card">
      <div class="axis-head"><div class="axis-dot" style="background:#4488ff"></div><div class="axis-name">Z Axis</div></div>
      <div class="axis-desc">Into room from wall.<br>0 = back wall<br>+{z_offset:.0f} cm = desk position</div>
    </div>
  </div>

  <div class="card">
    <div class="card-label">Known Issues</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;font-size:12px">
      <div style="background:#1a0f0e;border:1px solid #3d1a14;border-radius:8px;padding:14px">
        <div style="font-weight:600;color:var(--err);margin-bottom:8px">✗ Anomaly Markers (4)</div>
        <div style="color:#d08070;line-height:2">
          <div>504, 505, 508 — Z sign flipped during scan</div>
          <div>501 — Y below floor level (severe drift)</div>
        </div>
      </div>
      <div style="background:#0e1a13;border:1px solid #1a3d2a;border-radius:8px;padding:14px">
        <div style="font-weight:600;color:var(--ok);margin-bottom:8px">✓ Next Steps</div>
        <div style="color:#70c090;line-height:2">
          <div>Re-scan corner (504/505/508) facing head-on</div>
          <div>Physically measure tag 403 X-offset to confirm</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- 3D VIEWER -->
<div id="s-viewer" class="section">
  <div class="page-title">3D Viewer</div>
  <div class="page-sub">Drag to rotate · Scroll to zoom · Right-click to pan · Click marker for details</div>
  <div style="position:relative">
    <div id="vp">
      <div class="vp-hint">Drag · Scroll · Right-click pan</div>
      <div class="tt" id="tt">
        <div class="tt-id" id="tt-id">—</div>
        <div class="tt-row">X: <span id="tt-x">—</span></div>
        <div class="tt-row">Y: <span id="tt-y">—</span></div>
        <div class="tt-row">Z: <span id="tt-z">—</span></div>
        <div style="margin-top:7px;font-size:11px;color:var(--mut)" id="tt-loc">—</div>
        <div style="margin-top:4px;font-size:11px" id="tt-st">—</div>
      </div>
    </div>
  </div>
</div>

<!-- MARKER TABLE -->
<div id="s-table" class="section">
  <div class="page-title">Marker Position Data</div>
  <div class="page-sub">All coordinates in cm · Origin = room corner at floor level</div>
  <div class="card" style="padding:0;overflow:hidden">
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Status</th>
            <th>X (cm)</th>
            <th>Y (cm)</th>
            <th>Z (cm)</th>
            <th>X (inch)</th>
            <th>Y (ft)</th>
            <th>Z (ft)</th>
            <th>Normal X</th>
            <th>Normal Y</th>
            <th>Normal Z</th>
            <th>Surface</th>
            <th>Location</th>
            <th>Issue</th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- CODE REFERENCE -->
<div id="s-code" class="section">
  <div class="page-title">Code Reference</div>
  <div class="page-sub">Copy-paste ready · all units in cm</div>

  <div class="card">
    <div class="card-label">Coordinate System Setup</div>
    <div class="code-block">import numpy as np

SCALE_CM  = {scale}   # 1 model unit = {scale} cm

# Room corner at floor level = world origin (0, 0, 0) cm
# Tag 403 bottom-left = desk corner, measured from room corner:
#   X = {x_offset:.2f} cm  (28 inch along wall)
#   Y = {y_offset:.2f} cm  (2.9 ft desk height)
#   Z = {z_offset:.2f} cm  (6.4 ft from wall)

ORIGIN_CM = np.array([{ORIGIN_CM[0]:.2f}, {ORIGIN_CM[1]:.2f}, {ORIGIN_CM[2]:.2f}])  # cm</div>
  </div>

  <div class="card">
    <div class="card-label">Coordinate Transform Functions</div>
    <div class="code-block">def to_world_cm(model_pos_units):
    \"\"\"
    Convert model-space position (units) → world coordinates (cm).
    World origin = room corner at floor level.

    Returns [x_cm, y_cm, z_cm] where:
      x > 0  rightward along wall
      y > 0  upward  (0=floor, ~{y_offset:.0f}=desk)
      z > 0  into room away from wall
    \"\"\"
    model_cm = np.array(model_pos_units, dtype=float) * SCALE_CM
    return model_cm - ORIGIN_CM


def to_model_units(world_cm):
    \"\"\"Convert world coordinates (cm) → model units.\"\"\"
    model_cm = np.array(world_cm, dtype=float) + ORIGIN_CM
    return model_cm / SCALE_CM


# ── Usage examples ───────────────────────────────────────
# Tag 403 bottom-left corner (0,1,0) in model
print(to_world_cm([0, 1, 0]))
# → [ {x_offset:.2f}  {y_offset:.2f}  {z_offset:.2f}] cm  ✓

# Convert any gaze position from model to world
# gaze_world_cm = to_world_cm(gaze_model_units)</div>
  </div>

  <div class="card">
    <div class="card-label">Verification</div>
    <div class="code-block"># All three physical measurements should match:
w = to_world_cm([0, 1, 0])   # tag 403 bottom-left
assert abs(w[0] - {x_offset:.2f}) < 0.1,  f"X mismatch: {{w[0]:.2f}} != {x_offset:.2f}"
assert abs(w[1] - {y_offset:.2f}) < 0.1,  f"Y mismatch: {{w[1]:.2f}} != {y_offset:.2f}"
assert abs(w[2] - {z_offset:.2f}) < 0.1,  f"Z mismatch: {{w[2]:.2f}} != {z_offset:.2f}"
print("✓ All physical measurements verified")</div>
  </div>
</div>

</div><!-- /main -->

<script>
const MARKERS = {markers_js};
const ORIGIN_CM = {origin_js};
const SCALE = {scale};

// ── Navigation ─────────────────────────────────────────────────────────────
const secs = {{}};
document.querySelectorAll('.section').forEach(s => secs[s.id] = s);
function show(name, el) {{
  Object.values(secs).forEach(s => s.classList.remove('active'));
  secs['s-'+name].classList.add('active');
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  if(el) el.classList.add('active');
  if(name==='viewer' && !window._3d) init3D();
}}

// ── Marker Table ────────────────────────────────────────────────────────────
const tbody = document.getElementById('tbody');
MARKERS.forEach(m => {{
  const pc = m.status==='anchor' ? 'pill-warn' : m.status==='valid' ? 'pill-ok' : 'pill-err';
  const pt = m.status==='anchor' ? '⚓ Anchor' : m.status==='valid' ? '✓ Valid' : '✗ Anomaly';
  const yc = m.y_cm < -5 ? 'cell-err' : '';
  const zc = m.z_cm < -150 ? 'cell-err' : '';
  tbody.innerHTML += `<tr>
    <td style="font-weight:500;font-size:13px">${{m.id}}</td>
    <td><span class="pill ${{pc}}">${{pt}}</span></td>
    <td>${{m.x_cm > 0 ? '+' : ''}}${{m.x_cm}}</td>
    <td class="${{yc}}">${{m.y_cm > 0 ? '+' : ''}}${{m.y_cm}}</td>
    <td class="${{zc}}">${{m.z_cm > 0 ? '+' : ''}}${{m.z_cm}}</td>
    <td style="color:var(--mut)">${{m.x_in > 0 ? '+' : ''}}${{m.x_in}}</td>
    <td style="color:var(--mut)">${{m.y_ft > 0 ? '+' : ''}}${{m.y_ft}}</td>
    <td style="color:var(--mut)">${{m.z_ft > 0 ? '+' : ''}}${{m.z_ft}}</td>
    <td style="color:var(--mut);font-size:11px">${{m.nx}}</td>
    <td style="color:var(--mut);font-size:11px">${{m.ny}}</td>
    <td style="color:var(--mut);font-size:11px">${{m.nz}}</td>
    <td style="font-size:11px;color:var(--mut)">${{m.surface}}</td>
    <td class="note-cell">${{m.location}}</td>
    <td class="issue-cell">${{m.issues || '—'}}</td>
  </tr>`;
}});

// ── 3D Viewer ───────────────────────────────────────────────────────────────
function init3D() {{
  window._3d = true;
  const vp = document.getElementById('vp');
  const W = () => vp.clientWidth, H = () => vp.clientHeight;

  const renderer = new THREE.WebGLRenderer({{antialias:true}});
  renderer.setPixelRatio(devicePixelRatio);
  renderer.setSize(W(), H());
  renderer.setClearColor(0x090b0f, 1);
  vp.insertBefore(renderer.domElement, vp.firstChild);

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x090b0f, 0.018);
  const cam = new THREE.PerspectiveCamera(50, W()/H(), 0.01, 80);

  scene.add(new THREE.AmbientLight(0xffffff, 0.3));
  const sun = new THREE.DirectionalLight(0x8ab4ff, 0.7);
  sun.position.set(5,8,-3); scene.add(sun);

  // Grid at Y=0 (floor level)
  const grid = new THREE.GridHelper(10, 20, 0x151820, 0x151820);
  scene.add(grid);

  // Axes from origin (0,0,0) = room corner
  const axDefs = [[[1,0,0],0xff4444,6],[[0,1,0],0x44cc77,4],[[0,0,1],0x4488ff,6]];
  axDefs.forEach(([d,c,l]) => {{
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0,0,0), new THREE.Vector3(...d).multiplyScalar(l)]);
    scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({{color:c}})));
    const cap = new THREE.Mesh(new THREE.SphereGeometry(0.07,8,8),
      new THREE.MeshBasicMaterial({{color:c}}));
    cap.position.set(...new THREE.Vector3(...d).multiplyScalar(l).toArray());
    scene.add(cap);
  }});

  // Origin marker
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.12,0.19,48),
    new THREE.MeshBasicMaterial({{color:0xffbb30,side:THREE.DoubleSide,transparent:true,opacity:0.7}})
  );
  ring.rotation.x = -Math.PI/2; scene.add(ring);
  const origSph = new THREE.Mesh(new THREE.SphereGeometry(0.09,16,16),
    new THREE.MeshStandardMaterial({{color:0xffd060,emissive:0xffd060,emissiveIntensity:0.9}}));
  scene.add(origSph);

  // Desk outline (wireframe box)
  const deskW=0.711, deskH=0.884, deskD=1.951;
  const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(deskW,0.02,deskD));
  const deskTop = new THREE.LineSegments(edges,
    new THREE.LineBasicMaterial({{color:0x334455,transparent:true,opacity:0.5}}));
  deskTop.position.set(0.711/2+0.711/2, 0.884, 1.951/2);
  scene.add(deskTop);

  // Marker spheres
  const meshes = [];
  const raycaster = new THREE.Raycaster();
  const mouse2 = new THREE.Vector2();

  MARKERS.forEach(m => {{
    const pos = new THREE.Vector3(m.x_cm/100, m.y_cm/100, m.z_cm/100);
    const bad = m.status==='anomaly';
    const isAnc = m.status==='anchor';
    const col = isAnc ? 0xffd060 : bad ? 0xff3322 : 0x3ec88a;
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(isAnc?0.08:0.06,16,16),
      new THREE.MeshStandardMaterial({{color:col,emissive:col,
        emissiveIntensity:isAnc?0.7:bad?0.4:0.15,roughness:0.4}})
    );
    mesh.position.copy(pos);
    mesh.userData = m;
    scene.add(mesh); meshes.push(mesh);

    if(!bad) {{
      const lm = new THREE.LineBasicMaterial({{color:isAnc?0x443300:0x1a3a22,transparent:true,opacity:0.4}});
      const lg = new THREE.BufferGeometry().setFromPoints([pos.clone(),new THREE.Vector3(pos.x,0,pos.z)]);
      scene.add(new THREE.Line(lg,lm));
    }}
  }});

  // Orbit controls
  let drag=false,rc=false,lx=0,ly=0,theta=0.6,phi=0.4,rad=8,px=1,py=0.5;
  const recam=()=>{{
    cam.position.set(px+rad*Math.sin(theta)*Math.cos(phi),py+rad*Math.sin(phi),rad*Math.cos(theta)*Math.cos(phi));
    cam.lookAt(px,py,1);
  }};
  recam();
  renderer.domElement.addEventListener('mousedown',e=>{{drag=true;rc=e.button===2;lx=e.clientX;ly=e.clientY;}});
  window.addEventListener('mouseup',()=>drag=false);
  window.addEventListener('mousemove',e=>{{
    if(!drag) return;
    const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
    if(rc){{px-=dx*.006;py+=dy*.006;}}else{{theta-=dx*.007;phi=Math.max(-1.3,Math.min(1.3,phi+dy*.007));}}
    recam();
  }});
  renderer.domElement.addEventListener('wheel',e=>{{rad=Math.max(1.5,Math.min(20,rad+e.deltaY*.012));recam();}});
  renderer.domElement.addEventListener('contextmenu',e=>e.preventDefault());

  // Click pick
  const tt = document.getElementById('tt');
  renderer.domElement.addEventListener('click',e=>{{
    const rect=renderer.domElement.getBoundingClientRect();
    mouse2.x=((e.clientX-rect.left)/W())*2-1;
    mouse2.y=-((e.clientY-rect.top)/H())*2+1;
    raycaster.setFromCamera(mouse2,cam);
    const hits=raycaster.intersectObjects(meshes);
    if(hits.length){{
      const m=hits[0].object.userData;
      document.getElementById('tt-id').textContent=`Marker ${{m.id}}`;
      document.getElementById('tt-x').textContent=`${{m.x_cm>0?'+':''}}${{m.x_cm}} cm`;
      document.getElementById('tt-y').textContent=`${{m.y_cm>0?'+':''}}${{m.y_cm}} cm`;
      document.getElementById('tt-z').textContent=`${{m.z_cm>0?'+':''}}${{m.z_cm}} cm`;
      document.getElementById('tt-loc').textContent=m.location;
      document.getElementById('tt-st').innerHTML=m.status==='anomaly'
        ?'<span style="color:#ff4d3a">✗ Anomaly</span>'
        :m.status==='anchor'?'<span style="color:#ffb84a">⚓ Anchor</span>'
        :'<span style="color:#3ec88a">✓ Valid</span>';
      tt.style.display='block';
    }}else tt.style.display='none';
  }});

  let t=0;
  (function loop(){{
    requestAnimationFrame(loop);t+=0.018;
    ring.scale.setScalar(1+0.06*Math.sin(t));
    ring.material.opacity=0.45+0.25*Math.sin(t);
    renderer.render(scene,cam);
  }})();

  new ResizeObserver(()=>{{renderer.setSize(W(),H());cam.aspect=W()/H();cam.updateProjectionMatrix();}}).observe(vp);
}}
</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    print(f"[Report] Saved → {os.path.abspath(output_path)}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  AprilTag 3D Coordinate System Builder")
    print("=" * 65)

    # Quick self-check
    w = to_world_cm([0, 1, 0])  # tag 403 bottom-left
    assert abs(w[0] - X_OFFSET_CM) < 0.1, f"X mismatch: {w[0]:.2f}"
    assert abs(w[1] - Y_OFFSET_CM) < 0.1, f"Y mismatch: {w[1]:.2f}"
    assert abs(w[2] - Z_OFFSET_CM) < 0.1, f"Z mismatch: {w[2]:.2f}"
    print("  ✓ Physical measurements verified")

    print(f"\n  Loading {MODEL_CSV} ...")
    markers = load_markers(MODEL_CSV)
    print(f"  Loaded {len(markers)} markers")

    print_summary(markers)

    print(f"  Generating {OUTPUT_HTML} ...")
    generate_html(markers, OUTPUT_HTML)

    print("  Opening in browser...")
    webbrowser.open(f"file://{os.path.abspath(OUTPUT_HTML)}")
    print("\nDone ✓")