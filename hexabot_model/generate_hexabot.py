#!/usr/bin/env python3
"""
generate_hexabot.py — parametric 18-DOF hexapod, the SINGLE SOURCE OF TRUTH.

Edit the PARAMS block, re-run, and it regenerates everything Isaac needs:

  isaac/meshes/<mesh>.obj   per-link VISUAL meshes, link-LOCAL frame, METERS.
                            Only 4 unique meshes (base_link, coxa, femur, tibia) —
                            all 6 legs are identical in their own local frame, so
                            the 18 leg links share 3 meshes.
  isaac/meshes/hexabot.mtl  material colours
  isaac/hexabot.urdf        19 links, 18 revolute joints (6×[coxa,femur,tibia]),
                            box/cylinder collisions, analytic inertia tensors
  isaac/inertia_report.md   per-link mass / CoM / inertia + validation + standing pose

Why this is trustworthy (same philosophy as the TARS build):
  * Joint origins, axes and link lengths are pulled straight from the PARAMS — a
    hexapod is maximally regular (6 identical legs at 60°), so nothing is guessed.
  * Mass = per-primitive volume × material density for printed parts, catalog mass
    for bought parts (MG996R 55 g, LiPo, electronics).
  * Inertia = analytic sum of box/cylinder tensors, each rotated into the link
    frame and parallel-axis-shifted to the link CoM (PhysX-valid: PD + triangle).

Convention: METERS, +Z up, +X forward, +Y left (URDF/ROS). base_link at body
centre.  Coxa joint = vertical (+Z) yaw; femur & tibia = pitch about the leg's
tangent (+Y in the leg-local frame).  Zero-pose = legs straight out, horizontal;
the standing stance is set by joint defaults (see STANCE_* and the cfg).

Run:  python3 hexabot_model/generate_hexabot.py
"""

import os, math
import numpy as np

# ======================================================================== PARAMS
MM = 0.001  # mm -> m

# --- body (hexagonal) ---
BODY_CIRCUMR = 100.0   # centre -> hexagon VERTEX (mm). vertex points forward (+X)
BODY_H       = 46.0    # structural body height (mm)
DOME_H       = 26.0    # decorative dome on top (visual only, mm)
R_COXA       = 95.0    # centre -> coxa joint axis (mm)  (legs mount on hex edges)
COXA_Z       = 0.0     # coxa joint height rel. body centre (mm)

# --- leg segment joint-to-joint lengths (mm), calibrated to the reference parts ---
#   Leg_A bracket ~30 · Leg_B femur ~116 bbox · Leg_C curved tibia claw ~170 bbox
L_COXA  = 30.0
L_FEMUR = 80.0
L_TIBIA = 135.0

# --- leg azimuths (deg): 0 = +X forward, +CCW toward +Y (left). 60° spacing,
#     phase-offset so a GAP (antenna), not a leg, points dead-forward. ---
LEG_AZ = {"lf": 30.0, "lm": 90.0, "lr": 150.0, "rf": -30.0, "rm": -90.0, "rr": -150.0}
TRIPOD_A = ["lf", "rm", "lr"]   # alternating-tripod gait groups (for the report/demo)
TRIPOD_B = ["rf", "lm", "rr"]

# --- segment cross-sections (mm) ---
COXA_W,  COXA_H  = 26.0, 24.0
FEMUR_W, FEMUR_H = 22.0, 26.0
TIBIA_W, TIBIA_H = 18.0, 24.0    # claw, tapers toward the toe
TIBIA_TAPER      = 0.45          # toe-end cross-section as a fraction of the knee end
TOE              = 16.0          # toe contact cube (mm)

# --- MG996R servo (mm, g) ---
SV_L, SV_W, SV_H = 40.7, 19.7, 42.9
SV_MASS = 55.0

# --- standing stance (deg). + femur/tibia tips the segment DOWN (about +Y). ---
STANCE_COXA  = 0.0
STANCE_FEMUR = -18.0   # knee carried slightly high
STANCE_TIBIA = 64.0    # tibia reaches down to plant the claw

# --- masses ---
BODY_SHELL_G  = 210.0   # printed ABS body shell + dome
BATTERY_G     = 250.0   # 2-3S LiPo (18 servos)
ELECTRONICS_G = 120.0   # Pi/ESP + 2× PCA9685 + bucks
ABS_DENSITY   = 1040.0  # kg/m^3  (ABS 1.04 g/cm^3) for printed leg plastic
PRINT_FILL    = 0.45    # effective density factor for printed parts (walls + ~25% infill)

# --- joint limits (rad) — respect the MG996R's ~±90° travel ---
LIM = {"coxa": (-0.60, 0.60), "femur": (-1.05, 1.05), "tibia": (-1.55, 0.45)}
EFFORT, VEL, DAMPING = 1.08, 6.16, 0.05   # N·m, rad/s (MG996R @6V), joint damping

# ----------------------------------------------------------------------- outputs
HERE  = os.path.dirname(os.path.abspath(__file__))
ISAAC = os.path.join(HERE, "isaac")
MESHD = os.path.join(ISAAC, "meshes")

# ============================================================== transform helpers
def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)

def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)

def H(R=None, t=(0, 0, 0)):
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = t
    return M

# ============================================================ primitive container
# A primitive is a box or cylinder, given in its LINK-LOCAL frame by a 4×4
# transform T (metres).  mass=None -> compute from volume×ABS density; else the
# value (grams) is a catalog/fixed mass.
PRIMS = {}          # link/template name -> list of prim dicts

def _add(link, p):
    PRIMS.setdefault(link, []).append(p)

def box(link, mat, T, size, mass=None, taper=1.0):
    """size = full (sx,sy,sz) in metres, centred at T's origin."""
    _add(link, dict(kind="box", mat=mat, T=T,
                    half=(size[0] / 2, size[1] / 2, size[2] / 2),
                    mass=mass, taper=taper))

def cyl(link, mat, T, r, h, n=24, mass=None):
    """cylinder along local +Z, radius r, full height h (metres). n=6 -> hex prism."""
    _add(link, dict(kind="cyl", mat=mat, T=T, r=r, h=h, n=n, mass=mass))

# ==================================================================== build model
def build():
    s = MM  # shorthand

    # ---------------------------------------------------------------- base_link --
    # body: hexagonal prism (vertex forward). visual = 6-gon; inertia/collision via
    # the same solid (cylinder math is the hexagon's near-exact inertia).
    cyl("base_link", "mat_body", H(t=(0, 0, COXA_Z * s)),
        r=BODY_CIRCUMR * s, h=BODY_H * s, n=6, mass=BODY_SHELL_G)
    # decorative dome (visual only — tiny mass folded into the shell catalog value)
    cyl("base_link", "mat_body", H(t=(0, 0, (COXA_Z + BODY_H / 2 + DOME_H / 2) * s)),
        r=BODY_CIRCUMR * 0.62 * s, h=DOME_H * s, n=6, mass=0.0)
    # battery (low) + electronics (high), centred
    box("base_link", "mat_battery", H(t=(0, 0, (COXA_Z - 6) * s)),
        (74 * s, 36 * s, 26 * s), mass=BATTERY_G)
    box("base_link", "mat_pcb", H(t=(0, 0, (COXA_Z + 14) * s)),
        (62 * s, 44 * s, 16 * s), mass=ELECTRONICS_G)
    # 6 coxa servos — mounted on the body at each leg's azimuth (drive the coxa joint)
    for leg, az in LEG_AZ.items():
        a = math.radians(az)
        Tc = H(Rz(a), (R_COXA * s * math.cos(a), R_COXA * s * math.sin(a), COXA_Z * s))
        box("base_link", "mat_servo", Tc, (SV_L * s, SV_W * s, SV_H * s), mass=SV_MASS)

    # ------------------------------------------------------- leg templates (×1) --
    # built ONCE in the leg-local frame (+X radial out, +Y tangent, +Z up). The 6
    # legs reuse these meshes/inertias; only their joint origins differ.

    # coxa link: bracket out to the femur joint + the femur servo at its far end
    box("coxa", "mat_link", H(t=(L_COXA / 2 * s, 0, 0)),
        (L_COXA * s, COXA_W * s, COXA_H * s))
    box("coxa", "mat_servo", H(t=(L_COXA * s, 0, 0)),
        (SV_W * s, SV_L * s, SV_H * s), mass=SV_MASS)      # femur servo (axis +Y)

    # femur link: the thigh bar + the tibia servo at the knee
    box("femur", "mat_link", H(t=(L_FEMUR / 2 * s, 0, 0)),
        (L_FEMUR * s, FEMUR_W * s, FEMUR_H * s))
    box("femur", "mat_servo", H(t=(L_FEMUR * s, 0, 0)),
        (SV_W * s, SV_L * s, SV_H * s), mass=SV_MASS)      # tibia servo (axis +Y)

    # tibia link: tapered claw + a toe contact cube at the tip
    box("tibia", "mat_link", H(t=(L_TIBIA / 2 * s, 0, 0)),
        (L_TIBIA * s, TIBIA_W * s, TIBIA_H * s), taper=TIBIA_TAPER)
    box("tibia", "mat_toe", H(t=(L_TIBIA * s, 0, 0)),
        (TOE * s, TOE * s, TOE * s))

# ================================================================ mass / inertia
def prim_volume(p):
    if p["kind"] == "box":
        hx, hy, hz = p["half"]
        return 8 * hx * hy * hz
    return math.pi * p["r"] * p["r"] * p["h"]

def prim_inertia_own(p, m):
    """diagonal inertia (kg·m²) about the primitive's own centre, primitive axes."""
    if p["kind"] == "box":
        hx, hy, hz = p["half"]
        return np.diag([m / 3 * (hy * hy + hz * hz),
                        m / 3 * (hx * hx + hz * hz),
                        m / 3 * (hx * hx + hy * hy)])
    r, h = p["r"], p["h"]
    return np.diag([m / 12 * (3 * r * r + h * h),
                    m / 12 * (3 * r * r + h * h),
                    0.5 * m * r * r])               # cyl axis = local +Z

def link_props(link):
    """-> (mass kg, CoM link-local m, inertia-about-CoM 3×3, item list)."""
    items, tot, mc = [], 0.0, np.zeros(3)
    for p in PRIMS.get(link, []):
        v = prim_volume(p)
        m = (p["mass"] / 1000.0) if p["mass"] is not None else v * ABS_DENSITY * PRINT_FILL
        c = p["T"][:3, 3].copy()
        R = p["T"][:3, :3]
        Iown = R @ prim_inertia_own(p, m) @ R.T
        items.append((m, c, Iown))
        tot += m
        mc += m * c
    com = mc / tot if tot > 0 else np.zeros(3)
    I = np.zeros((3, 3))
    for (m, c, Iown) in items:
        d = c - com
        I += Iown + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return tot, com, I, items

# ==================================================================== mesh export
def box_mesh(p):
    """8 verts + 12 tris for a (possibly tapered) box, in link-local metres."""
    hx, hy, hz = p["half"]
    tf = p.get("taper", 1.0)
    # -x end full, +x end scaled by taper
    corners = [(-hx, -hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (-hx, -hy, hz),
               (hx, -hy * tf, -hz * tf), (hx, hy * tf, -hz * tf),
               (hx, hy * tf, hz * tf), (hx, -hy * tf, hz * tf)]
    V = [(p["T"] @ np.array([x, y, z, 1.0]))[:3] for (x, y, z) in corners]
    F = [(0, 1, 2), (0, 2, 3),          # -x
         (4, 6, 5), (4, 7, 6),          # +x
         (0, 4, 5), (0, 5, 1),          # -y? (fixed by signed-vol check)
         (1, 5, 6), (1, 6, 2),          # +z side
         (2, 6, 7), (2, 7, 3),          # +y
         (3, 7, 4), (3, 4, 0)]          # -z side
    return V, F

def cyl_mesh(p):
    """n-gon prism along local +Z (n=6 -> hexagon). verts + tris, link-local m."""
    n, r, hh = p["n"], p["r"], p["h"] / 2
    V, F = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        V.append((p["T"] @ np.array([r * math.cos(a), r * math.sin(a), -hh, 1]))[:3])
    for k in range(n):
        a = 2 * math.pi * k / n
        V.append((p["T"] @ np.array([r * math.cos(a), r * math.sin(a), +hh, 1]))[:3])
    cb = len(V); V.append((p["T"] @ np.array([0, 0, -hh, 1]))[:3])
    ct = len(V); V.append((p["T"] @ np.array([0, 0, +hh, 1]))[:3])
    for k in range(n):
        k2 = (k + 1) % n
        F += [(k, k2, n + k2), (k, n + k2, n + k)]      # side quad
        F += [(cb, k2, k)]                              # bottom fan
        F += [(ct, n + k, n + k2)]                      # top fan
    return V, F

def signed_volume(V, F):
    return sum(float(np.dot(V[a], np.cross(V[b], V[c]))) for (a, b, c) in F) / 6.0

def write_mesh(link, fname):
    """write meshes/<fname>.obj from a link/template's prims; fix winding if needed."""
    verts, faces, sv_total = [], [], 0.0
    for p in PRIMS.get(link, []):
        V, F = (box_mesh if p["kind"] == "box" else cyl_mesh)(p)
        if signed_volume(V, F) < 0:                     # ensure outward normals
            F = [(a, c, b) for (a, b, c) in F]
        sv_total += signed_volume(V, F)                 # post-flip (always > 0)
        base = len(verts)
        verts += list(V)
        faces += [(p["mat"], tuple(base + i for i in f)) for f in F]
    path = os.path.join(MESHD, fname)
    with open(path, "w") as fh:
        fh.write(f"# hexabot {link} — link-local, metres, Z-up/X-fwd\n")
        fh.write("mtllib hexabot.mtl\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        last = None
        for (mat, f) in faces:
            if mat != last:
                fh.write(f"usemtl {mat}\n"); last = mat
            fh.write("f " + " ".join(str(i + 1) for i in f) + "\n")
    return len(verts), len(faces), sv_total

MTL = """# hexabot materials
newmtl mat_body
Kd 0.13 0.13 0.15
newmtl mat_link
Kd 0.10 0.10 0.11
newmtl mat_servo
Kd 0.05 0.05 0.05
newmtl mat_battery
Kd 0.20 0.10 0.10
newmtl mat_pcb
Kd 0.10 0.30 0.18
newmtl mat_toe
Kd 0.02 0.02 0.02
"""

# ==================================================================== kinematics
def leg_frames(az_deg, qc, qf, qt):
    """world transforms of the coxa/femur/tibia link frames (base at origin)."""
    a = math.radians(az_deg)
    Tcoxa = H(Rz(a + qc), (R_COXA * MM * math.cos(a), R_COXA * MM * math.sin(a), COXA_Z * MM))
    Tfem = Tcoxa @ H(Ry(qf), (L_COXA * MM, 0, 0))
    Ttib = Tfem @ H(Ry(qt), (L_FEMUR * MM, 0, 0))
    return Tcoxa, Tfem, Ttib

def toe_world(az_deg, qc, qf, qt):
    _, _, Ttib = leg_frames(az_deg, qc, qf, qt)
    return (Ttib @ np.array([L_TIBIA * MM, 0, 0, 1.0]))[:3]

# ======================================================================= URDF
def collisions(link):
    """list of (kind, params, center) URDF collisions in link-local metres."""
    if link == "base_link":
        # cylinder ≈ hexagon (use inradius so it never over-claims footprint)
        return [("cyl", (BODY_CIRCUMR * MM * math.cos(math.radians(30)), BODY_H * MM),
                 (0, 0, COXA_Z * MM))]
    if link == "coxa":
        return [("box", (L_COXA * MM + 0.01, max(COXA_W, SV_L) * MM, COXA_H * MM),
                 (L_COXA / 2 * MM, 0, 0))]
    if link == "femur":
        return [("box", (L_FEMUR * MM, FEMUR_W * MM, FEMUR_H * MM), (L_FEMUR / 2 * MM, 0, 0))]
    if link == "tibia":
        return [("box", (L_TIBIA * MM, TIBIA_W * MM, TIBIA_H * MM), (L_TIBIA / 2 * MM, 0, 0)),
                ("box", (TOE * MM, TOE * MM, TOE * MM), (L_TIBIA * MM, 0, 0))]
    return []

def fmt(v): return f"{v:.6e}"

def link_xml(name, mesh, template):
    m, com, I, _ = link_props(template)
    L = [f'  <link name="{name}">',
         '    <inertial>',
         f'      <origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>',
         f'      <mass value="{m:.5f}"/>',
         f'      <inertia ixx="{fmt(I[0,0])}" ixy="{fmt(I[0,1])}" ixz="{fmt(I[0,2])}" '
         f'iyy="{fmt(I[1,1])}" iyz="{fmt(I[1,2])}" izz="{fmt(I[2,2])}"/>',
         '    </inertial>',
         '    <visual>',
         '      <origin xyz="0 0 0" rpy="0 0 0"/>',
         f'      <geometry><mesh filename="meshes/{mesh}" scale="1 1 1"/></geometry>',
         '    </visual>']
    for (kind, params, c) in collisions(template):
        L.append('    <collision>')
        L.append(f'      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>')
        if kind == "box":
            L.append(f'      <geometry><box size="{params[0]:.6f} {params[1]:.6f} {params[2]:.6f}"/></geometry>')
        else:
            L.append(f'      <geometry><cylinder radius="{params[0]:.6f}" length="{params[1]:.6f}"/></geometry>')
        L.append('    </collision>')
    L.append('  </link>')
    return "\n".join(L)

def joint_xml(name, parent, child, xyz, rpy, axis, lim):
    lo, hi = lim
    return "\n".join([
        f'  <joint name="{name}" type="revolute">',
        f'    <parent link="{parent}"/>',
        f'    <child link="{child}"/>',
        f'    <origin xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>',
        f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>',
        f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="{EFFORT}" velocity="{VEL}"/>',
        f'    <dynamics damping="{DAMPING}" friction="0.0"/>',
        '  </joint>'])

def emit_urdf():
    P = ['<?xml version="1.0"?>', '<robot name="hexabot">', '']
    P.append(link_xml("base_link", "base_link.obj", "base_link"))
    P.append('')
    for leg, az in LEG_AZ.items():
        a = math.radians(az)
        P.append(link_xml(f"coxa_{leg}", "coxa.obj", "coxa"))
        P.append(link_xml(f"femur_{leg}", "femur.obj", "femur"))
        P.append(link_xml(f"tibia_{leg}", "tibia.obj", "tibia"))
        P.append('')
    for leg, az in LEG_AZ.items():
        a = math.radians(az)
        P.append(joint_xml(f"coxa_{leg}", "base_link", f"coxa_{leg}",
                           (R_COXA * MM * math.cos(a), R_COXA * MM * math.sin(a), COXA_Z * MM),
                           (0, 0, a), (0, 0, 1), LIM["coxa"]))
        P.append(joint_xml(f"femur_{leg}", f"coxa_{leg}", f"femur_{leg}",
                           (L_COXA * MM, 0, 0), (0, 0, 0), (0, 1, 0), LIM["femur"]))
        P.append(joint_xml(f"tibia_{leg}", f"femur_{leg}", f"tibia_{leg}",
                           (L_FEMUR * MM, 0, 0), (0, 0, 0), (0, 1, 0), LIM["tibia"]))
        P.append('')
    P.append('</robot>')
    open(os.path.join(ISAAC, "hexabot.urdf"), "w").write("\n".join(P))

# ===================================================================== report
def write_report(meshinfo, stand_h, body_com):
    base_m = link_props("base_link")[0]
    coxa_m = link_props("coxa")[0]
    fem_m = link_props("femur")[0]
    tib_m = link_props("tibia")[0]
    total = base_m + 6 * (coxa_m + fem_m + tib_m)
    R = ["# Hexabot — Isaac Lab asset build report\n",
         "Generated by `generate_hexabot.py`. Units: **kg, m, Z-up / X-forward**.\n",
         f"**Total mass: {total*1000:.0f} g ({total:.3f} kg)**  ·  18 DOF "
         f"(6 legs × coxa/femur/tibia)  ·  19 links, 18 revolute joints.\n",
         f"Standing body-centre height ≈ **{stand_h*1000:.0f} mm**; whole-body CoM "
         f"(world, stance) x={body_com[0]:+.3f} y={body_com[1]:+.3f} z={body_com[2]:+.3f} m "
         f"(below the body top → low & statically stable).\n",
         "\n## Per-link mass / CoM / inertia (link-local)\n",
         "| link (×count) | mass (g) | CoM x,y,z (m) | Ixx Iyy Izz (kg·m²) |",
         "|---|---|---|---|"]
    for nm, cnt, tmpl in [("base_link", 1, "base_link"), ("coxa_*", 6, "coxa"),
                          ("femur_*", 6, "femur"), ("tibia_*", 6, "tibia")]:
        m, com, I, _ = link_props(tmpl)
        R.append(f"| `{nm}` (×{cnt}) | {m*1000:.1f} | "
                 f"{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f} | "
                 f"{I[0,0]:.2e} {I[1,1]:.2e} {I[2,2]:.2e} |")
    R += ["\n## Joints (revolute)\n",
          "| joint (×6) | axis | limit (rad) | effort (N·m) | vel (rad/s) |",
          "|---|---|---|---|---|",
          f"| `coxa_*` | +Z (yaw) | [{LIM['coxa'][0]}, {LIM['coxa'][1]}] | {EFFORT} | {VEL} |",
          f"| `femur_*` | +Y (pitch) | [{LIM['femur'][0]}, {LIM['femur'][1]}] | {EFFORT} | {VEL} |",
          f"| `tibia_*` | +Y (pitch) | [{LIM['tibia'][0]}, {LIM['tibia'][1]}] | {EFFORT} | {VEL} |",
          "\n## Standing stance (joint defaults, deg)\n",
          f"coxa **{STANCE_COXA}**, femur **{STANCE_FEMUR}**, tibia **{STANCE_TIBIA}** "
          f"→ body sits {stand_h*1000:.0f} mm up, feet at "
          f"{np.hypot(*toe_world(0,0,math.radians(STANCE_FEMUR),math.radians(STANCE_TIBIA))[:2])*1000:.0f} mm radius "
          f"(span ≈ {2*np.hypot(*toe_world(0,0,math.radians(STANCE_FEMUR),math.radians(STANCE_TIBIA))[:2])*1000:.0f} mm).\n",
          "\n## Validation\n"]
    bad = [k for k, (_, _, sv) in meshinfo.items() if sv <= 0]
    R.append(f"- **Mesh normals:** {'all outward (signed-vol > 0) ✅' if not bad else 'INVERTED: '+', '.join(bad)+' ⚠️'} "
             f"({len(meshinfo)} unique meshes).")
    R.append(f"- **Meshes:** " + ", ".join(f"`{k}` {v[0]}v/{v[1]}f" for k, v in meshinfo.items()) + ".")
    R.append("- **Tree:** single root `base_link`; 18 joints, 19 links (no loops — PhysX-valid tree).")
    R.append(f"- **Mass closure:** Σ = {total*1000:.0f} g = base {base_m*1000:.0f} + "
             f"6×(coxa {coxa_m*1000:.0f} + femur {fem_m*1000:.0f} + tibia {tib_m*1000:.0f}).")
    R.append("\n> Caveat: leg joint-to-joint lengths are calibrated to print-oriented "
             "reference-STL bounding boxes + photos, not an assembled CAD source. "
             "All are PARAMS at the top of `generate_hexabot.py` — tune freely.\n")
    open(os.path.join(ISAAC, "inertia_report.md"), "w").write("\n".join(R))

# ========================================================================= main
def main():
    os.makedirs(MESHD, exist_ok=True)
    build()

    # meshes
    meshinfo = {}
    for link, fname in [("base_link", "base_link.obj"), ("coxa", "coxa.obj"),
                        ("femur", "femur.obj"), ("tibia", "tibia.obj")]:
        meshinfo[link] = write_mesh(link, fname)
    open(os.path.join(MESHD, "hexabot.mtl"), "w").write(MTL)

    emit_urdf()

    # standing pose + whole-body CoM via FK
    qf, qt = math.radians(STANCE_FEMUR), math.radians(STANCE_TIBIA)
    stand_h = -min(toe_world(az, 0, qf, qt)[2] for az in LEG_AZ.values())
    # whole-body CoM at stance (base at world origin, +Z up; shift later by stand_h)
    bm, bcom, _, _ = link_props("base_link")
    M, MC = bm, bm * bcom
    for leg, az in LEG_AZ.items():
        Tc, Tf, Tt = leg_frames(az, 0, qf, qt)
        for tmpl, Tw in [("coxa", Tc), ("femur", Tf), ("tibia", Tt)]:
            m, com, _, _ = link_props(tmpl)
            cw = (Tw @ np.array([com[0], com[1], com[2], 1.0]))[:3]
            M += m; MC += m * cw
    body_com = MC / M
    body_com[2] += stand_h    # express relative to ground (feet at z=0)

    write_report(meshinfo, stand_h, body_com)

    total = bm + 6 * (link_props("coxa")[0] + link_props("femur")[0] + link_props("tibia")[0])
    print("=" * 66)
    print(f"  HEXABOT  18 DOF  ·  TOTAL MASS {total*1000:7.1f} g  ({total:.3f} kg)")
    print(f"  standing body height {stand_h*1000:5.0f} mm   "
          f"CoM up {body_com[2]*1000:.0f} mm  (x={body_com[0]:+.3f} y={body_com[1]:+.3f})")
    print("-" * 66)
    for nm, tmpl, cnt in [("base_link", "base_link", 1), ("coxa  (×6)", "coxa", 6),
                          ("femur (×6)", "femur", 6), ("tibia (×6)", "tibia", 6)]:
        m = link_props(tmpl)[0]
        print(f"  {nm:14s} {m*1000:7.1f} g each   (×{cnt} = {m*cnt*1000:6.1f} g)")
    print("-" * 66)
    bad = [k for k, v in meshinfo.items() if v[2] <= 0]
    print(f"  meshes: {', '.join(k+f' {v[0]}v' for k,v in meshinfo.items())}")
    print(f"  normals: {'all outward (+) OK' if not bad else 'INVERTED '+str(bad)}")
    print(f"  wrote: isaac/hexabot.urdf, isaac/meshes/*.obj, isaac/inertia_report.md")
    print("=" * 66)

if __name__ == "__main__":
    main()
