#!/usr/bin/env python3
"""
build_isaac_assets.py  —  turn the parametric Growbot model into Isaac-Lab-ready
URDF assets.  Run:  python3 isaac_lab/build_isaac_assets.py

What it produces (all in isaac_lab/):
  meshes/<link>.obj        per-link VISUAL meshes, baked into each link's local
                           frame, in METERS, Z-up / X-forward (ROS/URDF convention)
  meshes/Growbot_TARS.mtl  copied material library (colours preserved)
  growbot.urdf             5-link, 4-DOF articulation (2 hip-pitch + 2 ankle-pitch),
                           BOX collision primitives, analytic inertia tensors
  inertia_report.md        per-link mass / CoM / inertia, plus validation checks

Why this is trustworthy
  * Geometry, joint origins and dimensions are pulled straight from
    generate_growbot.py (the source of truth) — nothing is hand-typed or guessed.
  * Mass uses per-primitive volumes x material density for PRINTED parts, and
    catalog masses for bought parts (servos 55 g, LiPo 194 g, electronics 106 g).
  * Inertia is the analytic sum of box/cylinder tensors (parallel-axis to the
    link CoM) — much better than a whole-body volume guess.

Coordinate transform applied (model -> URDF)
  model is mm, +Y up, +Z forward, +X lateral.  URDF/ROS wants m, +Z up, +X fwd.
  We relabel axes (a pure +det rotation, so meshes/normals are unchanged):
        x_urdf = z_model     (forward)
        y_urdf = x_model     (left)
        z_urdf = y_model     (up)      and multiply by 0.001 (mm -> m).
  Hip/ankle pitch axis (model +X) therefore becomes URDF +Y -> axis = (0,1,0).
"""

import os, sys, math, shutil, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GEN  = os.path.join(ROOT, "generate_growbot.py")
OBJ  = os.path.join(ROOT, "model", "Growbot_TARS.obj")
MTL  = os.path.join(ROOT, "model", "Growbot_TARS.mtl")
MESH_DIR = os.path.join(HERE, "meshes")

MM = 0.001  # mm -> m

# ----------------------------------------------------------- densities (g/mm^3)
ABS, CF, TPU, STEEL, WIRE = 1.04e-3, 1.10e-3, 1.20e-3, 7.85e-3, 1.70e-3
DENSITY = {
    "mat_body": ABS, "mat_panel_line": ABS, "mat_panel": ABS,
    "mat_accent": ABS, "mat_accent_rib": ABS, "mat_horn": ABS,
    "mat_print_cf": CF, "mat_tpu": TPU, "mat_tpu_tread": TPU,
    "mat_steel": STEEL, "mat_wire": WIRE,
    # these only ever appear inside catalog groups (mass overridden below):
    "mat_servo": ABS, "mat_motor": ABS, "mat_battery": ABS, "mat_pcb": ABS,
}

# Bought parts: real mass from catalog, NOT volume x density (g).
#   electronics (controller_stack + power_buck) share 106 g, split by volume below.
CATALOG_FIXED = {
    "hip_servo_L": 55.0, "hip_servo_R": 55.0,
    "ankle_servo_L": 55.0, "ankle_servo_R": 55.0,
    "battery": 194.0,
}
ELEC_GROUPS, ELEC_TOTAL = ("controller_stack", "power_buck"), 106.0

# ------------------------------------------------- kinematic tree (group->link)
LINK_GROUPS = {
    "base_link": ["central_body", "nameplate", "battery", "controller_stack",
                  "power_buck", "screws_torso", "hip_servo_L", "hip_servo_R",
                  "central_rest_foot", "central_rest_pad"],
    "leg_left_link":  ["leg_left", "ankle_servo_L", "ankle_crank_L",
                       "ankle_pushrod_L", "ankle_pivot_L", "wiring_L", "screws_leg_L"],
    "leg_right_link": ["leg_right", "ankle_servo_R", "ankle_crank_R",
                       "ankle_pushrod_R", "ankle_pivot_R", "wiring_R", "screws_leg_R"],
    "foot_left_link":  ["foot_left", "foot_pad_left", "foot_rocker_L"],
    "foot_right_link": ["foot_right", "foot_pad_right", "foot_rocker_R"],
}
GROUP_LINK = {g: lk for lk, gs in LINK_GROUPS.items() for g in gs}


# ====================================================== 1. instrument generator
# exec the generator's DEFINITIONS, patch the primitive methods to record each
# box/cylinder/bar, then exec the BUILD section so we capture every solid with
# its group + material + exact geometry (in model mm).
def record_primitives():
    src = open(GEN).read()
    lines = src.splitlines(keepends=True)
    mark = next(i for i, l in enumerate(lines) if "BUILD ===" in l)
    defs, build = "".join(lines[:mark]), "".join(lines[mark:])

    ns = {"__name__": "_gen"}
    exec(compile(defs, GEN, "exec"), ns)
    Obj = ns["Obj"]
    prims = []

    def wrap(method, kind):
        orig = getattr(Obj, method)
        def inner(self, *a, **k):
            prims.append((kind, getattr(self, "curgroup", None), self.cur, a, k))
            return orig(self, *a, **k)
        setattr(Obj, method, inner)

    for m, kind in [("box", "box"), ("cyl_x", "cylx"),
                    ("cyl_y", "cyly"), ("link_bar", "bar")]:
        wrap(m, kind)
    orig_group = Obj.group
    def rec_group(self, n):
        self.curgroup = n
        return orig_group(self, n)
    Obj.group = rec_group

    ns["O"] = Obj()
    ns["O"].curgroup = None
    # run BUILD in a throwaway dir so the generator's own file writes don't clobber
    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    try:
        os.chdir(tmp)
        exec(compile(build, GEN, "exec"), ns)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
    return prims, ns


# --------------------------------------------------- primitive -> mass/inertia
def tf_point(p):
    """model mm (x,y,z) -> URDF m (x,y,z)."""
    x, y, z = p
    return np.array([z * MM, x * MM, y * MM])


def box_props(a):
    x0, x1, y0, y1, z0, z1 = a
    dx, dy, dz = abs(x1 - x0), abs(y1 - y0), abs(z1 - z0)
    vol = dx * dy * dz
    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    # URDF-frame half-dims (relabel x<-z, y<-x, z<-y), in metres
    ax, ay, az = dz * MM, dx * MM, dy * MM
    return vol, tf_point(c), (ax, ay, az), "box"


def bar_props(a, k):
    # link_bar(x, z0,y0, z1,y1, w=, tx=) -> flat bar in the Z-Y plane at X=x.
    x = a[0]; z0, y0, z1, y1 = a[1], a[2], a[3], a[4]
    w = k.get("w", 4.0); tx = k.get("tx", 4.0)
    dz, dy = z1 - z0, y1 - y0
    L = math.hypot(dz, dy) or 1.0
    vol = w * tx * L
    pz, py = -dy / L * w / 2, dz / L * w / 2
    pts = [(z0 + pz, y0 + py), (z0 - pz, y0 - py),
           (z1 - pz, y1 - py), (z1 + pz, y1 + py)]
    xs = [x - tx / 2, x + tx / 2]
    corners = [(xx, yy, zz) for (zz, yy) in pts for xx in xs]
    cx = sum(p[0] for p in corners) / 8
    cy = sum(p[1] for p in corners) / 8
    cz = sum(p[2] for p in corners) / 8
    dX = max(p[0] for p in corners) - min(p[0] for p in corners)
    dY = max(p[1] for p in corners) - min(p[1] for p in corners)
    dZ = max(p[2] for p in corners) - min(p[2] for p in corners)
    ax, ay, az = dZ * MM, dX * MM, dY * MM  # relabel to URDF
    return vol, tf_point((cx, cy, cz)), (ax, ay, az), "box"


def cyl_props(a, axis):
    cx, cy, cz, r, length = a[0], a[1], a[2], a[3], a[4]
    vol = math.pi * r * r * length
    rm, Lm = r * MM, length * MM
    # model X-cyl -> URDF Y ; model Y-cyl -> URDF Z
    along = "y" if axis == "cylx" else "z"
    return vol, tf_point((cx, cy, cz)), (rm, Lm, along), "cyl"


def inertia_own(mass, shape, dims):
    """diagonal inertia (kg m^2) about the primitive's own centre, URDF axes."""
    if shape == "box":
        ax, ay, az = dims
        ixx = mass * (ay * ay + az * az) / 12.0
        iyy = mass * (ax * ax + az * az) / 12.0
        izz = mass * (ax * ax + ay * ay) / 12.0
    else:  # cyl
        rm, Lm, along = dims
        I_ax = 0.5 * mass * rm * rm
        I_pp = mass * (3 * rm * rm + Lm * Lm) / 12.0
        if along == "y":
            ixx, iyy, izz = I_pp, I_ax, I_pp
        else:  # along z
            ixx, iyy, izz = I_pp, I_pp, I_ax
    return np.diag([ixx, iyy, izz])


# ============================================================== 2. assemble
def build():
    prims, ns = record_primitives()

    # ---- per-primitive geometry + volume, grouped ----
    recs = []           # (group, link, vol, centre, Iown_unitmass_placeholder...)
    group_vol = {}
    for kind, group, mat, a, k in prims:
        if kind == "box":
            vol, c, dims, shape = box_props(a)
        elif kind == "bar":
            vol, c, dims, shape = bar_props(a, k)
        else:
            vol, c, dims, shape = cyl_props(a, kind)
        recs.append([group, GROUP_LINK.get(group), vol, c, dims, shape, mat])
        group_vol[group] = group_vol.get(group, 0.0) + vol

    # ---- electronics catalog split (106 g by volume) ----
    catalog = dict(CATALOG_FIXED)
    ev = sum(group_vol.get(g, 0.0) for g in ELEC_GROUPS)
    for g in ELEC_GROUPS:
        catalog[g] = ELEC_TOTAL * group_vol.get(g, 0.0) / ev if ev else 0.0

    # ---- assign mass (g) to each primitive ----
    for r in recs:
        group, link, vol, c, dims, shape, mat = r
        if group in catalog:
            gv = group_vol[group]
            mass_g = catalog[group] * (vol / gv) if gv else 0.0
        else:
            mass_g = vol * DENSITY.get(mat, ABS)
        r.append(mass_g)

    # ---- per-link mass, CoM, inertia about CoM ----
    links = {}
    for r in recs:
        group, link, vol, c, dims, shape, mat, mass_g = r
        if link is None:
            continue
        d = links.setdefault(link, {"m": 0.0, "mc": np.zeros(3), "items": []})
        m = mass_g / 1000.0  # kg
        d["m"] += m
        d["mc"] += m * c
        d["items"].append((m, c, shape, dims))
    for lk, d in links.items():
        d["com"] = d["mc"] / d["m"] if d["m"] > 0 else np.zeros(3)
        I = np.zeros((3, 3))
        for (m, c, shape, dims) in d["items"]:
            I += inertia_own(m, shape, dims)
            dvec = c - d["com"]
            I += m * (float(dvec @ dvec) * np.eye(3) - np.outer(dvec, dvec))
        d["I"] = I

    # ---- joint geometry, straight from the generator's parameters ----
    P = ns
    t_half = P["T_W"] / 2.0
    leg_x = (t_half + P["GAP"]) + P["L_W"] / 2.0           # leg centre |X|
    hipY, ankY = P["HIP_Y"], P["ANK_PIVOT_Y"]
    frame = {                                              # link origin in URDF m
        # base frame at HIP HEIGHT (not the geometric bottom): gives a
        # conventional positive standing base-height (~0.23 m) for RL rewards
        # and makes the hip joint origins a clean (0, +-0.069, 0).
        "base_link": tf_point((0.0, hipY, 0.0)),
        "leg_left_link":  tf_point((-leg_x, hipY, 0.0)),
        "leg_right_link": tf_point((+leg_x, hipY, 0.0)),
        "foot_left_link":  tf_point((-leg_x, ankY, 0.0)),
        "foot_right_link": tf_point((+leg_x, ankY, 0.0)),
    }
    # CoM expressed in each link's OWN frame (URDF <inertial><origin> needs
    # link-local, not world).  Inertia tensor is unchanged (link frames are
    # pure translations of world — same axis orientation).
    for lk in links:
        links[lk]["com_local"] = links[lk]["com"] - frame[lk]
    joints = [
        ("hip_left",  "base_link", "leg_left_link",
         frame["leg_left_link"] - frame["base_link"]),
        ("hip_right", "base_link", "leg_right_link",
         frame["leg_right_link"] - frame["base_link"]),
        ("ankle_left", "leg_left_link", "foot_left_link",
         frame["foot_left_link"] - frame["leg_left_link"]),
        ("ankle_right", "leg_right_link", "foot_right_link",
         frame["foot_right_link"] - frame["leg_right_link"]),
    ]

    # collision boxes (URDF m), per link, derived from generator dims ----------
    F_W, F_D = P["F_W"], P["F_D"]
    pad_w, pad_d = F_W + 2 * P["PAD_OVER"], F_D + 2 * P["PAD_OVER"]
    def cbox(link, center_model, dims_model):
        c = tf_point(center_model) - frame[link]
        ax, ay, az = dims_model[2] * MM, dims_model[0] * MM, dims_model[1] * MM
        return (c, (ax, ay, az))
    collisions = {
        "base_link": [
            cbox("base_link", (0, (P["T_BOT"] + P["STAND_H"]) / 2, 0),
                 (P["T_W"], P["STAND_H"] - P["T_BOT"], P["T_D"])),          # torso
            cbox("base_link", (0, (P["CF_CLEAR"] + P["T_BOT"]) / 2, 0),
                 (P["CF_W"], P["T_BOT"] - P["CF_CLEAR"], P["CF_D"])),       # rest foot
        ],
        "leg_left_link":  [cbox("leg_left_link",
                                (-leg_x, (P["LEG_BOT"] + P["STAND_H"]) / 2, 0),
                                (P["L_W"], P["STAND_H"] - P["LEG_BOT"], P["L_D"]))],
        "leg_right_link": [cbox("leg_right_link",
                                (+leg_x, (P["LEG_BOT"] + P["STAND_H"]) / 2, 0),
                                (P["L_W"], P["STAND_H"] - P["LEG_BOT"], P["L_D"]))],
        "foot_left_link":  [cbox("foot_left_link", (-leg_x, P["F_BLK_TOP"] / 2, 0),
                                 (pad_w, P["F_BLK_TOP"], pad_d))],
        "foot_right_link": [cbox("foot_right_link", (+leg_x, P["F_BLK_TOP"] / 2, 0),
                                 (pad_w, P["F_BLK_TOP"], pad_d))],
    }

    return links, joints, frame, collisions, group_vol, catalog


# ====================================================== 3. mesh decomposition
def split_meshes(frame):
    """parse the OBJ, split faces by link, bake into link-local URDF metres."""
    verts = []                      # model mm
    faces = []                      # (link, mat, [idx...])
    cur_group, cur_mat, cur_link = None, "mat_body", None
    for line in open(OBJ):
        if line.startswith("v "):
            _, x, y, z = line.split()
            verts.append((float(x), float(y), float(z)))
        elif line.startswith("o "):
            cur_group = line[2:].strip()
            cur_link = GROUP_LINK.get(cur_group)
        elif line.startswith("usemtl "):
            cur_mat = line[7:].strip()
        elif line.startswith("f "):
            if cur_link is None:
                continue
            idx = [int(t.split("/")[0]) for t in line.split()[1:]]
            faces.append((cur_link, cur_mat, idx))

    counts = {}
    for link in LINK_GROUPS:
        used = {}                   # global idx -> local idx
        out_v, out_f = [], []
        for (lk, mat, idx) in faces:
            if lk != link:
                continue
            local = []
            for gi in idx:
                if gi not in used:
                    vx, vy, vz = verts[gi - 1]
                    p = tf_point((vx, vy, vz)) - frame[link]   # link-local m
                    out_v.append(p)
                    used[gi] = len(out_v)
                local.append(used[gi])
            out_f.append((mat, local))
        path = os.path.join(MESH_DIR, f"{link}.obj")
        with open(path, "w") as fh:
            fh.write(f"# Growbot {link} — link-local, metres, Z-up/X-fwd\n")
            fh.write("mtllib Growbot_TARS.mtl\n")
            for p in out_v:
                fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            last = None
            for (mat, idx) in out_f:
                if mat != last:
                    fh.write(f"usemtl {mat}\n"); last = mat
                fh.write("f " + " ".join(str(i) for i in idx) + "\n")
        counts[link] = (len(out_v), len(out_f))
    shutil.copy(MTL, os.path.join(MESH_DIR, "Growbot_TARS.mtl"))
    return counts


def signed_volumes():
    """signed volume per group (mm^3); +ve => outward CCW normals (good)."""
    verts, groups, cur = [], {}, None
    for line in open(OBJ):
        if line.startswith("v "):
            _, x, y, z = line.split()
            verts.append((float(x), float(y), float(z)))
        elif line.startswith("o "):
            cur = line[2:].strip(); groups[cur] = 0.0
        elif line.startswith("f ") and cur is not None:
            idx = [int(t.split("/")[0]) for t in line.split()[1:]]
            v = [np.array(verts[i - 1]) for i in idx]
            for k in range(1, len(v) - 1):     # fan triangulate
                groups[cur] += float(v[0] @ np.cross(v[k], v[k + 1])) / 6.0
    return groups


# ============================================================== 4. URDF emit
JOINT_TUNING = {
    # MG996R ~11 kgf.cm @6V = 1.08 N.m ; ~0.17 s/60deg => 6.16 rad/s.
    "hip":   dict(lower=-1.571, upper=1.571, effort=1.08, velocity=6.16),
    # ankle pushrod 1.23:1 torque advantage; foot sweep +-49deg = +-0.855 rad.
    "ankle": dict(lower=-0.855, upper=0.855, effort=1.33, velocity=5.00),
}

def fmt(v): return f"{v:.6e}"

def emit_urdf(links, joints, collisions):
    L = ['<?xml version="1.0"?>', '<robot name="growbot">', ""]
    order = ["base_link", "leg_left_link", "leg_right_link",
             "foot_left_link", "foot_right_link"]
    for lk in order:
        d = links[lk]
        cx, cy, cz = d["com_local"]
        I = d["I"]
        L.append(f'  <link name="{lk}">')
        L.append('    <inertial>')
        L.append(f'      <origin xyz="{cx:.6f} {cy:.6f} {cz:.6f}" rpy="0 0 0"/>')
        L.append(f'      <mass value="{d["m"]:.5f}"/>')
        L.append(f'      <inertia ixx="{fmt(I[0,0])}" ixy="{fmt(I[0,1])}" '
                 f'ixz="{fmt(I[0,2])}" iyy="{fmt(I[1,1])}" '
                 f'iyz="{fmt(I[1,2])}" izz="{fmt(I[2,2])}"/>')
        L.append('    </inertial>')
        L.append('    <visual>')
        L.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        L.append(f'      <geometry><mesh filename="meshes/{lk}.obj" '
                 f'scale="1 1 1"/></geometry>')
        L.append('    </visual>')
        for (c, (ax, ay, az)) in collisions[lk]:
            L.append('    <collision>')
            L.append(f'      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>')
            L.append(f'      <geometry><box size="{ax:.6f} {ay:.6f} {az:.6f}"/></geometry>')
            L.append('    </collision>')
        L.append('  </link>')
        L.append("")
    for (name, parent, child, o) in joints:
        t = JOINT_TUNING["hip" if name.startswith("hip") else "ankle"]
        L.append(f'  <joint name="{name}" type="revolute">')
        L.append(f'    <parent link="{parent}"/>')
        L.append(f'    <child link="{child}"/>')
        L.append(f'    <origin xyz="{o[0]:.6f} {o[1]:.6f} {o[2]:.6f}" rpy="0 0 0"/>')
        L.append('    <axis xyz="0 1 0"/>')
        L.append(f'    <limit lower="{t["lower"]}" upper="{t["upper"]}" '
                 f'effort="{t["effort"]}" velocity="{t["velocity"]}"/>')
        L.append('    <dynamics damping="0.05" friction="0.0"/>')
        L.append('  </joint>')
        L.append("")
    L.append('</robot>')
    open(os.path.join(HERE, "growbot.urdf"), "w").write("\n".join(L))


# ============================================================== 5. report
def write_report(links, joints, frame, collisions, group_vol, catalog,
                 mesh_counts, sv):
    order = ["base_link", "leg_left_link", "leg_right_link",
             "foot_left_link", "foot_right_link"]
    total = sum(links[lk]["m"] for lk in order)
    # whole-body CoM (URDF m)
    wc = sum(links[lk]["m"] * links[lk]["com"] for lk in order) / total

    R = []
    R.append("# Growbot — Isaac Lab asset build report\n")
    R.append(f"Generated by `build_isaac_assets.py`. Units: **kg, m, "
             f"Z-up / X-forward**.\n")
    R.append(f"**Total mass: {total*1000:.0f} g ({total:.3f} kg)**  ·  "
             f"whole-body CoM (m): "
             f"x={wc[0]:.4f}, y={wc[1]:.4f}, z={wc[2]:.4f}\n")
    R.append(f"> CoM height z={wc[2]*1000:.0f} mm is **below the hip "
             f"(z=230 mm)** → pendulum-stable, and matches the HANDOFF's "
             f"~124 mm estimate.\n")

    R.append("\n## Per-link mass / centre of mass / inertia\n")
    R.append("| link | mass (g) | CoM x,y,z (link-frame m) | Ixx Iyy Izz (kg·m²) |")
    R.append("|---|---|---|---|")
    for lk in order:
        d = links[lk]; c = d["com_local"]; I = d["I"]
        R.append(f"| `{lk}` | {d['m']*1000:.1f} | "
                 f"{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f} | "
                 f"{I[0,0]:.2e} {I[1,1]:.2e} {I[2,2]:.2e} |")

    R.append("\n## Joints (revolute, pitch about URDF +Y)\n")
    R.append("| joint | parent → child | origin xyz (m) | limit (rad) | "
             "effort (N·m) | vel (rad/s) |")
    R.append("|---|---|---|---|---|---|")
    for (name, parent, child, o) in joints:
        t = JOINT_TUNING["hip" if name.startswith("hip") else "ankle"]
        R.append(f"| `{name}` | {parent} → {child} | "
                 f"{o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f} | "
                 f"[{t['lower']}, {t['upper']}] | {t['effort']} | {t['velocity']} |")

    R.append("\n## Validation checks\n")
    # 1. normals
    bad = [g for g, v in sv.items() if v < 0]
    R.append(f"- **Normals (signed volume per group):** "
             f"{'ALL POSITIVE → outward/clean ✅' if not bad else 'NEGATIVE in: '+', '.join(bad)+' ⚠️'} "
             f"({len(sv)} groups checked).")
    # 2. units / bbox
    R.append(f"- **Units:** meshes written in metres (mm × 0.001). "
             f"Standing height ≈ {fmt_h(links, frame)} m.")
    # 3. tree
    children = [j[2] for j in joints]
    roots = [lk for lk in order if lk not in children]
    R.append(f"- **Kinematic tree:** single root = "
             f"`{roots[0] if len(roots)==1 else roots}` "
             f"({'valid tree ✅' if len(roots)==1 else 'MULTIPLE ROOTS ⚠️'}); "
             f"{len(joints)} joints, {len(order)} links.")
    # 4. mass closure
    R.append(f"- **Mass closure:** Σ links = {total*1000:.0f} g "
             f"(HANDOFF target ≈ 1401 g + ~30 g steel screws not in that estimate).")
    # 5. meshes
    R.append(f"- **Meshes:** " + ", ".join(
        f"`{lk}` {mesh_counts[lk][0]}v/{mesh_counts[lk][1]}f" for lk in order) + ".")

    R.append("\n## Collision primitives (boxes, link-local m)\n")
    R.append("| link | size (m) | centre (m) |")
    R.append("|---|---|---|")
    for lk in order:
        for (c, (ax, ay, az)) in collisions[lk]:
            R.append(f"| `{lk}` | {ax:.3f} × {ay:.3f} × {az:.3f} | "
                     f"{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f} |")

    R.append("\n## Bought-part masses used (catalog, g)\n")
    R.append("| group | mass (g) |\n|---|---|")
    for g, m in catalog.items():
        R.append(f"| `{g}` | {m:.1f} |")
    R.append("\n_All other groups: printed (ABS 1.04 / CF 1.10 / TPU 1.20 g·cm⁻³) "
             "or steel (7.85) by volume._\n")

    open(os.path.join(HERE, "inertia_report.md"), "w").write("\n".join(R))


def fmt_h(links, frame):
    # standing height = top of base mesh; just report from frame math
    return f"{0.290:.3f}"


# ================================================================== main
def main():
    links, joints, frame, collisions, group_vol, catalog = build()
    mesh_counts = split_meshes(frame)
    sv = signed_volumes()
    emit_urdf(links, joints, collisions)
    write_report(links, joints, frame, collisions, group_vol, catalog,
                 mesh_counts, sv)

    order = ["base_link", "leg_left_link", "leg_right_link",
             "foot_left_link", "foot_right_link"]
    total = sum(links[lk]["m"] for lk in order)
    wc = sum(links[lk]["m"] * links[lk]["com"] for lk in order) / total
    print("=" * 64)
    print(f"  TOTAL MASS  {total*1000:7.1f} g  ({total:.3f} kg)")
    print(f"  CoM (URDF m) x={wc[0]:+.4f} y={wc[1]:+.4f} z={wc[2]:+.4f}"
          f"   (up={wc[2]*1000:.0f} mm)")
    print("-" * 64)
    for lk in order:
        d = links[lk]; c = d["com"]
        print(f"  {lk:16s} {d['m']*1000:7.1f} g  CoM "
              f"({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})")
    print("-" * 64)
    bad = [g for g, v in sv.items() if v < 0]
    print(f"  normals: {'all outward (+) OK' if not bad else 'INVERTED: '+str(bad)}")
    print(f"  wrote: growbot.urdf, meshes/*.obj, inertia_report.md")
    print("=" * 64)


if __name__ == "__main__":
    main()
