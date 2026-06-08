#!/usr/bin/env python3
"""
render_hexabot.py — LOCAL verification renders (runs on this Mac; no GPU/Isaac).

Imports generate_hexabot.py so the geometry, kinematics and the open-loop tripod
gait are the SAME ones the Isaac asset uses — this is the proof that the model
stands and that the gait advances +X, before anything touches a GPU box.

Outputs (in hexabot_model/previews/):
  iso.png  top.png  front.png   the robot at its standing stance
  gait.gif                       one+ cycle of the alternating-tripod gait

Run:  python3 hexabot_model/previews/render_hexabot.py
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import generate_hexabot as gen          # the single source of truth

gen.build()                              # populate gen.PRIMS

# --------------------------------------------------------------- gait (shared)
# Alternating tripod. Each leg: half a cycle STANCE (foot planted, coxa sweeps to
# push the body +X), half SWING (femur/tibia lift + coxa returns). The coxa sweep
# is scaled by sin(azimuth) so every leg pushes the body the SAME way (+X).
COXA_AMP = 0.26     # rad, coxa sweep half-amplitude
LIFT     = 0.55     # rad, femur lift during swing
QF0, QT0 = math.radians(gen.STANCE_FEMUR), math.radians(gen.STANCE_TIBIA)
R_FOOT   = float(np.hypot(*gen.toe_world(0, 0, QF0, QT0)[:2]))   # stance foot radius
STRIDE_CYCLE = 4 * R_FOOT * math.sin(COXA_AMP)                   # body advance / cycle

def gait_pose(phase):
    """phase in [0,1) -> {leg: (qc, qf, qt)} for one full gait cycle."""
    pose = {}
    for leg, az in gen.LEG_AZ.items():
        th = math.radians(az)
        ph = phase if leg in gen.TRIPOD_A else (phase + 0.5) % 1.0
        if ph < 0.5:                              # STANCE
            s = ph / 0.5
            qc = COXA_AMP * math.sin(th) * (2 * s - 1)
            qf, qt = QF0, QT0
        else:                                     # SWING
            s = (ph - 0.5) / 0.5
            qc = COXA_AMP * math.sin(th) * (1 - 2 * s)
            lift = math.sin(math.pi * s)
            qf = QF0 - LIFT * lift
            qt = QT0 + 0.4 * LIFT * lift
        pose[leg] = (qc, qf, qt)
    return pose

STAND_POSE = {leg: (0.0, QF0, QT0) for leg in gen.LEG_AZ}
STAND_H = -min(gen.toe_world(az, 0, QF0, QT0)[2] for az in gen.LEG_AZ.values())

# ------------------------------------------------------------------- geometry
COLOR = {"mat_body": (0.20, 0.20, 0.23), "mat_link": (0.13, 0.13, 0.15),
         "mat_servo": (0.07, 0.07, 0.07), "mat_battery": (0.35, 0.13, 0.13),
         "mat_pcb": (0.10, 0.34, 0.20), "mat_toe": (0.02, 0.02, 0.02)}
LIGHT = np.array([0.4, 0.5, 0.85]); LIGHT = LIGHT / np.linalg.norm(LIGHT)

def _shade(tri, rgb):
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    nn = np.linalg.norm(n)
    d = 0.55 + 0.45 * max(0.0, float(n @ LIGHT) / nn) if nn > 1e-12 else 0.7
    return tuple(min(1.0, c * d) for c in rgb)

def _link_faces(template, Tw):
    out = []
    for p in gen.PRIMS[template]:
        q = dict(p); q["T"] = Tw @ p["T"]
        V, F = (gen.box_mesh if p["kind"] == "box" else gen.cyl_mesh)(q)
        V = [np.asarray(v) for v in V]
        for (a, b, c) in F:
            tri = np.array([V[a], V[b], V[c]])
            out.append((tri, _shade(tri, COLOR[p["mat"]])))
    return out

def robot_faces(pose, base_x=0.0):
    Tbase = gen.H(t=(base_x, 0.0, STAND_H))
    faces = _link_faces("base_link", Tbase)
    for leg, az in gen.LEG_AZ.items():
        Tc, Tf, Tt = gen.leg_frames(az, *pose[leg])
        faces += _link_faces("coxa", Tbase @ Tc)
        faces += _link_faces("femur", Tbase @ Tf)
        faces += _link_faces("tibia", Tbase @ Tt)
    return faces

# -------------------------------------------------------------------- drawing
def draw(ax, faces, xlim, ylim, zlim, elev, azim):
    ax.clear()
    polys = [f[0] for f in faces]
    cols = [f[1] for f in faces]
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.25),
                          linewidths=0.15)
    ax.add_collection3d(pc)
    # ground grid
    gx = np.linspace(xlim[0], xlim[1], 9)
    gy = np.linspace(ylim[0], ylim[1], 9)
    for x in gx:
        ax.plot([x, x], [gy[0], gy[-1]], [0, 0], color=(0.8, 0.8, 0.82), lw=0.4, zorder=0)
    for y in gy:
        ax.plot([gx[0], gx[-1]], [y, y], [0, 0], color=(0.8, 0.8, 0.82), lw=0.4, zorder=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

def still(name, elev, azim, title, arrow=False):
    fig = plt.figure(figsize=(7, 6), dpi=130)
    ax = fig.add_subplot(111, projection="3d")
    L = 0.34
    draw(ax, robot_faces(STAND_POSE), (-L, L), (-L, L), (0, 0.22), elev, azim)
    if arrow:  # mark the walking direction (+X = forward)
        ax.quiver(0.20, 0, 0.004, 0.11, 0, 0, color="tab:red", lw=2,
                  arrow_length_ratio=0.35, zorder=5)
        ax.text(0.345, 0, 0.004, "+X forward", color="tab:red", fontsize=9,
                ha="left", va="center", zorder=5)
    ax.set_title(title, fontsize=11, color="0.2")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)

def main():
    print(f"stance foot radius {R_FOOT*1000:.0f} mm · stride {STRIDE_CYCLE*1000:.0f} mm/cycle "
          f"· body height {STAND_H*1000:.0f} mm")
    still("iso.png",   22, -60, "hexabot — standing (iso)", arrow=True)
    still("top.png",   89, -90, "hexabot — top (tripod A=lf,rm,lr  B=rf,lm,rr)", arrow=True)
    still("front.png",  6,   0, "hexabot — front")

    # ---- gait GIF: robot walks +X across a fixed frame ----
    NCYC, FPC = 2, 22
    nfr = NCYC * FPC
    fig = plt.figure(figsize=(7, 4.2), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    frames = []
    for i in range(nfr):
        phase = (i % FPC) / FPC
        cyc = i / FPC
        bx = cyc * STRIDE_CYCLE
        draw(ax, robot_faces(gait_pose(phase), base_x=bx),
             (-0.34, 0.62), (-0.34, 0.34), (0, 0.22), 24, -62)
        ax.set_title(f"alternating-tripod gait — +X, {STRIDE_CYCLE*1000:.0f} mm/cycle",
                     fontsize=10, color="0.2")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(Image.fromarray(buf.copy()))
    plt.close(fig)
    gif = os.path.join(HERE, "gait.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=70, loop=0)
    print("  wrote gait.gif", f"({nfr} frames)")

if __name__ == "__main__":
    main()
