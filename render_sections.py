#!/usr/bin/env python3
"""Cross-section / cutaway renders of Growbot_TARS.obj.

Geometry is clipped against axis-aligned planes (real polygon clipping, so the
cut edges are clean and you see *into* the cavity), then painter-sorted.
Outputs (to previews/): section_front, section_hip, section_hip_top, section_ankle, wiring.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
import numpy as np

os.makedirs("previews", exist_ok=True)

COL = {  # legible colours for the cutaways
 "mat_body":(0.78,0.79,0.80), "mat_panel_line":(0.70,0.71,0.72),
 "mat_accent":(0.16,0.16,0.17),"mat_accent_rib":(0.24,0.24,0.25),
 "mat_tpu":(0.10,0.10,0.11),  "mat_tpu_tread":(0.16,0.16,0.17),
 "mat_panel":(0.62,0.63,0.65),
 "mat_motor":(0.28,0.33,0.45), "mat_steel":(0.70,0.73,0.78),
 "mat_battery":(0.15,0.62,0.45),"mat_pcb":(0.10,0.45,0.20),
 "mat_servo":(0.16,0.22,0.45), "mat_horn":(0.92,0.92,0.88),
 "mat_print_cf":(0.95,0.55,0.10),   # printed CF drive parts (amber = pops)
 "mat_wire":(0.82,0.14,0.14),       # servo wiring
}

def load():
    V=[]; F=[]; M=[]; G=[]; cur="mat_body"; grp="none"
    for ln in open("model/Growbot_TARS.obj"):
        if ln.startswith("v "):
            _,x,y,z=ln.split(); V.append((float(x),float(y),float(z)))
        elif ln.startswith("o "): grp=ln.split()[1]
        elif ln.startswith("usemtl"): cur=ln.split()[1]
        elif ln.startswith("f "):
            F.append([int(p.split('/')[0])-1 for p in ln.split()[1:]]); M.append(cur); G.append(grp)
    return np.array(V), F, M, G
V,F,M,G = load()

def clip(poly, axis, val, keep_less):
    """Sutherland–Hodgman against one axis-aligned half-space."""
    out=[]; n=len(poly)
    for i in range(n):
        a=poly[i]; b=poly[(i+1)%n]
        da=a[axis]-val; db=b[axis]-val
        ina = da<=0 if keep_less else da>=0
        inb = db<=0 if keep_less else db>=0
        if ina: out.append(a)
        if ina!=inb:
            t=da/(da-db); out.append(a+t*(b-a))
    return np.array(out) if len(out)>=3 else None

LIGHT=np.array([0.45,0.78,0.45]); LIGHT/=np.linalg.norm(LIGHT)

def render(fname, planes, screen, depth_axis, depth_sign, xlim, ylim,
           title="", figsize=(6,9), annot=None):
    polys=[]; cols=[]; deps=[]
    for f,m in zip(F,M):
        p=V[f].astype(float)
        for (ax,val,kl) in planes:
            p=clip(p,ax,val,kl)
            if p is None: break
        if p is None: continue
        nrm=np.cross(p[1]-p[0],p[2]-p[0]); nl=np.linalg.norm(nrm)
        sh=0.55
        if nl>0: sh=0.45+0.55*max(0,abs(np.dot(nrm/nl,LIGHT)))
        polys.append(p[:,screen]); cols.append(np.clip(np.array(COL.get(m,(.7,.7,.7)))*sh,0,1))
        deps.append(depth_sign*p[:,depth_axis].mean())
    order=np.argsort(deps)
    fig,ax=plt.subplots(figsize=figsize)
    if title: ax.set_title(title,fontsize=12,family="monospace",weight="bold")
    pc=PatchCollection([Polygon(polys[i],closed=True) for i in order],
        facecolors=[cols[i] for i in order],edgecolors=(0,0,0,0.30),linewidths=0.2)
    ax.add_collection(pc)
    if annot:
        for (txt,(dx,dy),(tx,ty)) in annot:
            ax.annotate(txt, xy=(dx,dy), xytext=(tx,ty), fontsize=8.5,
                family="monospace", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.4",alpha=0.92),
                arrowprops=dict(arrowstyle="->",color="0.15",lw=1.3))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join("previews",fname),dpi=125,bbox_inches="tight"); plt.close()
    print("wrote previews/"+fname)

# 1) FRONTAL section: cut away the front half (keep z<=1), look from +Z.
#    Shows battery, boards, both hip wheels + shafts, both ankle stacks.
render("section_front.png",
       planes=[(2,1.0,True)], screen=[0,1], depth_axis=2, depth_sign=+1,
       xlim=(-95,95), ylim=(-5,312), title="FRONTAL SECTION  (cut Z=0)", figsize=(7,9),
       annot=[
        ("2 hip servos, now COAXIAL\n(both on X at Z=0, back-to-back)\n-> both shown here", (-2,214), (24,252)),
        ("ankle servos x2 (one per leg)", (-69,80), (-95,120)),
       ])

# 2) HIP drive — sagittal cut through torso (keep x<=-1), look from +X (left side).
render("section_hip.png",
       planes=[(0,-1.0,True)], screen=[2,1], depth_axis=0, depth_sign=+1,
       xlim=(-55,55), ylim=(120,312), title="HIP DRIVE  (sagittal, left)")

# 2b) HIP top section at Y~220 — BOTH hip servos, now COAXIAL (back-to-back on the
#     X axis at Z=0) in the widened 82 mm torso. Symmetric -> simpler control.
render("section_hip_top.png",
       planes=[(1,223.0,True),(1,217.0,False)], screen=[0,2], depth_axis=1, depth_sign=-1,
       xlim=(-105,105), ylim=(-52,52), title="HIP — 2x MG996R COAXIAL (top section, Y=220)",
       figsize=(8,5),
       annot=[
        ("LEFT hip servo\n-> drives LEFT leg", (-18,0), (-101,-44)),
        ("RIGHT hip servo\n-> drives RIGHT leg", (18,0), (26,40)),
        ("coaxial & symmetric\n(both on X at Z=0)", (0,0), (-32,42)),
        ("left leg", (-69,0), (-103,20)),
        ("right leg", (69,0), (74,-22)),
       ])

# 3) ANKLE drive — sagittal cut through the left leg (keep x<=-58), from +X.
render("section_ankle.png",
       planes=[(0,-58.0,True)], screen=[2,1], depth_axis=0, depth_sign=+1,
       xlim=(-60,60), ylim=(-5,150), title="ANKLE DRIVE  (sagittal, left leg)")

# 4) ANKLE explainer — oblique cutaway from the OUTBOARD-front so the pushrod
#    (which runs on the outer side of the leg) is not hidden by the leg wall.
def render_ankle(fname, annot=None, explode=None,
                 title="ANKLE — MG996R in leg  ->  pushrod  ->  foot", figsize=(7.5,7)):
    explode = explode or {}
    base =[(0,-40.0,True),(1,118.0,True),(2,6.0,True)]    # context: left leg, lower, keep rod
    loose=[(0,40.0,True),(1,175.0,True)]                  # exploded parts: generous keep
    a=np.radians(24); kx,ky=np.cos(a)*0.7, np.sin(a)*0.7
    polys=[];cols=[];deps=[]
    for f,m,g in zip(F,M,G):
        p=V[f].astype(float)
        off=explode.get(g)
        if off is not None: p=p+np.array(off,float)
        for (ax,val,kl) in (loose if off is not None else base):
            p=clip(p,ax,val,kl)
            if p is None: break
        if p is None: continue
        nrm=np.cross(p[1]-p[0],p[2]-p[0]); nl=np.linalg.norm(nrm); sh=0.55
        if nl>0: sh=0.45+0.55*max(0,abs(np.dot(nrm/nl,LIGHT)))
        sx=-p[:,0]+kx*p[:,2]; sy=p[:,1]+ky*p[:,2]          # mirror X = outboard view
        polys.append(np.column_stack([sx,sy]))
        cols.append(np.clip(np.array(COL.get(m,(.7,.7,.7)))*sh,0,1))
        deps.append(-p[:,0].mean()+p[:,2].mean())           # near = outboard(-x)+front(+z)
    order=np.argsort(deps)
    fig,ax=plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=12,family="monospace",weight="bold")
    ax.add_collection(PatchCollection([Polygon(polys[i],closed=True) for i in order],
        facecolors=[cols[i] for i in order],edgecolors=(0,0,0,0.30),linewidths=0.2))
    allp=np.vstack(polys)
    ax.set_xlim(allp[:,0].min()-34, allp[:,0].max()+36)
    ax.set_ylim(allp[:,1].min()-10, allp[:,1].max()+14)
    if annot:
        for (txt,(dx,dy),(tx,ty)) in annot:
            ax.annotate(txt, xy=(dx,dy), xytext=(tx,ty), fontsize=8.5, family="monospace",
                ha="left", va="center", annotation_clip=False,
                bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.4",alpha=0.92),
                arrowprops=dict(arrowstyle="->",color="0.15",lw=1.3))
    ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join("previews",fname),dpi=130,bbox_inches="tight"); plt.close()
    print("wrote previews/"+fname)

# 5) WIRING — frontal view cut just in front of the cables (z<=17) so the red
#    ankle-servo cables are the frontmost thing: torso PCA9685 -> hip -> down leg.
render("wiring.png",
       planes=[(2,17.0,True)], screen=[0,1], depth_axis=2, depth_sign=+1,
       xlim=(-104,104), ylim=(-5,315), title="WIRING — ankle-servo cables (red)",
       figsize=(8,9),
       annot=[
        ("ankle-servo 3-wire cable\nup the leg cavity", (-66,150), (-103,150)),
        ("service loop at the hip\n(cable crosses the pitch joint)", (-40,227), (-103,262)),
        ("to PCA9685 (torso)\nthen XL4016 6V + common GND", (-9,185), (16,150)),
        ("hip-servo leads are short\n(servo sits by the board)", (-30,225), (40,250)),
       ])
