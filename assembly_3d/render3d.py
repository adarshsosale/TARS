#!/usr/bin/env python3
"""Growbot — TRUE-3D isometric cutaway & exploded renders.

Unlike the flat `section_*.png` slices, this rotates the model in 3D (azimuth +
elevation), projects orthographically, and painter's-sorts every face, so you
see the inner workings in three dimensions.  Reads ../Growbot_TARS.obj and writes
the PNGs next to this file (assembly_3d/).

  python3 assembly_3d/render3d.py        # (needs numpy + matplotlib)

Views written:
  iso_cutaway_3d.png      whole robot, near quarter removed -> internals in 3D
  hip_3d.png              hip cutaway: BOTH coaxial MG996R servos
  ankle_3d.png            ankle cutaway: hinge + pushrod fine geometry
  ankle_exploded_3d.png   ankle drive blown apart, coloured by PRINTED PART
  print_parts_3d.png      whole bot, bed-split halves lifted apart + part legend
"""
import os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OBJ  = os.path.join(HERE, "..", "Growbot_TARS.obj")
SPLIT_Y = 150.0

COL = {  # material base colours (for assembled / material views)
 "mat_body":(0.80,0.81,0.82),"mat_panel_line":(0.71,0.72,0.73),"mat_panel":(0.62,0.63,0.65),
 "mat_accent":(0.16,0.16,0.17),"mat_accent_rib":(0.24,0.24,0.25),
 "mat_tpu":(0.10,0.10,0.11),"mat_tpu_tread":(0.16,0.16,0.17),
 "mat_motor":(0.28,0.33,0.45),"mat_steel":(0.72,0.74,0.78),
 "mat_battery":(0.15,0.62,0.45),"mat_pcb":(0.10,0.45,0.20),
 "mat_servo":(0.17,0.23,0.47),"mat_horn":(0.92,0.92,0.88),
 "mat_print_cf":(0.96,0.56,0.11),"mat_wire":(0.86,0.13,0.13),
}

# ---- printed-PART map: which o-group prints as which physical part -----------
PARTC = {
 "torso_up":("torso upper half (ABS)",       (0.60,0.80,0.96)),
 "torso_lo":("torso lower half (ABS)",       (0.26,0.40,0.62)),
 "leg_up"  :("leg upper half (CF-ABS)",      (0.96,0.86,0.50)),
 "leg_lo"  :("leg lower half +clevis (CF-ABS)",(0.66,0.47,0.16)),
 "foot"    :("foot + hinge lug (CF-ABS)",    (0.55,0.74,0.62)),
 "crank"   :("crank (CF)",                   (0.98,0.60,0.12)),
 "pushrod" :("pushrod (CF)",                 (0.92,0.40,0.10)),
 "rocker"  :("foot rocker (CF)",             (0.99,0.80,0.22)),
 "pin"     :("hinge pin (CF / M3 bolt)",     (0.56,0.40,0.95)),
 "tpu"     :("TPU sole (TPU)",               (0.12,0.12,0.13)),
 "rest"    :("central rest foot (ABS)",      (0.70,0.72,0.74)),
 "servo"   :("MG996R servo (bought)",        (0.17,0.23,0.47)),
 "board"   :("battery / electronics (bought)",(0.13,0.50,0.30)),
 "screw"   :("M3 screws (steel)",            (0.74,0.76,0.80)),
 "wire"    :("servo wiring",                 (0.86,0.13,0.13)),
}
def part_key(g, ymean):
    # check the specific drive parts BEFORE the foot_/leg_ prefixes
    if "crank"   in g:             return "crank"
    if "pushrod" in g:             return "pushrod"
    if "rocker"  in g:             return "rocker"     # foot_rocker_* -> rocker, not foot
    if "pivot"   in g:             return "pin"
    if "servo"   in g:             return "servo"
    if g.startswith("wiring"):     return "wire"
    if g.startswith("screws"):     return "screw"
    if g in ("battery","controller_stack","power_buck"): return "board"
    if g == "central_rest_foot":   return "rest"
    if g == "central_rest_pad" or g.startswith("foot_pad"): return "tpu"
    if g.startswith("foot_"):      return "foot"
    if g.startswith("leg_"):       return "leg_up" if ymean > SPLIT_Y else "leg_lo"
    if g == "central_body":        return "torso_up" if ymean > SPLIT_Y else "torso_lo"
    if g == "nameplate":           return "torso_up"
    return "torso_lo"

def load():
    Vv=[];Ff=[];Mm=[];Gg=[];cur="mat_body";grp="none"
    for ln in open(OBJ):
        if ln.startswith("v "):
            _,x,y,z=ln.split(); Vv.append((float(x),float(y),float(z)))
        elif ln.startswith("o "):  grp=ln.split()[1]
        elif ln.startswith("usemtl"): cur=ln.split()[1]
        elif ln.startswith("f "):
            Ff.append([int(p.split('/')[0])-1 for p in ln.split()[1:]]); Mm.append(cur); Gg.append(grp)
    return np.array(Vv),Ff,Mm,Gg
V,F,M,G = load()

def viewmat(az, el):
    az,el=math.radians(az),math.radians(el)
    ca,sa,ce,se=math.cos(az),math.sin(az),math.cos(el),math.sin(el)
    Ry=np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
    Rx=np.array([[1,0,0],[0,ce,-se],[0,se,ce]])
    return Rx@Ry
LIGHT=np.array([-0.35,0.55,0.75]); LIGHT/=np.linalg.norm(LIGHT)

def clip(poly,axis,val,keep_less):
    out=[];n=len(poly)
    for i in range(n):
        a=poly[i]; b=poly[(i+1)%n]
        da=a[axis]-val; db=b[axis]-val
        ina=da<=0 if keep_less else da>=0
        inb=db<=0 if keep_less else db>=0
        if ina: out.append(a)
        if ina!=inb:
            t=da/(da-db); out.append(a+t*(b-a))
    return np.array(out) if len(out)>=3 else None

def render(fname, title, az=34, el=20, clips=(), explode=None, bypart=False,
           splitlift=0.0, figsize=(9,9), annot=(), legend=False):
    R=viewmat(az,el); explode=explode or {}
    polys=[];cols=[];deps=[];used={}
    for f,m,g in zip(F,M,G):
        p=V[f].astype(float)
        off=explode.get(g)
        if off is not None:
            p=p+np.array(off,float)
        else:
            ok=True
            for (ax,val,kl) in clips:
                p=clip(p,ax,val,kl)
                if p is None: ok=False; break
            if not ok: continue
        if splitlift and p[:,1].mean()>SPLIT_Y:
            p=p+np.array([0,splitlift,0],float)
        nrm=np.cross(p[1]-p[0],p[2]-p[0]); nl=np.linalg.norm(nrm)
        nc=R@(nrm/nl) if nl>0 else np.array([0,0,1.0])
        sh=0.42+0.58*abs(float(nc@LIGHT))
        pc=p@R.T
        polys.append(pc[:,:2]); deps.append(pc[:,2].mean())
        if bypart:
            lbl,c=PARTC[part_key(g,p[:,1].mean())]; used[lbl]=c
        else:
            c=COL.get(m,(0.7,0.7,0.7))
        cols.append(np.clip(np.array(c)*sh,0,1))
    order=np.argsort(deps)                      # far first (painter's)
    fig,ax=plt.subplots(figsize=figsize)
    ax.set_title(title,fontsize=12,family="monospace",weight="bold")
    ax.add_collection(PatchCollection([Polygon(polys[i],closed=True) for i in order],
        facecolors=[cols[i] for i in order],edgecolors=(0,0,0,0.22),linewidths=0.12))
    allp=np.vstack(polys)
    ax.set_xlim(allp[:,0].min()-10, allp[:,0].max()+10)
    ax.set_ylim(allp[:,1].min()-10, allp[:,1].max()+10)
    ax.set_aspect('equal'); ax.axis('off')
    if legend and used:
        order_lbl=[PARTC[k][0] for k in ("torso_up","torso_lo","leg_up","leg_lo","foot",
                   "rocker","crank","pushrod","pin","tpu","servo","board","screw","wire","rest")
                   if PARTC[k][0] in used]
        handles=[Patch(facecolor=used[l],edgecolor='0.3',label=l) for l in order_lbl]
        ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1.0,0.5),
                  fontsize=7.5,framealpha=0.95,title="printed / bought parts")
    for (txt,wpt,(dx,dy)) in annot:        # (dx,dy) = screen offset from projected point
        wp=np.array(wpt,float)@R.T
        ax.annotate(txt,xy=(wp[0],wp[1]),xytext=(wp[0]+dx,wp[1]+dy),fontsize=8,
            family="monospace",ha='left',va='center',annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="0.4",alpha=0.93),
            arrowprops=dict(arrowstyle="->",color="0.12",lw=1.2))
    plt.tight_layout(); plt.savefig(os.path.join(HERE,fname),dpi=130,bbox_inches="tight")
    plt.close(); print("wrote",fname)

# ============================================================ VIEWS ===========
# 1) whole-robot 3D quarter cutaway (remove the near +X/+Z quarter)
render("iso_cutaway_3d.png","GROWBOT — 3D quarter cutaway", az=34, el=18,
       clips=[(0,1.0,True),(2,1.0,True)], figsize=(8.5,10))

# 2) hip — both coaxial servos (cut front half away, crop to hip band)
render("hip_3d.png","HIP — 2x MG996R coaxial (3D cutaway)", az=40, el=16,
       clips=[(2,2.0,True),(1,262.0,True),(1,150.0,False)], figsize=(9,6.5),
       annot=[
        ("2x MG996R, COAXIAL & symmetric\n(one per leg, same X axis)", (0,236,0), (-30,46)),
        ("horn -> LEFT leg",  (-47,225,0), (-95,-12)),
        ("horn -> RIGHT leg", (47,225,0),  (28,-30)),
       ])

# 3) ankle — hinge + pushrod (cut front half away, crop low)
render("ankle_3d.png","ANKLE — hinge + pushrod (3D cutaway)", az=42, el=16,
       clips=[(2,2.0,True),(1,120.0,True),(0,-30.0,True)], figsize=(9,9),
       annot=[
        ("MG996R ankle servo", (-69,84,0), (-92,24)),
        ("crank on spline",    (-95,73,-4),(-80,16)),
        ("pushrod",            (-95,60,-9),(46,4)),
        ("foot rocker",        (-95,48,-6),(46,-20)),
        ("hinge pin = ankle axis", (-71,43,0), (-120,-10)),
        ("open ankle gap",     (-69,39,0), (-92,-42)),
       ])

# 4) ankle drive exploded, coloured by printed part
render("ankle_exploded_3d.png","ANKLE DRIVE — exploded by printed part", az=40, el=16,
       clips=[(0,-30.0,True),(1,120.0,True),(2,2.0,True)], bypart=True, legend=True,
       figsize=(10,8),
       explode={
        "ankle_servo_L":(0, 26, 0),
        "ankle_crank_L":(-30, 40, 0),
        "ankle_pushrod_L":(-46, 16, 0),
        "foot_rocker_L":(-30,-14, 0),
        "ankle_pivot_L":(-70, 0, 0),
        "foot_left":(0,-44, 0),
        "foot_pad_left":(0,-44, 0),
       })

# 5) whole-bot print-parts: lift the bed-split upper halves up, colour by part
render("print_parts_3d.png","PRINT PARTS — bed-split halves + what bolts to what",
       az=32, el=14, bypart=True, legend=True, splitlift=42.0, figsize=(10,10))
