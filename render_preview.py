#!/usr/bin/env python3
"""Quick flat-shaded preview renders of Growbot_TARS.obj (for visual checks)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

os.makedirs("previews", exist_ok=True)

colmap = {
    "mat_body":(0.72,0.73,0.74), "mat_panel_line":(0.62,0.63,0.64),
    "mat_accent":(0.12,0.12,0.13), "mat_accent_rib":(0.20,0.20,0.21),
    "mat_tpu":(0.06,0.06,0.06), "mat_tpu_tread":(0.11,0.11,0.11),
    "mat_joint":(0.5,0.51,0.56), "mat_panel":(0.6,0.61,0.63),
}
verts=[]; faces=[]; fmat=[]; cur="mat_body"
for line in open("model/Growbot_TARS.obj"):
    if line.startswith("v "):
        _,x,y,z=line.split(); verts.append((float(x),float(y),float(z)))
    elif line.startswith("usemtl"): cur=line.split()[1]
    elif line.startswith("f "):
        faces.append([int(p.split("/")[0]) for p in line.split()[1:]]); fmat.append(cur)
V=np.array(verts); P=V[:,[0,2,1]]   # map model-Y(up) -> matplotlib Z

def render(elev,azim,fname):
    fig=plt.figure(figsize=(5,8)); ax=fig.add_subplot(111,projection="3d")
    polys=[]; cols=[]; ln=np.array([0.5,0.7,0.6]); ln/=np.linalg.norm(ln)
    for f,m in zip(faces,fmat):
        polys.append(P[[i-1 for i in f]])
        o=V[[i-1 for i in f]]
        n=np.cross(o[1]-o[0],o[2]-o[0]); nn=np.linalg.norm(n); s=0.55
        if nn>0: s=0.4+0.6*max(0,abs(np.dot(n/nn,ln)))
        cols.append(np.clip(np.array(colmap.get(m,(0.7,0.7,0.7)))*s,0,1))
    ax.add_collection3d(Poly3DCollection(polys,facecolors=cols,
        edgecolors=(0,0,0,0.12),linewidths=0.15))
    ax.set_xlim(-100,100); ax.set_ylim(-100,100); ax.set_zlim(0,306)
    ax.set_box_aspect((200,200,306)); ax.view_init(elev=elev,azim=azim); ax.set_axis_off()
    plt.tight_layout(); plt.savefig(os.path.join("previews",fname),dpi=120,bbox_inches="tight"); plt.close()
    print("wrote previews/"+fname)

render(12,-72,"preview_front.png")
render(2,0,"preview_side.png")
render(20,-38,"preview_iso.png")
render(80,-90,"preview_top.png")
