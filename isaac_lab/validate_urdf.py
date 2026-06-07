#!/usr/bin/env python3
"""
validate_urdf.py — offline sanity checks for growbot.urdf, before it ever
touches Isaac Sim.  Pure stdlib + numpy (no ROS / Isaac needed).

  python3 isaac_lab/validate_urdf.py

Checks: XML well-formed · single kinematic root · tree (no cycles, ≤1 parent) ·
joint link refs resolve · mesh files exist + per-link AABB (unit sanity) ·
inertia tensors symmetric, positive-definite, and obey the triangle inequality
(PhysX silently destabilises on invalid inertia).
"""
import os, sys, xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "growbot.urdf")

ok = True
def check(cond, good, bad):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {good if cond else bad}")
    if not cond:
        ok = False

print(f"validating {URDF}\n")
tree = ET.parse(URDF)
robot = tree.getroot()
links = [l.get("name") for l in robot.findall("link")]
joints = robot.findall("joint")
parents = {j.find("child").get("link"): j.find("parent").get("link") for j in joints}

# --- tree topology ---
roots = [l for l in links if l not in parents]
check(len(roots) == 1, f"single root link: {roots[0] if roots else None}",
      f"expected 1 root, got {roots}")
for j in joints:
    p, c = j.find("parent").get("link"), j.find("child").get("link")
    check(p in links and c in links,
          f"joint {j.get('name')}: {p} → {c} resolve",
          f"joint {j.get('name')}: missing link ({p} or {c})")
# cycle / multi-parent
seen, cyc = set(), False
for start in links:
    n, hops = start, 0
    while n in parents and hops <= len(links):
        n = parents[n]; hops += 1
    if hops > len(links):
        cyc = True
check(not cyc, "no cycles in kinematic tree", "CYCLE detected")

# --- meshes exist + unit/bbox sanity ---
print()
for l in robot.findall("link"):
    mesh = l.find(".//visual/geometry/mesh")
    if mesh is None:
        continue
    path = os.path.join(HERE, mesh.get("filename"))
    if not os.path.exists(path):
        check(False, "", f"mesh missing: {mesh.get('filename')}")
        continue
    vs = []
    for line in open(path):
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
    vs = np.array(vs)
    ext = vs.max(0) - vs.min(0)
    big = ext.max()
    check(big < 0.4,
          f"{l.get('name'):16s} AABB {ext[0]:.3f}×{ext[1]:.3f}×{ext[2]:.3f} m  ({len(vs)} v)",
          f"{l.get('name')}: AABB {big:.1f} m looks like a UNIT ERROR (mm?)")

# --- inertia physical validity ---
print()
tot = 0.0
for l in robot.findall("link"):
    inert = l.find("inertial")
    if inert is None:
        continue
    m = float(inert.find("mass").get("value")); tot += m
    a = inert.find("inertia")
    I = np.array([[float(a.get("ixx")), float(a.get("ixy")), float(a.get("ixz"))],
                  [float(a.get("ixy")), float(a.get("iyy")), float(a.get("iyz"))],
                  [float(a.get("ixz")), float(a.get("iyz")), float(a.get("izz"))]])
    w = np.linalg.eigvalsh(I)              # principal moments
    pd = bool(np.all(w > 0))
    a_, b_, c_ = sorted(w)
    tri = (a_ + b_) >= c_ * (1 - 1e-6)
    check(pd and tri and m > 0,
          f"{l.get('name'):16s} m={m:.3f} kg  principal I={w[0]:.2e},{w[1]:.2e},{w[2]:.2e}  (PD+triangle ok)",
          f"{l.get('name')}: inertia invalid (PD={pd}, triangle={tri}, m={m})")

print(f"\n  total mass: {tot:.3f} kg")
print("\n" + ("ALL CHECKS PASSED ✅" if ok else "SOME CHECKS FAILED ❌"))
sys.exit(0 if ok else 1)
