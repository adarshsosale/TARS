# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone correctness checks for the CPG + symmetry mirror (no Isaac needed).

Run in the conda env (torch only):  python isaac_lab/tasks/hexabot/test_cpg.py

Verifies:
  1. Zero action (speed_scale=1) reproduces the analytical tripod gait
     `tripod_gait.gait_pose` joint targets exactly (the contract that makes BC +
     the imitation reward well-defined).
  2. speed_scale=0 collapses to the standing stance (coxa 0, femur/tibia at stance).
  3. The action / CPG-phase mirror permutations are involutions.
"""

import math
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from cpg import LEG_AZ_DEG, LEG_ORDER, STANCE_FEMUR, STANCE_TIBIA, TRIPOD_A, HexabotCPG  # noqa: E402


def _ref_gait_pose(phase, coxa_amp, lift):
    """Reference: a faithful copy of tripod_gait.gait_pose (which can't be imported
    without launching Isaac). Returns {leg: (qc, qf, qt)}."""
    pose = {}
    for leg, az in LEG_AZ_DEG.items():
        th = math.radians(az)
        ph = phase if leg in TRIPOD_A else (phase + 0.5) % 1.0
        if ph < 0.5:
            s = ph / 0.5
            qc = coxa_amp * math.sin(th) * (2 * s - 1)
            qf, qt = STANCE_FEMUR, STANCE_TIBIA
        else:
            s = (ph - 0.5) / 0.5
            qc = coxa_amp * math.sin(th) * (1 - 2 * s)
            k = math.sin(math.pi * s)
            qf = STANCE_FEMUR - lift * k
            qt = STANCE_TIBIA + 0.4 * lift * k
        pose[leg] = (qc, qf, qt)
    return pose


def _test_scan_regions():
    """Phase B/C: the 91-ray scan partitions into front/mid/rear over the body and a
    flat hit field gives identical region medians (the documented flat no-op)."""
    from regions import scan_region_masks

    res = 0.05
    # reproduce the grid_pattern x grid (size 0.6) + the +0.10 forward sensor offset
    xs = torch.arange(-0.3, 0.3 + 1e-9, res) + 0.10          # 13 -> body-frame [-0.20, 0.40]
    ys = torch.arange(-0.15, 0.15 + 1e-9, res)               # 7
    gx, _ = torch.meshgrid(xs, ys, indexing="xy")
    ray_x = gx.flatten()
    assert ray_x.numel() == 91, f"expected 91 rays, got {ray_x.numel()}"

    front, mid, rear = scan_region_masks(ray_x, half=0.20)
    counts = (int(front.sum()), int(mid.sum()), int(rear.sum()))
    assert counts == (21, 21, 21), f"body regions not 21/21/21: {counts}"
    n_look = int((~(front | mid | rear)).sum())
    assert n_look == 28, f"expected 28 lookahead rays excluded, got {n_look}"
    assert int((front & mid).sum()) == 0 and int((mid & rear).sum()) == 0 and int((front & rear).sum()) == 0
    # flat ground: identical hit heights -> region medians agree within 1 cm (no-op)
    hits = torch.full((1, 91), 0.037)
    nan = float("nan")

    def med(mask):
        v = torch.where(mask.unsqueeze(0), hits, torch.full_like(hits, nan))
        return torch.nanmedian(v, dim=1).values

    mf, mm, mr = med(front), med(mid), med(rear)
    assert (mf - mm).abs().max() < 0.01 and (mm - mr).abs().max() < 0.01, "flat region medians disagree"
    print("[OK] scan regions: body split 21/21/21, 28 lookahead excluded, flat medians agree")

    # symmetry scan-mirror (the algorithm HexabotRoughEnv runs on the live pattern):
    # reflect each ray's y and find its partner; check involution + (x equal, y mirrored).
    gx2, gy2 = torch.meshgrid(xs, ys, indexing="xy")
    xy = torch.stack([gx2.flatten(), gy2.flatten()], dim=1)   # (91, 2)
    n = xy.shape[0]
    tgt = xy.clone()
    tgt[:, 1] = -tgt[:, 1]
    mirror = torch.tensor([int(torch.argmin(torch.sum((xy - tgt[i]) ** 2, dim=1))) for i in range(n)])
    assert torch.equal(mirror[mirror], torch.arange(n)), "scan mirror is not an involution"
    assert torch.allclose(xy[mirror][:, 0], xy[:, 0]) and torch.allclose(xy[mirror][:, 1], -xy[:, 1]), \
        "scan mirror partners do not have equal x / mirrored y"
    print("[OK] 91-ray scan mirror: involution, partners have equal x and mirrored y")


def main():
    dev = "cpu"
    cfg = SimpleNamespace(
        cpg_f_base=1.0, cpg_coxa_amp=0.26, cpg_lift=0.55,
        cpg_kf=0.5, cpg_kmu=0.5, cpg_klift=0.6, cpg_kstance=0.35,
        cpg_coupling_strength=2.0,
    )
    n_act = HexabotCPG.ACTION_DIM     # 24: per-leg [d_freq, d_coxa_amp, d_lift, d_stance]
    # joint names in a deliberately scrambled order to test the name-based mapping
    joint_names = [f"{seg}_{lg}" for seg in ("femur", "tibia", "coxa") for lg in ("rm", "lf", "rr", "lm", "rf", "lr")]
    cpg = HexabotCPG(joint_names, num_envs=1, device=dev, cfg=cfg)

    psi = {lg: (0.0 if lg in TRIPOD_A else 0.5) for lg in LEG_ORDER}
    max_err = 0.0
    for phase in [0.0, 0.13, 0.27, 0.5, 0.62, 0.81, 0.99]:
        # drive the CPG phase to a known global phase
        cpg.theta[0] = torch.tensor([2 * math.pi * ((phase + psi[lg]) % 1.0) for lg in LEG_ORDER])
        tgt = cpg.joint_targets(torch.zeros(1, n_act), speed_scale=1.0)[0]
        ref = _ref_gait_pose(phase, 0.26, 0.55)
        for li, lg in enumerate(LEG_ORDER):
            qc = tgt[cpg._coxa_idx[li]].item()
            qf = tgt[cpg._femur_idx[li]].item()
            qt = tgt[cpg._tibia_idx[li]].item()
            rqc, rqf, rqt = ref[lg]
            max_err = max(max_err, abs(qc - rqc), abs(qf - rqf), abs(qt - rqt))
    assert max_err < 1e-5, f"CPG != analytical gait, max_err={max_err}"
    print(f"[OK] zero-action == analytical tripod gait (max joint err {max_err:.2e} rad)")

    # speed_scale=0 -> standing stance
    cpg.theta[0] = torch.rand(6) * 2 * math.pi
    tgt0 = cpg.joint_targets(torch.zeros(1, n_act), speed_scale=0.0)[0]
    for li in range(6):
        assert abs(tgt0[cpg._coxa_idx[li]].item()) < 1e-6
        assert abs(tgt0[cpg._femur_idx[li]].item() - STANCE_FEMUR) < 1e-6
        assert abs(tgt0[cpg._tibia_idx[li]].item() - STANCE_TIBIA) < 1e-6
    print("[OK] speed_scale=0 == standing stance")

    # mirror permutations are involutions
    a_idx = cpg.action_mirror_idx()
    p_idx = cpg.phase_mirror_idx()
    assert torch.equal(a_idx[a_idx], torch.arange(n_act)), "action mirror not an involution"
    assert torch.equal(p_idx[p_idx], torch.arange(12)), "phase mirror not an involution"
    print("[OK] action / phase mirror permutations are involutions")

    # --- Phase E: d_stance ride-height channel -------------------------------
    # +d_stance presses the femur down by kstance*d (raising the body) in BOTH
    # phase halves (it is a posture, not a gait amplitude), is one-sided
    # (negative values are a no-op -> can't reopen the belly-crawl), clamps at
    # d=1, and leaves coxa/tibia untouched.
    cpg.theta[0] = torch.tensor([2 * math.pi * ((0.27 + psi[lg]) % 1.0) for lg in LEG_ORDER])
    base_t = cpg.joint_targets(torch.zeros(1, n_act), speed_scale=1.0)[0]
    for d, expect in [(0.5, 0.5 * 0.35), (1.0, 0.35), (1.7, 0.35), (-0.8, 0.0)]:
        act = torch.zeros(1, n_act)
        act[0, 3::4] = d                      # d_stance of every leg (leg-major, 4 per leg)
        t = cpg.joint_targets(act, speed_scale=1.0)[0]
        for li in range(6):
            assert abs((t - base_t)[cpg._femur_idx[li]].item() - expect) < 1e-6, \
                f"d_stance={d}: femur offset != {expect}"
            assert abs((t - base_t)[cpg._coxa_idx[li]].item()) < 1e-6, "d_stance leaked into coxa"
            assert abs((t - base_t)[cpg._tibia_idx[li]].item()) < 1e-6, "d_stance leaked into tibia"
    print("[OK] d_stance: one-sided femur press-down, clamped at kstance, phase-independent")

    _test_scan_regions()
    print("\nALL CPG SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
