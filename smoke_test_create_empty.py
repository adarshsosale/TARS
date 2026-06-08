"""Bounded smoke test: open an empty Isaac Sim stage, step physics N times, exit cleanly."""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bounded create_empty smoke test.")
parser.add_argument("--steps", type=int, default=200, help="number of physics steps to run after reset")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext


def main() -> int:
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    t0 = time.time()
    sim.reset()
    t_reset = time.time() - t0
    print(f"[SMOKE] sim.reset() returned in {t_reset:.2f}s")
    print("[INFO]: Setup complete...")

    t0 = time.time()
    n = 0
    while simulation_app.is_running() and n < args_cli.steps:
        sim.step()
        n += 1
    elapsed = time.time() - t0
    fps = n / elapsed if elapsed > 0 else float("inf")
    print(f"[SMOKE] stepped {n} times in {elapsed:.2f}s ({fps:.0f} steps/sec)")
    print("[SMOKE] OK")
    return 0


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    raise SystemExit(rc)
