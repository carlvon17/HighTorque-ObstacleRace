"""Measure pretrained policies on this carpet, including low-speed dead zones."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sim.robot import Robot, VENDOR, build_model


def main():
    model, _ = build_model(no_obstacles=True, timestep=.002)
    results = []
    commands = [[v, 0, 0] for v in (.04, .08, .10, .12, .20)]
    commands += [[0, v, 0] for v in (-.16, -.12, .08, .10, .12, .16)]
    commands += [[.15, 0, .3], [0, 0, 0]]
    for policy in sorted((VENDOR / "Exported_policy").glob("*.pt")):
        for command in commands:
            robot = Robot(model, [.55, -1.25, 0], policy=policy)
            for _ in range(100):
                robot.step([0, 0, 0])
            initial = robot.pose.copy()
            minimum = 1.
            for _ in range(250):
                robot.step(command)
                minimum = min(minimum, float(robot.data.qpos[2]))
                if minimum < .2:
                    break
            elapsed = robot.data.time-2
            result = {"policy": policy.name, "command": command,
                      "average_world_vxy": ((robot.pose[:2]-initial[:2])/elapsed).tolist(),
                      "min_base_z": minimum, "fallen": minimum < .2}
            print(json.dumps(result), flush=True)
            results.append(result)
    output = ROOT / "output/sim/loco_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"duration_s": 5, "settle_s": 2, "results": results,
                                 "note": "Mean world displacement, including transient; not a fitted steady-state plant model."}, indent=2))


if __name__ == "__main__":
    main()
