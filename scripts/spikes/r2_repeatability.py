"""R2: is an attachment-assisted transfer repeatable? 10 fresh-client runs. Exit code 1 on failure."""
import sys

import numpy as np
import pybullet as p

from tests.conftest import make_basic_scene
from tests.test_execute import good_plan
from vplan.config import config_snapshot
from vplan.execute.executor import approval_token, execute, init_physics
from vplan.judge.judge import judge
from vplan.motion.trajectory import build_trajectory
from vplan.robot import HOME_Q

scene, cfgs = make_basic_scene(), config_snapshot()
traj = build_trajectory(good_plan(), scene, HOME_Q, cfg=cfgs["verifier"])
finals, results = [], []
for _ in range(10):
    cid = p.connect(p.DIRECT)
    h = init_physics(scene, cid, traj.q_start, cfgs["physics"])
    rec = execute(h, traj, scene, "cube_red", cfgs, approval_token(traj, scene, "cube_red", cfgs))
    p.disconnect(cid)
    finals.append(rec.final_poses["cube_red"].position)
    results.append(judge(rec, scene, "cube_red"))
spread = np.ptp(np.array(finals), axis=0)
wins = sum(r.success for r in results)
print("final position spread (m, per axis):", spread)
print("successes:", wins, "/ 10; failing predicates:", [k for r in results for k, v in r.predicates.items() if not v])
sys.exit(0 if (np.all(spread < 0.002) and wins == 10) else 1)
