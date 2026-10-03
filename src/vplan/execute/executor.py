"""Execute an approved trajectory under position control with finite gains (D21). Never resets joint state during a
run. Grasp = attachment constraint when the capture volume holds on the ACTUAL state (D10, simulation simplification)."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pybullet as p

from vplan.config import config_hash
from vplan.geometry import capture_check, pose_to_T, quat_to_R, rot_angle
from vplan.judge.kinematics import jacobian_s_current
from vplan.records import CandidateTrajectory, ContactSummary, ExecEvent, ExecutionRecord, Pose, Scene, scene_hash
from vplan.sim.scene import FINGER_OPEN_M, SceneHandles, build_scene

REPLAY_EVERY = 4  # 240 Hz physics -> 60 Hz replay


class StaleApproval(RuntimeError):
    """Trajectory, scene, target, config or the live world state differs from what was verified."""


def approval_token(traj: CandidateTrajectory, scene: Scene, target_id: str, cfgs: dict[str, dict]) -> str:
    payload = {"traj": traj.model_dump_json(), "scene": scene_hash(scene), "target": target_id,
               "configs": {k: config_hash(v) for k, v in sorted(cfgs.items())}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def init_physics(scene: Scene, client: int, q_start, phys: dict) -> SceneHandles:
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0, 0, phys["gravity_mps2"], physicsClientId=client)
    p.setTimeStep(phys["time_step_s"], physicsClientId=client)
    p.setPhysicsEngineParameter(numSolverIterations=phys["solver_iterations"], erp=phys["erp"],
                                contactERP=phys["contact_erp"], frictionERP=phys["friction_erp"], physicsClientId=client)
    h = build_scene(scene, client, np.asarray(q_start, dtype=float))
    dyn = dict(lateralFriction=phys["lateral_friction"], spinningFriction=phys["spinning_friction"],
               rollingFriction=phys["rolling_friction"], restitution=phys["restitution"])
    for bid in set(h.bodies.values()):
        p.changeDynamics(bid, -1, physicsClientId=client, **dyn)
    for link in [-1, *range(p.getNumJoints(h.robot, physicsClientId=client))]:
        p.changeDynamics(h.robot, link, physicsClientId=client, **dyn)
    return h


def attach_object(h: SceneHandles, body: int, max_force: float) -> tuple[int, tuple, tuple]:
    """Fixed constraint hand -> object at their CURRENT relative pose. createConstraint frames are relative to the
    parent link's COM frame (getLinkState[0:2]); the hand COM is 4 cm from its link frame (CR#2)."""
    hs = p.getLinkState(h.robot, h.hand_link, computeForwardKinematics=True, physicsClientId=h.client)
    obj_pos, obj_orn = p.getBasePositionAndOrientation(body, physicsClientId=h.client)
    inv_pos, inv_orn = p.invertTransform(hs[0], hs[1])
    rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, obj_pos, obj_orn)
    cid = p.createConstraint(h.robot, h.hand_link, body, -1, p.JOINT_FIXED, [0, 0, 0], rel_pos, [0, 0, 0],
                             rel_orn, [0, 0, 0, 1], physicsClientId=h.client)
    p.changeConstraint(cid, maxForce=max_force, physicsClientId=h.client)
    return cid, tuple(rel_pos), tuple(rel_orn)


def _arm_q(h: SceneHandles) -> np.ndarray:
    return np.array([p.getJointState(h.robot, j, physicsClientId=h.client)[0] for j in h.arm_joints])


def _check_start_state(h: SceneHandles, traj: CandidateTrajectory, scene: Scene, phys: dict) -> None:
    """Compare the LIVE world with the approved snapshot; nothing is reset first (CR#4)."""
    if np.max(np.abs(_arm_q(h) - np.array(traj.q_start))) > phys["start_state_tol_rad"]:
        raise StaleApproval("arm start state differs from the verified start state")
    for o in scene.objects:
        pos, orn = p.getBasePositionAndOrientation(h.bodies[o.id], physicsClientId=h.client)
        if (np.linalg.norm(np.array(pos) - np.array(o.pose.position)) > phys["start_state_tol_m"]
                or rot_angle(quat_to_R(orn), quat_to_R(o.pose.orientation)) > phys["start_state_tol_orient_rad"]):
            raise StaleApproval(f"object '{o.id}' moved since approval")


def _set_fingers(h: SceneHandles, width_each: float, force: float) -> None:
    for j in h.finger_joints:
        p.setJointMotorControl2(h.robot, j, p.POSITION_CONTROL, targetPosition=width_each, force=force,
                                physicsClientId=h.client)


def execute(h: SceneHandles, traj: CandidateTrajectory, scene: Scene, target_id: str, cfgs: dict[str, dict],
            token: str, realtime: bool = False) -> ExecutionRecord:
    phys, vcfg, jcfg = cfgs["physics"], cfgs["verifier"], cfgs["judge"]
    if token != approval_token(traj, scene, target_id, cfgs):
        raise StaleApproval("approval token mismatch (trajectory, scene, target or config changed)")
    _check_start_state(h, traj, scene, phys)
    client, dt = h.client, phys["time_step_s"]
    info = [p.getJointInfo(h.robot, j, physicsClientId=client) for j in h.arm_joints]
    lo, hi, forces = np.array([i[8] for i in info]), np.array([i[9] for i in info]), [i[10] for i in info]
    target_body = h.bodies[target_id]
    size = next(o.size for o in scene.objects if o.id == target_id)
    L = jcfg["jacobian_char_length_m"]
    min_margin, max_violation = np.full(7, np.inf), 0.0
    sigmas: list[float] = []
    agg: dict[tuple, list] = {}
    events: list[ExecEvent] = []
    tracking: dict[str, float] = {}
    q_rep: list[tuple] = []
    f_rep: list[tuple] = []
    o_rep: dict[str, list[Pose]] = {o.id: [] for o in scene.objects}
    constraint = None
    step = 0

    def link_name(body: int, link: int) -> str:
        return h.link_names.get(link, str(link)) if body == h.robot else "base"

    def physics_step(stage: str) -> None:
        nonlocal step, max_violation
        p.stepSimulation(physicsClientId=client)
        step += 1
        q = _arm_q(h)
        np.minimum(min_margin, np.minimum(q - lo, hi - q), out=min_margin)
        max_violation = max(max_violation, float(np.max(np.maximum(lo - q, q - hi))), 0.0)
        sigmas.append(float(np.linalg.svd(jacobian_s_current(h, L), compute_uv=False)[-1]))
        for c in p.getContactPoints(physicsClientId=client):
            a = (h.names.get(c[1], str(c[1])), link_name(c[1], c[3]))
            b = (h.names.get(c[2], str(c[2])), link_name(c[2], c[4]))
            a, b = sorted((a, b))
            e = agg.setdefault((a[0], a[1], b[0], b[1], stage), [step, 0, 0.0, 0.0])
            e[1] += 1
            e[2] = max(e[2], float(c[9]))
            e[3] = max(e[3], float(-c[8]))
        if step % REPLAY_EVERY == 0:
            q_rep.append(tuple(float(x) for x in q))
            f_rep.append(tuple(float(p.getJointState(h.robot, j, physicsClientId=client)[0]) for j in h.finger_joints))
            for o in scene.objects:
                pos, orn = p.getBasePositionAndOrientation(h.bodies[o.id], physicsClientId=client)
                o_rep[o.id].append(Pose(position=tuple(pos), orientation=tuple(orn)))
        if realtime:
            time.sleep(dt)

    def actual_capture() -> tuple[bool, dict[str, float], str]:
        ls = p.getLinkState(h.robot, h.tcp_link, computeForwardKinematics=True, physicsClientId=client)
        pos, orn = p.getBasePositionAndOrientation(target_body, physicsClientId=client)
        return capture_check(pose_to_T(ls[4], ls[5]), pose_to_T(pos, orn), size, vcfg)

    _set_fingers(h, FINGER_OPEN_M, phys["finger_force_n"])
    for seg in traj.segments:
        if seg.label == "lift" and constraint is None:
            ok, vals, why = actual_capture()
            if ok:
                _set_fingers(h, max(vals["half_width_m"] - phys["finger_squeeze_m"], 0.0), phys["finger_force_n"])
                for _ in range(int(phys["grasp_close_s"] / dt)):
                    physics_step("grasp")
                ok, vals, why = actual_capture()  # re-check after closing: fingers may have pushed the object (CR#5)
            if ok:
                constraint, rel_pos, rel_orn = attach_object(h, target_body, phys["attach_max_force_n"])
                events.append(ExecEvent(step=step, kind="attach", values=vals, rel_pos=rel_pos, rel_orn=rel_orn))
            else:
                events.append(ExecEvent(step=step, kind="attach_rejected", values=vals))
                _set_fingers(h, FINGER_OPEN_M, phys["finger_force_n"])
        if seg.label == "retreat" and constraint is not None:
            p.removeConstraint(constraint, physicsClientId=client)
            constraint = None
            events.append(ExecEvent(step=step, kind="release", values={}))
            _set_fingers(h, FINGER_OPEN_M, phys["finger_force_n"])
            for _ in range(int(phys["release_settle_s"] / dt)):
                physics_step("release")
        Q, T = np.array(seg.q), np.array(seg.t)
        for t in np.arange(0.0, T[-1] + dt, dt):
            q_target = np.array([np.interp(t, T, Q[:, j]) for j in range(7)])
            p.setJointMotorControlArray(h.robot, h.arm_joints, p.POSITION_CONTROL, targetPositions=q_target,
                                        forces=forces, positionGains=[phys["position_gain"]] * 7,
                                        velocityGains=[phys["velocity_gain"]] * 7, physicsClientId=client)
            physics_step(seg.label)
        for _ in range(int(phys["segment_settle_s"] / dt)):
            if np.max(np.abs(_arm_q(h) - Q[-1])) < phys["tracking_tol_rad"]:
                break
            physics_step(seg.label)
        tracking[seg.label] = float(np.max(np.abs(_arm_q(h) - Q[-1])))
    lin_hist: dict[str, list[float]] = {o.id: [] for o in scene.objects}
    ang_hist: dict[str, list[float]] = {o.id: [] for o in scene.objects}
    for _ in range(int(round(jcfg["rest_dwell_s"] / dt))):
        physics_step("rest")
        for o in scene.objects:
            v, w = p.getBaseVelocity(h.bodies[o.id], physicsClientId=client)
            lin_hist[o.id].append(float(np.linalg.norm(v)))
            ang_hist[o.id].append(float(np.linalg.norm(w)))
    final = {}
    for o in scene.objects:
        pos, orn = p.getBasePositionAndOrientation(h.bodies[o.id], physicsClientId=client)
        final[o.id] = Pose(position=tuple(pos), orientation=tuple(orn))
    contacts = tuple(ContactSummary(body_a=k[0], link_a=k[1], body_b=k[2], link_b=k[3], stage=k[4], first_step=v[0],
                                    count=v[1], max_normal_force_n=v[2], max_penetration_m=v[3]) for k, v in agg.items())
    return ExecutionRecord(scene_hash=scene_hash(scene), approval=token, dt_s=dt, n_physics_steps=step,
                           events=tuple(events), min_joint_margin_rad=tuple(float(x) for x in min_margin),
                           max_joint_violation_rad=max_violation, sigma_min_steps=tuple(sigmas), contacts=contacts,
                           tracking_error_rad=tracking, final_poses=final,
                           rest_lin_hist={k: tuple(v) for k, v in lin_hist.items()},
                           rest_ang_hist={k: tuple(v) for k, v in ang_hist.items()},
                           q_replay=tuple(q_rep), finger_replay=tuple(f_rep),
                           object_replay={k: tuple(v) for k, v in o_rep.items()})
