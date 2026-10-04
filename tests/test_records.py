import math

import pybullet as p
import pytest
from pydantic import ValidationError

from tests.conftest import make_basic_scene
from vplan.records import (PLAN_ADAPTER, CandidateTrajectory, PickPlacePlan, Pose, Refusal, Scene, Segment,
                           SubmitPlanArgs, scene_hash)

Q0 = (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785)
I16 = tuple(float(x) for x in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1))


def seg(label: str, q0=Q0, q1=Q0, carrying=False) -> dict:
    return {"label": label, "source_field": "grasp", "carrying": carrying, "q": [q0, q1], "t": [0.0, 1.0],
            "ik_idx": [1], "ik_targets": [I16]}


def valid_traj_dict() -> dict:
    labels = ["approach", "descend", "lift", "transport", "descend_place", "retreat"]
    return {"q_start": Q0, "segments": [seg(l) for l in labels]}


def test_plan_union_parses_pick_place_and_refusal() -> None:
    plan = PLAN_ADAPTER.validate_python({
        "kind": "pick_place", "target_id": "cube_red",
        "grasp": {"approach": "top", "yaw_deg": 0.0, "pregrasp_offset_m": 0.1},
        "transport_via": [[0.4, 0.0, 0.3]],
        "place": {"position": [0.45, 0.25, 0.03], "yaw_deg": 0.0, "release_height_m": 0.05}})
    assert isinstance(plan, PickPlacePlan)
    assert isinstance(PLAN_ADAPTER.validate_python(
        {"kind": "refusal", "reason": "target_unreachable", "explanation": "too far"}), Refusal)


def test_plan_rejects_out_of_range_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        PLAN_ADAPTER.validate_python({"kind": "pick_place", "target_id": "c",
                                      "grasp": {"approach": "top", "yaw_deg": 400.0, "pregrasp_offset_m": 0.1},
                                      "place": {"position": [0.4, 0.2, 0.03], "yaw_deg": 0.0, "release_height_m": 0.05}})
    with pytest.raises(ValidationError):
        PLAN_ADAPTER.validate_python({"kind": "refusal", "reason": "other", "explanation": "x", "extra": 1})


def test_tool_schema_has_object_root() -> None:
    schema = SubmitPlanArgs.model_json_schema()
    assert schema["type"] == "object" and "plan" in schema["properties"]


def test_scene_hash_stable_and_sensitive() -> None:
    s1 = make_basic_scene()
    assert scene_hash(s1) == scene_hash(Scene.model_validate_json(s1.model_dump_json()))
    d = s1.model_dump()
    d["table_top_z"] = 0.01
    assert scene_hash(Scene.model_validate(d)) != scene_hash(s1)


def test_scene_rejects_non_identity_base_and_duplicate_ids() -> None:
    d = make_basic_scene().model_dump()
    d["robot_base"] = {"position": (0.1, 0.0, 0.0), "orientation": (0, 0, 0, 1)}
    with pytest.raises(ValidationError):
        Scene.model_validate(d)
    d = make_basic_scene().model_dump()
    d["objects"][1]["id"] = d["objects"][0]["id"]
    with pytest.raises(ValidationError):
        Scene.model_validate(d)


def test_trajectory_structure_validators() -> None:
    CandidateTrajectory.model_validate(valid_traj_dict())  # valid
    d = valid_traj_dict()
    d["segments"] = []
    with pytest.raises(ValidationError):
        CandidateTrajectory.model_validate(d)  # empty
    d = valid_traj_dict()
    d["segments"][2]["label"] = "transport"
    with pytest.raises(ValidationError):
        CandidateTrajectory.model_validate(d)  # wrong order
    d = valid_traj_dict()
    d["segments"][1]["q"][0] = tuple(x + 0.1 for x in Q0)
    with pytest.raises(ValidationError):
        CandidateTrajectory.model_validate(d)  # boundary discontinuity


def test_segment_validators() -> None:
    Segment.model_validate(seg("approach"))
    bad = seg("approach")
    bad["q"] = [Q0, Q0[:6]]
    with pytest.raises(ValidationError):
        Segment.model_validate(bad)  # 6 joints
    bad = seg("approach")
    bad["q"] = [Q0, tuple([math.nan] * 7)]
    with pytest.raises(ValidationError):
        Segment.model_validate(bad)  # NaN
    bad = seg("approach")
    bad["t"] = [0.0, 0.0]
    with pytest.raises(ValidationError):
        Segment.model_validate(bad)  # non-increasing time
    bad = seg("approach")
    bad["ik_idx"] = [0]
    with pytest.raises(ValidationError):
        Segment.model_validate(bad)  # IK index must point past q[0] and end at the last sample


def test_scene_rejects_unknown_color() -> None:
    for bad in ("white", "Red"):
        d = make_basic_scene().model_dump()
        d["objects"][0]["color"] = bad
        with pytest.raises(ValidationError):
            Scene.model_validate(d)


@pytest.mark.parametrize("reserved", ["table", "bin_floor", "bin_wall_0", "bin_wall_3", "robot"])
def test_scene_rejects_reserved_body_ids(reserved: str) -> None:
    d = make_basic_scene().model_dump()
    d["objects"][0]["id"] = reserved
    with pytest.raises(ValidationError, match=reserved):
        Scene.model_validate(d)
    d = make_basic_scene(with_obstacle=True).model_dump()
    d["obstacles"][0]["id"] = reserved
    with pytest.raises(ValidationError, match=reserved):
        Scene.model_validate(d)


def test_records_reject_nan_and_inf() -> None:
    with pytest.raises(ValidationError):
        Pose(position=(math.nan, 0.0, 0.0))
    with pytest.raises(ValidationError):
        Pose(position=(0.0, math.inf, 0.0))
    d = make_basic_scene().model_dump()
    d["table_top_z"] = math.nan
    with pytest.raises(ValidationError):
        Scene.model_validate(d)


def test_pose_orientation_must_be_a_unit_quaternion() -> None:  # non-unit q would make _R a non-rotation
    for bad in ((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0 + 2e-5)):
        with pytest.raises(ValidationError, match="unit"):
            Pose(position=(0.0, 0.0, 0.0), orientation=bad)
    Pose(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0 + 5e-6))  # within 1e-5
    cid = p.connect(p.DIRECT)
    try:
        q = p.getQuaternionFromEuler((0.3, -1.1, 2.7), physicsClientId=cid)
        body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX, halfExtents=(0.02, 0.02, 0.02), physicsClientId=cid), basePosition=(0, 0, 1),
            baseOrientation=q, physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        for _ in range(50):
            p.stepSimulation(physicsClientId=cid)
        Pose(position=(0.0, 0.0, 0.0), orientation=tuple(q))
        Pose(position=(0.0, 0.0, 0.0), orientation=tuple(p.getBasePositionAndOrientation(body, physicsClientId=cid)[1]))
    finally:
        p.disconnect(cid)
