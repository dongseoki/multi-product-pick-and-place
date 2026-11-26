"""Sample geometric clearance along the day-four v0 scene path."""

from math import atan2, cos, pi, sin
from pathlib import Path

import mujoco
import numpy as np

from check_neck_grasp import BOTTLE, CLOSED, DRIVE, OPEN, START, Trial, vector
from check_v0_scene import SCENE, SUPPORTS

HAND_OFFSET = np.array([-0.00062, 0.054, -0.1385])  # hand origin to bottle bottom
PICKUP_YAW = -pi / 2
LIFT_HEIGHT = 0.060
APPROACH_HEIGHT = 0.045


def yaw_matrix(angle: float) -> np.ndarray:
    c, s = cos(angle), sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def check_layout(xml: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("standby").id)
    mujoco.mj_forward(model, data)
    pickup = data.body(BOTTLE).xpos.copy()
    placement = data.site("target_cell_center").xpos.copy()
    cross = pickup[0] * placement[1] - pickup[1] * placement[0]
    turn = atan2(cross, np.dot(pickup[:2], placement[:2]))

    drive = model.joint(DRIVE).id
    arm = [model.joint(joint).name for joint in model.actuator_trnid[:, 0] if joint != drive]
    coupled = {DRIVE: (1.0, 0.0)}
    for i in range(model.neq):
        if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT and model.eq_obj2id[i] == drive:
            offset, multiplier = model.eq_data[i, :2]
            coupled[model.joint(model.eq_obj1id[i]).name] = (multiplier, offset)
    if len(arm) != len(START) or len(coupled) != 6:
        raise RuntimeError("Unexpected arm or gripper drive configuration")
    resting = {frozenset(pair) for pair in SUPPORTS.values()}
    hand = model.body("gripper_base").id
    bottle_id = model.body(BOTTLE).id
    bottle_qpos = model.joint("bottle_free").qposadr[0]
    joint_ids = [model.joint(name).id for name in arm]
    qpos = model.jnt_qposadr[joint_ids]
    dofs = model.jnt_dofadr[joint_ids]
    low, high = model.jnt_range[joint_ids].T
    jp = np.zeros((3, model.nv))
    jr = np.zeros_like(jp)
    damping = 0.002 * np.eye(len(arm))
    q = np.array(START)
    standby = pickup + np.array([0, 0, APPROACH_HEIGHT])
    raised = pickup + np.array([0, 0, LIFT_HEIGHT])
    place_yaw = PICKUP_YAW + turn
    raised_place = yaw_matrix(turn) @ raised

    def solve(bottom: np.ndarray, yaw: float) -> None:
        nonlocal q
        goal_rot = yaw_matrix(yaw)
        goal_pos = bottom - goal_rot @ HAND_OFFSET
        for _ in range(100):
            data.qpos[qpos] = q
            mujoco.mj_kinematics(model, data)
            rot = data.xmat[hand].reshape(3, 3)
            residual = np.r_[goal_pos - data.xpos[hand], 0.5 * np.cross(rot.T, goal_rot.T).sum(axis=0)]
            if np.linalg.norm(residual) < 5e-6:
                return
            mujoco.mj_comPos(model, data)
            mujoco.mj_jacBody(model, data, jp, jr, hand)
            jac = np.r_[jp[:, dofs], jr[:, dofs]]
            delta = np.linalg.lstsq(
                np.r_[jac, damping], np.r_[residual, np.zeros(len(arm))], rcond=None,
            )[0]
            q = np.clip(q + np.clip(delta, -0.15, 0.15), low, high)
        raise RuntimeError(f"No arm pose found for {bottom} at yaw {yaw:.4f}")

    def verify(phase, tool_bottom, yaw, grip, bottle_bottom) -> None:
        solve(tool_bottom, yaw)
        for name, (multiplier, offset) in coupled.items():
            angle = grip * multiplier + offset
            joint = model.joint(name)
            if not joint.range[0] - 1e-9 <= angle <= joint.range[1] + 1e-9:
                raise RuntimeError(f"{phase}: {name} exceeds its URDF joint limits")
            data.joint(name).qpos[0] = angle
        data.qpos[bottle_qpos:bottle_qpos + 3] = bottle_bottom
        mujoco.mj_forward(model, data)
        trial = Trial()
        for contact in data.contact:
            a, b = map(int, contact.geom)
            if model.geom_bodyid[b] == bottle_id:
                a, b = b, a
            part, other = model.geom(a).name, model.geom(b).name
            if frozenset((part, other)) in resting:
                continue
            if model.geom_bodyid[a] == bottle_id:
                if part == "bottle_body" and other in {"support", "target_cell_floor"}:
                    continue
                if model.geom_contype[b] == 1:
                    link = model.body(model.geom_bodyid[b]).name
                    trial.record_contact(part, link, 0)
                    if not trial.forbidden:
                        continue
                else:
                    link = other
                raise RuntimeError(f"{phase}: {part} touched {link} at {bottle_bottom}")
            if model.geom_contype[b] == 1:
                a, b = b, a
            raise RuntimeError(
                f"{phase}: {model.body(model.geom_bodyid[a]).name} touched "
                f"{model.geom(b).name} at {tool_bottom}"
            )

    def linear(phase, start, end, yaw, grip, carry, parked_bottle) -> None:
        intervals = max(1, int(np.ceil(np.linalg.norm(end - start) / 0.001)))
        for frac in np.linspace(0, 1, intervals + 1):
            point = start + frac * (end - start)
            verify(phase, point, yaw, grip, point if carry else parked_bottle)
        print(f"PASS {phase}: {intervals + 1} poses")

    def print_pose(name) -> None:
        print(f"{name} q [rad]: {vector(round(float(value), 8) for value in q)}")

    verify("standby", standby, PICKUP_YAW, OPEN, pickup)
    print_pose("Standby")
    linear("pickup approach", standby, pickup, PICKUP_YAW, OPEN, False, pickup)
    print_pose("Pickup")
    for grip in np.linspace(OPEN, CLOSED, 66):
        verify("close", pickup, PICKUP_YAW, grip, pickup)
    print("PASS close: 66 poses")
    linear("lift", pickup, raised, PICKUP_YAW, CLOSED, True, None)

    for theta in np.linspace(0, turn, 321):
        point = yaw_matrix(theta) @ raised
        verify("transfer", point, PICKUP_YAW + theta, CLOSED, point)
    print("PASS transfer: 321 poses")
    linear("placement descent", raised_place, placement, place_yaw, CLOSED, True, None)
    print_pose("Placement")
    for grip in np.linspace(CLOSED, OPEN, 66):
        verify("open", placement, place_yaw, grip, placement)
    print("PASS open: 66 poses")
    linear("placement retreat", placement, raised_place, place_yaw, OPEN, False, placement)


if __name__ == "__main__":
    check_layout(SCENE)
