"""Sample geometric clearance for the day-four v0 layout before authoring its MJCF."""

from copy import deepcopy
from math import cos, pi, sin
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from check_mycobot_model import prepare_urdf
from check_neck_grasp import BOTTLE, CLOSED, OPEN, START, Trial, configure_model, vector

HAND_OFFSET = np.array([-0.00062, 0.054, -0.1385])  # hand origin to bottle bottom
PICKUP_YAW = -pi / 2
TURN = pi / 3
BOX_YAW = TURN - pi
CELL_OFFSET = 0.035
LIFT_HEIGHT = 0.060
BOX_DROP = 0.045
RETRACT = 0.045


def yaw_matrix(angle: float) -> np.ndarray:
    c, s = cos(angle), sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def add_placement(xml: Path, box_position: np.ndarray) -> None:
    tree = ET.parse(xml)
    world = tree.find("worldbody")
    if world is None:
        raise RuntimeError("Configured robot scene has no worldbody")
    source = world.find(f"body[@name='{BOTTLE}']")
    if source is None:
        raise RuntimeError("Configured robot scene has no target bottle")
    box = ET.SubElement(
        world, "body", name="packing_box", pos=vector(box_position),
        quat=vector((cos(BOX_YAW / 2), 0, 0, sin(BOX_YAW / 2))),
    )

    def block(name, half_size, center) -> None:
        ET.SubElement(
            box, "geom", name=name, type="box", pos=vector(center),
            size=vector(half_size), contype="4", conaffinity="3",
        )

    block("conveyor_belt", (0.100, 0.100, 0.010), (0, 0, -0.010))
    block("target_cell_floor", (0.0725, 0.0725, 0.0025), (0, 0, 0.0025))
    for i, coordinate in enumerate((-0.070, 0, 0.070)):
        block(f"box_wall_x{i}", (0.0025, 0.0725, 0.020), (coordinate, 0, 0.025))
        block(f"box_wall_y{i}", (0.0725, 0.0025, 0.020), (0, coordinate, 0.025))
    for i, (x, y) in enumerate(
        ((-CELL_OFFSET, -CELL_OFFSET), (-CELL_OFFSET, CELL_OFFSET), (CELL_OFFSET, -CELL_OFFSET)),
        start=1,
    ):
        name = f"bottle_placed_{i}"
        placed = ET.SubElement(box, "body", name=name, pos=vector((x, y, 0.005)))
        for original in source.findall("geom"):
            geom = deepcopy(original)
            geom.set("name", f"{name}_{original.attrib['name']}")
            geom.set("contype", "4")
            geom.set("conaffinity", "3")
            placed.append(geom)
    tree.write(xml, encoding="utf-8")


def check_layout(xml: Path, arm, coupled, pickup, placement) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
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
    standby = pickup - np.array([RETRACT, 0, 0])
    raised = pickup + np.array([0, 0, LIFT_HEIGHT])
    place_yaw = PICKUP_YAW + TURN
    raised_place = yaw_matrix(TURN) @ raised

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

    for theta in np.linspace(0, TURN, 241):
        point = yaw_matrix(theta) @ raised
        verify("transfer", point, PICKUP_YAW + theta, CLOSED, point)
    print("PASS transfer: 241 poses")
    linear("placement descent", raised_place, placement, place_yaw, CLOSED, True, None)
    print_pose("Placement")
    for grip in np.linspace(CLOSED, OPEN, 66):
        verify("open", placement, place_yaw, grip, placement)
    print("PASS open: 66 poses")
    linear("placement retreat", placement, raised_place, place_yaw, OPEN, False, placement)


def main() -> None:
    with TemporaryDirectory(prefix="mycobot-v0-layout-") as directory:
        urdf, source = prepare_urdf(Path(directory))
        initial, arm, coupled = configure_model(urdf, source)
        pose = mujoco.MjData(initial)
        mujoco.mj_forward(initial, pose)
        pickup = pose.body(BOTTLE).xpos.copy()
        placement = yaw_matrix(TURN) @ pickup - (0, 0, BOX_DROP)
        box = placement - yaw_matrix(BOX_YAW) @ (CELL_OFFSET, CELL_OFFSET, 0.005)
        for name, position in (("Pickup bottom", pickup), ("Box bottom", box), ("Target floor", placement)):
            print(f"{name} [m]: {vector(round(float(value), 8) for value in position)}")
        xml = urdf.with_suffix(".xml")
        add_placement(xml, box)
        check_layout(xml, arm, coupled, pickup, placement)


if __name__ == "__main__":
    main()
