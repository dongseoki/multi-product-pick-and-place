"""Simulate one commanded close, lift, and hold with the pinned myCobot URDF."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from math import cos, sin, tau
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from xml.etree import ElementTree as ET

import mujoco

from check_mycobot_model import DRIVE, prepare_urdf

# Single-bottle test poses (radians in manufacturer arm-joint order).
START = (-0.00625177, -1.62534365, 2.17351433, -0.55637027, 0.00625150, -0.00005529)
LIFT = (-0.00625419, -1.36386027, 1.88549590, -0.52983521, 0.00625395, -0.00005531)
OPEN, CLOSED = 0.15, -0.50
DT = 0.001
HOLD_START = 1.8
END = 2.3
MIN_LIFT_M = 0.020
FINGERS = {"gripper_left1", "gripper_right1"}
BOTTLE = "bottle_target"
NECK = "bottle_neck"


def vector(numbers) -> str:
    return " ".join(map(str, numbers))


def configure_model(urdf: Path, source: ET.Element):
    raw = mujoco.MjModel.from_xml_path(str(urdf))
    xml = urdf.with_suffix(".xml")
    mujoco.mj_saveLastXML(str(xml), raw)
    tree = ET.parse(xml)
    root = tree.getroot()
    world = root.find("worldbody")
    asset = root.find("asset")
    if world is None or asset is None:
        raise RuntimeError("Imported model must have a worldbody and mesh assets")

    actuators = ET.SubElement(root, "actuator")
    equality = ET.SubElement(root, "equality")
    arm = []
    coupled = {DRIVE: (1.0, 0.0)}
    for joint in source.findall("joint"):
        if joint.get("type") != "revolute":
            continue
        name = joint.attrib["name"]
        mimic = joint.find("mimic")
        if mimic is not None:
            if mimic.attrib["joint"] != DRIVE:
                raise RuntimeError(f"Unexpected coupling on {name}")
            multiplier = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
            coupled[name] = (multiplier, offset)
            ET.SubElement(
                equality, "joint", joint1=name, joint2=DRIVE,
                polycoef=vector((offset, multiplier, 0, 0, 0)), solref="0.004 1",
            )
            continue
        is_gripper = name == DRIVE
        if not is_gripper:
            arm.append(name)
        ET.SubElement(
            actuators, "position", name=name, joint=name,
            kp="20" if is_gripper else "300",
            kv="0.25" if is_gripper else "3",
            forcerange="-1 1" if is_gripper else "-12 12",
        )

    if len(arm) != len(START) or len(coupled) != 6:
        raise RuntimeError("Unexpected arm or gripper joint count")

    posed = mujoco.MjData(raw)
    for name, angle in zip(arm, START, strict=True):
        posed.joint(name).qpos[0] = angle
    mujoco.mj_forward(raw, posed)
    hand = posed.body("gripper_base")
    # Align the upright bottle neck with the inner pads in the starting pose.
    bottom = hand.xpos + hand.xmat.reshape(3, 3) @ (-0.00062, 0.054, -0.0085)
    bottom[2] -= 0.130

    ET.SubElement(
        root, "option", timestep=str(DT), integrator="implicitfast",
        iterations="80", cone="elliptic",
    )
    for body in world.iter("body"):
        body.set("gravcomp", "1")
    for geom in world.iter("geom"):
        # Isolate robot–bottle and robot–support contacts from robot self-collision.
        geom.set("contype", "1")
        geom.set("conaffinity", "6")
    ET.SubElement(
        world, "geom", name="support", type="box", size="0.035 0.035 0.010",
        pos=vector((bottom[0], bottom[1], bottom[2] - 0.010)),
        contype="4", conaffinity="2",
    )

    shoulder = (
        (radius * cos(tau * i / 32), radius * sin(tau * i / 32), z)
        for radius, z in ((0.025, -0.005), (0.012, 0.005)) for i in range(32)
    )
    ET.SubElement(
        asset, "mesh", name="shoulder_taper",
        vertex=vector(value for point in shoulder for value in point),
    )
    bottle = ET.SubElement(world, "body", name=BOTTLE, pos=vector(bottom))
    ET.SubElement(bottle, "freejoint", name="bottle_free")
    for name, center, mass, shape in (
        ("bottle_body", "0.055", "0.110", {"type": "cylinder", "size": "0.025 0.055"}),
        ("bottle_shoulder", "0.115", "0.010", {"type": "mesh", "mesh": "shoulder_taper"}),
        (NECK, "0.130", "0.020", {"type": "cylinder", "size": "0.012 0.010"}),
        ("bottle_cap", "0.145", "0.010", {"type": "cylinder", "size": "0.015 0.005"}),
    ):
        ET.SubElement(
            bottle, "geom", name=name, pos=f"0 0 {center}", mass=mass,
            contype="2", conaffinity="5", friction="0.8 0.005 0.001", condim="4", **shape,
        )
    tree.write(xml, encoding="utf-8")
    return mujoco.MjModel.from_xml_path(str(xml)), arm, coupled


@dataclass
class Trial:
    peak_lift: float = 0.0
    hold_lift: float = float("inf")
    two_finger_hold: int = 0
    hold_steps: int = 0
    support_hold_steps: int = 0
    forbidden: dict[tuple[str, str], float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            not self.forbidden
            and self.hold_steps > 0
            and self.two_finger_hold == self.hold_steps
            and self.support_hold_steps == 0
            and self.hold_lift >= MIN_LIFT_M
        )


def run_trial(model, arm, coupled, trace=None) -> Trial:
    data = mujoco.MjData(model)
    for name, angle in zip(arm, START, strict=True):
        data.joint(name).qpos[0] = angle
    for name, (multiplier, offset) in coupled.items():
        data.joint(name).qpos[0] = OPEN * multiplier + offset
    mujoco.mj_forward(model, data)

    bottle_id = model.body(BOTTLE).id
    support_id = model.geom("support").id
    initial_z = data.body(BOTTLE).xpos[2]
    result = Trial()

    def observe(holding: bool) -> None:
        fingers = set()
        supported = False
        for contact in data.contact:
            a, b = contact.geom
            body_a, body_b = model.geom_bodyid[[a, b]]
            if bottle_id not in (body_a, body_b):
                continue
            if body_b == bottle_id:
                a, b = b, a
            if b == support_id:
                supported = True
                continue
            part = model.geom(a).name
            link = model.body(model.geom_bodyid[b]).name
            if part == NECK and link in FINGERS:
                fingers.add(link)
            else:
                result.forbidden.setdefault((part, link), data.time)

        height = data.body(BOTTLE).xpos[2] - initial_z
        result.peak_lift = max(result.peak_lift, height)
        if holding:
            result.hold_lift = min(result.hold_lift, height)
            result.hold_steps += 1
            result.two_finger_hold += fingers == FINGERS
            result.support_hold_steps += supported

    observe(False)
    if trace is not None:
        trace.append((data.time, data.qpos.copy(), data.qvel.copy()))
    for step in range(round(END / DT)):
        time = step * DT
        close = min(max((time - 0.20) / 0.55, 0.0), 1.0)
        lift = min(max((time - 0.95) / 0.85, 0.0), 1.0)
        for name, start, end in zip(arm, START, LIFT, strict=True):
            data.actuator(name).ctrl[0] = start + (end - start) * lift
        data.actuator(DRIVE).ctrl[0] = OPEN + (CLOSED - OPEN) * close
        # After reset only the actuators move the robot; the bottle is always free.
        mujoco.mj_step(model, data)
        observe(time >= HOLD_START)
        if trace is not None and (step + 1) % 10 == 0:
            trace.append((data.time, data.qpos.copy(), data.qvel.copy()))
    if abs(data.time - END) > DT or any(w.number for w in data.warning):
        raise RuntimeError("Simulation reported a warning or reset during the trial")
    return result


def show_replay(model, trace) -> None:
    """Replay sampled simulation states without altering the measured trial."""
    from mujoco import viewer

    data = mujoco.MjData(model)
    for name, color in (
        ("bottle_body", (0.25, 0.64, 0.86, 1)),
        ("bottle_shoulder", (0.86, 0.49, 0.19, 1)),
        (NECK, (0.25, 0.75, 0.39, 1)),
        ("bottle_cap", (0.85, 0.24, 0.21, 1)),
    ):
        model.geom(name).rgba = color

    def set_frame(frame) -> None:
        data.time, qpos, qvel = frame
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)

    set_frame(trace[0])
    with viewer.launch_passive(model, data) as window:
        with window.lock():
            window.cam.lookat[:] = data.geom(NECK).xpos
            window.cam.distance = 0.31
            window.cam.azimuth = 55
            window.cam.elevation = -17
            window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        print("Viewer opened: replaying at 0.25x. Close the window to exit.", flush=True)
        # Pause on the initial and final pose so the viewer has time to inspect both.
        period = 1.0 + END * 4 + 2.0
        start = monotonic()
        while window.is_running():
            replay_time = ((monotonic() - start) % period - 1.0) / 4
            index = min(max(round(replay_time / (10 * DT)), 0), len(trace) - 1)
            with window.lock():
                set_frame(trace[index])
            window.sync()
            sleep(1 / 60)


def print_result(model, result: Trial) -> None:

    print(f"Model: {model.nu} position actuators, {model.neq} joint couplings")
    print(
        "Bottle lift (peak / hold minimum): "
        f"{result.peak_lift * 1000:.2f} / {result.hold_lift * 1000:.2f} mm"
    )
    print(f"Two-finger neck contact during hold: {result.two_finger_hold}/{result.hold_steps} steps")
    print(f"Support contacts during hold: {result.support_hold_steps}/{result.hold_steps} steps")
    contacts = ", ".join(
        f"{part}:{link}@{time:.3f}s"
        for (part, link), time in sorted(result.forbidden.items(), key=lambda item: item[1])
    )
    print(f"Forbidden bottle contacts: {contacts or 'none'}")
    print("PASS: neck-only grasp and lift" if result.passed else "FAIL: neck-only grasp and lift")


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="replay the trial in the MuJoCo GUI")
    args = parser.parse_args()
    trace = [] if args.viewer else None
    with TemporaryDirectory(prefix="mycobot-grasp-") as scratch:
        urdf, source = prepare_urdf(Path(scratch))
        model, arm, coupled = configure_model(urdf, source)
        result = run_trial(model, arm, coupled, trace)
        print_result(model, result)
        if trace is not None:
            show_replay(model, trace)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
