"""Run a kinematic smoke check against the pinned manufacturer model."""

from concurrent.futures import ThreadPoolExecutor
from math import dist
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from xml.etree import ElementTree as ET

import mujoco

REVISION = "25bec49559ba119395f4d335d3da230bdad494d8"
SOURCE = f"https://raw.githubusercontent.com/elephantrobotics/mycobot_ros2/{REVISION}"
URDF = "mycobot_description/urdf/mycobot_280_m5/mycobot_280_m5_adaptive_gripper.urdf"
COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
DRIVE = "gripper_controller"


def prepare_urdf(directory: Path) -> tuple[Path, ET.Element]:
    import trimesh

    path = directory / "mycobot.urdf"
    urlretrieve(f"{SOURCE}/{URDF}", path)
    tree = ET.parse(path)
    robot = tree.getroot()
    ET.SubElement(
        ET.SubElement(robot, "mujoco"),
        "compiler",
        discardvisual="true",
        fusestatic="false",
        strippath="false",
    )
    meshes = list(robot.iter("mesh"))
    sources = {mesh.attrib["filename"].removeprefix("package://") for mesh in meshes}
    converted = {source: f"mesh_{i}.obj" for i, source in enumerate(sorted(sources))}

    def convert(source: str) -> None:
        obj = directory / converted[source]
        dae = obj.with_suffix(".dae")
        urlretrieve(f"{SOURCE}/{source}", dae)
        unit = ET.parse(dae).find("c:asset/c:unit", COLLADA_NS)
        geometry = trimesh.load_scene(dae).to_geometry()
        if unit is not None:
            geometry.apply_scale(float(unit.attrib["meter"]))
        geometry.export(obj)

    with ThreadPoolExecutor(max_workers=5) as pool:
        for _ in pool.map(convert, sorted(sources)):
            pass
    for mesh in meshes:
        source = mesh.attrib["filename"].removeprefix("package://")
        mesh.set("filename", converted[source])
    tree.write(path, encoding="utf-8")
    return path, robot


def check_kinematics(model: mujoco.MjModel, robot: ET.Element) -> None:
    data = mujoco.MjData(model)
    joints = robot.findall("joint")
    arm = [
        joint.attrib["name"]
        for joint in joints
        if joint.get("type") == "revolute"
        and joint.find("mimic") is None
        and joint.get("name") != DRIVE
    ]
    mimics = [
        (joint.attrib["name"], mimic)
        for joint in joints
        if (mimic := joint.find("mimic")) is not None
    ]
    if (len(arm), len(mimics), model.njnt, model.nmesh) != (6, 5, 12, 15):
        raise RuntimeError("Unexpected arm, gripper joint, or mesh count")

    def set_joint(name: str, angle: float) -> None:
        joint_id = model.joint(name).id
        low, high = model.jnt_range[joint_id]
        if not low <= angle <= high:
            raise RuntimeError(f"Test angle outside limits: {name}={angle}")
        data.qpos[model.jnt_qposadr[joint_id]] = angle

    mujoco.mj_forward(model, data)
    base = data.body("gripper_base")
    before = base.xmat.copy()
    set_joint(arm[0], 0.20)
    mujoco.mj_forward(model, data)
    if dist(before, base.xmat) <= 1e-6:
        raise RuntimeError("Arm joint did not change gripper orientation")

    fingers = ("gripper_left1", "gripper_right1")

    def move_gripper(angle: float) -> tuple[float, float, float]:
        set_joint(DRIVE, angle)
        for name, mimic in mimics:
            if mimic.attrib["joint"] != DRIVE:
                raise RuntimeError(f"Unexpected mimic driver for {name}")
            multiplier = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
            set_joint(name, angle * multiplier + offset)
        mujoco.mj_forward(model, data)
        return tuple(data.body(fingers[0]).xpos - data.body(fingers[1]).xpos)

    if dist(move_gripper(-0.40), move_gripper(0.10)) <= 1e-6:
        raise RuntimeError("Gripper joint positions did not change relative finger position")

    print(f"PASS: {len(arm)} arm joints, {len(mimics) + 1} gripper joints, {model.nmesh} meshes")
    print("PASS: arm rotation and coupled finger poses change when qpos is set")
    print(f"Imported control: actuators={model.nu}, equality_constraints={model.neq}")
    print("NOT TESTED: commanded actuation, bottle contact, lift")


def main() -> None:
    with TemporaryDirectory(prefix="mycobot-smoke-") as scratch:
        path, robot = prepare_urdf(Path(scratch))
        check_kinematics(mujoco.MjModel.from_xml_path(str(path)), robot)


if __name__ == "__main__":
    main()
