"""Check the v0 MJCF's occupancy and idle stability, with optional rendering."""

from argparse import ArgumentParser
from math import cos, radians
from pathlib import Path
from struct import pack
from zlib import compress, crc32

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "scenes" / "v0.xml"
SUPPORTS = {
    "bottle_target": ("bottle_body", "support"),
    "bottle_placed_1": ("bottle_placed_1_body", "cell_floor_1"),
    "bottle_placed_2": ("bottle_placed_2_body", "cell_floor_2"),
    "bottle_placed_3": ("bottle_placed_3_body", "cell_floor_3"),
}
EXPECTED_CELLS = {(-1, -1), (-1, 1), (1, -1)}
MAX_DRIFT = 0.001
MAX_TILT_DEG = 1.0
IDLE_SECONDS = 1.0


def require(condition, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_scene(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("standby").id)
    mujoco.mj_forward(model, data)

    bottles = {
        model.body(i).name for i in range(model.nbody) if model.body(i).name.startswith("bottle_")
    }
    require(bottles == set(SUPPORTS), f"Wrong bottle inventory: {sorted(bottles)}")
    ids = [model.body(name).id for name in SUPPORTS]
    masses = model.body_mass[ids]
    require(np.allclose(masses, 0.150, atol=1e-9, rtol=0), f"Wrong bottle masses: {masses}")
    target = model.body("bottle_target").id
    initial = data.xpos[ids].copy()
    for name, body_id in zip(SUPPORTS, ids, strict=True):
        qadr = model.jnt_qposadr[model.body_jntadr[body_id]]
        require(np.allclose(data.qpos[qadr:qadr + 7], model.qpos0[qadr:qadr + 7]),
                f"Standby keyframe does not match declared initial pose of {name}")

    box = model.body("packing_box").id
    rotation = data.xmat[box].reshape(3, 3)
    target_center = rotation.T @ (data.site("target_cell_center").xpos - data.xpos[box])
    require(np.allclose(target_center, (0.035, 0.035, 0.005), atol=1e-7),
            f"Target center outside expected box cell: {target_center}")

    def check_occupancy() -> None:
        support = model.geom("support").id
        require(np.allclose(data.xpos[target, :2], data.geom_xpos[support, :2], atol=MAX_DRIFT, rtol=0),
                "Target bottle is not on the pickup support")
        placed = (data.xpos[ids[1:]] - data.xpos[box]) @ rotation
        cells = {tuple(np.sign(point[:2])) for point in placed}
        require(cells == EXPECTED_CELLS and np.allclose(np.abs(placed[:, :2]), 0.035, atol=MAX_DRIFT, rtol=0),
                f"Expected one bottle centered in each non-target cell, got {placed.tolist()}")

    permitted = {frozenset(pair) for pair in SUPPORTS.values()}

    def actual_contacts() -> set[frozenset[str]]:
        active = {
            frozenset(model.geom(int(g)).name for g in contact.geom)
            for contact in data.contact if contact.dist <= 0
        }
        require(active <= permitted, f"Unexpected contact at {data.time:.3f} s: {active - permitted}")
        return active

    check_occupancy()
    actual_contacts()
    peak_drift = 0.0
    tilt_limit = cos(radians(MAX_TILT_DEG))
    for _ in range(round(IDLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
        actual_contacts()
        drift = np.linalg.norm(data.xpos[ids] - initial, axis=1)
        peak_drift = max(peak_drift, float(np.max(drift)))
        require(peak_drift <= MAX_DRIFT, f"Bottle moved {peak_drift * 1000:.3f} mm while idle")
        require(np.all(data.xmat[ids, 8] >= tilt_limit), f"Bottle tilted by more than {MAX_TILT_DEG:g}°")
    mujoco.mj_forward(model, data)
    check_occupancy()
    require(actual_contacts() == permitted, "A bottle is not supported by its intended base at the end")
    require(abs(data.time - IDLE_SECONDS) < 1e-8 and not any(w.number for w in data.warning),
            "Simulation terminated early or issued a warning")

    print(f"PASS: {len(bottles)} bottles, total {sum(masses):.3f} kg")
    print("PASS: 3 occupied cells, 1 empty target cell (+x, +y)")
    print(f"PASS: standby held for {data.time:.3f} s; max bottle drift {peak_drift * 1000:.3f} mm")
    print("PASS: only bottle-base support contacts; all 4 bottles supported after settling")
    return data


def screenshot(model: mujoco.MjModel, data: mujoco.MjData, path: Path) -> None:
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.102, 0.075, 0.144]
    camera.distance = 0.69
    camera.azimuth = 126
    camera.elevation = -49
    width, height = 1000, 750
    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return pack(">I", len(payload)) + tag + payload + pack(">I", crc32(tag + payload))

    rows = b"".join(b"\x00" + row.tobytes() for row in rgb)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", compress(rows)) + chunk(b"IEND", b"")
    )
    print(f"Saved screenshot: {path}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, help="write a PNG of the settled scene")
    parser.add_argument("--viewer", action="store_true", help="open the verified scene in the MuJoCo viewer")
    args = parser.parse_args()
    if not (SCENE.parent / "_mycobot" / "robot.xml").is_file():
        parser.error("Missing robot assets. Run scripts/prepare_v0_robot.py as described in docs/scene.md.")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = check_scene(model)
    if args.screenshot:
        screenshot(model, data, args.screenshot)
    if args.viewer:
        from mujoco import viewer

        mujoco.mj_resetDataKeyframe(model, data, model.key("standby").id)
        mujoco.mj_forward(model, data)
        print("Opening MuJoCo viewer at the standby keyframe.", flush=True)
        viewer.launch(model, data)


if __name__ == "__main__":
    main()
