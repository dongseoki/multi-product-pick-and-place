"""Build the local manufacturer robot include for scenes/v0.xml."""

from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from check_mycobot_model import prepare_urdf
from check_neck_grasp import configure_model

OUTPUT = Path(__file__).resolve().parents[1] / "scenes" / "_mycobot"


def main() -> None:
    with TemporaryDirectory(prefix="mycobot-v0-build-") as directory:
        urdf, source = prepare_urdf(Path(directory))
        configure_model(urdf, source)
        root = ET.parse(urdf.with_suffix(".xml")).getroot()
        include = ET.Element("mujocoinclude")
        for name in ("asset", "worldbody", "actuator", "equality"):
            section = root.find(name)
            if section is None:
                raise RuntimeError(f"Missing robot model section: {name}")
            if name == "worldbody":
                # The scene owns the support and all bottles, not the robot include.
                for item in list(section):
                    if item.tag != "body" or item.attrib.get("name") != "g_base":
                        section.remove(item)
                for geom in section.iter("geom"):
                    geom.set("rgba", "0.71 0.74 0.79 1")
            include.append(section)

        OUTPUT.mkdir(exist_ok=True)
        for mesh in include.iter("mesh"):
            file = mesh.get("file")
            if file is not None:
                asset = Path(directory) / file
                (OUTPUT / asset.name).write_bytes(asset.read_bytes())
                mesh.set("file", f"_mycobot/{asset.name}")
        ET.indent(include)
        ET.ElementTree(include).write(OUTPUT / "robot.xml", encoding="utf-8")
    print(f"Prepared {OUTPUT / 'robot.xml'}")


if __name__ == "__main__":
    main()
