"""根据 scene_objects.yaml 计算吸附/放置用的目标点（base_link 坐标系）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory


def _load_cfg(path: Path | None) -> dict[str, Any]:
    if path is not None and path.is_file():
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    share = Path(get_package_share_directory("cs612_moveit_config"))
    default = share / "config" / "scene_objects.yaml"
    return yaml.safe_load(default.read_text(encoding="utf-8"))


def compute_targets(cfg: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    """返回顶面中心、吸附前上方点、箱内上方放置点（单位 m）。"""
    r = cfg["rect_pickup"]
    cx, cy, cz = r["center_xyz"]
    _, _, sz = r["size_xyz"]
    clear = float(r.get("approach_clearance", 0.05))
    top_z = cz + 0.5 * sz
    top_center = (cx, cy, top_z)
    approach = (cx, cy, top_z + clear)
    c = cfg["carton_box"]
    place_cfg = cfg.get("place_target") or {}
    configured_place = place_cfg.get("tcp_xyz") or place_cfg.get("xyz")
    if isinstance(configured_place, list) and len(configured_place) == 3:
        place = (float(configured_place[0]), float(configured_place[1]), float(configured_place[2]))
    else:
        px, py = c["center_xy"]
        fz = float(c["floor_top_z"])
        ph = float(c["place_height_above_floor"])
        place = (float(px), float(py), fz + ph)
    return {
        "top_center": top_center,
        "approach": approach,
        "place_tcp": place,
    }


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="打印吸附/放置目标点（与 my_world.sdf 一致）")
    p.add_argument("--config", type=Path, default=None, help="覆盖默认 scene_objects.yaml")
    args = p.parse_args()
    cfg = _load_cfg(args.config)
    t = compute_targets(cfg)
    print(f"参考系: {cfg.get('reference_frame', 'base_link')}")
    print("  矩形块顶面中心 (TCP 应对准该点略上方再下压): "
          f"({t['top_center'][0]:.4f}, {t['top_center'][1]:.4f}, {t['top_center'][2]:.4f})")
    print("  吸附前预定位 (顶面中心 + clearance): "
          f"({t['approach'][0]:.4f}, {t['approach'][1]:.4f}, {t['approach'][2]:.4f})")
    print("  放入箱内上方 (TCP 目标近似，需根据姿态微调): "
          f"({t['place_tcp'][0]:.4f}, {t['place_tcp'][1]:.4f}, {t['place_tcp'][2]:.4f})")
    print("说明: 实际 TCP 为 suction_cup_link，需 IK 求解姿态；此处仅为位置参考。")


if __name__ == "__main__":
    main()
