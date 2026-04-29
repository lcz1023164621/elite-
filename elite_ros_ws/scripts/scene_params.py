#!/usr/bin/env python3
import argparse
import os
import shlex


# ============================================================
# 统一场景配置文件
#
# 以后要改物料箱 / 包装箱位置、尺寸、测试安全点，只改这个文件。
# start_pick_scene.sh、add_boxes_to_planning_scene.py、
# test_tool0_points.py 都从这里读取。
#
# 坐标单位：m
# 尺寸单位：m
# ============================================================


# =========================
# 1. 包装箱六种规格
# =========================
PACKING_BOX_SPECS = {
    # a. 最大 450 x 450 x 210 mm
    "a": {
        "length": 0.45,
        "width": 0.45,
        "height": 0.21,
    },

    # a100. 后期可能的 450 x 450 x 100 mm
    "a100": {
        "length": 0.45,
        "width": 0.45,
        "height": 0.10,
    },

    # b. 最小 160 x 120 x 185 mm
    "b": {
        "length": 0.16,
        "width": 0.12,
        "height": 0.185,
    },

    # c. 常用款 260 x 260 x 260 mm
    "c": {
        "length": 0.26,
        "width": 0.26,
        "height": 0.26,
    },

    # d. 310 x 220 x 160 mm
    "d": {
        "length": 0.31,
        "width": 0.22,
        "height": 0.16,
    },

    # e. 310 x 265 x 220 mm
    "e": {
        "length": 0.31,
        "width": 0.265,
        "height": 0.22,
    },

    # f. 360 x 360 x 360 mm
    "f": {
        "length": 0.36,
        "width": 0.36,
        "height": 0.36,
    },
}


# =========================
# 2. 物料箱参数
# =========================
MATERIAL_BOX_DEFAULT = {
    # 建议先用 0.65，比 0.8 更容易被机械臂稳定够到
    # 如果你想恢复之前位置，改回 x=0.8
    "x": 0.65,
    "y": 0.0,
    "z": 0.0,

    "length": 0.6,
    "width": 1.0,
    "height": 0.3,

    "wall_thickness": 0.015,
    "wall_alpha": 1.0,
}


# =========================
# 3. 包装箱参数
# =========================
PACKING_BOX_DEFAULT = {
    "spec": "c",

    # 建议先用 y=0.55，比 0.8 更容易规划
    # 如果你想恢复之前位置，改回 y=0.8
    "x": 0.0,
    "y": 0.55,
    "z": 0.0,

    "wall_thickness": 0.01,
    "wall_alpha": 1.0,
}


# =========================
# 4. 测试运动参数
# =========================
TEST_DEFAULT = {
    # 高空安全中转点
    "safe_x": 0.30,
    "safe_y": -0.45,
    "safe_z": 0.95,

    # 箱口上方高度余量
    # material_above_z = material_z + material_height + above_clearance
    # packing_above_z = packing_z + packing_height + above_clearance
    "above_clearance": 0.55,

    # tool0 位置允许误差
    "goal_tolerance": 0.02,
}


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _get_str_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return str(default)
    return str(value)


def get_scene(pack_spec: str | None = None) -> dict:
    """
    Python 脚本统一从这里读取场景参数。
    也支持环境变量覆盖，方便临时测试。
    """

    spec = pack_spec or _get_str_env("PACK_BOX_SPEC", PACKING_BOX_DEFAULT["spec"])

    if spec not in PACKING_BOX_SPECS:
        raise ValueError(
            f"未知包装箱规格：{spec}，可选：{list(PACKING_BOX_SPECS.keys())}"
        )

    spec_size = PACKING_BOX_SPECS[spec]

    material = {
        "x": _get_float_env("MATERIAL_BOX_X", MATERIAL_BOX_DEFAULT["x"]),
        "y": _get_float_env("MATERIAL_BOX_Y", MATERIAL_BOX_DEFAULT["y"]),
        "z": _get_float_env("MATERIAL_BOX_Z", MATERIAL_BOX_DEFAULT["z"]),

        "length": _get_float_env("MATERIAL_BOX_L", MATERIAL_BOX_DEFAULT["length"]),
        "width": _get_float_env("MATERIAL_BOX_W", MATERIAL_BOX_DEFAULT["width"]),
        "height": _get_float_env("MATERIAL_BOX_H", MATERIAL_BOX_DEFAULT["height"]),

        "wall_thickness": _get_float_env(
            "MATERIAL_WALL_THICKNESS",
            MATERIAL_BOX_DEFAULT["wall_thickness"],
        ),
        "wall_alpha": _get_float_env(
            "MATERIAL_WALL_ALPHA",
            MATERIAL_BOX_DEFAULT["wall_alpha"],
        ),
    }

    packing = {
        "spec": spec,

        "x": _get_float_env("PACK_BOX_X", PACKING_BOX_DEFAULT["x"]),
        "y": _get_float_env("PACK_BOX_Y", PACKING_BOX_DEFAULT["y"]),
        "z": _get_float_env("PACK_BOX_Z", PACKING_BOX_DEFAULT["z"]),

        "length": _get_float_env("PACK_BOX_L", spec_size["length"]),
        "width": _get_float_env("PACK_BOX_W", spec_size["width"]),
        "height": _get_float_env("PACK_BOX_H", spec_size["height"]),

        "wall_thickness": _get_float_env(
            "PACK_WALL_THICKNESS",
            PACKING_BOX_DEFAULT["wall_thickness"],
        ),
        "wall_alpha": _get_float_env(
            "PACK_WALL_ALPHA",
            PACKING_BOX_DEFAULT["wall_alpha"],
        ),
    }

    test = {
        "safe_x": _get_float_env("TEST_SAFE_X", TEST_DEFAULT["safe_x"]),
        "safe_y": _get_float_env("TEST_SAFE_Y", TEST_DEFAULT["safe_y"]),
        "safe_z": _get_float_env("TEST_SAFE_Z", TEST_DEFAULT["safe_z"]),

        "above_clearance": _get_float_env(
            "TEST_ABOVE_CLEARANCE",
            TEST_DEFAULT["above_clearance"],
        ),

        "goal_tolerance": _get_float_env(
            "TEST_GOAL_TOLERANCE",
            TEST_DEFAULT["goal_tolerance"],
        ),
    }

    return {
        "material_box": material,
        "packing_box": packing,
        "test": test,
    }


def _emit_bash(scene: dict) -> None:
    """
    给 bash 脚本使用：

        eval "$(python3 scripts/scene_params.py --bash --spec c)"
    """

    material = scene["material_box"]
    packing = scene["packing_box"]
    test = scene["test"]

    items = {
        "MATERIAL_BOX_X": material["x"],
        "MATERIAL_BOX_Y": material["y"],
        "MATERIAL_BOX_Z": material["z"],
        "MATERIAL_BOX_L": material["length"],
        "MATERIAL_BOX_W": material["width"],
        "MATERIAL_BOX_H": material["height"],
        "MATERIAL_WALL_THICKNESS": material["wall_thickness"],
        "MATERIAL_WALL_ALPHA": material["wall_alpha"],

        "PACK_BOX_SPEC": packing["spec"],
        "PACK_BOX_X": packing["x"],
        "PACK_BOX_Y": packing["y"],
        "PACK_BOX_Z": packing["z"],
        "PACK_BOX_L": packing["length"],
        "PACK_BOX_W": packing["width"],
        "PACK_BOX_H": packing["height"],
        "PACK_WALL_THICKNESS": packing["wall_thickness"],
        "PACK_WALL_ALPHA": packing["wall_alpha"],

        "TEST_SAFE_X": test["safe_x"],
        "TEST_SAFE_Y": test["safe_y"],
        "TEST_SAFE_Z": test["safe_z"],
        "TEST_ABOVE_CLEARANCE": test["above_clearance"],
        "TEST_GOAL_TOLERANCE": test["goal_tolerance"],
    }

    for key, value in items.items():
        print(f"export {key}={shlex.quote(str(value))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bash", action="store_true")
    parser.add_argument("--spec", default=None)
    args = parser.parse_args()

    scene = get_scene(args.spec)

    if args.bash:
        _emit_bash(scene)


if __name__ == "__main__":
    main()
