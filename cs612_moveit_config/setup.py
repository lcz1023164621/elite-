import os
from collections import defaultdict
from glob import glob
from pathlib import Path

from setuptools import setup
from setuptools.command.develop import develop as _develop

package_name = "cs612_moveit_config"
setup_dir = Path(__file__).resolve().parent
project_root = setup_dir.parent


class _ColconCompatibleDevelop(_develop):
    """
    兼容部分环境下 colcon 对 setup.py develop 追加的参数：
    --uninstall --editable --build-directory
    """

    user_options = list(_develop.user_options) + [
        ("uninstall", None, "compat no-op for colcon"),
        ("editable", None, "compat no-op for colcon"),
        ("build-directory=", None, "compat no-op for colcon"),
        ("script-dir=", None, "compat option for setup.cfg [develop]"),
        ("install-dir=", None, "compat option for setup.cfg [develop]"),
    ]
    boolean_options = list(getattr(_develop, "boolean_options", [])) + ["uninstall", "editable"]

    def initialize_options(self):
        super().initialize_options()
        self.uninstall = False
        self.editable = False
        self.build_directory = None
        self.script_dir = None
        self.install_dir = None

    def run(self):
        if self.uninstall:
            # colcon 的清理阶段；此处无需做实际卸载，返回成功即可。
            return
        super().run()


def collect_tree_data_files(relative_dir: str) -> list:
    """Install project_root/<relative_dir> into share/<package>/<relative_dir>."""
    base = project_root / relative_dir
    if not base.is_dir():
        return []
    by_dest: defaultdict[str, list[str]] = defaultdict(list)
    for path in base.rglob("*"):
        if path.is_file():
            rel_under = path.relative_to(base)
            dest = os.path.join("share", package_name, relative_dir, str(rel_under.parent))
            rel_to_setup = os.path.relpath(path, setup_dir)
            by_dest[dest].append(rel_to_setup)
    return list(by_dest.items())


data_files = [
    ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
    (os.path.join("share", package_name), ["package.xml"]),
    (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    (os.path.join("share", package_name, "config"), glob("config/*")),
    # 仅打包 shell 辅助脚本；Python 入口由 scripts= 安装到 lib/<pkg>/，避免重复打包
    (os.path.join("share", package_name, "scripts"), glob("scripts/*.sh")),
]
data_files.extend(collect_tree_data_files("my_arms"))
data_files.extend(collect_tree_data_files("worlds"))
data_files.extend(collect_tree_data_files("arms_models"))

# 使用 scripts= 安装可执行文件：直接 import main，不依赖 setuptools 生成的
# importlib.metadata 包装器（在 colcon 安装不完整或缺 dist-info 时会 PackageNotFoundError）。
_scripts_dir = setup_dir / "scripts"
_console_scripts = [
    str(_scripts_dir / "cs612_joint_states_bridge"),
    str(_scripts_dir / "cs612_gz_pose_bridge"),
    str(_scripts_dir / "cs612_trajectory_action_bridge"),
    str(_scripts_dir / "cs612_pick_place_demo"),
    str(_scripts_dir / "cs612_motion_smoke_test"),
    str(_scripts_dir / "cs612_print_scene_targets"),
    str(_scripts_dir / "cs612_auto_pick_place"),
    str(_scripts_dir / "cs612_world_markers"),
    str(_scripts_dir / "cs612_static_arm_state"),
    str(_scripts_dir / "cs612_planning_scene_spawner"),
    str(_scripts_dir / "cs612_system_watchdog"),
    str(_scripts_dir / "cs612_base_anchor_guard"),
    str(_scripts_dir / "cs612_regression_check"),
    str(_scripts_dir / "cs612_scene_alignment_check"),
]

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=data_files,
    install_requires=["setuptools", "pyyaml"],
    zip_safe=False,
    scripts=_console_scripts,
    maintainer="TODO",
    maintainer_email="todo@example.com",
    description="MoveIt 2 configuration for CS612 arm",
    license="BSD-3-Clause",
    cmdclass={"develop": _ColconCompatibleDevelop},
)
