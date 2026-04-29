"""Monitor base pose drift and try to re-anchor CS612 model in Gazebo."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any

import rclpy
from rclpy.node import Node


def _find_gz_bin() -> str:
    if shutil.which("/usr/bin/ign"):
        return "/usr/bin/ign"
    if shutil.which("/usr/bin/gz"):
        return "/usr/bin/gz"
    return shutil.which("ign") or shutil.which("gz") or "ign"


def _json_stream(stream) -> Iterator[dict[str, Any]]:
    dec = json.JSONDecoder()
    buf = ""
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            stripped = buf.lstrip()
            if stripped != buf:
                buf = stripped
            if not buf:
                break
            try:
                obj, idx = dec.raw_decode(buf)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                yield obj
            buf = buf[idx:]


def _extract_pose(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        if {"position", "orientation"}.issubset(payload):
            pos = payload.get("position")
            ori = payload.get("orientation")
            if isinstance(pos, dict) and isinstance(ori, dict):
                if {"x", "y", "z"}.issubset(pos) and {"x", "y", "z", "w"}.issubset(ori):
                    return payload
        for v in payload.values():
            p = _extract_pose(v)
            if p is not None:
                return p
    if isinstance(payload, list):
        for v in payload:
            p = _extract_pose(v)
            if p is not None:
                return p
    return None


class BaseAnchorGuard(Node):
    def __init__(self) -> None:
        super().__init__("cs612_base_anchor_guard")
        self.declare_parameter("model_name", "cs612")
        self.declare_parameter("world_name", "arm_world")
        self.declare_parameter("base_pose_topic", "/model/cs612/pose")
        self.declare_parameter("drift_tolerance_xy", 0.01)
        self.declare_parameter("drift_tolerance_z", 0.01)
        self.declare_parameter("max_recover_attempts", 3)

        self._gz = _find_gz_bin()
        self._recover_attempts = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def _monitor_loop(self) -> None:
        topic = str(self.get_parameter("base_pose_topic").value)
        while rclpy.ok() and not self._stop.is_set():
            proc: subprocess.Popen[str] | None = None
            try:
                proc = subprocess.Popen(
                    [self._gz, "topic", "-e", "-t", topic, "--json-output"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                if proc.stdout is None:
                    raise RuntimeError("gz topic stdout unavailable")
                for obj in _json_stream(proc.stdout):
                    pose = _extract_pose(obj)
                    if pose is None:
                        continue
                    pos = pose["position"]
                    x = float(pos.get("x", 0.0))
                    y = float(pos.get("y", 0.0))
                    z = float(pos.get("z", 0.0))
                    if self._is_drifted(x, y, z):
                        self._try_reanchor(x, y, z)
            except Exception as exc:
                if not self._stop.is_set():
                    self.get_logger().warn(f"base anchor monitor异常，将重试: {exc}")
            finally:
                if proc is not None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=0.5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            time.sleep(1.0)

    def _is_drifted(self, x: float, y: float, z: float) -> bool:
        tol_xy = max(0.001, float(self.get_parameter("drift_tolerance_xy").value))
        tol_z = max(0.001, float(self.get_parameter("drift_tolerance_z").value))
        return abs(x) > tol_xy or abs(y) > tol_xy or abs(z) > tol_z

    def _try_reanchor(self, x: float, y: float, z: float) -> None:
        max_attempts = int(self.get_parameter("max_recover_attempts").value)
        if self._recover_attempts >= max_attempts:
            self.get_logger().error(
                f"base_link 漂移持续存在且已超过恢复次数: ({x:.4f}, {y:.4f}, {z:.4f})"
            )
            return
        self._recover_attempts += 1
        world = str(self.get_parameter("world_name").value)
        model = str(self.get_parameter("model_name").value)
        self.get_logger().warn(
            f"检测到底座漂移 ({x:.4f}, {y:.4f}, {z:.4f})，执行重锚定尝试 #{self._recover_attempts}"
        )
        req = (
            f'name: "{model}" '
            "position {x: 0 y: 0 z: 0} "
            "orientation {x: 0 y: 0 z: 0 w: 1}"
        )
        cmd = [
            self._gz,
            "service",
            "-s",
            f"/world/{world}/set_pose",
            "--reqtype",
            "ignition.msgs.Pose",
            "--reptype",
            "ignition.msgs.Boolean",
            "--timeout",
            "1000",
            "--req",
            req,
        ]
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3.0)
        except Exception as exc:
            self.get_logger().warn(f"重锚定命令执行失败: {exc}")

    def destroy_node(self) -> bool:
        self._stop.set()
        self._thread.join(timeout=1.0)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = BaseAnchorGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
