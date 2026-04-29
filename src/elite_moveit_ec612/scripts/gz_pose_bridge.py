#!/usr/bin/env python3
"""Read Gazebo pose topics with native `ign topic` and republish them as ROS PoseStamped."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def _find_gz_executable() -> str:
    if shutil.which("/usr/bin/ign"):
        return "/usr/bin/ign"
    if shutil.which("/usr/bin/gz"):
        return "/usr/bin/gz"
    found = shutil.which("ign")
    if found:
        return found
    found = shutil.which("gz")
    return found or "ign"


def _extract_pose_dict(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        position = payload.get("position")
        orientation = payload.get("orientation")
        if isinstance(position, dict) and isinstance(orientation, dict):
            pos_keys = {"x", "y", "z"}
            ori_keys = {"x", "y", "z", "w"}
            if pos_keys.issubset(position) and ori_keys.issubset(orientation):
                return payload
        for value in payload.values():
            found = _extract_pose_dict(value)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _extract_pose_dict(value)
            if found is not None:
                return found
    return None


def _json_objects_from_stream(stream) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
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
                obj, idx = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                yield obj
            buf = buf[idx:]


class GazeboPoseBridge(Node):
    def __init__(self) -> None:
        super().__init__("cs612_gz_pose_bridge")
        self.declare_parameter("frame_id", "base_link")
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._gz = _find_gz_executable()
        self._stop = threading.Event()
        self._procs: list[subprocess.Popen[str]] = []
        self._threads: list[threading.Thread] = []
        self._seen_topics: set[str] = set()

        self._bridges = [
            (
                "/model/rect_pickup/pose",
                self.create_publisher(PoseStamped, "/model/rect_pickup/pose", qos_profile_sensor_data),
            ),
            (
                "/model/carton_box/pose",
                self.create_publisher(PoseStamped, "/model/carton_box/pose", qos_profile_sensor_data),
            ),
        ]

        for topic, pub in self._bridges:
            thread = threading.Thread(
                target=self._bridge_topic,
                args=(topic, pub),
                daemon=True,
                name=f"gz-pose-{topic.rsplit('/', 2)[-2]}",
            )
            thread.start()
            self._threads.append(thread)

    def _bridge_topic(self, topic: str, pub) -> None:
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
                self._procs.append(proc)
                if proc.stdout is None:
                    raise RuntimeError(f"{os.path.basename(self._gz)} topic stdout is unavailable")
                for obj in _json_objects_from_stream(proc.stdout):
                    pose = _extract_pose_dict(obj)
                    if pose is None:
                        continue
                    msg = PoseStamped()
                    msg.header.frame_id = self._frame_id
                    msg.header.stamp = self.get_clock().now().to_msg()
                    position = pose["position"]
                    orientation = pose["orientation"]
                    msg.pose.position = Point(
                        x=float(position.get("x", 0.0)),
                        y=float(position.get("y", 0.0)),
                        z=float(position.get("z", 0.0)),
                    )
                    msg.pose.orientation = Quaternion(
                        x=float(orientation.get("x", 0.0)),
                        y=float(orientation.get("y", 0.0)),
                        z=float(orientation.get("z", 0.0)),
                        w=float(orientation.get("w", 1.0)),
                    )
                    pub.publish(msg)
                    if topic not in self._seen_topics:
                        self._seen_topics.add(topic)
                        p = msg.pose.position
                        self.get_logger().info(
                            f"已桥接 {topic} -> ROS PoseStamped: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
                        )
                if not self._stop.is_set():
                    stderr = ""
                    if proc.stderr is not None:
                        stderr = proc.stderr.read().strip()
                    if stderr:
                        self.get_logger().warn(f"`{os.path.basename(self._gz)} topic` 退出，topic={topic}: {stderr}")
            except Exception as exc:
                if not self._stop.is_set():
                    self.get_logger().warn(f"桥接 {topic} 失败，将重试: {exc}")
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
                    try:
                        self._procs.remove(proc)
                    except ValueError:
                        pass
            if not self._stop.is_set():
                time.sleep(1.0)

    def destroy_node(self) -> bool:
        self._stop.set()
        for proc in list(self._procs):
            try:
                proc.terminate()
            except Exception:
                pass
        for thread in self._threads:
            thread.join(timeout=0.5)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = GazeboPoseBridge()
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
