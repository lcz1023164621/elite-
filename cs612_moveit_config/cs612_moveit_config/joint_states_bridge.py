"""将 Gazebo 经 ros_gz_bridge 发布的 joint_states 规范为 URDF 关节名并发布到 /joint_states。"""
from __future__ import annotations

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState

# 必须与 robot_state_publisher 对 /joint_states 的订阅 QoS 一致（Humble 下为 SensorData/BEST_EFFORT）。
# 使用 RELIABLE 发布时，部分环境下与 RSP 的 BEST_EFFORT 订阅匹配失败 → 整臂 TF 缺失、RViz 全红。
# depth 过小在 Gazebo Model 关节状态较大时易触发 DDS “sequence size exceeds remaining buffer”。
_JOINT_STATES_PUBLISHER_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    # 采用 RELIABLE，兼容 RSP 可能的默认可靠订阅配置，避免“整链 TF 缺失”。
    # RELIABLE 发布可被 BEST_EFFORT 订阅端接收；反向不成立。
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

_JOINT_STATES_SUBSCRIBER_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

# 与 URDF 中可动关节名一致（顺序固定，便于 robot_state_publisher）
_ARM = ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6")


def _strip_scope(name: str) -> str:
    if "::" in name:
        return name.split("::")[-1]
    return name


def _canonical_joint_name(raw: str) -> str | None:
    """将 Gazebo / 桥接可能输出的 joint1、Joint_2、model::Joint3 等映射到 URDF 中的 JointN。"""
    key = _strip_scope(raw)
    if key in _ARM:
        return key
    compact = key.replace("_", "").lower()
    alias = {
        "shoulderpanjoint": "Joint1",
        "shoulderliftjoint": "Joint2",
        "elbowjoint": "Joint3",
        "wrist1joint": "Joint4",
        "wrist2joint": "Joint5",
        "wrist3joint": "Joint6",
    }
    if compact in alias:
        return alias[compact]
    if compact.startswith("joint") and len(compact) > 5:
        suf = compact[5:]
        if suf.isdigit():
            cand = f"Joint{int(suf)}"
            if cand in _ARM:
                return cand
    return None


class JointStatesBridge(Node):
    def __init__(self) -> None:
        super().__init__("cs612_joint_states_bridge")
        # robot_state_publisher 对 joint_states 要求 name 与 position 等长；发布端与 RSP 默认 SensorDataQoS 对齐。
        self._pub = self.create_publisher(JointState, "joint_states", _JOINT_STATES_PUBLISHER_QOS)
        self.create_subscription(JointState, "joint_states_gz", self._cb, _JOINT_STATES_SUBSCRIBER_QOS)
        self._logged_first = False

        self._last: JointState | None = None
        self._logged_seed = False
        # 在 Gazebo 首帧前先发布全零姿态，避免 RViz / RobotModel 因整链缺 TF 而全红。
        # 这里必须使用稳定时钟：use_sim_time=true 且 /clock 尚未桥接时，ROS 时间不会前进，
        # 普通定时器不会触发，/joint_states 也就不会发布，RViz 会报 link1..6 无 TF。
        # 收到真实 joint_states_gz 后会立刻切换为仿真状态。
        self.create_timer(
            1.0 / 20.0,
            self._tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def _make_zero(self) -> JointState:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        for j in _ARM:
            out.name.append(j)
            out.position.append(0.0)
            out.velocity.append(0.0)
            out.effort.append(0.0)
        return out

    def _build_from_gz(self, msg: JointState) -> JointState:
        pos: dict[str, float] = {}
        vel: dict[str, float] = {}
        eff: dict[str, float] = {}
        # Gazebo Model 关节状态常含 world_fixed 等与 URDF 可动关节无关的项：必须按名称对齐，
        # 禁止在「有 name 但含额外关节」时用 position[0..5] 误映射到 Joint1..6。
        if msg.name:
            for i, raw in enumerate(msg.name):
                key = _canonical_joint_name(raw)
                if key is None:
                    continue
                if i < len(msg.position):
                    pos[key] = float(msg.position[i])
                if i < len(msg.velocity):
                    vel[key] = float(msg.velocity[i])
                if i < len(msg.effort):
                    eff[key] = float(msg.effort[i])

        # 仅当完全没有关节名、且仅有 6 个标量时，才按顺序对应 Joint1..6
        if not pos and (not msg.name or len(msg.name) == 0) and len(msg.position) >= len(_ARM):
            for i, jn in enumerate(_ARM):
                pos[jn] = float(msg.position[i])
            if len(msg.velocity) >= len(_ARM):
                for i, jn in enumerate(_ARM):
                    vel[jn] = float(msg.velocity[i])
            if len(msg.effort) >= len(_ARM):
                for i, jn in enumerate(_ARM):
                    eff[jn] = float(msg.effort[i])

        out = JointState()
        out.header = msg.header
        if out.header.stamp.sec == 0 and out.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()
        for j in _ARM:
            out.name.append(j)
            out.position.append(pos.get(j, 0.0))
            out.velocity.append(vel.get(j, 0.0))
            out.effort.append(eff.get(j, 0.0))
        return out

    def _publish_stamped(self, template: JointState) -> None:
        stamped = JointState()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.name = list(template.name)
        stamped.position = list(template.position)
        stamped.velocity = list(template.velocity)
        stamped.effort = list(template.effort)
        self._pub.publish(stamped)

    def _cb(self, msg: JointState) -> None:
        # 仅更新状态，由 _tick 统一按固定频率发布，QoS 与 RSP 一致且首帧即有 TF
        if not self._logged_first and msg.name:
            sample = next((n for n in msg.name if _canonical_joint_name(n)), msg.name[0])
            self.get_logger().info(
                f"已收到 Gazebo 关节状态（示例关节名: {sample}），/joint_states 将与仿真同步。"
            )
            self._logged_first = True
        self._last = self._build_from_gz(msg)

    def _tick(self) -> None:
        if self._last is None:
            self._publish_stamped(self._make_zero())
            if not self._logged_seed:
                self.get_logger().info(
                    "尚未收到 Gazebo 关节状态，先发布零位 /joint_states 以保持 TF 链完整。"
                )
                self._logged_seed = True
            return
        self._publish_stamped(self._last)


def main() -> None:
    rclpy.init()
    node = JointStatesBridge()
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
