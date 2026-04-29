"""Helpers for extracting model poses from Gazebo Pose_V TF bridge topics."""
from __future__ import annotations

from geometry_msgs.msg import Point, PoseStamped, Quaternion
from tf2_msgs.msg import TFMessage


def _frame_tokens(frame_id: str) -> list[str]:
    cleaned = (frame_id or "").replace("::", "/").strip("/")
    return [part for part in cleaned.split("/") if part]


def _matches_model(frame_id: str, model_name: str) -> bool:
    tokens = _frame_tokens(frame_id)
    return model_name in tokens


def _rank_model_frame(frame_id: str, model_name: str) -> int:
    tokens = _frame_tokens(frame_id)
    if not tokens:
        return 100
    if tokens[-1] == model_name:
        return 0
    if len(tokens) >= 2 and tokens[-2] == model_name and tokens[-1] in ("base", "box_link"):
        return 1
    if model_name in tokens:
        return 5
    return 100


def extract_model_pose(msg: TFMessage, model_name: str) -> PoseStamped | None:
    """Return the best PoseStamped for a Gazebo model from a TFMessage.

    Gazebo PosePublisher may expose either model frames (``rect_pickup``) or
    link frames (``rect_pickup::box_link``). The rect and carton links in this
    project have zero local offset, so link frames are valid fallbacks.
    """
    best = None
    best_rank = 100
    for transform in msg.transforms:
        child = transform.child_frame_id or ""
        if not _matches_model(child, model_name):
            continue
        rank = _rank_model_frame(child, model_name)
        if rank >= best_rank:
            continue
        ps = PoseStamped()
        ps.header = transform.header
        if not ps.header.frame_id:
            ps.header.frame_id = "world"
        ps.pose.position = Point(
            x=float(transform.transform.translation.x),
            y=float(transform.transform.translation.y),
            z=float(transform.transform.translation.z),
        )
        ps.pose.orientation = Quaternion(
            x=float(transform.transform.rotation.x),
            y=float(transform.transform.rotation.y),
            z=float(transform.transform.rotation.z),
            w=float(transform.transform.rotation.w),
        )
        best = ps
        best_rank = rank
        if rank == 0:
            break
    return best
