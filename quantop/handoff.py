"""检测路径交接状态与跨帧变化归因。

模型团队需要回查一次“由缺样转为可信”的变化到底来自：

- 半径改变（补交的同材带让位置从大窗口缩回小窗口）；
- 秩改变（同半径、同定标下样本集合或样本数变化，选中秩项或见证移动）；
- 定标改变（换版只作用于归它管辖的读头；分位物理值可能不变，但平台区
  见证换了读头——这是更难查的情况，必须以见证身份而不是分位值为准）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .operator import DeliveryState, PositionResult


class ChangeCause(str, Enum):
    RADIUS = "radius"
    RANK = "rank"
    CALIBRATION = "calibration"
    STABLE = "stable"


@dataclass(frozen=True)
class Transition:
    """同一位置相邻两帧之间的交接变化记录。"""

    row: int
    col: int
    from_frame: str
    to_frame: str
    cause: ChangeCause
    radius_changed: bool
    calibration_changed: bool
    quantile_value_changed: bool
    witness_changed: bool
    target_value_changed: bool
    out_of_tolerance_changed: bool
    became_deliverable: bool
    detail: str


def _key(result: PositionResult) -> tuple[int, int]:
    return result.row, result.col


def attribute_transition(
    previous: PositionResult, current: PositionResult
) -> Transition:
    """对同一位置的前后帧结果做单一主因归因。

    归因优先级与物理含义一致：先看邻域选择是否变化，再看定标版是否切换，
    最后才是同版同半径下的秩变化。定标改变时即使分位值保持原值，只要
    见证身份换了也必须显式记录，避免复查人员按旧测量点追查。
    """
    if (previous.row, previous.col) != (current.row, current.col):
        raise ValueError("只能对同一位置做交接归因")

    radius_changed = previous.radius != current.radius
    calibration_changed = (
        previous.calibration_version != current.calibration_version
    )
    quantile_changed = previous.quantile_value != current.quantile_value
    witness_changed = previous.witness_id != current.witness_id
    target_changed = previous.target_value != current.target_value
    tolerance_changed = previous.out_of_tolerance != current.out_of_tolerance
    became_deliverable = (
        previous.delivery is not DeliveryState.DELIVERED
        and current.delivery is DeliveryState.DELIVERED
    )

    if radius_changed:
        cause = ChangeCause.RADIUS
        detail = "补交资料使所选物理半径改变，邻域样本集合随之替换"
    elif calibration_changed:
        cause = ChangeCause.CALIBRATION
        if not quantile_changed and witness_changed:
            detail = "定标换版后分位物理值不变，但平台区并列使见证换到另一读头"
        elif quantile_changed:
            detail = "定标换版改变校正后物理值，分位值与/或见证随之改变"
        else:
            detail = "定标换版未改变该位置的分位值与见证"
    elif quantile_changed or witness_changed:
        cause = ChangeCause.RANK
        detail = "同半径同定标下秩次或并列见证改变"
    else:
        cause = ChangeCause.STABLE
        detail = "校正后分位、见证与目标读数均无变化"

    return Transition(
        row=current.row,
        col=current.col,
        from_frame=previous.frame_id,
        to_frame=current.frame_id,
        cause=cause,
        radius_changed=radius_changed,
        calibration_changed=calibration_changed,
        quantile_value_changed=quantile_changed,
        witness_changed=witness_changed,
        target_value_changed=target_changed,
        out_of_tolerance_changed=tolerance_changed,
        became_deliverable=became_deliverable,
        detail=detail,
    )


def build_handoff(frames_results: dict[str, list[PositionResult]]):
    """按帧序列生成每位置的结果时间线与相邻帧过渡。

    返回 ``(indexed, transitions)``：``indexed`` 按位置汇总各帧结果，
    ``transitions`` 只包含相邻两帧都有结果的位置。
    """
    indexed: dict[tuple[int, int], dict[str, PositionResult]] = {}
    frame_order = list(frames_results)
    for frame_id, results in frames_results.items():
        for result in results:
            indexed.setdefault(_key(result), {})[frame_id] = result

    transitions: list[Transition] = []
    for cell, by_frame in indexed.items():
        for earlier, later in zip(frame_order, frame_order[1:]):
            if earlier in by_frame and later in by_frame:
                transitions.append(
                    attribute_transition(by_frame[earlier], by_frame[later])
                )
    transitions.sort(key=lambda t: (t.row, t.col, t.to_frame))
    return indexed, transitions
