"""交接状态与跨帧变化归因。

模型团队回查一次“由缺样转为可信”的变化时，需要区分三类来源，三者
互斥并按下列优先级归因（一次变化可能同时伴随多种现象，归因取最根
本、最先发生的那一类）：

1. ``radius`` 半径改变：补交资料使选中窗口从大半径缩回小半径；
2. ``rank`` 秩改变：同一定标版下窗口未变，但分位秩落到不同见证
   （含平台并列样本之间的身份切换）；
3. ``calibration`` 定标改变：窗口与码值样本不变，仅定标版切换，
   分位值可能恰好不变但见证换了读头。
"""
from __future__ import annotations

from dataclasses import dataclass

from .core import PositionResult
from .geometry import CellId

RADIUS_CHANGE = "radius"
RANK_CHANGE = "rank"
CALIBRATION_CHANGE = "calibration"
NO_CHANGE = "none"


@dataclass(frozen=True)
class ChangeRecord:
    cell: CellId
    change_type: str
    radius_before: int | None
    radius_after: int | None
    value_before: int | None
    value_after: int | None
    witness_before: list[int] | None
    witness_after: list[int] | None
    calibration_before: str
    calibration_after: str

    def as_dict(self) -> dict:
        return {
            "cell": list(self.cell.as_tuple()),
            "change_type": self.change_type,
            "radius_before": self.radius_before,
            "radius_after": self.radius_after,
            "quantile_before": self.value_before,
            "quantile_after": self.value_after,
            "witness_before": self.witness_before,
            "witness_after": self.witness_after,
            "calibration_before": self.calibration_before,
            "calibration_after": self.calibration_after,
        }


def attribute_change(
    before: PositionResult,
    after: PositionResult,
    calibration_before: str,
    calibration_after: str,
) -> ChangeRecord:
    """比较同一位置两个帧的结果，给出单一归因。"""
    if before.cell != after.cell:
        raise ValueError("只能比较同一位置的两次结果")

    change = NO_CHANGE
    if before.selected_radius != after.selected_radius:
        change = RADIUS_CHANGE
    elif calibration_before != calibration_after:
        # 定标切换即使分位值恰好相同也单独成类：见证可能已换读头。
        change = CALIBRATION_CHANGE
    elif before.witness != after.witness:
        change = RANK_CHANGE
    elif before.quantile_value != after.quantile_value:
        change = RANK_CHANGE

    return ChangeRecord(
        cell=after.cell,
        change_type=change,
        radius_before=before.selected_radius,
        radius_after=after.selected_radius,
        value_before=before.quantile_value,
        value_after=after.quantile_value,
        witness_before=None
        if before.witness is None else list(before.witness.as_tuple()),
        witness_after=None
        if after.witness is None else list(after.witness.as_tuple()),
        calibration_before=calibration_before,
        calibration_after=calibration_after,
    )


def build_handoff(
    frame_id: str,
    calibration_id: str,
    results,
) -> dict:
    """整帧交接包：逐位置结果附分量级确定性标记。"""
    return {
        "frame": frame_id,
        "calibration": calibration_id,
        "positions": [result.as_handoff() for result in results],
        "releasable": [
            list(result.cell.as_tuple())
            for result in results
            if result.releasable()
        ],
        "pending": [
            list(result.cell.as_tuple())
            for result in results
            if not result.tolerance_certain
        ],
    }
