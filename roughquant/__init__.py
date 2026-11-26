"""包入口：分块自适应分位滤波算子。"""
from .aggregator import AdaptiveQuantileFilter, ScratchBudget
from .core import (
    MaterialSpec,
    PositionResult,
    Sample,
    UnknownSample,
    evaluate_position,
    quantile_rank,
)
from .geometry import CellId, Rect, SurfaceMap
from .handoff import (
    CALIBRATION_CHANGE,
    NO_CHANGE,
    RADIUS_CHANGE,
    RANK_CHANGE,
    ChangeRecord,
    attribute_change,
    build_handoff,
)
from .pipeline import build_filter, build_surface, evaluate_local

__all__ = [
    "AdaptiveQuantileFilter",
    "ScratchBudget",
    "MaterialSpec",
    "PositionResult",
    "Sample",
    "UnknownSample",
    "evaluate_position",
    "quantile_rank",
    "CellId",
    "Rect",
    "SurfaceMap",
    "CALIBRATION_CHANGE",
    "NO_CHANGE",
    "RADIUS_CHANGE",
    "RANK_CHANGE",
    "ChangeRecord",
    "attribute_change",
    "build_handoff",
    "build_filter",
    "build_surface",
    "evaluate_local",
]
