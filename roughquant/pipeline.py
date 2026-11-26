"""从采集交接记录构建检测流水线。

同时提供两条等价路径，便于交叉核对：

* :func:`build_filter` 汇合端路径，读头只暴露频次/身份受控应答；
* :func:`evaluate_local` 本地完整张量路径，直接走核心算法原语。
"""
from __future__ import annotations

import numpy as np

from .aggregator import AdaptiveQuantileFilter
from .core import MaterialSpec, evaluate_position
from .geometry import CellId, SurfaceMap
from .heads import HeadStore


def build_surface(records, arrays) -> SurfaceMap:
    return SurfaceMap(
        grid=records["grid"],
        spacing=records["spacing"],
        classes=arrays["classes"],
        halves=records["halves"],
    )


def build_material_specs(records) -> dict[int, MaterialSpec]:
    return {
        int(material): MaterialSpec.from_record(material, spec)
        for material, spec in records["materials"].items()
    }


def frame_sources(frame: dict, arrays):
    return {name: arrays[frame[name]] for name in ("left", "right")}


def build_head_stores(records, arrays, frame: dict,
                      overrides=None, unavailable=None):
    """按帧声明的定标版构建读头私有存储。"""
    overrides = overrides or {}
    unavailable = unavailable or set()
    stores = {}
    for name, tensor_name in (("left", frame["left"]), ("right", frame["right"])):
        r0, _r1, c0, _c1 = records["halves"][name]
        limit = records["head_limits"][name]["max_response_entries"]
        stores[name] = HeadStore(
            head=name,
            frame_id=frame["id"],
            calibration_id=frame["calibration"],
            grid=records["grid"],
            codes=arrays[tensor_name],
            max_response_entries=limit,
            origin=(r0, c0),
            reply_calibration_override=overrides.get(name),
            unavailable=name in unavailable,
        )
    return stores


def build_filter(records, arrays, frame: dict, peak_scratch_bytes: int,
                 overrides=None, unavailable=None) -> AdaptiveQuantileFilter:
    surface = build_surface(records, arrays)
    status = arrays[frame["status"]]
    return AdaptiveQuantileFilter(
        surface=surface,
        material_specs=build_material_specs(records),
        head_stores=build_head_stores(
            records, arrays, frame, overrides=overrides, unavailable=unavailable
        ),
        calibration_tables=records["calibration"],
        frame_id=frame["id"],
        calibration_id=frame["calibration"],
        status=status,
        peak_scratch_bytes=peak_scratch_bytes,
    )


def _local_head_arrays(records, arrays, frame: dict):
    """本地路径下把读头局部码张量贴回全局格网。"""
    grid = records["grid"]
    global_codes = {}
    for name, (r0, r1, c0, c1) in records["halves"].items():
        if (r1 - r0, c1 - c0) != arrays[frame[name]].shape:
            raise ValueError(f"读头 {name} 张量形状与管辖区域不符")
        canvas = np.full(grid, -1, dtype=arrays[frame[name]].dtype)
        canvas[r0:r1, c0:c1] = arrays[frame[name]]
        global_codes[name] = canvas
    return global_codes


def evaluate_local(records, arrays, frame: dict, cell: CellId):
    """不经过读头协议的本地直算，结果应与汇合端路径完全一致。"""
    surface = build_surface(records, arrays)
    specs = build_material_specs(records)
    status = arrays[frame["status"]]
    global_codes = _local_head_arrays(records, arrays, frame)
    tables = records["calibration"][frame["calibration"]]

    def head_of(cand: CellId) -> str:
        return surface.owner_of(cand)

    def loader(cand: CellId, head: str) -> int:
        # 每个读取点沿自己的拥有者和定标表换算，不能跨读头查表。
        return int(tables[head][int(global_codes[head][cand.row, cand.col])])

    def table_range(head: str):
        values = tables[head]
        return min(values), max(values)

    spec = specs[surface.material_at(cell)]
    return evaluate_position(
        surface, cell, spec, status, loader, head_of, table_range
    )
