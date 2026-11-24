"""把采集交接记录装配成算子输入，并产出检测路径交接结果。

装配职责：

- 从 ``capture.json`` 的 records/tensors 构造格网、材料配置、定标册；
- 每个读头只装载自己辖区的原始量化码，帧与定标版绑定；
- ``query_replay`` 描述同一查询先后收到的右读头定标版：网关先按记录
  回放旧版应答，再重发新版应答。算子拒绝旧版结果，只有新版重算齐备
  后才允许交付。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import CalibrationBook, CalibrationTable
from .geometry import Grid, ScratchBudget
from .handoff import build_handoff
from .heads import (
    FrequencyReply,
    HeadPort,
    IdentityReply,
    LocalReadHead,
)
from .operator import (
    ChunkedAdaptiveQuantileFilter,
    MaterialConfig,
    PositionResult,
    UndecidedReason,
)
from caseio import load_case


def build_grid(records: dict) -> Grid:
    rows, cols = records["grid"]
    spacing_row, spacing_col = records["spacing"]
    return Grid(
        rows=int(rows),
        cols=int(cols),
        spacing_row=int(spacing_row),
        spacing_col=int(spacing_col),
    )


def build_calibration_book(records: dict) -> CalibrationBook:
    book = CalibrationBook()
    for version, per_head in records["calibration"].items():
        for head_name, table_values in per_head.items():
            book.add(
                CalibrationTable(
                    version=str(version),
                    head=str(head_name),
                    table=tuple(int(v) for v in table_values),
                )
            )
    return book


def build_materials(records: dict) -> dict[int, MaterialConfig]:
    configs = {}
    for material, spec in records["materials"].items():
        q_num, q_den = spec["q"]
        configs[int(material)] = MaterialConfig(
            material=int(material),
            radii=tuple(int(r) for r in spec["radii"]),
            minimum_samples=int(spec["minimum"]),
            q_num=int(q_num),
            q_den=int(q_den),
            threshold=int(spec["threshold"]),
        )
    return configs


def _embed_tile(full_shape, tile, half):
    r0, r1, c0, c1 = half
    canvas = np.zeros(full_shape, dtype=tile.dtype)
    canvas[r0:r1, c0:c1] = tile
    return canvas


def build_frame_heads(
    frame: dict,
    records: dict,
    tensors: dict,
    grid: Grid,
    version: str,
) -> dict[str, LocalReadHead]:
    """为一帧构造各读头；读头只持有自己辖区的码片。"""
    heads: dict[str, LocalReadHead] = {}
    limits = records["head_limits"]
    status = tensors[frame["status"]]
    for head_name, tensor_name in ((frame["left"], "left"), (frame["right"], "right")):
        half = records["halves"][tensor_name]
        head = LocalReadHead(
            name=tensor_name,
            grid=grid,
            bound_calibration=version,
            max_response_entries=int(limits[tensor_name]["max_response_entries"]),
        )
        head.bind_classes(tensors["classes"])
        embedded = _embed_tile((grid.rows, grid.cols), tensors[head_name], half)
        head.load_frame(
            frame_id=frame["id"],
            codes=embedded,
            status=status,
            half=half,
        )
        heads[tensor_name] = head
    return heads


@dataclass
class ReplayEvent:
    head: str
    operation: str
    reply_calibration: str
    accepted: bool


class ReplayGateway(HeadPort):
    """带 ``query_replay`` 规则的读头网关。

    命中回放规则的查询先返回旧定标版应答（算子必须拒绝），随后的重发
    才返回绑定新版的应答。同一读头同时持有新旧两版实例，旧版仅用于
    回放，不参与任何新版目标的计算。
    """

    def __init__(
        self,
        name: str,
        current: LocalReadHead,
        legacy: LocalReadHead | None,
        replay_rules: list[dict],
        frame_id: str,
    ) -> None:
        self.name = name
        self._current = current
        self._legacy = legacy
        self.bound_calibration = current.bound_calibration
        self.bound_grid = current.bound_grid
        self._rules = [
            rule
            for rule in replay_rules
            if rule["head"] == name and rule["request"] == frame_id
        ]
        # 每个命中规则的窗口只在首次接触时回放一次旧版，之后无论分页还是
        # 身份查询都走绑定新版；重发由调用方重新发起评估完成。
        self._replayed_windows: set[tuple] = set()
        self.events: list[ReplayEvent] = []

    def _matching_rule(self, request) -> dict | None:
        """按窗口键精确匹配该位置实际查阅窗口的回放规则。"""
        if request.window_key is None:
            return None
        target_cell, _radius, _material = request.window_key
        for rule in self._rules:
            rq, cq = rule["query"]
            if self._current.bound_grid.cell_id(rq, cq) == target_cell:
                return rule
        return None

    def _should_replay_stale(self, request) -> dict | None:
        rule = self._matching_rule(request)
        if rule is None or self._legacy is None:
            return None
        key = request.window_key
        if key in self._replayed_windows:
            return None
        self._replayed_windows.add(key)
        return rule

    def frequency_page(self, request):
        rule = self._should_replay_stale(request)
        if rule is not None:
            legacy_reply = self._legacy.frequency_page(request)
            stale = self._stamp(legacy_reply, rule["reply_calibration"])
            self.events.append(
                ReplayEvent(self.name, "frequency", stale.calibration_version, False)
            )
            return stale
        reply = self._current.frequency_page(request)
        self.events.append(
            ReplayEvent(self.name, "frequency", reply.calibration_version, True)
        )
        return reply

    def identity_page(self, request):
        rule = self._should_replay_stale(request)
        if rule is not None:
            legacy_reply = self._legacy.identity_page(request)
            stale = self._stamp(legacy_reply, rule["reply_calibration"])
            self.events.append(
                ReplayEvent(self.name, "identity", stale.calibration_version, False)
            )
            return stale
        reply = self._current.identity_page(request)
        self.events.append(
            ReplayEvent(self.name, "identity", reply.calibration_version, True)
        )
        return reply

    @staticmethod
    def _stamp(reply, version: str):
        if isinstance(reply, FrequencyReply):
            return FrequencyReply(
                frame_id=reply.frame_id, head=reply.head,
                calibration_version=version, r0=reply.r0, r1=reply.r1,
                c0=reply.c0, c1=reply.c1, material=reply.material,
                entries=reply.entries,
                total_distinct_codes=reply.total_distinct_codes,
                offset=reply.offset, complete=reply.complete,
            )
        return IdentityReply(
            frame_id=reply.frame_id, head=reply.head,
            calibration_version=version, code=reply.code,
            sample_ids=reply.sample_ids, total_for_code=reply.total_for_code,
            ordinal=reply.ordinal, complete=reply.complete,
        )

    def read_raw_code(self, frame_id: str, row: int, col: int) -> int:
        return self._current.read_raw_code(frame_id, row, col)

    def owns_cell(self, frame_id: str, row: int, col: int) -> bool:
        return self._current.owns_cell(frame_id, row, col)


def run_capture(capture_path: Path, capacity_path: Path, output_path: Path | None = None):
    """端到端跑一组合成采集资料，返回可 JSON 序列化的交接结果。"""
    records, tensors = load_case(capture_path)
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))

    grid = build_grid(records)
    book = build_calibration_book(records)
    materials = build_materials(records)
    queries = [tuple(q) for q in records["queries"]]
    replay_rules = records.get("query_replay", [])

    frames_results: dict[str, list[PositionResult]] = {}
    frame_meta = {}
    gateway_events: dict[str, list[ReplayEvent]] = {}
    for frame in records["frames"]:
        version = frame["calibration"]
        current_heads = build_frame_heads(frame, records, tensors, grid, version)
        legacy_heads: dict[str, LocalReadHead] = {}
        for replay in replay_rules:
            if replay["request"] != frame["id"]:
                continue
            legacy_version = replay["reply_calibration"]
            if legacy_version == version:
                continue
            legacy_frame = dict(frame, calibration=legacy_version, status="complete")
            legacy_heads = build_frame_heads(
                legacy_frame, records, tensors, grid, legacy_version
            )
        heads: dict[str, HeadPort] = {}
        gateways: dict[str, ReplayGateway] = {}
        for name, current in current_heads.items():
            legacy = legacy_heads.get(name)
            gateway = ReplayGateway(
                name=name, current=current, legacy=legacy,
                replay_rules=replay_rules, frame_id=frame["id"],
            )
            heads[name] = gateway
            gateways[name] = gateway

        scratch = ScratchBudget(int(capacity["peak_scratch_bytes"]))
        operator = ChunkedAdaptiveQuantileFilter(
            grid=grid,
            classes=tensors["classes"],
            status=tensors[frame["status"]],
            materials=materials,
            heads=heads,
            calibration_book=book,
            frame_id=frame["id"],
            calibration_version=version,
            scratch=scratch,
            page_limit=min(
                int(records["head_limits"][h]["max_response_entries"])
                for h in records["head_limits"]
            ),
            interior_rows=max(1, capacity["grid"][0] // 64),
        )
        first_pass = list(operator.evaluate_positions(queries))
        # 命中旧版回放的位置：网关已前进到新版应答，完整重算一次。
        results = []
        for result in first_pass:
            if UndecidedReason.STALE_CALIBRATION in result.undecided_reasons:
                rerun = operator.evaluate_position(result.row, result.col)
                if (
                    UndecidedReason.STALE_CALIBRATION
                    not in rerun.undecided_reasons
                ):
                    results.append(rerun)
                else:
                    results.append(result)
            else:
                results.append(result)
        frames_results[frame["id"]] = results
        frame_meta[frame["id"]] = {
            "calibration": version,
            "status": frame["status"],
            "scratch_peak_bytes": scratch.peak_bytes,
            "scratch_limit_bytes": scratch.limit_bytes,
        }
        for gateway in gateways.values():
            gateway_events.setdefault(frame["id"], []).extend(gateway.events)

    indexed, transitions = build_handoff(frames_results)
    payload = _serialize(
        records, frames_results, transitions, frame_meta, gateway_events
    )
    if output_path is not None:
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return payload


def _serialize(records, frames_results, transitions, frame_meta, gateway_events):
    def result_to_dict(result: PositionResult):
        return {
            "row": result.row,
            "col": result.col,
            "sample_id": result.sample_id,
            "material": result.material,
            "frame": result.frame_id,
            "calibration": result.calibration_version,
            "target_status": result.target_status,
            "selected_radius": result.radius,
            "valid_count": result.valid_count,
            "pending_potential": result.pending_potential,
            "quantile_value": result.quantile_value,
            "witness_id": result.witness_id,
            "target_value": result.target_value,
            "out_of_tolerance": result.out_of_tolerance,
            "delivery": result.delivery.value,
            "undecided_reasons": [r.value for r in result.undecided_reasons],
            "radius_chain": [
                {"radius": r, "valid": v, "pending": p}
                for r, v, p in result.radius_chain
            ],
        }

    return {
        "queries": records["queries"],
        "frames": frame_meta,
        "results": {
            frame_id: [result_to_dict(r) for r in results]
            for frame_id, results in frames_results.items()
        },
        "transitions": [
            {
                "row": t.row,
                "col": t.col,
                "from_frame": t.from_frame,
                "to_frame": t.to_frame,
                "cause": t.cause.value,
                "radius_changed": t.radius_changed,
                "calibration_changed": t.calibration_changed,
                "quantile_value_changed": t.quantile_value_changed,
                "witness_changed": t.witness_changed,
                "target_value_changed": t.target_value_changed,
                "out_of_tolerance_changed": t.out_of_tolerance_changed,
                "became_deliverable": t.became_deliverable,
                "detail": t.detail,
            }
            for t in transitions
        ],
        "replay_events": {
            frame_id: [
                {
                    "head": e.head,
                    "operation": e.operation,
                    "reply_calibration": e.reply_calibration,
                    "accepted": e.accepted,
                }
                for e in events
            ]
            for frame_id, events in gateway_events.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把采集测量片接入自适应分位滤波检测路径"
    )
    parser.add_argument("capture", type=Path, help="capture.json 路径")
    parser.add_argument("capacity", type=Path, help="capacity.json 路径")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="交接结果输出 JSON（缺省只打印摘要）",
    )
    args = parser.parse_args()
    payload = run_capture(args.capture, args.capacity, args.output)
    delivered = sum(
        1
        for frame_results in payload["results"].values()
        for item in frame_results
        if item["delivery"] == "delivered"
    )
    total = sum(len(items) for items in payload["results"].values())
    print(
        f"帧 {len(payload['results'])}，位置结果 {total} 条，"
        f"可交付 {delivered} 条，过渡 {len(payload['transitions'])} 条"
    )
    if args.output is not None:
        print(f"交接结果已写入 {args.output}")


if __name__ == "__main__":
    main()
