"""检测路径命令行入口：输出校正后分位、见证与超差交接状态。

用法::

    python -m roughquant.detect data/capture.json data/capacity.json out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from caseio import load_case

from .geometry import CellId
from .handoff import attribute_change, build_handoff
from .pipeline import build_filter, evaluate_local


def _frames_by_id(records):
    return {frame["id"]: frame for frame in records["frames"]}


def run(capture_path: Path, capacity_path: Path, out_dir: Path) -> dict:
    records, arrays = load_case(capture_path)
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    budget = int(capacity["peak_scratch_bytes"])
    frames = _frames_by_id(records)
    queries = [CellId(int(r), int(c)) for r, c in records["queries"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    handoff: dict = {"frames": {}, "changes": [], "replays": []}
    previous_by_cell: dict[tuple[int, int], tuple] = {}

    for frame in (frames[fid] for fid in (f["id"] for f in records["frames"])):
        runner = build_filter(records, arrays, frame, peak_scratch_bytes=budget)
        results = runner.evaluate_chunk([q.as_tuple() for q in queries])

        # 本地直算与汇合端应答路径必须逐字段一致。
        for query, result in zip(queries, results):
            local = evaluate_local(records, arrays, frame, query)
            if local.as_handoff() != result.as_handoff():
                raise AssertionError(
                    f"帧 {frame['id']} 位置 {query} 两条路径结果不一致"
                )

        package = build_handoff(frame["id"], frame["calibration"], results)
        package["scratch_peak_bytes"] = runner.budget.peak
        handoff["frames"][frame["id"]] = package

        for result in results:
            key = result.cell.as_tuple()
            if key in previous_by_cell:
                prior_result, prior_cal = previous_by_cell[key]
                change = attribute_change(
                    prior_result, result, prior_cal, frame["calibration"]
                )
                handoff["changes"].append(change.as_dict())
            previous_by_cell[key] = (result, frame["calibration"])

    # query_replay：同一查询先后收到旧版与新版右读头应答。
    for event in records.get("query_replay", []):
        frame = frames[event["request"]]
        overrides = (
            {event["head"]: event["reply_calibration"]}
            if event["reply_calibration"] != frame["calibration"]
            else None
        )
        runner = build_filter(
            records, arrays, frame, peak_scratch_bytes=budget,
            overrides=overrides,
        )
        cell = CellId(*event["query"])
        result = runner.evaluate(cell)
        handoff["replays"].append({
            "request_frame": event["request"],
            "head": event["head"],
            "reply_calibration": event["reply_calibration"],
            "expected_calibration": frame["calibration"],
            "cell": list(cell.as_tuple()),
            "tolerance_certain": result.tolerance_certain,
            "reasons": list(result.reasons),
        })

    out_path = out_dir / "handoff.json"
    out_path.write_text(
        json.dumps(handoff, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return handoff


def main():
    parser = argparse.ArgumentParser(description="分块自适应分位滤波检测路径")
    parser.add_argument("capture", type=Path)
    parser.add_argument("capacity", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    handoff = run(args.capture, args.capacity, args.out_dir)
    for frame_id, package in handoff["frames"].items():
        print(f"帧 {frame_id}（定标 {package['calibration']}）")
        for pos in package["positions"]:
            flag = "可发布" if pos["certainty"]["tolerance"] else "待判定"
            print(
                f"  位置 {pos['cell']} 半径 {pos['selected_radius']} "
                f"分位 {pos['quantile_value']} 见证 {pos['witness']}"
                f"({pos['witness_head']}) 超差 {pos['out_of_tolerance']} "
                f"偏差 {pos['deviation']} [{flag}]"
            )
    for change in handoff["changes"]:
        print(
            f"变化 {change['cell']}: {change['change_type']} "
            f"{change['calibration_before']} -> {change['calibration_after']}"
        )
    for replay in handoff["replays"]:
        print(
            f"重放 {replay['cell']} 收到 {replay['reply_calibration']} "
            f"应答（期望 {replay['expected_calibration']}）："
            f"{'待判定' if not replay['tolerance_certain'] else '可发布'}"
        )


if __name__ == "__main__":
    main()
