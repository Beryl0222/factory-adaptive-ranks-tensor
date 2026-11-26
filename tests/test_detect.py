"""合成采集资料的端到端检测路径测试。

覆盖：
* 缺样补交大窗口缩回小窗口（半径改变）；
* 定标平台横跨两个读头，分位值不变但见证换读头（定标改变）；
* 待判定的分量级标记与可证明发布；
* 汇合端受控应答与本地直算两条路径逐字段一致；
* 旧版应答/答复未齐时按受影响范围交付；
* 读头隐私边界：原始码窗口不能被外部取走。
"""
import json
from pathlib import Path
import unittest

from roughquant.geometry import CellId
from roughquant.handoff import (
    CALIBRATION_CHANGE,
    NO_CHANGE,
    RADIUS_CHANGE,
    attribute_change,
)
from roughquant.pipeline import build_filter, evaluate_local

DATA = Path(__file__).resolve().parents[1] / "data"


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        from caseio import load_case

        self.records, self.arrays = load_case(DATA / "capture.json")
        self.capacity = json.loads(
            (DATA / "capacity.json").read_text(encoding="utf-8")
        )
        self.budget = self.capacity["peak_scratch_bytes"]
        self.frames = {f["id"]: f for f in self.records["frames"]}
        self.queries = [CellId(*q) for q in self.records["queries"]]

    def _run_frame(self, frame_id, **kwargs):
        frame = self.frames[frame_id]
        runner = build_filter(
            self.records, self.arrays, frame, self.budget, **kwargs
        )
        results = {
            result.cell: result
            for result in runner.evaluate_chunk(
                [q.as_tuple() for q in self.queries]
            )
        }
        return runner, results

    def test_pending_frame_radius_growth_and_component_certainty(self):
        _, results = self._run_frame("first")
        q1 = results[CellId(1, 3)]
        # 半径 2 仅 2 个有效样本，不足门槛 3，扩到半径 4（10 个）。
        self.assertEqual(q1.selected_radius, 4)
        self.assertEqual(q1.valid_count, 10)
        self.assertEqual(q1.unknown_count, 2)
        # 补交后半径 2 可达标，窗口必然缩回 ⇒ 半径分量待判定。
        self.assertFalse(q1.radius_certain)
        self.assertFalse(q1.quantile_certain)
        self.assertFalse(q1.tolerance_certain)
        self.assertFalse(q1.blocked)
        # 目标读数本身存在。
        self.assertEqual(q1.target_physical, 5)

        q2 = results[CellId(2, 4)]
        # 该窗口内没有未读样本，即使全帧未齐也可独立发布。
        self.assertEqual(q2.selected_radius, 2)
        self.assertTrue(q2.radius_certain)
        self.assertTrue(q2.quantile_certain)
        self.assertTrue(q2.tolerance_certain)
        self.assertFalse(q2.out_of_tolerance)

    def test_patched_frame_shrinks_radius_after_strip_arrives(self):
        _, pending = self._run_frame("first")
        _, patched = self._run_frame("patched")
        q1_before = pending[CellId(1, 3)]
        q1_after = patched[CellId(1, 3)]
        self.assertEqual(q1_before.selected_radius, 4)
        self.assertEqual(q1_after.selected_radius, 2)
        change = attribute_change(
            q1_before, q1_after, "c1", "c1"
        )
        self.assertEqual(change.change_type, RADIUS_CHANGE)
        # 补交后所有分量均可发布，超差结论保持。
        self.assertTrue(q1_after.tolerance_certain)
        self.assertTrue(q1_after.out_of_tolerance)
        self.assertEqual(q1_after.deviation, 2)

    def test_plateau_spans_heads_witness_switches_under_new_calibration(self):
        _, c1 = self._run_frame("patched")
        _, c2 = self._run_frame("new-range")
        before = c1[CellId(2, 4)]
        after = c2[CellId(2, 4)]
        # 两张定标表下分位物理值都是 3，但见证从右读头换到左读头。
        self.assertEqual(before.quantile_value, after.quantile_value)
        self.assertEqual(before.witness, CellId(2, 4))
        self.assertEqual(before.witness_head, "right")
        self.assertEqual(after.witness, CellId(2, 3))
        self.assertEqual(after.witness_head, "left")
        change = attribute_change(before, after, "c1", "c2")
        self.assertEqual(change.change_type, CALIBRATION_CHANGE)
        # 若用分位值代替见证，复查会停在错误的测量点。
        self.assertNotEqual(change.witness_before, change.witness_after)

    def test_unchanged_position_reports_no_change(self):
        _, first = self._run_frame("first")
        _, patched = self._run_frame("patched")
        change = attribute_change(
            first[CellId(3, 6)], patched[CellId(3, 6)], "c1", "c1"
        )
        self.assertEqual(change.change_type, NO_CHANGE)

    def test_tolerance_decision_math(self):
        _, results = self._run_frame("patched")
        # (0,1) 材料 2：目标 6，分位 3，阈值 3，偏差恰好 3 ⇒ 超差。
        q3 = results[CellId(0, 1)]
        self.assertEqual(q3.quantile_value, 3)
        self.assertEqual(q3.deviation, 3)
        self.assertTrue(q3.out_of_tolerance)
        # (3,6)：目标与分位同为 2，偏差 0。
        q4 = results[CellId(3, 6)]
        self.assertEqual(q4.deviation, 0)
        self.assertFalse(q4.out_of_tolerance)

    def test_aggregator_matches_local_path_everywhere(self):
        for frame_id in ("first", "patched", "new-range"):
            frame = self.frames[frame_id]
            _, results = self._run_frame(frame_id)
            for query in self.queries:
                local = evaluate_local(self.records, self.arrays, frame, query)
                self.assertEqual(
                    local.as_handoff(),
                    results[query].as_handoff(),
                    msg=f"帧 {frame_id} 位置 {query} 两条路径不一致",
                )

    def test_stale_calibration_reply_marks_affected_scope_pending(self):
        # new-range 帧请求 c2，右读头却用 c1 应答（query_replay 第一条）。
        runner, results = self._run_frame(
            "new-range", overrides={"right": "c1"}
        )
        result = results[CellId(2, 4)]
        self.assertFalse(result.tolerance_certain)
        self.assertFalse(result.radius_certain)
        self.assertTrue(any("定标版不符" in reason for reason in result.reasons))

    def test_correct_replay_calibration_releases_position(self):
        # 读头按请求帧声明的 c2 应答，无版本覆盖。
        runner, results = self._run_frame("new-range")
        result = results[CellId(2, 4)]
        self.assertTrue(result.tolerance_certain)

    def test_unavailable_reply_is_pending_not_silent(self):
        _, results = self._run_frame("new-range", unavailable={"left"})
        # 只校验依赖左读头证据的位置：(3,6) 是纯右读头邻域，不受影响。
        affected = {CellId(1, 3), CellId(2, 4), CellId(0, 1)}
        for cell in affected:
            self.assertFalse(
                results[cell].tolerance_certain,
                msg=f"位置 {cell} 依赖左读头却被静默发布",
            )
        self.assertTrue(
            all("答复未齐" in r for r in results[CellId(2, 4)].reasons)
        )

    def test_raw_codes_never_leave_head(self):
        runner, _ = self._run_frame("patched")
        store = runner.heads["left"]
        with self.assertRaises(PermissionError):
            _ = store._codes  # noqa: B018 - 隐私守卫必须拦截
        # 受控应答仍然可用。
        reply = store.frequency(
            runner.surface.physical_window(CellId(1, 3), 2),
            1, runner.surface, runner.status, "patched",
        )
        self.assertIsInstance(reply.counts, dict)
        self.assertNotIn("codes", reply.__dict__)

    def test_identity_reply_uses_lexicographic_ordinals(self):
        runner, _ = self._run_frame("patched")
        # 半径 4 窗口跨多行；右读头材料 1 的码值 2 命中三个身份，
        # 只询问第 1、2 次序（分页/部分次序请求）。
        rect = runner.surface.physical_window(CellId(2, 4), 4)
        reply = runner.heads["right"].identities(
            rect, 1, 2, (1, 2), runner.surface, runner.status, "patched"
        )
        cells = list(reply.cells)
        self.assertEqual(cells, [CellId(1, 4), CellId(2, 5)])
        self.assertEqual(cells, sorted(cells))

    def test_scratch_peak_bounded_and_released_per_position(self):
        runner, _ = self._run_frame("patched")
        self.assertLessEqual(runner.budget.peak, self.budget)
        # 位置处理完后工作区归零，不残留邻域副本。
        self.assertEqual(runner.budget.current, 0)


if __name__ == "__main__":
    unittest.main()
