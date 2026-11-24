"""端到端管线：合成采集、旧版回放拒绝、容量预算与交接产物测试。"""
import json
import unittest
from pathlib import Path

from quantop.pipeline import run_capture

DATA = Path(__file__).resolve().parents[1] / "data"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = run_capture(
            DATA / "capture.json", DATA / "capacity.json"
        )

    def test_every_query_has_a_result_per_frame(self):
        queries = {tuple(q) for q in self.payload["queries"]}
        for frame_id, results in self.payload["results"].items():
            cells = {(r["row"], r["col"]) for r in results}
            self.assertEqual(cells, queries)

    def test_pending_frame_marks_affected_position_undecided(self):
        first = {
            (r["row"], r["col"]): r
            for r in self.payload["results"]["first"]
        }
        # (1,3) 的 r=2 窗口含待补，且待补可能让大窗口缩回小窗口。
        q1 = first[(1, 3)]
        self.assertEqual(q1["delivery"], "undecided")
        self.assertIn("radius_may_shrink", q1["undecided_reasons"])
        self.assertIn("pending_inside_window", q1["undecided_reasons"])
        # 资料齐备且无待补影响的位置在首帧即可发布，无需全幅停等。
        self.assertEqual(first[(2, 4)]["delivery"], "delivered")

    def test_patched_frame_shrinks_radius_and_delivers(self):
        patched = {
            (r["row"], r["col"]): r
            for r in self.payload["results"]["patched"]
        }
        q1 = patched[(1, 3)]
        self.assertEqual(q1["selected_radius"], 2)
        self.assertEqual(q1["delivery"], "delivered")
        self.assertEqual(q1["quantile_value"], 3)
        self.assertEqual(q1["witness_id"], 11)
        self.assertTrue(q1["out_of_tolerance"])

    def test_plateau_witness_switches_while_quantile_value_holds(self):
        new_range = {
            (r["row"], r["col"]): r
            for r in self.payload["results"]["new-range"]
        }
        patched = {
            (r["row"], r["col"]): r
            for r in self.payload["results"]["patched"]
        }
        self.assertEqual(
            patched[(2, 4)]["quantile_value"],
            new_range[(2, 4)]["quantile_value"],
        )
        self.assertNotEqual(
            patched[(2, 4)]["witness_id"],
            new_range[(2, 4)]["witness_id"],
        )
        transition = next(
            t for t in self.payload["transitions"]
            if (t["row"], t["col"]) == (2, 4)
            and t["to_frame"] == "new-range"
        )
        self.assertEqual(transition["cause"], "calibration")
        self.assertTrue(transition["witness_changed"])
        self.assertFalse(transition["quantile_value_changed"])

    def test_stale_c1_reply_rejected_then_c2_rerun_delivers(self):
        events = self.payload["replay_events"]["new-range"]
        stale = [e for e in events if not e["accepted"]]
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["head"], "right")
        self.assertEqual(stale[0]["reply_calibration"], "c1")
        # 重算后该位置按 c2 交付，没有遗留旧版原因。
        q2 = next(
            r for r in self.payload["results"]["new-range"]
            if (r["row"], r["col"]) == (2, 4)
        )
        self.assertEqual(q2["delivery"], "delivered")
        self.assertEqual(q2["calibration"], "c2")
        self.assertNotIn("stale_calibration", q2["undecided_reasons"])

    def test_transition_causes_cover_radius_and_calibration_classes(self):
        causes = {t["cause"] for t in self.payload["transitions"]}
        # 半径缩回（补交同材带）与定标换版在本数据上都有；rank 类由
        # test_handoff 在同半径同版的受控构造下覆盖。
        self.assertIn("radius", causes)
        self.assertIn("calibration", causes)
        radius_transitions = [
            t for t in self.payload["transitions"] if t["cause"] == "radius"
        ]
        self.assertTrue(all(t["became_deliverable"] for t in radius_transitions))

    def test_scratch_peak_within_capacity(self):
        for frame_id, meta in self.payload["frames"].items():
            self.assertLessEqual(
                meta["scratch_peak_bytes"], meta["scratch_limit_bytes"]
            )

    def test_payload_is_json_serializable(self):
        json.dumps(self.payload, ensure_ascii=False)

    def test_output_file_written(self):
        out = Path("/tmp/handoff_unittest.json")
        run_capture(DATA / "capture.json", DATA / "capacity.json", out)
        self.assertTrue(out.exists())
        parsed = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("transitions", parsed)


if __name__ == "__main__":
    unittest.main()
