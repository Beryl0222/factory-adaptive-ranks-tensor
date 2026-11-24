"""旧定标版应答治理：严格模式下无新版重发必须保持待判定。"""
import unittest
from pathlib import Path

from caseio import load_case

from quantop.geometry import ScratchBudget
from quantop.heads import FrequencyReply, FrequencyRequest, HeadPort
from quantop.operator import (
    ChunkedAdaptiveQuantileFilter,
    DeliveryState,
    UndecidedReason,
)
from quantop.pipeline import (
    build_calibration_book,
    build_frame_heads,
    build_grid,
    build_materials,
)

DATA = Path(__file__).resolve().parents[1] / "data"


class StaleOnlyGateway(HeadPort):
    """永远回旧版应答的网关：模拟新版重发始终未到达。"""

    def __init__(self, current, stale_version: str):
        self._current = current
        self.name = current.name
        self.bound_grid = current.bound_grid
        self.bound_calibration = current.bound_calibration
        self._stale_version = stale_version

    def _restamp(self, reply):
        return FrequencyReply(
            frame_id=reply.frame_id, head=reply.head,
            calibration_version=self._stale_version, r0=reply.r0, r1=reply.r1,
            c0=reply.c0, c1=reply.c1, material=reply.material,
            entries=reply.entries, total_distinct_codes=reply.total_distinct_codes,
            offset=reply.offset, complete=reply.complete,
        )

    def frequency_page(self, request: FrequencyRequest) -> FrequencyReply:
        return self._restamp(self._current.frequency_page(request))

    def identity_page(self, request):
        reply = self._current.identity_page(request)
        return type(reply)(
            frame_id=reply.frame_id, head=reply.head,
            calibration_version=self._stale_version, code=reply.code,
            sample_ids=reply.sample_ids, total_for_code=reply.total_for_code,
            ordinal=reply.ordinal, complete=reply.complete,
        )

    def read_raw_code(self, frame_id, row, col):
        return self._current.read_raw_code(frame_id, row, col)

    def owns_cell(self, frame_id, row, col):
        return self._current.owns_cell(frame_id, row, col)


class StaleCalibrationTests(unittest.TestCase):
    def test_position_remains_undecided_without_new_version_reply(self):
        records, tensors = load_case(DATA / "capture.json")
        grid = build_grid(records)
        book = build_calibration_book(records)
        materials = build_materials(records)
        new_frame = next(f for f in records["frames"] if f["id"] == "new-range")
        heads = build_frame_heads(new_frame, records, tensors, grid, "c2")
        # 右读头被卡在旧版 c1，新版应答永不到达。
        heads["right"] = StaleOnlyGateway(heads["right"], "c1")

        operator = ChunkedAdaptiveQuantileFilter(
            grid=grid, classes=tensors["classes"],
            status=tensors[new_frame["status"]], materials=materials,
            heads=heads, calibration_book=book, frame_id="new-range",
            calibration_version="c2", scratch=ScratchBudget(1 << 20),
            page_limit=12,
        )
        result = operator.evaluate_position(2, 4)
        self.assertEqual(result.delivery, DeliveryState.UNDECIDED)
        self.assertIn(UndecidedReason.STALE_CALIBRATION,
                      result.undecided_reasons)
        # 旧版基准处理新版目标的结果不得当作承诺分位发布。
        self.assertIsNone(result.quantile_value)
        self.assertIsNone(result.witness_id)


if __name__ == "__main__":
    unittest.main()
