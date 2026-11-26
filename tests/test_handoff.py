"""跨帧变化归因与分量级交付的单元测试。"""
import unittest

from roughquant.core import PositionResult
from roughquant.geometry import CellId, Rect
from roughquant.handoff import (
    CALIBRATION_CHANGE,
    NO_CHANGE,
    RADIUS_CHANGE,
    RANK_CHANGE,
    attribute_change,
)


def _result(cell, radius, value, witness, head, status=1, target=9):
    return PositionResult(
        cell=CellId(*cell), material=1, selected_radius=radius,
        rectangle=Rect(0, 1, 0, 1), valid_count=4, unknown_count=0,
        quantile_rank=2, quantile_value=value, witness=CellId(*witness),
        witness_head=head, target_physical=target if status == 1 else None,
        target_status=status,
        deviation=abs(target - value) if status == 1 else None,
        out_of_tolerance=(abs(target - value) >= 1) if status == 1 else None,
        radius_certain=True, quantile_certain=True, witness_certain=True,
        tolerance_certain=status == 1, blocked=False, reasons=(),
    )


class AttributionTests(unittest.TestCase):
    def test_witness_move_within_plateau_is_rank_change(self):
        before = _result((1, 1), 4, 5, (1, 2), "left")
        after = _result((1, 1), 4, 5, (1, 3), "right")
        change = attribute_change(before, after, "c1", "c1")
        self.assertEqual(change.change_type, RANK_CHANGE)
        # 分位值相同但见证换了读头，正是“更难查”的平台情况。
        self.assertEqual(change.value_before, change.value_after)
        self.assertNotEqual(change.witness_before, change.witness_after)

    def test_value_move_is_rank_change(self):
        before = _result((1, 1), 4, 5, (1, 2), "left")
        after = _result((1, 1), 4, 6, (1, 2), "left")
        self.assertEqual(
            attribute_change(before, after, "c1", "c1").change_type,
            RANK_CHANGE,
        )

    def test_radius_change_takes_priority(self):
        before = _result((1, 1), 4, 5, (1, 2), "left")
        after = _result((1, 1), 2, 6, (1, 9), "left")
        self.assertEqual(
            attribute_change(before, after, "c1", "c1").change_type,
            RADIUS_CHANGE,
        )

    def test_calibration_change_flagged_even_when_value_unchanged(self):
        before = _result((1, 1), 4, 5, (1, 2), "left")
        after = _result((1, 1), 4, 5, (1, 3), "right")
        self.assertEqual(
            attribute_change(before, after, "c1", "c2").change_type,
            CALIBRATION_CHANGE,
        )

    def test_no_change(self):
        before = _result((1, 1), 4, 5, (1, 2), "left")
        self.assertEqual(
            attribute_change(before, before, "c1", "c1").change_type,
            NO_CHANGE,
        )

    def test_unread_target_blocks_only_tolerance_component(self):
        # 分位本身已可发布，但目标点未读，超差结论必须待判定。
        result = _result((1, 1), 4, 5, (1, 2), "left", status=0)
        self.assertTrue(result.quantile_certain)
        self.assertFalse(result.tolerance_certain)
        self.assertIsNone(result.out_of_tolerance)
        # 仍存在可发布分量，无需全幅停等。
        self.assertTrue(result.releasable())

    def test_different_cells_cannot_be_compared(self):
        before = _result((1, 1), 4, 5, (1, 2), "left")
        other = _result((2, 2), 4, 5, (1, 2), "left")
        with self.assertRaises(ValueError):
            attribute_change(before, other, "c1", "c1")


if __name__ == "__main__":
    unittest.main()
