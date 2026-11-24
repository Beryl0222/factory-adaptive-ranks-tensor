"""主算子：自适应选窗、分位秩、交付证明与超差判定测试。"""
import unittest

import numpy as np

from quantop.calibration import CalibrationBook, CalibrationTable
from quantop.geometry import Grid, ScratchBudget
from quantop.heads import (
    FrequencyReply,
    FrequencyRequest,
    HeadPort,
    IdentityReply,
    IdentityRequest,
    LocalReadHead,
    STATUS_PENDING,
    STATUS_READ,
    STATUS_VOID,
)
from quantop.operator import (
    ChunkedAdaptiveQuantileFilter,
    DeliveryState,
    MaterialConfig,
    UndecidedReason,
)


def two_head_setup(status=None, left_codes=None, right_codes=None,
                   left_table=(0, 1, 3, 3, 5, 7),
                   right_table=(0, 2, 3, 3, 4, 8)):
    grid = Grid(rows=4, cols=6, spacing_row=2, spacing_col=2)
    classes = np.ones((4, 6), dtype=np.int8)
    if status is None:
        status = np.full((4, 6), STATUS_READ, dtype=np.int8)
    if left_codes is None:
        left_codes = np.zeros((4, 3), dtype=np.int8)
    if right_codes is None:
        right_codes = np.zeros((4, 3), dtype=np.int8)

    def embed(tile, half):
        r0, r1, c0, c1 = half
        canvas = np.zeros((4, 6), dtype=tile.dtype)
        canvas[r0:r1, c0:c1] = tile
        return canvas

    left = LocalReadHead("left", grid, "c1", 12)
    left.bind_classes(classes)
    left.load_frame("f", embed(left_codes, (0, 4, 0, 3)), status,
                    (0, 4, 0, 3))
    right = LocalReadHead("right", grid, "c1", 12)
    right.bind_classes(classes)
    right.load_frame("f", embed(right_codes, (0, 4, 3, 6)), status,
                     (0, 4, 3, 6))
    book = CalibrationBook()
    book.add(CalibrationTable("c1", "left", left_table))
    book.add(CalibrationTable("c1", "right", right_table))
    return grid, classes, status, {"left": left, "right": right}, book


class RecordingHead(HeadPort):
    """记录所有外发请求，断言完整窗口码值不会离开读头。"""

    def __init__(self, inner: HeadPort):
        self._inner = inner
        self.name = inner.name
        self.bound_calibration = inner.bound_calibration
        self.bound_grid = inner.bound_grid
        self.frequency_requests = []
        self.identity_requests = []

    def frequency_page(self, request: FrequencyRequest) -> FrequencyReply:
        self.frequency_requests.append(request)
        return self._inner.frequency_page(request)

    def identity_page(self, request: IdentityRequest) -> IdentityReply:
        self.identity_requests.append(request)
        return self._inner.identity_page(request)

    def read_raw_code(self, frame_id, row, col):
        return self._inner.read_raw_code(frame_id, row, col)

    def owns_cell(self, frame_id, row, col):
        return self._inner.owns_cell(frame_id, row, col)


def make_operator(heads, book, grid, classes, status, materials, budget=None):
    return ChunkedAdaptiveQuantileFilter(
        grid=grid, classes=classes, status=status, materials=materials,
        heads=heads, calibration_book=book, frame_id="f",
        calibration_version="c1",
        scratch=budget or ScratchBudget(1 << 20),
        page_limit=12, interior_rows=2,
    )


class AdaptiveSelectionTests(unittest.TestCase):
    def test_first_radius_meeting_threshold_is_chosen(self):
        grid, classes, status, heads, book = two_head_setup()
        materials = {
            1: MaterialConfig(1, radii=(2, 4, 8), minimum_samples=5,
                              q_num=1, q_den=2, threshold=2),
        }
        op = make_operator(heads, book, grid, classes, status, materials)
        result = op.evaluate_position(1, 2)
        # r=2 -> 行列各 1 格，9 格但门槛 5 已满足，应选第一个半径。
        self.assertEqual(result.radius, 2)
        self.assertEqual(result.valid_count, 9)
        self.assertEqual(result.delivery, DeliveryState.DELIVERED)

    def test_grows_to_larger_radius_when_small_window_below_threshold(self):
        grid, classes, status, heads, book = two_head_setup()
        # 间距 2：r=2 只覆盖行列各 1 格（9 格），r=4 覆盖行列各 2 格（25 格）。
        # 只在 r=4 才触及的位置 (3,4) 放第二个读数，逼算子放大窗口。
        status[:] = STATUS_PENDING
        status[1, 2] = STATUS_READ
        status[3, 4] = STATUS_READ
        materials = {
            1: MaterialConfig(1, radii=(2, 4), minimum_samples=2,
                              q_num=1, q_den=1, threshold=1),
        }
        op = make_operator(heads, book, grid, classes, status, materials)
        result = op.evaluate_position(1, 2)
        self.assertEqual(result.radius, 4)
        self.assertEqual(result.valid_count, 2)
        self.assertIn(UndecidedReason.PENDING_INSIDE_WINDOW,
                      result.undecided_reasons)
        self.assertEqual(result.delivery, DeliveryState.UNDECIDED)

    def test_insufficient_when_max_radius_below_threshold(self):
        grid, classes, status, heads, book = two_head_setup()
        status[:] = STATUS_VOID
        status[1, 2] = STATUS_READ
        materials = {
            1: MaterialConfig(1, radii=(2,), minimum_samples=3,
                              q_num=1, q_den=2, threshold=1),
        }
        op = make_operator(heads, book, grid, classes, status, materials)
        result = op.evaluate_position(1, 2)
        self.assertEqual(result.delivery, DeliveryState.INSUFFICIENT)
        self.assertIsNone(result.quantile_value)

    def test_pending_may_shrink_radius_blocks_delivery(self):
        grid, classes, status, heads, book = two_head_setup()
        # r=2 窗口（行0-2、列1-3）里放一个待补，使补交后小窗口可达门槛。
        status[:] = STATUS_READ
        status[0, 1] = STATUS_PENDING
        materials = {
            1: MaterialConfig(1, radii=(2, 4), minimum_samples=9,
                              q_num=1, q_den=2, threshold=100),
        }
        op = make_operator(heads, book, grid, classes, status, materials)
        result = op.evaluate_position(1, 2)
        self.assertEqual(result.radius, 4)  # 当前只有大窗口达标
        self.assertIn(UndecidedReason.RADIUS_MAY_SHRINK,
                      result.undecided_reasons)
        self.assertEqual(result.delivery, DeliveryState.UNDECIDED)

    def test_zero_values_are_valid_samples(self):
        left_codes = np.zeros((4, 3), dtype=np.int8)
        right_codes = np.zeros((4, 3), dtype=np.int8)
        grid, classes, status, heads, book = two_head_setup(
            left_codes=left_codes, right_codes=right_codes
        )
        materials = {
            1: MaterialConfig(1, radii=(2,), minimum_samples=3,
                              q_num=1, q_den=2, threshold=1),
        }
        op = make_operator(heads, book, grid, classes, status, materials)
        result = op.evaluate_position(1, 2)
        self.assertEqual(result.valid_count, 9)
        self.assertEqual(result.quantile_value, 0)
        self.assertEqual(result.delivery, DeliveryState.DELIVERED)


class QuantileAndWitnessTests(unittest.TestCase):
    def test_quantile_uses_ceil_rational_index_and_stable_witness(self):
        # 手工布置：r=2 窗口 9 格，左右码不同，制造平台与并列。
        left_codes = np.array(
            [[1, 2, 0], [2, 3, 0], [0, 1, 2], [3, 0, 1]], dtype=np.int8
        )
        right_codes = np.array(
            [[1, 2, 4], [3, 2, 0], [1, 4, 2], [0, 3, 1]], dtype=np.int8
        )
        grid, classes, status, heads, book = two_head_setup(
            left_codes=left_codes, right_codes=right_codes
        )
        materials = {
            1: MaterialConfig(1, radii=(2,), minimum_samples=1,
                              q_num=1, q_den=2, threshold=2),
        }
        recording = {name: RecordingHead(h) for name, h in heads.items()}
        op = make_operator(recording, book, grid, classes, status, materials)
        result = op.evaluate_position(1, 2)

        # 独立复算窗口物理值序列。
        physical = []
        for r in range(0, 3):
            for c in range(1, 4):
                owner = "left" if c < 3 else "right"
                code = int((left_codes if owner == "left" else right_codes)[r, c - (0 if owner == "left" else 3)])
                physical.append(book.get("c1", owner).to_physical(code))
        physical.sort()
        n = len(physical)
        k1 = (1 * n + 2 - 1) // 2  # ceil(n/2)
        self.assertEqual(result.quantile_value, physical[k1 - 1])
        self.assertEqual(result.out_of_tolerance,
                         abs(result.target_value - result.quantile_value) >= 2)
        # 见证身份必须落在选中物理值的并列组内。
        wr, wc = divmod(result.witness_id, 6)
        self.assertEqual(classes[wr, wc], 1)
        # 隐私：外发的只有频次页与身份页，没有整窗码值数组。
        for head in recording.values():
            self.assertTrue(head.frequency_requests)
            for req in head.frequency_requests:
                self.assertIsInstance(req, FrequencyRequest)
            for req in head.identity_requests:
                self.assertLessEqual(req.limit, 12)

    def test_grid_mismatch_refuses_construction(self):
        grid, classes, status, heads, book = two_head_setup()
        materials = {1: MaterialConfig(1, (2,), 1, 1, 2, 1)}
        foreign = Grid(rows=4, cols=6, spacing_row=9, spacing_col=9)
        heads["left"].bound_grid = foreign
        with self.assertRaises(ValueError):
            make_operator(heads, book, grid, classes, status, materials)


class ChunkingTests(unittest.TestCase):
    def test_band_budget_is_released_after_each_block(self):
        grid, classes, status, heads, book = two_head_setup()
        materials = {1: MaterialConfig(1, (2, 4), 3, 1, 2, 1)}
        budget = ScratchBudget(1 << 16)
        op = make_operator(heads, book, grid, classes, status, materials, budget)
        results = list(op.evaluate_positions([(0, 2), (2, 2), (3, 2)]))
        self.assertEqual(len(results), 3)
        self.assertEqual(budget.current_bytes, 0)  # 所有 band 工作区已释放
        self.assertGreater(budget.peak_bytes, 0)


if __name__ == "__main__":
    unittest.main()
