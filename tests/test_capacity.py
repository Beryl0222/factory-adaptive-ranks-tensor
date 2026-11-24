"""制造端规模下的工作区预算与分块行为测试。

按 ``capacity.json`` 的 4096x3072 格网生成合成测量片，确认算子峰值工作区
不超过公布预算，并且不依赖每个位置的完整邻域副本（身份只经分页协议
按需拉取）。
"""
import json
import unittest
from pathlib import Path

import numpy as np

from quantop.calibration import CalibrationBook, CalibrationTable
from quantop.geometry import Grid, ScratchBudget
from quantop.heads import LocalReadHead, STATUS_READ
from quantop.operator import (
    ChunkedAdaptiveQuantileFilter,
    DeliveryState,
    MaterialConfig,
)

DATA = Path(__file__).resolve().parents[1] / "data"


class CapacityScaleTests(unittest.TestCase):
    def test_peak_scratch_within_published_budget(self):
        capacity = json.loads((DATA / "capacity.json").read_text(encoding="utf-8"))
        rows, cols = capacity["grid"]
        rng = np.random.default_rng(capacity["seed"])
        grid = Grid(rows=rows, cols=cols, spacing_row=3, spacing_col=2)
        classes = rng.integers(1, capacity["materials"] + 1, size=(rows, cols),
                               dtype=np.int8)
        code_max = 1 << capacity["code_bits"]
        left_codes = rng.integers(0, code_max, size=(rows, cols // 2),
                                  dtype=np.int16)
        right_codes = rng.integers(0, code_max, size=(rows, cols - cols // 2),
                                   dtype=np.int16)
        status = np.full((rows, cols), STATUS_READ, dtype=np.int8)

        def embed(tile, c0):
            canvas = np.zeros((rows, cols), dtype=tile.dtype)
            canvas[:, c0:c0 + tile.shape[1]] = tile
            return canvas

        mid = cols // 2
        left = LocalReadHead("left", grid, "c1", 12)
        left.bind_classes(classes)
        left.load_frame("f", embed(left_codes, 0), status,
                        (0, rows, 0, mid))
        right = LocalReadHead("right", grid, "c1", 12)
        right.bind_classes(classes)
        right.load_frame("f", embed(right_codes, mid), status,
                         (0, rows, mid, cols))

        book = CalibrationBook()
        table_values = tuple(
            sorted(int(v) for v in rng.integers(0, 1000, size=code_max))
        )
        book.add(CalibrationTable("c1", "left", table_values))
        book.add(CalibrationTable("c1", "right", table_values))

        materials = {
            material: MaterialConfig(
                material=material,
                radii=tuple(capacity["physical_radius"]),
                minimum_samples=8,
                q_num=1,
                q_den=2,
                threshold=50,
            )
            for material in range(1, capacity["materials"] + 1)
        }
        budget = ScratchBudget(capacity["peak_scratch_bytes"])
        operator = ChunkedAdaptiveQuantileFilter(
            grid=grid, classes=classes, status=status, materials=materials,
            heads={"left": left, "right": right}, calibration_book=book,
            frame_id="f", calibration_version="c1", scratch=budget,
            page_limit=12, interior_rows=64,
        )
        # 抽样位置，避免全量 1200 万点在测试里跑太久；跨多个 band。
        sample_positions = [
            (r, c)
            for r in range(128, rows, max(1, rows // 8))
            for c in range(256, cols, cols // 4)
        ][:32]
        results = list(operator.evaluate_positions(sample_positions))
        self.assertEqual(len(results), len(sample_positions))
        self.assertTrue(
            all(r.delivery is DeliveryState.DELIVERED for r in results)
        )
        # 工作区在所有 band 处理完后必须全部归还。
        self.assertEqual(budget.current_bytes, 0)
        self.assertLessEqual(budget.peak_bytes, capacity["peak_scratch_bytes"])
        # 峰值远小于“每位置邻域副本”式实现会需要的量级。
        full_grid_bytes = rows * cols * 4
        self.assertLess(budget.peak_bytes, full_grid_bytes // 16)


if __name__ == "__main__":
    unittest.main()
