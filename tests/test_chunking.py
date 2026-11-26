"""读头预算拆分、大规模分块与工作区预算测试。"""
import json
from pathlib import Path
import unittest

import numpy as np

from roughquant.aggregator import AdaptiveQuantileFilter
from roughquant.core import MaterialSpec
from roughquant.geometry import CellId, SurfaceMap
from roughquant.heads import HeadStore, SplitRequired

DATA = Path(__file__).resolve().parents[1] / "data"


def _large_surface(rows=96, cols=96, materials=12):
    rng = np.random.default_rng(442)
    classes = rng.integers(1, materials + 1, size=(rows, cols)).astype(np.int8)
    halves = {
        "left": (0, rows, 0, cols // 2),
        "right": (0, rows, cols // 2, cols),
    }
    surface = SurfaceMap(
        grid=(rows, cols), spacing=(3, 2),
        classes=classes, halves=halves,
    )
    return surface, classes, halves, rng


class SplitTests(unittest.TestCase):
    def test_head_rejects_oversized_frequency_response(self):
        surface, classes, halves, rng = _large_surface(12, 12)
        codes = rng.integers(0, 9, size=(12, 6)).astype(np.int8)
        store = HeadStore(
            "left", "f", "c1", (12, 12), codes,
            max_response_entries=2, origin=(0, 0),
        )
        status = np.ones((12, 12), dtype=np.int8)
        rect = surface.physical_window(CellId(6, 2), 40)
        with self.assertRaises(SplitRequired):
            store.frequency(rect, material=1, surface=surface,
                            status=status, frame_id="f")

    def test_chunked_query_equals_full_window_counts(self):
        surface, classes, halves, rng = _large_surface(24, 24)
        left_codes = rng.integers(0, 9, size=(24, 12)).astype(np.int8)
        right_codes = rng.integers(0, 9, size=(24, 12)).astype(np.int8)
        status = np.ones((24, 24), dtype=np.int8)
        tables = {
            "c1": {
                "left": list(range(9)),
                "right": list(range(9)),
            }
        }
        specs = {
            m: MaterialSpec(m, (5, 10, 20, 40), minimum=4,
                            q=__import__("fractions").Fraction(1, 2),
                            threshold=2)
            for m in range(1, 13)
        }
        # 故意把应答预算压到 3，迫使每个窗口递归拆分。
        stores = {
            "left": HeadStore("left", "f", "c1", (24, 24), left_codes,
                              max_response_entries=3),
            "right": HeadStore("right", "f", "c1", (24, 24), right_codes,
                               max_response_entries=3, origin=(0, 12)),
        }
        runner = AdaptiveQuantileFilter(
            surface=surface, material_specs=specs, head_stores=stores,
            calibration_tables=tables, frame_id="f", calibration_id="c1",
            status=status, peak_scratch_bytes=4096,
        )
        result = runner.evaluate(CellId(10, 10))
        self.assertIsNotNone(result.selected_radius)
        self.assertTrue(result.radius_certain)
        self.assertTrue(result.witness_certain)

        # 与无预算限制的参考实现对比分位值与见证。
        ref_stores = {
            "left": HeadStore("left", "f", "c1", (24, 24), left_codes,
                              max_response_entries=10_000),
            "right": HeadStore("right", "f", "c1", (24, 24), right_codes,
                               max_response_entries=10_000, origin=(0, 12)),
        }
        ref = AdaptiveQuantileFilter(
            surface=surface, material_specs=specs, head_stores=ref_stores,
            calibration_tables=tables, frame_id="f", calibration_id="c1",
            status=status, peak_scratch_bytes=4096,
        )
        expected = ref.evaluate(CellId(10, 10))
        self.assertEqual(result.quantile_value, expected.quantile_value)
        self.assertEqual(result.witness, expected.witness)
        self.assertEqual(result.selected_radius, expected.selected_radius)


class WitnessMergeTests(unittest.TestCase):
    def test_cross_head_plateau_target_in_large_group_tail(self):
        # 大网格上两个读头各持同物理值 5 的码值，右读头样本远多于左读头，
        # 目标秩落在右读头大组后段；应答预算与分页都压到很小，验证 k 路
        # 归并不被分页/组交错误导，且窗口被限制在目标邻域而非整幅。
        rows, cols = 40, 40
        center = CellId(20, 20)
        classes = np.ones((rows, cols), dtype=np.int8)
        halves = {
            "left": (0, rows, 0, cols // 2),
            "right": (0, rows, cols // 2, cols),
        }
        surface = SurfaceMap((rows, cols), (3, 2), classes, halves)
        left_codes = np.full((rows, cols // 2), 4, dtype=np.int8)
        right_codes = np.full((rows, cols // 2), 1, dtype=np.int8)
        status = np.ones((rows, cols), dtype=np.int8)
        tables = {"c1": {"left": [0, 0, 0, 0, 5], "right": [0, 5]}}
        from fractions import Fraction
        spec = MaterialSpec(1, (20,), minimum=4, q=Fraction(1, 2), threshold=2)
        stores = {
            "left": HeadStore("left", "f", "c1", (rows, cols), left_codes,
                              max_response_entries=4),
            "right": HeadStore("right", "f", "c1", (rows, cols), right_codes,
                               max_response_entries=4, origin=(0, cols // 2)),
        }
        runner = AdaptiveQuantileFilter(
            surface=surface, material_specs={1: spec}, head_stores=stores,
            calibration_tables=tables, frame_id="f", calibration_id="c1",
            status=status, peak_scratch_bytes=1_048_576, identity_page_size=3,
        )
        result = runner.evaluate(center)
        window = result.rectangle
        self.assertLess(window.area(), rows * cols)  # 没有扩成整幅
        ordered = sorted(window.cells())
        rank = -((-(Fraction(1, 2) * len(ordered))).__floor__())
        self.assertEqual(result.quantile_value, 5)
        self.assertEqual(result.quantile_rank, rank)
        # 见证即窗口内全局字典序第 rank 个格点（所有样本同物理值）。
        self.assertEqual(result.witness, ordered[rank - 1])
        # 目标秩落在右读头大组（左读头列止于 19）。
        self.assertEqual(result.witness_head, "right")
        self.assertGreaterEqual(result.witness.col, cols // 2)

        # 与无预算、大分页的参考实现一致。
        ref_stores = {
            "left": HeadStore("left", "f", "c1", (rows, cols), left_codes,
                              max_response_entries=10_000),
            "right": HeadStore("right", "f", "c1", (rows, cols), right_codes,
                               max_response_entries=10_000, origin=(0, cols // 2)),
        }
        ref = AdaptiveQuantileFilter(
            surface=surface, material_specs={1: spec}, head_stores=ref_stores,
            calibration_tables=tables, frame_id="f", calibration_id="c1",
            status=status, peak_scratch_bytes=1_048_576, identity_page_size=1024,
        )
        self.assertEqual(ref.evaluate(center).witness, result.witness)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.capacity = json.loads(
            (DATA / "capacity.json").read_text(encoding="utf-8")
        )

    def test_peak_does_not_grow_with_neighborhood_size(self):
        capacity = self.capacity
        surface, classes, halves, rng = _large_surface(
            capacity["grid"][0] // 8, capacity["grid"][1] // 8
        )
        rows, cols = surface.rows, surface.cols
        left_codes = rng.integers(0, 2 ** capacity["code_bits"],
                                  size=(rows, cols // 2)).astype(np.int16)
        right_codes = rng.integers(0, 2 ** capacity["code_bits"],
                                   size=(rows, cols // 2)).astype(np.int16)
        status = np.ones((rows, cols), dtype=np.int8)
        radii = tuple(capacity["physical_radius"])
        from fractions import Fraction
        specs = {
            m: MaterialSpec(m, radii, minimum=8, q=Fraction(1, 2),
                            threshold=4)
            for m in range(1, 13)
        }
        table = list(range(2 ** capacity["code_bits"]))
        tables = {"c1": {"left": table, "right": table}}
        stores = {
            "left": HeadStore("left", "f", "c1", (rows, cols), left_codes,
                              max_response_entries=12),
            "right": HeadStore("right", "f", "c1", (rows, cols), right_codes,
                               max_response_entries=12, origin=(0, cols // 2)),
        }
        runner = AdaptiveQuantileFilter(
            surface=surface, material_specs=specs, head_stores=stores,
            calibration_tables=tables, frame_id="f", calibration_id="c1",
            status=status,
            peak_scratch_bytes=capacity["peak_scratch_bytes"],
        )
        cells = [(r, c) for r in range(20, rows, 40)
                 for c in range(20, cols, 40)]
        results = runner.evaluate_chunk(cells)
        self.assertTrue(all(r.selected_radius is not None for r in results))
        # 工作区峰值远小于“每位置完整邻域副本”的朴素预算，且逐位置释放。
        self.assertLessEqual(runner.budget.peak, capacity["peak_scratch_bytes"])
        self.assertEqual(runner.budget.current, 0)
        naive = max(r.rectangle.area() for r in results) * 8
        self.assertLess(runner.budget.peak, naive)

    def test_scratch_grows_sublinearly_to_neighborhood(self):
        # 同参数只放大物理半径：邻域像素上百倍增长时，工作区峰值只随
        # 不同码值条目与身份缓存增长，不存在按邻域大小物化的副本。
        def build(radius, n=80):
            rng = np.random.default_rng(7)
            classes = np.ones((n, n), dtype=np.int8)
            halves = {"l": (0, n, 0, n // 2), "r": (0, n, n // 2, n)}
            surface = SurfaceMap((n, n), (3, 2), classes, halves)
            lc = rng.integers(0, 8, (n, n // 2)).astype(np.int8)
            rc = rng.integers(0, 8, (n, n // 2)).astype(np.int8)
            status = np.ones((n, n), dtype=np.int8)
            tables = {"c": {"l": list(range(8)), "r": list(range(8))}}
            from fractions import Fraction
            spec = MaterialSpec(
                1, (radius,), minimum=4, q=Fraction(1, 2), threshold=2
            )
            stores = {
                "l": HeadStore("l", "f", "c", (n, n), lc, 12),
                "r": HeadStore("r", "f", "c", (n, n), rc, 12,
                               origin=(0, n // 2)),
            }
            return AdaptiveQuantileFilter(
                surface=surface, material_specs={1: spec},
                head_stores=stores, calibration_tables=tables,
                frame_id="f", calibration_id="c", status=status,
                peak_scratch_bytes=self.capacity["peak_scratch_bytes"],
            )

        observations = []
        for radius in (10, 40, 100):
            runner = build(radius)
            result = runner.evaluate(CellId(40, 40))
            observations.append(
                (result.rectangle.area(), runner.budget.peak)
            )
        small_area, small_peak = observations[0]
        big_area, big_peak = observations[-1]
        self.assertGreater(big_area // small_area, 50)
        # 亚线性：工作区峰值增幅严格小于邻域面积增幅，因为不存在按
        # 邻域大小物化的码值副本，只保留码值频次与有限的身份缓存。
        self.assertLess(
            big_peak / max(1, small_peak),
            big_area / small_area,
        )
        # 且峰值绝对值始终是 KB 级，远小于朴素邻域副本。
        self.assertLess(big_peak, big_area * 8 // 4)


if __name__ == "__main__":
    unittest.main()
