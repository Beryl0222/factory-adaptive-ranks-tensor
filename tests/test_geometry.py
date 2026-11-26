"""几何窗口与材料缝夹断的单元测试。"""
import unittest

import numpy as np

from roughquant.geometry import CellId, Rect, SurfaceMap


class GeometryTests(unittest.TestCase):
    def setUp(self):
        classes = np.array([
            [1, 1, 2],
            [1, 2, 2],
            [1, 2, 2],
        ], dtype=np.int8)
        halves = {"a": (0, 3, 0, 2), "b": (0, 3, 2, 3)}
        self.surface = SurfaceMap(
            grid=(3, 3), spacing=(3, 2), classes=classes, halves=halves
        )

    def test_physical_window_closed_and_clipped(self):
        # 半径 2：行方向 2//3=0 只含中心行；列方向 2//2=1 各扩一列。
        rect = self.surface.physical_window(CellId(1, 1), 2)
        self.assertEqual((rect.r0, rect.r1, rect.c0, rect.c1), (1, 2, 0, 3))
        # 中心格点包含在闭合窗口内。
        self.assertTrue(rect.contains(CellId(1, 1)))

    def test_window_does_not_cross_grid_boundary(self):
        rect = self.surface.physical_window(CellId(0, 0), 6)
        self.assertEqual((rect.r0, rect.r1, rect.c0, rect.c1), (0, 3, 0, 3))

    def test_owner_partition(self):
        self.assertEqual(self.surface.owner_of(CellId(0, 0)), "a")
        self.assertEqual(self.surface.owner_of(CellId(2, 2)), "b")

    def test_ring_decomposition_partitions_outer(self):
        inner = Rect(1, 2, 1, 2)
        outer = Rect(0, 3, 0, 3)
        bands = inner.ring_within(outer)
        ring_cells = {cell for band in bands for cell in band.cells()}
        outer_cells = set(outer.cells())
        self.assertEqual(ring_cells, outer_cells - set(inner.cells()))

    def test_halves_must_tile_grid(self):
        with self.assertRaises(ValueError):
            SurfaceMap(
                grid=(2, 2), spacing=(1, 1),
                classes=np.ones((2, 2), dtype=np.int8),
                halves={"a": (0, 2, 0, 1)},  # 缺一列
            )


if __name__ == "__main__":
    unittest.main()
