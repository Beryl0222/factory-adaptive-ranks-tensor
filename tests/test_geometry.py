"""几何窗口、分块与工作区预算测试。"""
import unittest

from quantop.geometry import Grid, ScratchBudget, band_with_halo


class GridTests(unittest.TestCase):
    def setUp(self):
        # 行间距 3、列间距 2，与采集记录一致。
        self.grid = Grid(rows=4, cols=7, spacing_row=3, spacing_col=2)

    def test_cell_id_is_row_major_dictionary_order(self):
        self.assertEqual(self.grid.cell_id(0, 0), 0)
        self.assertEqual(self.grid.cell_id(1, 3), 10)
        self.assertEqual(self.grid.cell_id(3, 6), 27)
        self.assertEqual(self.grid.id_to_cell(18), (2, 4))

    def test_window_is_closed_rectangle_in_physical_units(self):
        # 半径 2：行轴 2//3=0 格，列轴 2//2=1 格。
        self.assertEqual(self.grid.window_bounds(2, 4, 2), (2, 2, 3, 5))
        # 半径 4：行轴 4//3=1 格，列轴 4//2=2 格。
        self.assertEqual(self.grid.window_bounds(2, 4, 4), (1, 3, 2, 6))

    def test_window_clipped_to_grid(self):
        self.assertEqual(self.grid.window_bounds(0, 0, 7), (0, 2, 0, 3))

    def test_band_halo_uses_row_spacing(self):
        halo, start, end = band_with_halo(self.grid, 0, interior_rows=2, max_radius_phys=7)
        self.assertEqual(halo, 7 // 3)
        self.assertEqual((start, end), (0, 2))

    def test_band_iteration_is_id_ascending(self):
        ids = [cell_id for _, _, cell_id in self.grid.iter_band(1, 2, 3, 4)]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(ids, [10, 11, 17, 18])


class ScratchBudgetTests(unittest.TestCase):
    def test_allocation_and_release_accounting(self):
        budget = ScratchBudget(100)
        budget.allocate(40)
        budget.allocate(30)
        self.assertEqual(budget.peak_bytes, 70)
        budget.release(30)
        self.assertEqual(budget.current_bytes, 40)
        self.assertEqual(budget.peak_bytes, 70)

    def test_over_budget_raises(self):
        budget = ScratchBudget(10)
        with self.assertRaises(MemoryError):
            budget.allocate(11)

    def test_cannot_release_more_than_held(self):
        budget = ScratchBudget(10)
        budget.allocate(4)
        with self.assertRaises(AssertionError):
            budget.release(5)


if __name__ == "__main__":
    unittest.main()
