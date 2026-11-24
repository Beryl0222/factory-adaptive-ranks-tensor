"""定标表单调性、平台区与版本归属测试。"""
import unittest

from quantop.calibration import CalibrationBook, CalibrationTable


class CalibrationTests(unittest.TestCase):
    def test_table_allows_plateau(self):
        table = CalibrationTable("c1", "right", (0, 2, 3, 3, 4, 8))
        self.assertEqual(table.to_physical(2), 3)
        self.assertEqual(table.to_physical(3), 3)  # 平台：两个码同物理值

    def test_non_monotonic_table_rejected(self):
        with self.assertRaises(ValueError):
            CalibrationTable("cx", "left", (0, 5, 3))

    def test_code_out_of_range(self):
        table = CalibrationTable("c1", "left", (0, 1, 3))
        with self.assertRaises(KeyError):
            table.to_physical(9)

    def test_book_scopes_by_version_and_head(self):
        book = CalibrationBook()
        book.add(CalibrationTable("c1", "left", (0, 1)))
        book.add(CalibrationTable("c2", "left", (0, 2)))
        self.assertTrue(book.has("c1", "left"))
        self.assertFalse(book.has("c1", "right"))
        self.assertEqual(book.get("c2", "left").to_physical(1), 2)
        with self.assertRaises(KeyError):
            book.get("c3", "left")


if __name__ == "__main__":
    unittest.main()
