"""采集地图与物理窗口的几何描述。

所有矩形在内部一律用半开区间 ``[r0, r1) × [c0, c1)`` 表达；对调用方
承诺的邻域仍是闭合矩形，半开表达只是为了方便 Python 切片。
"""
from __future__ import annotations

from dataclasses import dataclass

# 状态约定：1 为实际读数（物理值 0 也是合法读数），0 为可能补交的未读
# 样本，-1 为确认永久无读数。
STATUS_UNREAD = 0
STATUS_BLANK = -1
STATUS_READ = 1


@dataclass(frozen=True, order=True)
class CellId:
    """全局样本身份，按 (行, 列) 字典序排序。"""

    row: int
    col: int

    def as_tuple(self) -> tuple[int, int]:
        return self.row, self.col


@dataclass(frozen=True)
class Rect:
    """半开轴对齐矩形。"""

    r0: int
    r1: int
    c0: int
    c1: int

    def cells(self):
        for r in range(self.r0, self.r1):
            for c in range(self.c0, self.c1):
                yield CellId(r, c)

    def contains(self, cell: CellId) -> bool:
        return self.r0 <= cell.row < self.r1 and self.c0 <= cell.col < self.c1

    def area(self) -> int:
        return (self.r1 - self.r0) * (self.c1 - self.c0)

    def ring_within(self, outer: "Rect") -> tuple["Rect", "Rect", "Rect", "Rect"]:
        """返回 ``outer`` 相对本矩形的四条边带（用于增量累加窗口）。"""
        if not (outer.r0 <= self.r0 and outer.c0 <= self.c0
                and outer.r1 >= self.r1 and outer.c1 >= self.c1):
            raise ValueError("外环必须包含内环")
        top = Rect(outer.r0, self.r0, outer.c0, outer.c1)
        bottom = Rect(self.r1, outer.r1, outer.c0, outer.c1)
        left = Rect(self.r0, self.r1, outer.c0, self.c0)
        right = Rect(self.r0, self.r1, self.c1, outer.c1)
        return top, bottom, left, right


class SurfaceMap:
    """采集地图：格点间距、材料图与读头所有者。"""

    def __init__(self, grid, spacing, classes, halves):
        self.rows, self.cols = int(grid[0]), int(grid[1])
        self.row_pitch, self.col_pitch = int(spacing[0]), int(spacing[1])
        if classes.shape != (self.rows, self.cols):
            raise ValueError("材料图尺寸与格网不一致")
        self.classes = classes
        self.halves = {
            name: tuple(int(v) for v in box) for name, box in halves.items()
        }
        self._validate_halves()

    def _validate_halves(self):
        covered = set()
        for r0, r1, c0, c1 in self.halves.values():
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if (r, c) in covered:
                        raise ValueError("读头管辖区域重叠")
                    covered.add((r, c))
        if covered != {(r, c) for r in range(self.rows) for c in range(self.cols)}:
            raise ValueError("读头管辖区域必须铺满整幅格网")

    def owner_of(self, cell: CellId) -> str:
        for name, (r0, r1, c0, c1) in self.halves.items():
            if r0 <= cell.row < r1 and c0 <= cell.col < c1:
                return name
        raise KeyError(f"格点 {cell} 没有归属读头")

    def material_at(self, cell: CellId) -> int:
        return int(self.classes[cell.row, cell.col])

    def physical_window(self, center: CellId, radius: int) -> Rect:
        """闭合物理半径对应的半开格点矩形。

        横纵跨度分别用各自的实际格点间距比较：只有物理距离不超过半径
        的整格点才入选，因此窗口天然按材料图夹断，不会跨过材料缝。
        """
        r_half = radius // self.row_pitch
        c_half = radius // self.col_pitch
        r0 = max(0, center.row - r_half)
        r1 = min(self.rows, center.row + r_half + 1)
        c0 = max(0, center.col - c_half)
        c1 = min(self.cols, center.col + c_half + 1)
        return Rect(r0, r1, c0, c1)
