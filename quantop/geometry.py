"""格网几何、闭合矩形物理窗口、分块与工作区预算。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Grid:
    """行列格网与实际格点间距（物理单位）。

    样本 ID 按全局行列字典序：``id = row * cols + col``。
    窗口用同一个物理半径分别比两轴的绝对间距，因此是轴对齐矩形，
    横纵跨度都以物理单位计算。
    """

    rows: int
    cols: int
    spacing_row: int
    spacing_col: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("格网规模必须为正")
        if self.spacing_row <= 0 or self.spacing_col <= 0:
            raise ValueError("格点间距必须为正")

    @property
    def size(self) -> int:
        return self.rows * self.cols

    def cell_id(self, row: int, col: int) -> int:
        """全局稳定样本 ID（行优先字典序）。"""
        return row * self.cols + col

    def id_to_cell(self, cell_id: int) -> tuple[int, int]:
        return divmod(cell_id, self.cols)

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def window_bounds(self, row: int, col: int, radius_phys: int) -> tuple[int, int, int, int]:
        """返回闭合轴对齐矩形的行列边界 ``(r0, r1, c0, c1)``（含端点）。

        物理跨度判定：``|dr| * spacing_row <= radius`` 且
        ``|dc| * spacing_col <= radius``，再裁剪到格网范围。
        """
        if radius_phys < 0:
            raise ValueError("物理半径不能为负")
        r_half = radius_phys // self.spacing_row
        c_half = radius_phys // self.spacing_col
        r0 = max(0, row - r_half)
        r1 = min(self.rows - 1, row + r_half)
        c0 = max(0, col - c_half)
        c1 = min(self.cols - 1, col + c_half)
        return r0, r1, c0, c1

    def iter_band(self, r0: int, r1: int, c0: int, c1: int):
        """遍历闭合矩形内的全部格点（行优先，即 ID 升序）。"""
        for r in range(r0, r1 + 1):
            base = r * self.cols
            for c in range(c0, c1 + 1):
                yield r, c, base + c


def band_with_halo(
    grid: Grid, band_start: int, interior_rows: int, max_radius_phys: int
) -> tuple[int, int, int]:
    """给出分块的内行区间与物理晕圈行数。

    返回 ``(halo_rows, inner_start, inner_end)``，其中 ``inner_end`` 为
    左闭右开边界；晕圈是最大半径在行轴上可能触及的格点行数。
    """
    halo_rows = max_radius_phys // grid.spacing_row
    inner_end = min(grid.rows, band_start + interior_rows)
    return halo_rows, band_start, inner_end


class ScratchBudget:
    """工作区预算记账器。

    制造端容量文件给出 ``peak_scratch_bytes``。只统计算子新分配的解包、
    转换、归约、版本暂存与交接缓冲；只读输入和已消费输出由调用方持有，
    不计入。
    """

    def __init__(self, limit_bytes: int) -> None:
        if limit_bytes <= 0:
            raise ValueError("工作区预算必须为正")
        self.limit_bytes = int(limit_bytes)
        self.current_bytes = 0
        self.peak_bytes = 0

    def allocate(self, nbytes: int) -> int:
        """登记一笔新分配，超预算立即抛出（不允许静默越过容量）。"""
        nbytes = int(nbytes)
        if nbytes < 0:
            raise ValueError("分配字节数不能为负")
        if self.current_bytes + nbytes > self.limit_bytes:
            raise MemoryError(
                f"工作区超预算：需要 {self.current_bytes + nbytes} 字节，"
                f"上限 {self.limit_bytes} 字节"
            )
        self.current_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        return self.current_bytes

    def release(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        if nbytes > self.current_bytes:
            raise AssertionError("释放量超过已登记工作区")
        self.current_bytes -= nbytes

    def fraction_used(self) -> float:
        return self.peak_bytes / self.limit_bytes
