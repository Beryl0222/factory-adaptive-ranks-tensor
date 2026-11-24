"""读头侧资料协议。

汇合端只能向读头询问两类资料：

1. 指定窗口、指定材料类属下，按量化码计数的频次（分页返回，避免把
   完整窗口的码值搬出读头）；
2. 某个码值下、按全局样本 ID 排序的指定次序样本身份（分页返回）。

每个读头只持有自己的原始量化码，只应答与自己绑定的定标版和格网。
应答自带版本声明，落在旧版或资料未齐时，由汇合端按可能受影响范围
判定交付。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

from .geometry import Grid


# 状态码：1 实际读数；0 仍可能补交；-1 确认永久无读数。
STATUS_READ = 1
STATUS_PENDING = 0
STATUS_VOID = -1


@dataclass(frozen=True)
class FrequencyRequest:
    """窗口码值频次查询。

    窗口为闭合轴对齐矩形（行列均含端点），只统计该读头辖区内、同材料、
    状态为实际读数的样本。分页以码值升序进行，``offset`` 是跳过的码值数。
    ``window_key`` 是“位置 + 实际查阅窗口”的稳定标识，供网关把同一查询
    的先后应答（如换版重放）关联起来。
    """

    frame_id: str
    calibration_version: str
    r0: int
    r1: int
    c0: int
    c1: int
    material: int
    offset: int = 0
    limit: int = 12
    window_key: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class CodeCount:
    code: int
    count: int


@dataclass(frozen=True)
class FrequencyReply:
    frame_id: str
    head: str
    calibration_version: str
    r0: int
    r1: int
    c0: int
    c1: int
    material: int
    entries: tuple[CodeCount, ...]
    total_distinct_codes: int
    offset: int
    complete: bool
    """资料是否齐备（帧状态允许应答且窗口在本读头格网内）。"""


@dataclass(frozen=True)
class IdentityRequest:
    """某码值下样本身份查询。

    ``ordinal`` 是该码值样本序列（按全局 ID 升序）的起始下标，
    ``limit`` 是本页最多返回的身份数。``window_key`` 与频次查询共用。
    """

    frame_id: str
    calibration_version: str
    r0: int
    r1: int
    c0: int
    c1: int
    material: int
    code: int
    ordinal: int = 0
    limit: int = 12
    window_key: tuple[int, int, int] | None = None


@dataclass(frozen=True)
class IdentityReply:
    frame_id: str
    head: str
    calibration_version: str
    code: int
    sample_ids: tuple[int, ...]
    total_for_code: int
    ordinal: int
    complete: bool


class HeadPort(abc.ABC):
    """汇合端视角的读头接口（可被模拟实现或真实网关实现）。"""

    name: str
    bound_calibration: str
    bound_grid: Grid

    @abc.abstractmethod
    def frequency_page(self, request: FrequencyRequest) -> FrequencyReply:
        """返回一页码值频次；窗口不在辖区时返回空页。"""

    @abc.abstractmethod
    def identity_page(self, request: IdentityRequest) -> IdentityReply:
        """返回一页某码值下的样本身份。"""

    @abc.abstractmethod
    def read_raw_code(self, frame_id: str, row: int, col: int) -> int:
        """读取本读头辖区内单个格点的原始量化码（不离开读头边界语义）。"""

    @abc.abstractmethod
    def owns_cell(self, frame_id: str, row: int, col: int) -> bool:
        """判定格点是否归本读头在该帧的辖区。"""


@dataclass
class _HeadTiles:
    """单个读头在单个帧上持有的本地原始资料。"""

    codes: object  # 全网格形状的整数数组，非辖区值无意义
    status: object  # 全网格状态数组
    row_lo: int
    row_hi: int  # 含端点
    col_lo: int
    col_hi: int  # 含端点


class LocalReadHead(HeadPort):
    """本地模拟读头。

    读头只读取自己辖区的原始量化码；频次与身份都在辖区、材料、状态
    三重过滤后给出，未读（0）与确认无读数（-1）样本不进计数，也没有
    秩次。零值是合法读数，照常参与。
    """

    def __init__(
        self,
        name: str,
        grid: Grid,
        bound_calibration: str,
        max_response_entries: int,
    ) -> None:
        self.name = name
        self.bound_grid = grid
        self.bound_calibration = bound_calibration
        self.max_response_entries = max_response_entries
        self._tiles: dict[str, _HeadTiles] = {}
        self._classes = None

    def load_frame(
        self,
        frame_id: str,
        codes,
        status,
        half: tuple[int, int, int, int],
    ) -> None:
        r0, r1, c0, c1 = half
        if codes.shape != (self.bound_grid.rows, self.bound_grid.cols):
            raise ValueError(f"读头 {self.name} 的码片形状与绑定格网不符")
        if status.shape != codes.shape:
            raise ValueError("状态图与码片形状不一致")
        self._tiles[frame_id] = _HeadTiles(
            codes=codes,
            status=status,
            row_lo=r0,
            row_hi=r1 - 1,
            col_lo=c0,
            col_hi=c1 - 1,
        )

    def _clipped_window(
        self, frame_id: str, r0: int, r1: int, c0: int, c1: int
    ) -> tuple[_HeadTiles, int, int, int, int] | None:
        tile = self._tiles.get(frame_id)
        if tile is None:
            return None
        cr0 = max(r0, tile.row_lo)
        cr1 = min(r1, tile.row_hi)
        cc0 = max(c0, tile.col_lo)
        cc1 = min(c1, tile.col_hi)
        if cr0 > cr1 or cc0 > cc1:
            return tile, 0, -1, 0, -1
        return tile, cr0, cr1, cc0, cc1

    def _iter_valid_cells(self, tile: _HeadTiles, classes, cr0, cr1, cc0, cc1, material):
        cols = self.bound_grid.cols
        for r in range(cr0, cr1 + 1):
            row_codes = tile.codes[r]
            row_status = tile.status[r]
            row_class = classes[r]
            base = r * cols
            for c in range(cc0, cc1 + 1):
                if row_class[c] != material:
                    continue
                if row_status[c] != STATUS_READ:
                    continue
                yield base + c, int(row_codes[c])

    def frequency_page(self, request: FrequencyRequest) -> FrequencyReply:
        clipped = self._clipped_window(
            request.frame_id, request.r0, request.r1, request.c0, request.c1
        )
        if clipped is None:
            raise KeyError(f"读头 {self.name} 没有帧 {request.frame_id} 的资料")
        tile, cr0, cr1, cc0, cc1 = clipped
        # 窗口与本读头辖区不相交：合法空应答。
        if cr0 > cr1 or cc0 > cc1:
            return FrequencyReply(
                frame_id=request.frame_id,
                head=self.name,
                calibration_version=self.bound_calibration,
                r0=request.r0,
                r1=request.r1,
                c0=request.c0,
                c1=request.c1,
                material=request.material,
                entries=(),
                total_distinct_codes=0,
                offset=request.offset,
                complete=True,
            )
        counts: dict[int, int] = {}
        for _cell_id, code in self._iter_valid_cells(
            tile, self._require_classes(), cr0, cr1, cc0, cc1, request.material
        ):
            counts[code] = counts.get(code, 0) + 1
        ordered = sorted(counts.items())
        total = len(ordered)
        page = ordered[request.offset : request.offset + request.limit]
        entries = tuple(CodeCount(code=code, count=count) for code, count in page)
        # 帧是否可能补交由状态图（0）表达；读头对当前持有的资料总能给出
        # 齐备应答。未读样本不进计数，待判定由汇合端的可交付性证明负责。
        return FrequencyReply(
            frame_id=request.frame_id,
            head=self.name,
            calibration_version=self.bound_calibration,
            r0=request.r0,
            r1=request.r1,
            c0=request.c0,
            c1=request.c1,
            material=request.material,
            entries=entries,
            total_distinct_codes=total,
            offset=request.offset,
            complete=request.limit <= self.max_response_entries,
        )

    def identity_page(self, request: IdentityRequest) -> IdentityReply:
        clipped = self._clipped_window(
            request.frame_id, request.r0, request.r1, request.c0, request.c1
        )
        if clipped is None:
            raise KeyError(f"读头 {self.name} 没有帧 {request.frame_id} 的资料")
        tile, cr0, cr1, cc0, cc1 = clipped
        ids: list[int] = []
        if not (cr0 > cr1 or cc0 > cc1):
            for cell_id, code in self._iter_valid_cells(
                tile, self._require_classes(), cr0, cr1, cc0, cc1, request.material
            ):
                if code == request.code:
                    ids.append(cell_id)
        ids.sort()
        total = len(ids)
        page = tuple(ids[request.ordinal : request.ordinal + request.limit])
        return IdentityReply(
            frame_id=request.frame_id,
            head=self.name,
            calibration_version=self.bound_calibration,
            code=request.code,
            sample_ids=page,
            total_for_code=total,
            ordinal=request.ordinal,
            complete=request.limit <= self.max_response_entries,
        )

    # 材料图是静态工艺图，读头可据此过滤同材候选。
    def bind_classes(self, classes) -> None:
        if classes.shape != (self.bound_grid.rows, self.bound_grid.cols):
            raise ValueError("材料图形状与绑定格网不符")
        self._classes = classes

    def read_raw_code(self, frame_id: str, row: int, col: int) -> int:
        tile = self._tiles.get(frame_id)
        if tile is None:
            raise KeyError(f"读头 {self.name} 没有帧 {frame_id} 的资料")
        if not self.owns_cell(frame_id, row, col):
            raise PermissionError(f"格点 ({row},{col}) 不在读头 {self.name} 辖区")
        return int(tile.codes[row, col])

    def owns_cell(self, frame_id: str, row: int, col: int) -> bool:
        tile = self._tiles.get(frame_id)
        if tile is None:
            raise KeyError(f"读头 {self.name} 没有帧 {frame_id} 的资料")
        return (
            tile.row_lo <= row <= tile.row_hi
            and tile.col_lo <= col <= tile.col_hi
        )

    def _require_classes(self):
        if self._classes is None:
            raise RuntimeError(f"读头 {self.name} 尚未绑定材料图")
        return self._classes
