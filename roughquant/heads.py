"""读头侧资料与隐私边界。

读头只保存自己管辖格点的原始量化码，外部只能通过两类受控应答访问：

1. 给定窗口与材料的**按码值频次**（不含未读与确认无读数样本）；
2. 给定码值下按全局行列字典序指定次序的**样本身份**。

完整窗口的码值永远不离开读头。每次应答都绑定请求帧声明的格网与
定标版；版本不符或答复缺失时抛出对应异常，由汇合端按受影响范围
把交付标记为待判定，而不是静默采用旧版基准。
"""
from __future__ import annotations

from dataclasses import dataclass

from .geometry import CellId, Rect, STATUS_READ, SurfaceMap


class StaleCalibrationReply(RuntimeError):
    """读头答复落在旧定标版（或与请求帧声明版本不符）。"""

    def __init__(self, head: str, expected: str, replied: str):
        super().__init__(
            f"读头 {head} 应答定标版 {replied} 与请求版 {expected} 不符"
        )
        self.head = head
        self.expected = expected
        self.replied = replied


class ReplyUnavailable(RuntimeError):
    """读头答复未齐。"""


class SplitRequired(RuntimeError):
    """频次应答条目超出读头预算，调用方必须拆分窗口后重问。"""

    def __init__(self, entries: int, limit: int):
        super().__init__(f"应答条目 {entries} 超过上限 {limit}")
        self.entries = entries
        self.limit = limit


@dataclass(frozen=True)
class FrequencyReply:
    head: str
    rect: Rect
    calibration: str
    counts: dict[int, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@dataclass(frozen=True)
class IdentityReply:
    head: str
    rect: Rect
    code: int
    calibration: str
    ordinals: tuple[int, ...]
    cells: tuple[CellId, ...]


class HeadStore:
    """单个读头在单个帧下的私有码值存储。"""

    def __init__(
        self,
        head: str,
        frame_id: str,
        calibration_id: str,
        grid,
        codes,
        max_response_entries: int,
        origin=(0, 0),
        reply_calibration_override: str | None = None,
        unavailable: bool = False,
    ):
        self.head = head
        self.frame_id = frame_id
        self.calibration_id = calibration_id
        self.grid = tuple(int(v) for v in grid)
        self.max_response_entries = int(max_response_entries)
        self._codes = codes
        # 局部张量左上角在全局格网中的 (行, 列) 偏移。
        self._origin = (int(origin[0]), int(origin[1]))
        self._override = reply_calibration_override
        self._unavailable = unavailable

    def __getattribute__(self, name):
        # 阻止任何外部代码直接取走原始码值窗口；汇合端只能走受控应答。
        if name == "_codes":
            raise PermissionError("原始量化码不能离开读头，请使用频次/身份应答")
        return object.__getattribute__(self, name)

    # -- 内部访问（读头自身计算时使用，绕过隐私守卫） ----------
    def _code_at(self, cell: CellId) -> int:
        codes = object.__getattribute__(self, "_codes")
        r0, c0 = self._origin
        return int(codes[cell.row - r0, cell.col - c0])

    def _check_frame(self, frame_id: str, grid):
        if self._unavailable:
            raise ReplyUnavailable(f"读头 {self.head} 在帧 {frame_id} 的答复未齐")
        if frame_id != self.frame_id:
            raise StaleCalibrationReply(self.head, frame_id, self.calibration_id)
        if tuple(int(v) for v in grid) != self.grid:
            raise StaleCalibrationReply(self.head, frame_id, self.calibration_id)
        if self._override is not None:
            raise StaleCalibrationReply(
                self.head, self.calibration_id, self._override
            )

    # -- 受控应答 ------------------------------------------------
    def frequency(
        self,
        rect: Rect,
        material: int,
        surface: SurfaceMap,
        status,
        frame_id: str,
    ) -> FrequencyReply:
        self._check_frame(frame_id, (surface.rows, surface.cols))
        counts: dict[int, int] = {}
        for cell in rect.cells():
            if (surface.owner_of(cell) != self.head
                    or surface.material_at(cell) != material
                    or int(status[cell.row, cell.col]) != STATUS_READ):
                continue
            code = self._code_at(cell)
            counts[code] = counts.get(code, 0) + 1
        if len(counts) > self.max_response_entries:
            raise SplitRequired(len(counts), self.max_response_entries)
        return FrequencyReply(self.head, rect, self.calibration_id, counts)

    def identities(
        self,
        rect: Rect,
        material: int,
        code: int,
        ordinals,
        surface: SurfaceMap,
        status,
        frame_id: str,
    ) -> IdentityReply:
        """返回该码值下指定 1 基次序的样本身份（全局行列字典序）。"""
        self._check_frame(frame_id, (surface.rows, surface.cols))
        wanted = sorted(int(o) for o in ordinals)
        matched: list[CellId] = []
        for cell in rect.cells():
            if (surface.owner_of(cell) != self.head
                    or surface.material_at(cell) != material
                    or int(status[cell.row, cell.col]) != STATUS_READ):
                continue
            if self._code_at(cell) == code:
                matched.append(cell)  # 矩形扫描本身即行列字典序
        for ordinal in wanted:
            if not 1 <= ordinal <= len(matched):
                raise IndexError(
                    f"次序 {ordinal} 超出码值 {code} 的样本数 {len(matched)}"
                )
        return IdentityReply(
            self.head, rect, code, self.calibration_id,
            tuple(wanted), tuple(matched[o - 1] for o in wanted),
        )
