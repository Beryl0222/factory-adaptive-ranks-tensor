"""汇合端：分块自适应分位滤波算子。

算子只通过读头的受控应答取得资料，因此合成大图时不需要为每个位置
物化完整邻域码值副本：

* 每个读头按闭合矩形 + 材料返回码值频次；条目超预算时由汇合端按行
  带拆分重问，频次在汇合端合并；
* 秩见证只在确定目标秩后，按码值分页询问样本身份，按物理值与稳定
  ID 做跨读头归并（定标平台可横跨两个读头）；
* 半径序列嵌套，大窗口通过四条边带增量累加，不重复搬运内圈；
* 工作区计入频次表、未读清单与分页身份缓冲，受 ``peak_scratch_bytes``
  约束，且与邻域总像素数无关。
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .core import (
    MaterialSpec,
    PositionResult,
    UnknownSample,
    certify_quantile_bracket,
    quantile_rank,
    witness_invariant,
)
from .geometry import (
    CellId,
    Rect,
    STATUS_BLANK,
    STATUS_READ,
    STATUS_UNREAD,
    SurfaceMap,
)
from .heads import (
    FrequencyReply,
    HeadStore,
    IdentityReply,
    ReplyUnavailable,
    SplitRequired,
    StaleCalibrationReply,
)

_INT_BYTES = 8  # 工作区按 64 位整数记账，保守上界


class ScratchBudget:
    """模拟制造端工作区预算：追踪峰值占用，支持位置级释放。"""

    def __init__(self, limit_bytes: int):
        self.limit = int(limit_bytes)
        self.current = 0
        self.peak = 0

    def charge(self, n_items: int) -> None:
        self.current += max(0, n_items) * _INT_BYTES
        if self.current > self.limit:
            raise MemoryError(
                f"工作区占用 {self.current} 字节超过预算 {self.limit}"
            )
        self.peak = max(self.peak, self.current)

    def release(self, n_items: int) -> None:
        self.current = max(0, self.current - max(0, n_items) * _INT_BYTES)


@dataclass
class WindowEvidence:
    """单个窗口经读头应答合并后的证据。"""

    rect: Rect
    # 读头 -> 码值 -> 计数（码值不出读头，这里只保留频次聚合）
    histograms: dict[str, dict[int, int]]
    valid_samples: tuple[Sample, ...]
    unknowns: tuple[UnknownSample, ...]


def merge_physical_histogram(
    histograms: dict[str, dict[int, int]],
    tables: dict[str, list[int]],
) -> dict[int, int]:
    """把各读头的码值频次按其定标表换算为物理值频次。"""
    merged: dict[int, int] = {}
    for head, counts in histograms.items():
        table = tables[head]
        for code, count in counts.items():
            physical = int(table[int(code)])
            merged[physical] = merged.get(physical, 0) + count
    return merged


def quantile_from_histogram(histogram: dict[int, int], q: Fraction) -> tuple[int, int]:
    """物理值频次 → (分位物理值, 1 基秩)。"""
    total = sum(histogram.values())
    if total == 0:
        raise ValueError("空样本集无法定义分位")
    rank = quantile_rank(total, q)
    cumulative = 0
    for physical in sorted(histogram):
        cumulative += histogram[physical]
        if cumulative >= rank:
            return physical, rank
    raise AssertionError("分位秩越界")


class AdaptiveQuantileFilter:
    """可复用的分块自适应分位滤波算子。"""

    def __init__(
        self,
        surface: SurfaceMap,
        material_specs: dict[int, MaterialSpec],
        head_stores: dict[str, HeadStore],
        calibration_tables: dict[str, dict[str, list[int]]],
        frame_id: str,
        calibration_id: str,
        status,
        peak_scratch_bytes: int,
        identity_page_size: int = 32,
    ):
        self.surface = surface
        self.specs = material_specs
        self.heads = dict(head_stores)
        if set(self.heads) != set(surface.halves):
            raise ValueError("读头集合必须与采集地图的所有者完全一致")
        self.tables = calibration_tables[calibration_id]
        if set(self.tables) != set(self.heads):
            raise ValueError("定标表必须覆盖且仅覆盖全部读头")
        for head, table in self.tables.items():
            if any(a > b for a, b in zip(table, table[1:])):
                raise ValueError(f"读头 {head} 的定标表必须单调不减")
        self.frame_id = frame_id
        self.calibration_id = calibration_id
        self.status = status
        self.budget = ScratchBudget(peak_scratch_bytes)
        self.page_size = identity_page_size
        # 单位置内的身份应答记忆化缓存，随位置结束清空；缓存条目计入
        # 工作区预算，避免同一 (头, 码, 次序) 被归并重复计费。
        self._identity_cache: dict[tuple, CellId] = {}
        self._table_ranges = {
            head: (min(table), max(table)) for head, table in self.tables.items()
        }

    # -- 读头应答（含超预算自动分块） ---------------------------
    def _frequency_chunked(self, head: str, rect: Rect, material: int) -> dict[int, int]:
        store = self.heads[head]
        try:
            reply = store.frequency(
                rect, material, self.surface, self.status, self.frame_id
            )
            return dict(reply.counts)
        except SplitRequired:
            merged: dict[int, int] = {}
            # 递归二分矩形，直到每个应答都在读头条目预算内；拆分只发生
            # 在汇合端，读头始终只返回频次，不返回码值窗口。
            for sub in self._split_rect(rect):
                reply = store.frequency(
                    sub, material, self.surface, self.status, self.frame_id
                )
                for code, count in reply.counts.items():
                    merged[code] = merged.get(code, 0) + count
            return merged

    @staticmethod
    def _split_rect(rect: Rect):
        if rect.r1 - rect.r0 > 1:
            mid = (rect.r0 + rect.r1) // 2
            yield from AdaptiveQuantileFilter._split_rect(
                Rect(rect.r0, mid, rect.c0, rect.c1))
            yield from AdaptiveQuantileFilter._split_rect(
                Rect(mid, rect.r1, rect.c0, rect.c1))
        elif rect.c1 - rect.c0 > 1:
            mid = (rect.c0 + rect.c1) // 2
            yield Rect(rect.r0, rect.r1, rect.c0, mid)
            yield Rect(rect.r0, rect.r1, mid, rect.c1)
        else:
            yield rect

    def _ask_rect(self, rect: Rect, material: int) -> dict[str, dict[int, int]]:
        answer: dict[str, dict[int, int]] = {}
        for head, store in self.heads.items():
            r0, r1, c0, c1 = self.surface.halves[head]
            clipped = Rect(
                max(rect.r0, r0), min(rect.r1, r1),
                max(rect.c0, c0), min(rect.c1, c1),
            )
            if clipped.r0 >= clipped.r1 or clipped.c0 >= clipped.c1:
                continue
            counts = self._frequency_chunked(head, clipped, material)
            if counts:
                answer[head] = counts
        return answer

    @staticmethod
    def _merge_into(
        target: dict[str, dict[int, int]],
        extra: dict[str, dict[int, int]],
    ) -> None:
        for head, counts in extra.items():
            bucket = target.setdefault(head, {})
            for code, count in counts.items():
                bucket[code] = bucket.get(code, 0) + count

    def _collect_unknowns(self, rect: Rect, material: int) -> list[UnknownSample]:
        unknowns: list[UnknownSample] = []
        for cand in rect.cells():
            if (self.surface.material_at(cand) == material
                    and int(self.status[cand.row, cand.col]) == STATUS_UNREAD):
                head = self.surface.owner_of(cand)
                low, high = self._table_ranges[head]
                unknowns.append(UnknownSample(cand, head, low, high))
        return unknowns

    def _build_evidence(
        self,
        center: CellId,
        spec: MaterialSpec,
    ) -> tuple[list, dict[str, dict[int, int]], int]:
        """逐半径增量收集证据（边带只问一次），达标即停。

        返回 (各阶段摘要, 累计频次, 工作区已记账条目数)。每个阶段记录
        半径、矩形、有效数与未读数；累计频次在位置结束后统一释放。
        """
        material = spec.material
        stages = []
        accumulated: dict[str, dict[int, int]] = {}
        previous: Rect | None = None
        charged = 0
        for radius in spec.radii:
            rect = self.surface.physical_window(center, radius)
            if previous is None:
                extra = self._ask_rect(rect, material)
            else:
                extra = {}
                for band in previous.ring_within(rect):
                    if band.area() == 0:
                        continue
                    self._merge_into(extra, self._ask_rect(band, material))
            added = sum(len(c) for c in extra.values())
            self.budget.charge(added)
            charged += added
            self._merge_into(accumulated, extra)
            unknowns = self._collect_unknowns(rect, material)
            self.budget.charge(len(unknowns))
            charged += len(unknowns)
            n_valid = sum(sum(c.values()) for c in accumulated.values())
            stages.append((radius, rect, n_valid, len(unknowns)))
            if n_valid >= spec.minimum:
                break
            previous = rect
        return stages, accumulated, charged

    # -- 秩见证：分页身份应答 + 跨读头稳定归并 ------------------
    def _identity_page(
        self, head: str, rect: Rect, material: int, code: int, ordinals
    ) -> IdentityReply:
        return self.heads[head].identities(
            rect, material, code, ordinals,
            self.surface, self.status, self.frame_id,
        )

    def resolve_witness(
        self,
        rect: Rect,
        material: int,
        target_value: int,
        target_rank: int,
        histograms: dict[str, dict[int, int]],
    ) -> tuple[CellId, str]:
        """从所有并列于目标物理值的样本中找回正确见证。

        先统计物理值严格更小的样本数；目标秩落在并列于目标物理值的
        样本区间内。各 (读头, 码) 组的格点互不相交，组内按全局行列
        字典序排列。把这些有序序列做 k 路归并、取第 ``ordinal_in_tie``
        个身份：用“小于候选身份的跨组计数”二分，全程只调用读头允许的
        “指定次序取身份”应答，不请求任何码值窗口。
        """
        smaller = 0
        tied: list[tuple[HeadStore, int, int]] = []  # (读头, 码值, 该头该码计数)
        for head, counts in histograms.items():
            table = self.tables[head]
            for code, count in counts.items():
                physical = int(table[int(code)])
                if physical < target_value:
                    smaller += count
                elif physical == target_value:
                    tied.append((self.heads[head], int(code), count))

        ordinal_in_tie = target_rank - smaller  # 1 基，按全局 ID 排序
        total_tied = sum(item[2] for item in tied)
        if not 1 <= ordinal_in_tie <= total_tied:
            raise AssertionError("目标秩落在平台区间之外")

        def identity_at(group_index: int, ordinal: int) -> CellId:
            store, code, _count = tied[group_index]
            key = (store.head, rect.r0, rect.r1, rect.c0, rect.c1,
                   material, code, ordinal)
            cached = self._identity_cache.get(key)
            if cached is not None:
                return cached
            reply = self._identity_page(
                store.head, rect, material, code, (ordinal,)
            )
            cell = reply.cells[0]
            self._identity_cache[key] = cell
            self.budget.charge(1)
            return cell

        def count_strict_less(candidate: CellId, exclude: int) -> int:
            """其他组中字典序严格小于 candidate 的元素数（各组内二分）。"""
            total = 0
            for idx, (store, code, count) in enumerate(tied):
                if idx == exclude:
                    continue
                lo, hi = 0, count
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if identity_at(idx, mid) < candidate:
                        lo = mid
                    else:
                        hi = mid - 1
                total += lo
            return total

        # 标准 k 路归并取第 ordinal_in_tie 个元素。各组内部有序，组间在
        # 全局字典序上可能交错，因此维护每组游标，每轮取游标处最小身份；
        # 该身份的全局位置 = 它在本组的次序 + 其他组严格更小计数，可一次
        # 跳跃而不必逐元素往返。全程只调用“指定次序取身份”受控应答。
        cursors = [1] * len(tied)
        while True:
            best_idx = -1
            best_cell: CellId | None = None
            for idx, (store, code, count) in enumerate(tied):
                if cursors[idx] > count:
                    continue
                cand = identity_at(idx, cursors[idx])
                if best_cell is None or cand < best_cell:
                    best_cell, best_idx = cand, idx
            if best_cell is None:
                raise AssertionError("k 路归并提前耗尽")
            ordinal_here = cursors[best_idx]
            global_pos = ordinal_here + count_strict_less(
                best_cell, best_idx
            )
            if global_pos == ordinal_in_tie:
                return best_cell, tied[best_idx][0].head
            # global_pos 严格小于目标：该身份确定排在目标之前，游标前进。
            # 归并游标单调推进，global_pos 不可能大于当前目标位置，因为
            # 更大的元素属于尚未成为最小值的其他组，不会在本轮被选中。
            cursors[best_idx] += 1

    # -- 单点主流程 ----------------------------------------------
    def evaluate(self, cell: CellId) -> PositionResult:
        material = self.surface.material_at(cell)
        spec = self.specs[material]
        try:
            return self._evaluate(cell, spec)
        except StaleCalibrationReply as exc:
            return self._affected_result(
                cell, spec, f"定标版不符（{exc}），受影响范围按待判定交付"
            )
        except ReplyUnavailable as exc:
            return self._affected_result(cell, spec, f"读头答复未齐：{exc}")

    def _affected_result(self, cell: CellId, spec: MaterialSpec, reason: str) -> PositionResult:
        material = spec.material
        max_rect = self.surface.physical_window(cell, spec.radii[-1])
        unknown_count = sum(
            1
            for cand in max_rect.cells()
            if self.surface.material_at(cand) == material
            and int(self.status[cand.row, cand.col]) == STATUS_UNREAD
        )
        target_status = int(self.status[cell.row, cell.col])
        return PositionResult(
            cell=cell, material=material, selected_radius=None,
            rectangle=max_rect, valid_count=0, unknown_count=unknown_count,
            quantile_rank=None, quantile_value=None, witness=None,
            witness_head=None, target_physical=None,
            target_status=target_status, deviation=None, out_of_tolerance=None,
            radius_certain=False, quantile_certain=False,
            witness_certain=False, tolerance_certain=False, blocked=False,
            reasons=(reason,),
        )

    def _evaluate(self, cell: CellId, spec: MaterialSpec) -> PositionResult:
        material = spec.material
        start_charge = self.budget.current
        self._identity_cache.clear()
        try:
            return self.__evaluate(cell, spec)
        finally:
            # 位置完成即释放频次表、未读清单与身份缓存，峰值与邻域像素无关。
            self._identity_cache.clear()
            self.budget.current = start_charge

    def __evaluate(self, cell: CellId, spec: MaterialSpec) -> PositionResult:
        material = spec.material
        stages, accumulated, _charged = self._build_evidence(cell, spec)
        target_status = int(self.status[cell.row, cell.col])
        target_head = self.surface.owner_of(cell)
        max_rect = self.surface.physical_window(cell, spec.radii[-1])

        selected_stage = next(
            (stage for stage in stages if stage[2] >= spec.minimum), None
        )
        if selected_stage is None:
            unknown_count = stages[-1][3]
            has_potential = any(
                self.surface.material_at(cand) == material
                and int(self.status[cand.row, cand.col]) != STATUS_BLANK
                for cand in max_rect.cells()
            )
            return PositionResult(
                cell=cell, material=material, selected_radius=None,
                rectangle=max_rect, valid_count=0, unknown_count=unknown_count,
                quantile_rank=None, quantile_value=None, witness=None,
                witness_head=None, target_physical=None,
                target_status=target_status, deviation=None,
                out_of_tolerance=None, radius_certain=False,
                quantile_certain=False, witness_certain=False,
                tolerance_certain=False, blocked=not has_potential,
                reasons=(
                    "最大物理半径内有效样本仍不足门槛",
                    "等待同材质未读样本补交" if has_potential
                    else "同材质候选均为确认无读数，结论永久阻塞",
                ),
            )

        radius, rect, n_valid, n_unknown = selected_stage
        # 达标即停后，accumulated 恰好等于选中窗口的频次；未读清单只需
        # 该窗口内的部分，单独重建以便认证。
        unknowns = tuple(self._collect_unknowns(rect, material))
        physical_hist = merge_physical_histogram(accumulated, self.tables)
        value, rank = quantile_from_histogram(physical_hist, spec.q)
        witness, witness_head = self.resolve_witness(
            rect, material, value, rank, accumulated
        )

        # 半径认证：所有更小半径在全部补交后的潜力仍不足门槛。
        radius_certain = all(
            stage_valid + stage_unknown < spec.minimum
            for _r, _rect, stage_valid, stage_unknown in stages[:-1]
        )
        bracket_stable, low, high = certify_quantile_bracket(
            physical_hist, unknowns, spec.q
        )
        value_certain = radius_certain and bracket_stable
        witness_certain = radius_certain and witness_invariant(
            n_valid, rank, witness, value, unknowns, spec.q
        )

        reasons = []
        if not radius_certain:
            reasons.append("未读样本补交后窗口可能缩回更小半径")
        elif not bracket_stable:
            reasons.append(f"分位值可能落在 [{low}, {high}] 区间")
        if radius_certain and not witness_certain:
            reasons.append("平台并列样本可能改变秩见证身份")

        target_physical = None
        if target_status == STATUS_READ:
            target_code = self._read_target_code(cell, target_head)
            target_physical = int(self.tables[target_head][target_code])

        deviation = None
        out_of_tol = None
        tolerance_certain = False
        if target_physical is not None:
            deviation = abs(target_physical - value)
            out_of_tol = deviation >= spec.threshold
            if radius_certain:
                if bracket_stable:
                    tolerance_certain = True
                else:
                    lo_dev = abs(target_physical - low)
                    hi_dev = abs(target_physical - high)
                    if lo_dev >= spec.threshold and hi_dev >= spec.threshold:
                        out_of_tol, tolerance_certain = True, True
                        reasons.append("分位值区间两端均超差，超差结论可发布")
                    elif lo_dev < spec.threshold and hi_dev < spec.threshold:
                        out_of_tol, tolerance_certain = False, True
                        reasons.append("分位值区间两端均合格，合格结论可发布")
        else:
            reasons.append("目标点尚无读数，超差结论待判定")

        return PositionResult(
            cell=cell, material=material, selected_radius=radius,
            rectangle=rect, valid_count=n_valid,
            unknown_count=n_unknown, quantile_rank=rank,
            quantile_value=value, witness=witness, witness_head=witness_head,
            target_physical=target_physical, target_status=target_status,
            deviation=deviation, out_of_tolerance=out_of_tol,
            radius_certain=radius_certain, quantile_certain=value_certain,
            witness_certain=witness_certain,
            tolerance_certain=tolerance_certain, blocked=False,
            reasons=tuple(reasons),
        )

    def _read_target_code(self, cell: CellId, head: str) -> int:
        # 目标点自身的码值由其拥有者读头以单点频次应答提供，
        # 汇合端不持有任何读头的原始码窗口。
        rect = Rect(cell.row, cell.row + 1, cell.col, cell.col + 1)
        reply: FrequencyReply = self.heads[head].frequency(
            rect, self.surface.material_at(cell),
            self.surface, self.status, self.frame_id,
        )
        codes = list(reply.counts)
        if len(codes) != 1:
            raise ReplyUnavailable(f"目标点 {cell} 缺少单点读数应答")
        return int(codes[0])

    def evaluate_chunk(self, cells) -> list[PositionResult]:
        """分块入口：逐位置评估，单位置工作区在结束时释放。"""
        return [self.evaluate(CellId(int(r), int(c))) for r, c in cells]
