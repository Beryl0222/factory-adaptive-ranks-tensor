"""分块自适应分位滤波主算子。

每个被请求位置的处理流程：

1. 按已公布的物理半径序列从小到大开窗（闭合轴对齐矩形，横纵跨度均按
   物理单位计算），只把同材料类属的实际读数作为样本；
2. 选第一个有效样本数达到该材料样本门槛的邻域。未读样本没有秩次、
   确认无读数样本同样不入秩；已知零值是合法读数，照常参与；
3. 各读头只通过码值频次分页和码值样本身份分页提供资料，完整窗口码值
   不离开读头。汇合端按各读头绑定的定标表把码换算成校正后物理值；
4. 秩按校正后物理值升序，物理值并列时按稳定样本 ID 升序；
5. 分位取距首项起第 ``ceil(q*n)`` 项（q 为有理数，1 基），见证身份从
   选中物理值的全部并列样本中归并找回；
6. 对所有可能补交都保持承诺输出时才允许交付，否则标记待判定。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .calibration import CalibrationBook
from .geometry import Grid, ScratchBudget
from .heads import (
    FrequencyReply,
    FrequencyRequest,
    HeadPort,
    STATUS_PENDING,
    STATUS_READ,
)
from .merge import (
    HeadCodeStream,
    IncompleteReplyError,
    StaleCalibrationError,
    kth_identity,
)


class DeliveryState(str, Enum):
    DELIVERED = "delivered"          # 证明后续可达资料不影响结果
    UNDECIDED = "undecided"          # 待判定：未齐资料可能改变邻域或见证
    INSUFFICIENT = "insufficient"    # 最大半径仍达不到样本门槛


class UndecidedReason(str, Enum):
    TARGET_NOT_READ = "target_not_read"
    RADIUS_MAY_SHRINK = "radius_may_shrink"
    PENDING_INSIDE_WINDOW = "pending_inside_window"
    INCOMPLETE_REPLY = "incomplete_reply"
    STALE_CALIBRATION = "stale_calibration"
    NO_ELIGIBLE_RADIUS = "no_eligible_radius"


@dataclass(frozen=True)
class MaterialConfig:
    material: int
    radii: tuple[int, ...]          # 已公布物理半径，升序
    minimum_samples: int
    q_num: int                      # 分位有理数 q = q_num / q_den
    q_den: int
    threshold: int

    def __post_init__(self) -> None:
        if not self.radii or list(self.radii) != sorted(self.radii):
            raise ValueError("半径序列必须非空且升序")
        if self.minimum_samples <= 0:
            raise ValueError("样本门槛必须为正")
        if not (0 < self.q_num <= self.q_den):
            raise ValueError("分位 q 必须满足 0 < q <= 1")
        if self.threshold < 0:
            raise ValueError("超差阈值不能为负")

    def quantile_index_1based(self, n: int) -> int:
        """ceil(q*n)，整数有理运算，避免浮点。"""
        return (self.q_num * n + self.q_den - 1) // self.q_den


@dataclass
class HeadCode:
    """某读头在一个物理值桶内的码与计数。"""

    head: str
    code: int
    count: int


@dataclass
class CodeBucket:
    """同一物理值下、跨读头的码值分桶（频次为各读头之和）。"""

    physical: int
    members: list[HeadCode]
    total: int


@dataclass
class PositionResult:
    row: int
    col: int
    material: int
    frame_id: str
    calibration_version: str
    sample_id: int
    target_status: int
    # 选择结果（待判定时给出当前资料下的最佳估计，可交付时为承诺值）
    radius: int | None
    valid_count: int
    quantile_value: int | None
    witness_id: int | None
    target_value: int | None
    out_of_tolerance: bool | None
    # 交接状态
    delivery: DeliveryState
    undecided_reasons: tuple[UndecidedReason, ...] = ()
    # 可回查的同材候选与待补交潜力
    pending_potential: int = 0
    radius_chain: tuple[tuple[int, int, int], ...] = ()
    """每个半径的 (radius, valid_count, pending_count)，供模型团队回查。"""

    @property
    def publishable(self) -> bool:
        return self.delivery is DeliveryState.DELIVERED


class ChunkedAdaptiveQuantileFilter:
    """分块自适应分位滤波算子。

    算子只依赖 ``HeadPort`` 抽象，不直接接触读头原始码片；真实部署时
    把网关适配成该接口即可，单测里可注入模拟端口。
    """

    def __init__(
        self,
        grid: Grid,
        classes,
        status,
        materials: dict[int, MaterialConfig],
        heads: dict[str, HeadPort],
        calibration_book: CalibrationBook,
        frame_id: str,
        calibration_version: str,
        scratch: ScratchBudget,
        page_limit: int = 12,
        interior_rows: int = 64,
    ) -> None:
        if classes.shape != (grid.rows, grid.cols):
            raise ValueError("材料图形状与格网不符")
        if status.shape != (grid.rows, grid.cols):
            raise ValueError("状态图形状与格网不符")
        for name, head in heads.items():
            # 绑定版本与目标版本不一致的混合输入不在此静默重算：运行时
            # 每次应答仍逐次校验版本，落在旧版的应答转为待判定原因。
            if head.bound_grid != grid:
                raise ValueError(f"读头 {name} 绑定格网与算子格网不一致")
        self.grid = grid
        self.classes = classes
        self.status = status
        self.materials = materials
        self.heads = heads
        self.book = calibration_book
        self.frame_id = frame_id
        self.calibration_version = calibration_version
        self.scratch = scratch
        self.page_limit = page_limit
        self.interior_rows = interior_rows

    # ------------------------------------------------------------------
    # 分块入口
    # ------------------------------------------------------------------
    def evaluate_positions(self, positions):
        """按行分块处理请求位置，逐块申请/释放工作区并产出结果。"""
        ordered = sorted(positions)
        index = 0
        max_radius = max(cfg.radii[-1] for cfg in self.materials.values())
        while index < len(ordered):
            band_start = (ordered[index][0] // self.interior_rows) * self.interior_rows
            band_end = min(self.grid.rows, band_start + self.interior_rows)
            block = []
            while index < len(ordered) and ordered[index][0] < band_end:
                block.append(ordered[index])
                index += 1
            halo = max_radius // self.grid.spacing_row
            band_rows = (band_end - band_start) + 2 * halo
            band_cells = band_rows * self.grid.cols
            # 两个 int32 工作数组的预算登记（有效计数、待补计数按窗口复用）。
            band_bytes = band_cells * 4 * 2
            self.scratch.allocate(band_bytes)
            try:
                for row, col in block:
                    yield self.evaluate_position(row, col)
            finally:
                self.scratch.release(band_bytes)

    # ------------------------------------------------------------------
    # 单位置处理
    # ------------------------------------------------------------------
    def evaluate_position(self, row: int, col: int) -> PositionResult:
        material = int(self.classes[row, col])
        config = self.materials[material]
        cell_id = self.grid.cell_id(row, col)
        target_status = int(self.status[row, col])
        reasons: list[UndecidedReason] = []

        chain: list[tuple[int, int, int]] = []
        selected_radius: int | None = None
        selected_counts: tuple[int, int] | None = None
        bounds_selected: tuple[int, int, int, int] | None = None
        for radius in config.radii:
            r0, r1, c0, c1 = self.grid.window_bounds(row, col, radius)
            valid, pending = self._window_counts(r0, r1, c0, c1, material)
            chain.append((radius, valid, pending))
            if selected_radius is None and valid >= config.minimum_samples:
                selected_radius = radius
                selected_counts = (valid, pending)
                bounds_selected = (r0, r1, c0, c1)
                # 第一个达标的半径即选中，后续半径只用于回查链，不再选。
        if selected_radius is None:
            return PositionResult(
                row=row, col=col, material=material, frame_id=self.frame_id,
                calibration_version=self.calibration_version, sample_id=cell_id,
                target_status=target_status, radius=None, valid_count=chain[-1][1],
                quantile_value=None, witness_id=None, target_value=None,
                out_of_tolerance=None, delivery=DeliveryState.INSUFFICIENT,
                undecided_reasons=(UndecidedReason.NO_ELIGIBLE_RADIUS,),
                pending_potential=chain[-1][2],
                radius_chain=tuple(chain),
            )

        radius = selected_radius
        valid_count, pending_in_window = selected_counts
        r0, r1, c0, c1 = bounds_selected
        window_key = (cell_id, radius, material)

        # 资料未齐 / 旧版应答：按可能受影响范围标记。
        protocol_reasons = self._check_protocol_health(
            r0, r1, c0, c1, material, window_key
        )
        reasons.extend(protocol_reasons)

        # 目标点自身读数（只能向其归属读头索取原始码，再按绑定表换算）。
        target_value = None
        if target_status == STATUS_READ:
            owner = self._owner_head(row, col)
            target_code = self.heads[owner].read_raw_code(self.frame_id, row, col)
            target_value = self.book.get(
                self.calibration_version, owner
            ).to_physical(target_code)
        else:
            reasons.append(UndecidedReason.TARGET_NOT_READ)

        # 分位与见证（当前资料下的最佳估计）。
        quantile_value = witness_id = None
        try:
            quantile_value, witness_id = self._quantile_and_witness(
                config, r0, r1, c0, c1, material, window_key
            )
        except (IncompleteReplyError, StaleCalibrationError) as exc:
            if isinstance(exc, StaleCalibrationError):
                reasons.append(UndecidedReason.STALE_CALIBRATION)
            else:
                reasons.append(UndecidedReason.INCOMPLETE_REPLY)

        # 可交付性证明：只有无任何待判定原因时才发布。
        self._delivery_proof(config, chain, pending_in_window, reasons)

        out_of_tolerance = None
        if quantile_value is not None and target_value is not None:
            out_of_tolerance = abs(target_value - quantile_value) >= config.threshold

        deliverable = not reasons
        return PositionResult(
            row=row, col=col, material=material, frame_id=self.frame_id,
            calibration_version=self.calibration_version, sample_id=cell_id,
            target_status=target_status, radius=radius, valid_count=valid_count,
            quantile_value=quantile_value, witness_id=witness_id,
            target_value=target_value, out_of_tolerance=out_of_tolerance,
            delivery=DeliveryState.DELIVERED if deliverable else DeliveryState.UNDECIDED,
            undecided_reasons=tuple(dict.fromkeys(reasons)),
            pending_potential=pending_in_window,
            radius_chain=tuple(chain),
        )

    # ------------------------------------------------------------------
    # 窗口统计（同材料、同状态），读头协议不可见的部分由静态图本地完成
    # ------------------------------------------------------------------
    def _window_counts(self, r0, r1, c0, c1, material):
        valid = pending = 0
        classes = self.classes
        status = self.status
        for r in range(r0, r1 + 1):
            row_class = classes[r]
            row_status = status[r]
            for c in range(c0, c1 + 1):
                if row_class[c] != material:
                    continue
                state = row_status[c]
                if state == STATUS_READ:
                    valid += 1
                elif state == STATUS_PENDING:
                    pending += 1
                # STATUS_VOID：永久无读数，既不入秩也不算补交潜力。
        return valid, pending

    def _check_protocol_health(self, r0, r1, c0, c1, material, window_key):
        """探一页频次，确认各读头应答版本正确且资料齐备。

        窗口与某读头辖区不相交时该读头返回合法空页，不算未齐；只有版本
        不符或非空窗口声明不完整才影响交付。
        """
        reasons: list[UndecidedReason] = []
        for head in self.heads.values():
            reply = head.frequency_page(
                FrequencyRequest(
                    frame_id=self.frame_id,
                    calibration_version=self.calibration_version,
                    r0=r0, r1=r1, c0=c0, c1=c1, material=material,
                    offset=0, limit=self.page_limit, window_key=window_key,
                )
            )
            if reply.calibration_version != self.calibration_version:
                reasons.append(UndecidedReason.STALE_CALIBRATION)
            elif not reply.complete and reply.total_distinct_codes > 0:
                reasons.append(UndecidedReason.INCOMPLETE_REPLY)
        return reasons

    def _owner_head(self, row: int, col: int) -> str:
        for name, head in self.heads.items():
            if head.owns_cell(self.frame_id, row, col):
                return name
        raise KeyError(f"格点 ({row},{col}) 没有归属读头")

    # ------------------------------------------------------------------
    # 分位值与秩见证
    # ------------------------------------------------------------------
    def _collect_buckets(self, r0, r1, c0, c1, material, window_key):
        """向各读头分页收集窗口内同材码值频次，按物理值聚合成桶。"""
        buckets: list[CodeBucket] = []
        index: dict[int, CodeBucket] = {}
        for head_name, head in self.heads.items():
            table = self.book.get(self.calibration_version, head_name)
            offset = 0
            while True:
                reply: FrequencyReply = head.frequency_page(
                    FrequencyRequest(
                        frame_id=self.frame_id,
                        calibration_version=self.calibration_version,
                        r0=r0, r1=r1, c0=c0, c1=c1, material=material,
                        offset=offset, limit=self.page_limit,
                        window_key=window_key,
                    )
                )
                if reply.calibration_version != self.calibration_version:
                    raise StaleCalibrationError(
                        head_name, reply.calibration_version, self.calibration_version
                    )
                if not reply.complete:
                    raise IncompleteReplyError(head_name, -1)
                for entry in reply.entries:
                    physical = table.to_physical(entry.code)
                    bucket = index.get(physical)
                    if bucket is None:
                        bucket = CodeBucket(physical=physical, members=[], total=0)
                        index[physical] = bucket
                        buckets.append(bucket)
                    bucket.members.append(
                        HeadCode(head=head_name, code=entry.code, count=entry.count)
                    )
                    bucket.total += entry.count
                offset += len(reply.entries)
                if offset >= reply.total_distinct_codes:
                    break
                if not reply.entries:
                    break
        buckets.sort(key=lambda b: b.physical)
        return buckets

    def _quantile_and_witness(
        self, config, r0, r1, c0, c1, material, window_key
    ):
        buckets = self._collect_buckets(r0, r1, c0, c1, material, window_key)
        n = sum(bucket.total for bucket in buckets)
        if n == 0:
            raise IncompleteReplyError("*", -1)
        k1 = config.quantile_index_1based(n)  # 1 基
        # 定位第 k1 项所在的物理值桶。
        cumulative = 0
        chosen = None
        for bucket in buckets:
            cumulative += bucket.total
            if cumulative >= k1:
                chosen = bucket
                break
        if chosen is None:  # pragma: no cover
            raise AssertionError("分位定位失败")
        ordinal_in_tie = k1 - (cumulative - chosen.total) - 1  # 桶内 0 基次序

        streams: list[HeadCodeStream] = []
        for member in chosen.members:
            head = self.heads[member.head]
            streams.append(
                HeadCodeStream(
                    head_name=member.head,
                    frame_id=self.frame_id,
                    calibration_version=self.calibration_version,
                    r0=r0, r1=r1, c0=c0, c1=c1, material=material,
                    code=member.code, page_limit=self.page_limit,
                    identity_call=head.identity_page,
                    window_key=window_key,
                )
            )
        witness_id = self._witness_in_tie(streams, ordinal_in_tie)
        return chosen.physical, witness_id

    def _witness_in_tie(self, streams, ordinal_in_tie):
        return kth_identity(streams, ordinal_in_tie)

    # ------------------------------------------------------------------
    # 可交付性证明
    # ------------------------------------------------------------------
    def _delivery_proof(self, config, chain, pending_in_window, reasons):
        # 1) 更小半径：逐个检查选中半径之前的邻域。即使这些窗口内所有待补
        #    同材格点全部补交仍达不到门槛，邻域选择才证明不会缩回；否则
        #    一份被遮挡同材带的补交就可能让位置从大窗口缩回小窗口。
        for radius, valid, pending in chain:
            if valid >= config.minimum_samples:
                break  # 到达选中半径，更小半径已全部检查完
            if valid + pending >= config.minimum_samples:
                reasons.append(UndecidedReason.RADIUS_MAY_SHRINK)
        # 2) 选中窗口内若存在待补同材格点，任何合法补交码都可能改变秩
        #    （样本数 n 变化还会改变 ceil(q*n) 本身），不能承诺秩见证。
        if pending_in_window > 0:
            reasons.append(UndecidedReason.PENDING_INSIDE_WINDOW)
