"""分位滤波的纯算法核心与可证明的待判定规则。

本模块不关心样本来自本地张量还是读头应答，只接受已校正的物理样本，
因此本地路径与汇合端路径共用同一套半径选择、秩、分位与判定逻辑。

待判定（pending）规则按分量给出，只有证明后续可达资料不影响该分量时
才标记为确定（certain）：

* 半径：当前选中的窗口若在所有未读样本补交后仍满足样本门槛，则窗口
  不会再缩回（半径序列单调，窗口只可能缩小），半径分量可发布。
* 分位值：所有未读样本各取其读头定标表的最小/最大物理值这两种极端
  补交，若两端的分位物理值相等，则任何合法补交的分位值都相同。
* 见证：用秩区间引理精确证明目标秩在任意合法补交下都落在同一样本上。
* 超差：需要目标读数固定且分位值确定；目标本身未读时结论必为待判定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from .geometry import CellId, Rect, STATUS_READ, STATUS_UNREAD, SurfaceMap


@dataclass(frozen=True)
class MaterialSpec:
    material: int
    radii: tuple[int, ...]
    minimum: int
    q: Fraction
    threshold: int

    @classmethod
    def from_record(cls, material: int, record: dict) -> "MaterialSpec":
        num, den = record["q"]
        q = Fraction(int(num), int(den))
        if not (0 < q <= 1):
            raise ValueError(f"材料 {material} 的分位 q 必须满足 0<q<=1")
        radii = tuple(int(r) for r in record["radii"])
        if any(a >= b for a, b in zip(radii, radii[1:])):
            raise ValueError(f"材料 {material} 的物理半径序列必须严格递增")
        if int(record["minimum"]) < 1:
            raise ValueError(f"材料 {material} 的样本门槛必须为正")
        return cls(
            material=int(material),
            radii=radii,
            minimum=int(record["minimum"]),
            q=q,
            threshold=int(record["threshold"]),
        )


@dataclass
class Sample:
    """一条已校正样本：物理微单位值与稳定身份。"""

    cell: CellId
    physical: int

    @property
    def key(self) -> tuple[int, CellId]:
        # 秩定义：校正后物理值升序，同值用稳定样本 ID（行列字典序）。
        return self.physical, self.cell


@dataclass
class UnknownSample:
    """可能补交的样本；其物理值落在所属读头定标表的值域内。"""

    cell: CellId
    head: str
    low: int
    high: int


@dataclass
class RadiusOutcome:
    radius: int
    rect: Rect
    valid: tuple[Sample, ...]
    unknowns: tuple[UnknownSample, ...]

    @property
    def n_valid(self) -> int:
        return len(self.valid)

    @property
    def n_unknown(self) -> int:
        return len(self.unknowns)


@dataclass
class PositionResult:
    """单个请求位置的完整交接结果。"""

    cell: CellId
    material: int
    selected_radius: int | None
    rectangle: Rect | None
    valid_count: int
    unknown_count: int
    quantile_rank: int | None
    quantile_value: int | None
    witness: CellId | None
    witness_head: str | None
    target_physical: int | None
    target_status: int
    deviation: int | None
    out_of_tolerance: bool | None
    radius_certain: bool
    quantile_certain: bool
    witness_certain: bool
    tolerance_certain: bool
    blocked: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def releasable(self) -> bool:
        """该位置是否存在可发布分量（无需全幅停等）。"""
        return self.radius_certain or self.quantile_certain or self.tolerance_certain

    def as_handoff(self) -> dict:
        return {
            "cell": list(self.cell.as_tuple()),
            "material": self.material,
            "selected_radius": self.selected_radius,
            "rectangle": None
            if self.rectangle is None
            else [self.rectangle.r0, self.rectangle.r1,
                  self.rectangle.c0, self.rectangle.c1],
            "valid_count": self.valid_count,
            "unknown_count": self.unknown_count,
            "quantile_rank": self.quantile_rank,
            "quantile_value": self.quantile_value,
            "witness": None if self.witness is None else list(self.witness.as_tuple()),
            "witness_head": self.witness_head,
            "target_physical": self.target_physical,
            "target_status": self.target_status,
            "deviation": self.deviation,
            "out_of_tolerance": self.out_of_tolerance,
            "certainty": {
                "radius": self.radius_certain,
                "quantile": self.quantile_certain,
                "witness": self.witness_certain,
                "tolerance": self.tolerance_certain,
            },
            "blocked": self.blocked,
            "reasons": list(self.reasons),
        }


def quantile_rank(n: int, q: Fraction) -> int:
    """距首项起第 ceil(q*n) 项（1 基）。"""
    return -((-q * n).__floor__())  # ceil(Fraction*int)，避免浮点


def order_samples(samples) -> list[Sample]:
    return sorted(samples, key=lambda s: s.key)


def pick_quantile(samples, q: Fraction) -> tuple[int, CellId, int]:
    """返回 (分位物理值, 见证身份, 1 基秩)。样本必须非空。"""
    ordered = order_samples(samples)
    rank = quantile_rank(len(ordered), q)
    chosen = ordered[rank - 1]
    return chosen.physical, chosen.cell, rank


def certify_radius(stages, selected_index: int, minimum: int) -> bool:
    """证明选中半径在任意合法补交后都不会缩回。

    半径窗口随半径序列嵌套。选中第 ``selected_index`` 个半径（首个有效
    数达门槛者）不会缩回，当且仅当每个更小半径在“全部未读样本补交”
    后的潜力（有效数 + 未读数）仍严格小于门槛。
    """
    for outcome in stages[:selected_index]:
        if outcome.n_valid + outcome.n_unknown >= minimum:
            return False
    return True


def _extreme_quantile(valid, unknowns, q: Fraction, *, take_high: bool) -> int:
    """所有未读样本同时取定标表极值时的分位值。

    分位值关于任一插入值单调，因此“全部取最小/最大”两种极端补交
    夹住了任意合法补交（含部分补交）的分位值。
    """
    values = [s.physical for s in valid]
    values.extend(u.high if take_high else u.low for u in unknowns)
    values.sort()
    rank = quantile_rank(len(values), q)
    return values[rank - 1]


def _kth_merged(counted, extras, k: int) -> int:
    """两个有序多重集合合并后的第 k 小（1 基），不物化合并数组。

    ``counted`` 为按值排序的 (值, 计数) 序列，``extras`` 为按值排序
    的标量序列。
    """
    i = j = 0
    consumed = 0
    while True:
        if i < len(counted):
            counted_value = counted[i][0]
            extra_value = extras[j] if j < len(extras) else None
            if extra_value is None or counted_value <= extra_value:
                run = counted[i][1]
                if consumed + run >= k:
                    return counted_value
                consumed += run
                i += 1
                continue
        if j >= len(extras):
            raise AssertionError("k 超出合并序列长度")
        value = extras[j]
        consumed += 1
        if consumed == k:
            return value
        j += 1


def certify_quantile_bracket(histogram, unknowns, q: Fraction) -> tuple[bool, int, int]:
    """频次直方图版本的极值夹住证明，供汇合端直接使用。"""
    counted = sorted((int(value), int(count)) for value, count in histogram.items())
    n = sum(count for _, count in counted) + len(unknowns)
    rank = quantile_rank(n, q)
    low = _kth_merged(counted, sorted(u.low for u in unknowns), rank)
    high = _kth_merged(counted, sorted(u.high for u in unknowns), rank)
    return low == high, low, high


def certify_quantile_value(outcome: RadiusOutcome, q: Fraction) -> tuple[bool, int, int]:
    """样本列表版本的极值夹住证明（本地完整张量路径使用）。"""
    histogram: dict[int, int] = {}
    for sample in outcome.valid:
        histogram[sample.physical] = histogram.get(sample.physical, 0) + 1
    return certify_quantile_bracket(histogram, outcome.unknowns, q)


def certify_witness(outcome: RadiusOutcome, q: Fraction) -> bool:
    """精确见证不变性证明（秩区间引理）。

    对当前第 k 小的样本 W，记任意补交中排在 W 之前的未读样本数为 x、
    补交总数为 a，则 W 的秩为 k+x，目标秩为 ceil(q(n+a))。W 在所有
    合法补交下都保持见证，当且仅当对每个 a，x 可取的整段区间
    ``[x_min(a), x_max(a)]`` 都唯一满足该等式。
    """
    ordered = order_samples(outcome.valid)
    n = len(ordered)
    if n == 0:
        return False
    k = quantile_rank(n, q)
    witness = ordered[k - 1]
    return witness_invariant(n, k, witness.cell, witness.physical,
                             outcome.unknowns, q)


def witness_invariant(n: int, k: int, witness: CellId, witness_value: int,
                      unknowns, q: Fraction) -> bool:
    """秩区间引理的独立实现，输入只需见证身份与未读样本值域。"""
    before_low = sum(u.low < witness_value or
                     (u.low == witness_value and u.cell < witness)
                     for u in unknowns)
    before_high = sum(u.high < witness_value or
                      (u.high == witness_value and u.cell < witness)
                      for u in unknowns)
    total = len(unknowns)

    for a in range(total + 1):
        x_min = max(0, a - (total - before_low))
        x_max = min(a, before_high)
        target_minus_k = quantile_rank(n + a, q) - k
        if x_min != x_max or x_min != target_minus_k:
            return False
    return True


def evaluate_position(
    surface: SurfaceMap,
    cell: CellId,
    spec: MaterialSpec,
    status,
    sample_loader,
    head_of_cell,
    table_range,
) -> PositionResult:
    """本地完整张量路径的单点评估。

    ``sample_loader(cell)`` 返回该格点的整数量化码；``head_of_cell``
    给出读头名；``table_range(head)`` 返回该读头定标表的 (最小, 最大)
    物理值，用于界定未读样本补交后的取值范围。
    """
    material = surface.material_at(cell)
    if material != spec.material:
        raise ValueError(f"位置 {cell} 材料为 {material}，与规格 {spec.material} 不符")

    stages: list[RadiusOutcome] = []
    selected_index: int | None = None
    reasons: list[str] = []
    for idx, radius in enumerate(spec.radii):
        rect = surface.physical_window(cell, radius)
        valid: list[Sample] = []
        unknowns: list[UnknownSample] = []
        for cand in rect.cells():
            if surface.material_at(cand) != material:
                continue  # 邻域只收同一类属，材料缝天然不跨入
            st = int(status[cand.row, cand.col])
            if st == STATUS_READ:
                # 已知零值与其他码一样参与，未读/确认无读数没有秩次。
                head = head_of_cell(cand)
                valid.append(Sample(cand, int(sample_loader(cand, head))))
            elif st == STATUS_UNREAD:
                head = head_of_cell(cand)
                low, high = table_range(head)
                unknowns.append(UnknownSample(cand, head, low, high))
        outcome = RadiusOutcome(radius, rect, tuple(valid), tuple(unknowns))
        stages.append(outcome)
        if selected_index is None and outcome.n_valid >= spec.minimum:
            selected_index = idx

    target_status = int(status[cell.row, cell.col])
    target_head = head_of_cell(cell)
    target_physical = (
        int(sample_loader(cell, target_head))
        if target_status == STATUS_READ
        else None
    )

    if selected_index is None:
        last = stages[-1]
        blocked = not _any_potential(surface, cell, material, status, last.rect)
        reasons.append("最大物理半径内有效样本仍不足门槛")
        if blocked:
            reasons.append("同材质候选均为确认无读数，结论永久阻塞")
        else:
            reasons.append("等待同材质未读样本补交")
        return PositionResult(
            cell=cell, material=material, selected_radius=None,
            rectangle=last.rect, valid_count=0,
            unknown_count=last.n_unknown,
            quantile_rank=None, quantile_value=None, witness=None,
            witness_head=None, target_physical=target_physical,
            target_status=target_status, deviation=None,
            out_of_tolerance=None, radius_certain=False,
            quantile_certain=False, witness_certain=False,
            tolerance_certain=False, blocked=blocked,
            reasons=tuple(reasons),
        )

    chosen = stages[selected_index]
    value, witness, rank = pick_quantile(chosen.valid, spec.q)
    radius_certain = certify_radius(stages, selected_index, spec.minimum)
    bracket_stable, low, high = certify_quantile_value(chosen, spec.q)
    # 窗口可能缩回时，最终样本集会整体改变，当前窗口的夹住结论不能外推。
    value_certain = radius_certain and bracket_stable
    witness_certain = radius_certain and certify_witness(chosen, spec.q)
    if not radius_certain:
        reasons.append("未读样本补交后窗口可能缩回更小半径")
    elif not bracket_stable:
        reasons.append(f"分位值可能落在 [{low}, {high}] 区间")
    if radius_certain and not witness_certain:
        reasons.append("平台并列样本可能改变秩见证身份")

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
        cell=cell, material=material, selected_radius=chosen.radius,
        rectangle=chosen.rect, valid_count=chosen.n_valid,
        unknown_count=chosen.n_unknown, quantile_rank=rank,
        quantile_value=value, witness=witness,
        witness_head=head_of_cell(witness), target_physical=target_physical,
        target_status=target_status, deviation=deviation,
        out_of_tolerance=out_of_tol, radius_certain=radius_certain,
        quantile_certain=value_certain, witness_certain=witness_certain,
        tolerance_certain=tolerance_certain, blocked=False,
        reasons=tuple(reasons),
    )


def _any_potential(surface, cell, material, status, rect) -> bool:
    """最大窗口内是否还存在可能补交或可能有效的同材质候选。"""
    for cand in rect.cells():
        if surface.material_at(cand) != material:
            continue
        if int(status[cand.row, cand.col]) != -1:
            return True
    return False
