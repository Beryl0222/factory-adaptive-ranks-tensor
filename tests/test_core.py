"""分位秩、平台见证与待判定可证明规则的单元测试。"""
from fractions import Fraction
import unittest

from roughquant.core import (
    MaterialSpec,
    RadiusOutcome,
    Sample,
    UnknownSample,
    certify_quantile_bracket,
    certify_quantile_value,
    certify_radius,
    pick_quantile,
    quantile_rank,
    witness_invariant,
)
from roughquant.geometry import CellId

Q_HALF = Fraction(1, 2)
Q_23 = Fraction(2, 3)


def cell(k):
    return CellId(k // 10, k % 10)


class QuantileTests(unittest.TestCase):
    def test_ceil_rank_definition(self):
        # 第 ceil(q*n) 项，1 基；q=1/2, n=3 -> 第 2 项。
        self.assertEqual(quantile_rank(3, Q_HALF), 2)
        self.assertEqual(quantile_rank(4, Q_HALF), 2)
        self.assertEqual(quantile_rank(3, Q_23), 2)
        self.assertEqual(quantile_rank(1, Fraction(1, 1)), 1)

    def test_tie_break_by_stable_id(self):
        samples = [
            Sample(cell(20), 5),
            Sample(cell(10), 5),  # 同物理值，(1,0) 字典序在前
            Sample(cell(5), 1),
        ]
        value, witness, rank = pick_quantile(samples, Q_HALF)
        self.assertEqual(value, 5)
        self.assertEqual(witness, cell(10))
        self.assertEqual(rank, 2)

    def test_known_zero_participates(self):
        # 已知零值与其他读数一样参与排序。
        samples = [Sample(cell(1), 0), Sample(cell(2), 9), Sample(cell(3), 9)]
        value, witness, _ = pick_quantile(samples, Q_23)
        self.assertEqual(value, 9)
        self.assertEqual(witness, cell(2))

    def test_quantile_bracket_stable_and_uncertain(self):
        histogram = {2: 1, 5: 2}  # n=3, 秩 2 -> 当前分位 5
        unknown = UnknownSample(cell(9), "h", low=0, high=9)
        stable, low, high = certify_quantile_bracket(
            histogram, [unknown], Q_HALF
        )
        self.assertFalse(stable)
        # 全部取最小 -> 插入 0，序列 [0,2,5,5]，秩 2 -> 2
        # 全部取最大 -> 插入 9，序列 [2,5,5,9]，秩 2 -> 5
        self.assertEqual((low, high), (2, 5))

        stable, low, high = certify_quantile_bracket(
            histogram, [], Q_HALF
        )
        self.assertTrue(stable)

    def test_bracket_certifies_despite_swing_around_threshold(self):
        # 未读样本无论取何值，分位恒为 5（平台足够宽）。
        histogram = {5: 3}
        unknown = UnknownSample(cell(9), "h", low=0, high=9)
        stable, low, high = certify_quantile_bracket(
            histogram, [unknown], Q_HALF
        )
        self.assertTrue(stable)
        self.assertEqual((low, high), (5, 5))


class WitnessInvariantTests(unittest.TestCase):
    def _make_outcome(self, values, unknowns):
        samples = tuple(Sample(cell(10 + i), v) for i, v in enumerate(values))
        return RadiusOutcome(
            radius=1, rect=None, valid=samples, unknowns=tuple(unknowns)
        ), samples

    def test_witness_switches_inside_plateau_is_pending(self):
        # 两个同值 5 的样本；未读样本也是 5，但稳定 ID 排在两者之后，
        # 补交后秩从 1 移到 2，见证必然换人。
        outcome, samples = self._make_outcome(
            [5, 5],
            [UnknownSample(cell(99), "h", low=5, high=5)],
        )
        # n=2, q=1/2 -> k=1；补交后 n=3 -> 秩 2，见证换人。
        self.assertFalse(witness_invariant(
            len(samples), 1, samples[0].cell, 5, outcome.unknowns, Q_HALF
        ))

    def test_witness_holds_when_tied_unknown_always_sorts_before(self):
        # 同值未读样本恒排在见证之前时，秩平移后见证身份保持不变。
        outcome, samples = self._make_outcome(
            [5, 5],
            [UnknownSample(cell(1), "h", low=5, high=5)],
        )
        self.assertTrue(witness_invariant(
            len(samples), 1, samples[0].cell, 5, outcome.unknowns, Q_HALF
        ))

    def test_witness_invariant_when_unknowns_cannot_reach_rank(self):
        # 未读样本值域全部高于见证，且分位秩不会因补交而后移到见证之后。
        outcome, samples = self._make_outcome(
            [1, 2, 9],
            [UnknownSample(cell(1), "h", low=8, high=9)],
        )
        # 当前秩 2 见证为 2；插入值 >=8，新序列秩 ceil(4/2)=2，仍是 2。
        self.assertTrue(witness_invariant(
            len(samples), 2, samples[1].cell, 2, outcome.unknowns, Q_HALF
        ))

    def test_no_unknowns_always_invariant(self):
        outcome, samples = self._make_outcome([1, 2, 3], [])
        self.assertTrue(witness_invariant(
            len(samples), 2, samples[1].cell, 2, outcome.unknowns, Q_HALF
        ))

    def test_radius_certification_uses_all_potential_samples(self):
        spec = MaterialSpec(2, (1, 2), minimum=3, q=Q_HALF, threshold=1)
        small, _ = self._make_outcome(
            [1, 2],
            [UnknownSample(cell(1), "h", low=0, high=9)],
        )
        large, _ = self._make_outcome(
            [1, 2, 3, 4, 5],
            [UnknownSample(cell(2), "h", low=0, high=9)],
        )
        # 小半径潜力 3 已达门槛，补交后窗口会缩回 ⇒ 半径不确定。
        self.assertFalse(certify_radius([small, large], 1, spec.minimum))
        small_empty, _ = self._make_outcome([1, 2], [])
        # 小半径潜力 2 严格小于门槛，任何补交都无法让它达标 ⇒ 确定。
        self.assertTrue(certify_radius([small_empty, large], 1, spec.minimum))
        # 选中首个半径时没有更小窗口，自然确定。
        self.assertTrue(certify_radius([large], 0, spec.minimum))

    def test_value_and_witness_certification_agree_on_full_data(self):
        outcome, samples = self._make_outcome([1, 2, 3, 4], [])
        stable, low, high = certify_quantile_value(outcome, Q_HALF)
        self.assertTrue(stable)
        self.assertEqual(low, high)
        self.assertTrue(witness_invariant(
            len(samples), 2, samples[1].cell, 2, outcome.unknowns, Q_HALF
        ))


if __name__ == "__main__":
    unittest.main()
