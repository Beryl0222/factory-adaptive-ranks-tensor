"""交接状态与跨帧变化归因测试。"""
import unittest

from quantop.handoff import ChangeCause, attribute_transition, build_handoff
from quantop.operator import DeliveryState, PositionResult


def make_result(frame="f1", version="c1", radius=2, quantile=3, witness=10,
                target=5, oot=True, delivery=DeliveryState.DELIVERED):
    return PositionResult(
        row=1, col=3, material=1, frame_id=frame,
        calibration_version=version, sample_id=10, target_status=1,
        radius=radius, valid_count=4, quantile_value=quantile,
        witness_id=witness, target_value=target, out_of_tolerance=oot,
        delivery=delivery,
    )


class AttributionTests(unittest.TestCase):
    def test_radius_change_takes_precedence(self):
        previous = make_result(radius=4)
        current = make_result(frame="f2", radius=2)
        transition = attribute_transition(previous, current)
        self.assertEqual(transition.cause, ChangeCause.RADIUS)
        self.assertTrue(transition.radius_changed)
        self.assertTrue(transition.became_deliverable is False)

    def test_undecided_to_delivered_records_became_deliverable(self):
        previous = make_result(
            radius=4, delivery=DeliveryState.UNDECIDED, quantile=None,
            witness=None, oot=None,
        )
        current = make_result(frame="f2", radius=2)
        transition = attribute_transition(previous, current)
        self.assertTrue(transition.became_deliverable)
        self.assertEqual(transition.cause, ChangeCause.RADIUS)

    def test_calibration_plateau_same_quantile_switched_witness(self):
        # 更难查的情况：分位物理值不变，见证换到另一读头。
        previous = make_result(version="c1", quantile=3, witness=18, target=3, oot=False)
        current = make_result(frame="f2", version="c2", quantile=3, witness=17,
                              target=4, oot=False)
        transition = attribute_transition(previous, current)
        self.assertEqual(transition.cause, ChangeCause.CALIBRATION)
        self.assertFalse(transition.quantile_value_changed)
        self.assertTrue(transition.witness_changed)
        self.assertIn("平台区", transition.detail)

    def test_rank_change_under_same_radius_and_calibration(self):
        previous = make_result(quantile=3, witness=11)
        current = make_result(frame="f2", quantile=4, witness=12)
        transition = attribute_transition(previous, current)
        self.assertEqual(transition.cause, ChangeCause.RANK)

    def test_stable_when_nothing_moves(self):
        previous = make_result()
        current = make_result(frame="f2")
        transition = attribute_transition(previous, current)
        self.assertEqual(transition.cause, ChangeCause.STABLE)
        self.assertFalse(transition.out_of_tolerance_changed)

    def test_build_handoff_indexes_per_cell_and_adjacent_frames(self):
        r1 = make_result(frame="f1")
        r2 = make_result(frame="f2")
        other = PositionResult(
            row=0, col=0, material=1, frame_id="f1",
            calibration_version="c1", sample_id=0, target_status=1,
            radius=2, valid_count=4, quantile_value=1, witness_id=0,
            target_value=1, out_of_tolerance=False,
            delivery=DeliveryState.DELIVERED,
        )
        indexed, transitions = build_handoff(
            {"f1": [r1, other], "f2": [r2]}
        )
        self.assertEqual(set(indexed), {(1, 3), (0, 0)})
        self.assertEqual(len(transitions), 1)
        self.assertEqual((transitions[0].row, transitions[0].col), (1, 3))


if __name__ == "__main__":
    unittest.main()
