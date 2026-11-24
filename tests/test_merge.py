"""跨读头稳定汇合：平台区并列、分页归并、旧版应答拒绝测试。"""
import unittest

from quantop.heads import IdentityReply, IdentityRequest
from quantop.merge import (
    HeadCodeStream,
    IncompleteReplyError,
    StaleCalibrationError,
    kth_identity,
    merge_streams,
)


class FakeHead:
    """只记录身份序列的假读头，可控制分页大小、版本与齐备性。"""

    def __init__(self, name, ids, version="c1", page_size=2, complete=True):
        self.name = name
        self._ids = sorted(ids)
        self._version = version
        self._page_size = page_size
        self._complete = complete
        self.requests = []

    def identity_page(self, request: IdentityRequest) -> IdentityReply:
        self.requests.append(request)
        if not self._complete:
            return IdentityReply(
                frame_id=request.frame_id, head=self.name,
                calibration_version=self._version, code=request.code,
                sample_ids=(), total_for_code=len(self._ids),
                ordinal=request.ordinal, complete=False,
            )
        page = tuple(
            self._ids[request.ordinal : request.ordinal + request.limit]
        )
        return IdentityReply(
            frame_id=request.frame_id, head=self.name,
            calibration_version=self._version, code=request.code,
            sample_ids=page, total_for_code=len(self._ids),
            ordinal=request.ordinal, complete=True,
        )


def make_stream(head, code=3, page_size=None):
    return HeadCodeStream(
        head_name=head.name, frame_id="f", calibration_version="c1",
        r0=0, r1=9, c0=0, c1=9, material=1, code=code,
        page_limit=page_size or head._page_size,
        identity_call=head.identity_page,
    )


class MergeTests(unittest.TestCase):
    def test_merge_is_global_id_order_across_heads(self):
        left = FakeHead("left", [3, 17, 25], page_size=2)
        right = FakeHead("right", [12, 18], page_size=1)
        merged = list(merge_streams([make_stream(left), make_stream(right)]))
        self.assertEqual(merged, [3, 12, 17, 18, 25])

    def test_kth_identity_inside_plateau_group(self):
        left = FakeHead("left", [3, 16, 17], page_size=2)
        right = FakeHead("right", [12, 18], page_size=1)
        streams = [make_stream(left), make_stream(right)]
        self.assertEqual(kth_identity(streams, 0), 3)
        # 每次取 kth 都重新构造流，避免消费状态串扰。
        self.assertEqual(
            kth_identity([make_stream(left), make_stream(right)], 1), 12
        )
        self.assertEqual(
            kth_identity([make_stream(left), make_stream(right)], 4), 18
        )

    def test_pagination_does_not_drop_page_boundary_ids(self):
        head = FakeHead("solo", list(range(0, 10, 2)), page_size=2)
        stream = make_stream(head, page_size=2)
        self.assertEqual(list(merge_streams([stream])), [0, 2, 4, 6, 8])

    def test_stale_calibration_reply_is_surfaced_for_rejection(self):
        stale = FakeHead("old", [1, 2], version="c0", page_size=2)
        stream = make_stream(stale)
        # 流按 c1 发起，读头以 c0 应答。
        with self.assertRaises(StaleCalibrationError):
            list(merge_streams([stream]))

    def test_incomplete_reply_is_surfaced(self):
        broken = FakeHead("broken", [1, 2], complete=False)
        stream = make_stream(broken)
        with self.assertRaises(IncompleteReplyError):
            kth_identity([stream], 0)

    def test_kth_out_of_range(self):
        head = FakeHead("h", [1], page_size=1)
        with self.assertRaises(IndexError):
            kth_identity([make_stream(head)], 1)


if __name__ == "__main__":
    unittest.main()
