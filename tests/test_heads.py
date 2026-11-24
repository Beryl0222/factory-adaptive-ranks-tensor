"""读头协议：辖区隔离、分页、状态过滤与越权访问测试。"""
import unittest

import numpy as np

from quantop.geometry import Grid
from quantop.heads import (
    FrequencyRequest,
    IdentityRequest,
    LocalReadHead,
    STATUS_PENDING,
    STATUS_READ,
    STATUS_VOID,
)


def build_head(name="left", page_limit=2):
    grid = Grid(rows=3, cols=4, spacing_row=1, spacing_col=1)
    classes = np.array(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.int8,
    )
    codes = np.array(
        [
            [0, 2, 9, 9],
            [2, 3, 9, 9],
            [1, 2, 9, 9],
        ],
        dtype=np.int8,
    )
    status = np.array(
        [
            [STATUS_READ, STATUS_READ, STATUS_READ, STATUS_READ],
            [STATUS_PENDING, STATUS_READ, STATUS_READ, STATUS_VOID],
            [STATUS_READ, STATUS_READ, STATUS_READ, STATUS_READ],
        ],
        dtype=np.int8,
    )
    head = LocalReadHead(name, grid, "c1", max_response_entries=page_limit)
    head.bind_classes(classes)
    head.load_frame(
        frame_id="f",
        codes=codes,
        status=status,
        half=(0, 3, 0, 2),  # 左读头辖区：前两列
    )
    return head, grid


class HeadProtocolTests(unittest.TestCase):
    def test_frequency_excludes_pending_and_void_includes_zero(self):
        head, _ = build_head()
        reply = head.frequency_page(
            FrequencyRequest("f", "c1", 0, 2, 0, 1, material=1, limit=12)
        )
        counts = {entry.code: entry.count for entry in reply.entries}
        # (0,0)=0 零值读数参与；(1,0) 待补、(2 列) 非本辖区均不参与。
        self.assertEqual(counts, {0: 1, 1: 1, 2: 2, 3: 1})
        self.assertEqual(reply.total_distinct_codes, 4)

    def test_frequency_paging_never_exceeds_entry_limit(self):
        head, _ = build_head(page_limit=2)
        seen = []
        offset = 0
        while True:
            reply = head.frequency_page(
                FrequencyRequest(
                    "f", "c1", 0, 2, 0, 1, material=1, offset=offset, limit=2
                )
            )
            self.assertLessEqual(len(reply.entries), 2)
            seen.extend(entry.code for entry in reply.entries)
            offset += len(reply.entries)
            if offset >= reply.total_distinct_codes:
                break
        self.assertEqual(seen, [0, 1, 2, 3])

    def test_identity_returns_global_ids_sorted_and_paged(self):
        head, _ = build_head(page_limit=1)
        ids = []
        ordinal = 0
        while True:
            reply = head.identity_page(
                IdentityRequest(
                    "f", "c1", 0, 2, 0, 1, material=1, code=2,
                    ordinal=ordinal, limit=1,
                )
            )
            ids.extend(reply.sample_ids)
            ordinal += len(reply.sample_ids)
            if ordinal >= reply.total_for_code:
                break
        # code=2 的材质 1 读数：(0,1)->id1、(2,1)->id9；(1,0) 待补被排除。
        self.assertEqual(ids, [1, 9])

    def test_window_outside_territory_is_legal_empty_reply(self):
        head, _ = build_head()
        reply = head.frequency_page(
            FrequencyRequest("f", "c1", 0, 2, 2, 3, material=1, limit=12)
        )
        self.assertEqual(reply.entries, ())
        self.assertEqual(reply.total_distinct_codes, 0)
        self.assertTrue(reply.complete)

    def test_read_raw_code_enforces_ownership(self):
        head, _ = build_head()
        self.assertEqual(head.read_raw_code("f", 1, 1), 3)
        with self.assertRaises(PermissionError):
            head.read_raw_code("f", 1, 2)  # 列 2 不属左读头

    def test_reply_declares_bound_calibration_only(self):
        head, _ = build_head()
        reply = head.frequency_page(
            FrequencyRequest("f", "c9", 0, 2, 0, 1, material=1, limit=12)
        )
        # 即使请求声明别的版本，读头只按自己绑定的版本应答。
        self.assertEqual(reply.calibration_version, "c1")


if __name__ == "__main__":
    unittest.main()
