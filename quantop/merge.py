"""跨读头并列样本的稳定汇合。

定标平台会把多个码（甚至横跨两个读头）映射到同一物理值。被选中的物理
值对应的全部并列样本必须按全局稳定 ID 汇合后，才能找回真正的秩见证；
只拿分位物理值会让复查人员找错测量点。

汇合端不能把完整窗口的码值搬出读头，身份只能逐页询问，因此这里实现
分页有序流上的 k 路归并，只保留各流当前页，不物化整组身份。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Iterator

from .heads import IdentityRequest


class IncompleteReplyError(RuntimeError):
    """读头应答未齐（帧资料不完整或分页被拒）。"""

    def __init__(self, head: str, code: int) -> None:
        super().__init__(f"读头 {head} 对码 {code} 的身份应答未齐")
        self.head = head
        self.code = code


class StaleCalibrationError(RuntimeError):
    """读头应答落在旧定标版，不能用于新版目标。"""

    def __init__(self, head: str, reply_version: str, expected: str) -> None:
        super().__init__(
            f"读头 {head} 以旧版 {reply_version} 应答，期望 {expected}"
        )
        self.head = head
        self.reply_version = reply_version
        self.expected_version = expected


@dataclass
class HeadCodeStream:
    """某读头在某窗口、某材料、某码值下的有序身份流（分页惰性拉取）。"""

    head_name: str
    frame_id: str
    calibration_version: str
    r0: int
    r1: int
    c0: int
    c1: int
    material: int
    code: int
    page_limit: int
    identity_call: Callable[[IdentityRequest], object]
    window_key: tuple[int, int, int] | None = None
    _total: int | None = None
    _page_ids: tuple[int, ...] = ()
    _page_origin: int = 0
    _cursor: int = 0

    def _load_page(self) -> None:
        reply = self.identity_call(
            IdentityRequest(
                frame_id=self.frame_id,
                calibration_version=self.calibration_version,
                r0=self.r0, r1=self.r1, c0=self.c0, c1=self.c1,
                material=self.material,
                code=self.code,
                ordinal=self._cursor,
                limit=self.page_limit,
                window_key=self.window_key,
            )
        )
        if not reply.complete:
            raise IncompleteReplyError(self.head_name, self.code)
        if reply.calibration_version != self.calibration_version:
            raise StaleCalibrationError(
                self.head_name, reply.calibration_version, self.calibration_version
            )
        self._total = reply.total_for_code
        self._page_ids = tuple(reply.sample_ids)
        self._page_origin = self._cursor

    @property
    def total(self) -> int:
        if self._total is None:
            self._load_page()
        return self._total  # type: ignore[return-value]

    def peek(self) -> int | None:
        if self._cursor >= self.total:
            return None
        # 当前页已消费到页尾，下一次访问按新游标翻页。
        if not self._page_ids or self._cursor >= self._page_origin + len(self._page_ids):
            self._load_page()
        return self._page_ids[self._cursor - self._page_origin]

    def pop(self) -> int | None:
        value = self.peek()
        if value is not None:
            self._cursor += 1
            if self._cursor == self._page_origin + len(self._page_ids):
                # 页尾：清空缓冲，下次 peek 触发翻页，而不是错位重载。
                self._page_ids = ()
        return value


def merge_streams(streams: list[HeadCodeStream]) -> Iterator[int]:
    """把多个读头的有序身份流稳定归并为全局 ID 升序流。"""
    heap: list[tuple[int, int, HeadCodeStream]] = []
    for order, stream in enumerate(streams):
        value = stream.peek()
        if value is not None:
            heapq.heappush(heap, (value, order, stream))
    while heap:
        value, order, stream = heapq.heappop(heap)
        yield value
        stream.pop()
        nxt = stream.peek()
        if nxt is not None:
            heapq.heappush(heap, (nxt, order, stream))


def kth_identity(streams: list[HeadCodeStream], k: int) -> int:
    """返回归并后第 k（从 0 起）个样本 ID。

    每个流只在内存里保留当前一页；堆归并保证全局 ID 升序，且同一读头
    内部顺序也由读头保证，跨读头同值时取 ID 更小者，与字典序秩定义一致。
    """
    if k < 0:
        raise IndexError("秩下标不能为负")
    total = sum(stream.total for stream in streams)
    if k >= total:
        raise IndexError(f"并列组只有 {total} 个样本，取不到第 {k} 个")
    for value in merge_streams(streams):
        if k == 0:
            return value
        k -= 1
    raise AssertionError("归并流提前耗尽")  # pragma: no cover
