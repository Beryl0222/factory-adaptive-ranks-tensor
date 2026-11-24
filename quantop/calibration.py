"""读头定标表：整数码到整数物理值的单调映射，允许平台区。

定标表按读头分别绑定，换版只适用于归该版管辖的读头值。表可包含平台区
（多个不同码映射到相同物理值），因此秩次在物理值并列时必须借助稳定
样本 ID 区分，物理值本身无法唯一定位见证。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationTable:
    """单个读头在某一版本下的定标表。

    ``table[code]`` 是该整数码校正后的物理值（整数微单位，可精确比较）。
    表必须单调非降，构造时校验。
    """

    version: str
    head: str
    table: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.table:
            raise ValueError(f"读头 {self.head} 的定标表为空")
        prev = self.table[0]
        for value in self.table[1:]:
            if value < prev:
                raise ValueError(
                    f"读头 {self.head} 的 {self.version} 定标表不单调"
                )
            prev = value

    @property
    def code_count(self) -> int:
        return len(self.table)

    def to_physical(self, code: int) -> int:
        if not 0 <= code < len(self.table):
            raise KeyError(f"码 {code} 超出读头 {self.head} 定标表范围")
        return self.table[code]


class CalibrationBook:
    """各读头、各版本定标表的归属册。"""

    def __init__(self) -> None:
        # book[version][head] -> CalibrationTable
        self._book: dict[str, dict[str, CalibrationTable]] = {}

    def add(self, table: CalibrationTable) -> None:
        self._book.setdefault(table.version, {})[table.head] = table

    def versions(self) -> tuple[str, ...]:
        return tuple(self._book)

    def heads_of(self, version: str) -> tuple[str, ...]:
        return tuple(self._book[version])

    def get(self, version: str, head: str) -> CalibrationTable:
        try:
            return self._book[version][head]
        except KeyError as exc:
            raise KeyError(f"定标版 {version} 下没有读头 {head} 的表") from exc

    def has(self, version: str, head: str) -> bool:
        return version in self._book and head in self._book[version]
