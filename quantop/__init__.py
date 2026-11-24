"""分块自适应分位滤波算子（quantop）。

模块划分：

- ``geometry``：格网、物理跨度窗口、分块与工作区预算；
- ``calibration``：按读头绑定的单调定标表（允许平台区）；
- ``heads``：读头侧资料协议（码值频次分页、码值样本身份分页）；
- ``merge``：跨读头并列样本的稳定汇合；
- ``operator``：分块自适应分位滤波主算子与可交付性证明；
- ``handoff``：交接状态与跨帧变化归因。

设计约束见 README。所有秩次仅基于校正后的整数物理值与全局稳定样本 ID，
浮点不参与任何判定。
"""

from __future__ import annotations

__all__ = [
    "geometry",
    "calibration",
    "heads",
    "merge",
    "operator",
    "handoff",
    "pipeline",
]
