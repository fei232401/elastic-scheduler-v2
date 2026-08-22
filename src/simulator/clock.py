"""仿真时钟 —— tick 驱动（spec Step 6）。

默认 tick=5s，duration_ticks 决定总时长。engine 按 tick 推进。
"""
from __future__ import annotations


class Clock:
    def __init__(self, duration_ticks: int, tick_seconds: int = 5) -> None:
        self.duration_ticks = duration_ticks
        self.tick_seconds = tick_seconds
        self._tick = 0

    @property
    def tick(self) -> int:
        """当前 tick 序号（0-based，已完成的 tick 数）。"""
        return self._tick

    @property
    def now_s(self) -> float:
        return self._tick * self.tick_seconds

    @property
    def now_min(self) -> float:
        return self.now_s / 60.0

    @property
    def done(self) -> bool:
        return self._tick >= self.duration_ticks

    def advance(self) -> None:
        """推进一个 tick。"""
        self._tick += 1
