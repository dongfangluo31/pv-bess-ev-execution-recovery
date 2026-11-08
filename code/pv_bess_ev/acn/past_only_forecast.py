# EN: Strictly causal lag-L persistence forecaster (P24 / P48) using only observations prior to the decision index.
# 中文：严格因果滞后持续性预测器（P24 / P48），仅读取当前决策时刻之前的历史实测序列。

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

class InsufficientHistoryError(ValueError):
    pass

@dataclass(frozen=True)
class CausalPersistenceForecaster:
    lag: int
    require_full_lag: bool = True

    def __post_init__(self) -> None:
        if self.lag < 1:
            raise ValueError('lag must be positive')

    @property
    def name(self) -> str:
        return f'P{self.lag}'

    # EN: Recursive lag lookup: offsets beyond `lag` wrap onto earlier forecasted steps rather than reading future observations.
    # 中文：递推滞后映射：当预测步偏移超过滞后阶数 lag 时自动回卷引用本轮前序预测值，杜绝未来信息泄漏。
    def forecast(self, history: np.ndarray, horizon: int) -> np.ndarray:
        history = np.asarray(history, dtype=float).reshape(-1)
        if horizon < 1:
            raise ValueError('horizon must be positive')
        if not np.all(np.isfinite(history)):
            raise ValueError('history contains non-finite values')
        origin = history.size
        if self.require_full_lag and origin < self.lag:
            raise InsufficientHistoryError(f'{self.name} requires {self.lag} observations; received {origin}')
        fallback = float(np.mean(history)) if origin else 0.0
        out = np.empty(horizon, dtype=float)
        for offset in range(horizon):
            source = origin + offset - self.lag
            if source < 0:
                value = fallback
            elif source < origin:
                value = history[source]
            else:
                value = out[source - origin]
            out[offset] = value
        return out

    def rolling_one_step(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError('values contain non-finite observations')
        predictions = np.full(values.shape, np.nan, dtype=float)
        if values.size > self.lag:
            predictions[self.lag:] = values[:-self.lag]
        valid = np.isfinite(predictions)
        return (predictions, valid)

def p24() -> CausalPersistenceForecaster:
    return CausalPersistenceForecaster(24)

def p48() -> CausalPersistenceForecaster:
    return CausalPersistenceForecaster(48)
