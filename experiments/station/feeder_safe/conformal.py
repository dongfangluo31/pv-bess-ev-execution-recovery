# EN: Past-only asymmetric split conformal prediction and online Adaptive Conformal Inference (ACI) interval calibration.
# 中文：纯历史非对称分裂保形预测（Split Conformal）与在线自适应保形推断（ACI）区间校准。

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
import numpy as np

def finite_sample_rank(n: int, miscoverage: float) -> int:
    if n < 1:
        raise ValueError('at least one calibration score is required')
    if not 0 < miscoverage < 1:
        raise ValueError('miscoverage must lie in (0, 1)')
    return min(n, max(1, math.ceil((n + 1) * (1 - miscoverage))))

def finite_sample_quantile(scores: np.ndarray, miscoverage: float) -> float:
    values = np.asarray(scores, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('no finite calibration scores')
    rank = finite_sample_rank(values.size, miscoverage)
    return float(np.partition(values, rank - 1)[rank - 1])

@dataclass(frozen=True)
class FrozenIntervals:
    origin: int
    alpha_lower: float
    alpha_upper: float
    calibration_size: int
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lower_offset: float
    upper_offset: float

    def __post_init__(self) -> None:
        for array in (self.point, self.lower, self.upper):
            array.setflags(write=False)

    @property
    def alpha(self) -> float:
        return self.alpha_lower + self.alpha_upper

class AsymmetricConformal:

    def __init__(self, residuals: np.ndarray, miscoverage: float=0.1, rolling_window: int | None=None) -> None:
        residuals = np.asarray(residuals, dtype=float).reshape(-1)
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size == 0:
            raise ValueError('calibration residuals cannot be empty')
        if not 0 < miscoverage < 1:
            raise ValueError('miscoverage must lie in (0, 1)')
        if rolling_window is not None and rolling_window < 1:
            raise ValueError('rolling window must be positive')
        self.target_miscoverage = float(miscoverage)
        self.rolling_window = rolling_window
        maxlen = rolling_window
        self._lower_scores = deque(-residuals, maxlen=maxlen)
        self._upper_scores = deque(residuals, maxlen=maxlen)

    @property
    def alpha(self) -> float:
        return self.alpha_lower + self.alpha_upper

    @property
    def alpha_lower(self) -> float:
        return self.target_miscoverage / 2.0

    @property
    def alpha_upper(self) -> float:
        return self.target_miscoverage / 2.0

    @property
    def calibration_size(self) -> int:
        return len(self._lower_scores)

    def _offsets(self, alpha_lower: float, alpha_upper: float) -> tuple[float, float]:
        q_lower = -finite_sample_quantile(np.asarray(self._lower_scores), alpha_lower)
        q_upper = finite_sample_quantile(np.asarray(self._upper_scores), alpha_upper)
        return (q_lower, q_upper)

    def freeze_horizon(self, point: np.ndarray, origin: int) -> FrozenIntervals:
        point = np.array(point, dtype=float, copy=True).reshape(-1)
        if point.size == 0 or not np.all(np.isfinite(point)):
            raise ValueError('point forecast must be finite and nonempty')
        alpha_lower = float(self.alpha_lower)
        alpha_upper = float(self.alpha_upper)
        lower_offset, upper_offset = self._offsets(alpha_lower, alpha_upper)
        return FrozenIntervals(origin=int(origin), alpha_lower=alpha_lower, alpha_upper=alpha_upper, calibration_size=self.calibration_size, point=point, lower=point + lower_offset, upper=point + upper_offset, lower_offset=lower_offset, upper_offset=upper_offset)

    def observe(self, actual: float, prediction: float, frozen: FrozenIntervals) -> None:
        if not (math.isfinite(actual) and math.isfinite(prediction)):
            raise ValueError('observation and prediction must be finite')
        if self.rolling_window is not None:
            self._append_residual(float(actual - prediction))

    def _append_residual(self, residual: float) -> None:
        self._lower_scores.append(-residual)
        self._upper_scores.append(residual)

class AdaptiveConformal(AsymmetricConformal):

    def __init__(self, residuals: np.ndarray, target_miscoverage: float=0.1, gamma: float=0.02, rolling_window: int | None=None, alpha_min: float=0.005, alpha_max: float=0.3) -> None:
        super().__init__(residuals, target_miscoverage, rolling_window)
        if gamma <= 0 or not 0 < alpha_min < alpha_max < 1:
            raise ValueError('ACI gamma/alpha limits are invalid')
        self.gamma = float(gamma)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        initial_tail = target_miscoverage / 2.0
        self._alpha_lower = float(np.clip(initial_tail, alpha_min, alpha_max))
        self._alpha_upper = float(np.clip(initial_tail, alpha_min, alpha_max))

    @property
    def alpha(self) -> float:
        return self._alpha_lower + self._alpha_upper

    @property
    def alpha_lower(self) -> float:
        return self._alpha_lower

    @property
    def alpha_upper(self) -> float:
        return self._alpha_upper

    def observe(self, actual: float, prediction: float, frozen: FrozenIntervals) -> None:
        if frozen.origin < 0 or frozen.point.size == 0:
            raise ValueError('invalid frozen horizon snapshot')
        if not (math.isfinite(actual) and math.isfinite(prediction)):
            raise ValueError('observation and prediction must be finite')
        lower_miss = float(actual < frozen.lower[0])
        upper_miss = float(actual > frozen.upper[0])
        target_tail = self.target_miscoverage / 2.0
        self._alpha_lower = float(np.clip(self._alpha_lower + self.gamma * (target_tail - lower_miss), self.alpha_min, self.alpha_max))
        self._alpha_upper = float(np.clip(self._alpha_upper + self.gamma * (target_tail - upper_miss), self.alpha_min, self.alpha_max))
        self._append_residual(float(actual - prediction))

@dataclass(frozen=True)
class CalibrationPressureScaler:
    scale_kw: float
    quantile: float
    calibration_size: int
    clip: bool = False

    @classmethod
    def fit(cls, ev_upper_cal: np.ndarray, pv_lower_cal: np.ndarray, quantile: float=1.0, epsilon: float=1e-09, clip: bool=False) -> 'CalibrationPressureScaler':
        ev = np.asarray(ev_upper_cal, dtype=float).reshape(-1)
        pv = np.asarray(pv_lower_cal, dtype=float).reshape(-1)
        if ev.shape != pv.shape or ev.size == 0:
            raise ValueError('calibration EV/PV arrays must be nonempty and aligned')
        if not np.all(np.isfinite(ev)) or not np.all(np.isfinite(pv)):
            raise ValueError('calibration arrays contain non-finite values')
        if quantile != 1.0:
            raise ValueError('pressure scale must use the calibration maximum')
        deficit = np.maximum(ev - pv, 0.0)
        scale = max(float(np.max(deficit)), float(epsilon))
        return cls(scale, float(quantile), int(ev.size), clip)

    def transform(self, adverse_net_kw: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(adverse_net_kw, dtype=float), 0.0) / self.scale_kw
        return np.clip(values, 0.0, 1.0) if self.clip else values
