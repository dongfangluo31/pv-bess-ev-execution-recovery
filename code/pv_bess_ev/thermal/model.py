# EN: IEC 60076-7 / IEEE C57.91 two-gradient ONAN distribution-transformer top-oil and winding hot-spot thermal model.
# 中文：基于 IEC 60076-7 / IEEE C57.91 的油浸自冷（ONAN）配电变压器顶层油温与绕组热点双梯度热模型。

from dataclasses import dataclass, fields
from math import expm1, hypot, isfinite, sqrt

def finite(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (not isfinite(value)):
        raise ValueError(name + ' must be a finite real scalar')

@dataclass(frozen=True)
class Parameters:
    oil_rise_rated_k: float
    hotspot_gradient_rated_k: float
    load_to_no_load_loss_ratio: float
    oil_exponent: float
    winding_exponent: float
    oil_time_constant_min: float
    winding_time_constant_min: float
    k11: float
    k21: float
    k22: float

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            finite(f.name, value)
            if value <= 0:
                raise ValueError(f.name + ' must be positive')
        if self.k21 < 1:
            raise ValueError('This two-gradient realization requires k21 >= 1')

    @property
    def time_constants_min(self):
        return (self.k11 * self.oil_time_constant_min, self.k22 * self.winding_time_constant_min, self.oil_time_constant_min / self.k22)

@dataclass(frozen=True)
class State:
    oil_c: float
    h1_k: float
    h2_k: float

    def __post_init__(self):
        for f in fields(self):
            finite(f.name, getattr(self, f.name))
        if self.oil_c <= -273.15 or self.h1_k < 0 or self.h2_k < 0:
            raise ValueError('Invalid absolute oil temperature or gradient state')

    @property
    def hotspot_c(self):
        return self.oil_c + self.h1_k - self.h2_k

# EN: Steady-state top-oil temperature and fast/slow winding gradients (h1_k, h2_k) at per-unit RMS load factor current_pu.
# 中文：给定标幺有效值负载率 current_pu 与环境温度 ambient_c 下的稳态顶层油温及绕组快/慢温升梯度（h1_k, h2_k）。
def equilibrium(parameters: Parameters, current_pu: float, ambient_c: float) -> State:
    finite('current_pu', current_pu)
    finite('ambient_c', ambient_c)
    if current_pu < 0 or ambient_c <= -273.15:
        raise ValueError('RMS current pu must be nonnegative and ambient physical')
    ratio = parameters.load_to_no_load_loss_ratio
    oil_rise = parameters.oil_rise_rated_k * ((1 + ratio * current_pu ** 2) / (1 + ratio)) ** parameters.oil_exponent
    gradient = parameters.hotspot_gradient_rated_k * current_pu ** parameters.winding_exponent
    return State(ambient_c + oil_rise, parameters.k21 * gradient, (parameters.k21 - 1) * gradient)

# EN: Exact constant-current step integration over `seconds` using `-expm1(-dt / tau)` for numerical precision at short steps.
# 中文：采用 `-expm1(-dt / tau)` 对恒定电流步长进行精确指数积分，避免短步长下的浮点相消误差。
def advance(parameters: Parameters, state: State, *, current_pu: float, ambient_c: float, seconds: float) -> State:
    finite('seconds', seconds)
    if seconds < 0:
        raise ValueError('Cannot integrate a negative interval')
    target = equilibrium(parameters, current_pu, ambient_c)
    dt_min = seconds / 60
    output = []
    for old, goal, tau in zip((state.oil_c, state.h1_k, state.h2_k), (target.oil_c, target.h1_k, target.h2_k), parameters.time_constants_min):
        output.append(old + (goal - old) * -expm1(-dt_min / tau))
    return State(*output)

def current_pu_from_balanced_terminal(*, p_mw: float, q_mvar: float, voltage_ll_kv: float, nominal_current_ka: float) -> float:
    for name, value in [('p_mw', p_mw), ('q_mvar', q_mvar), ('voltage_ll_kv', voltage_ll_kv), ('nominal_current_ka', nominal_current_ka)]:
        finite(name, value)
    if voltage_ll_kv <= 0 or nominal_current_ka <= 0:
        raise ValueError('Actual line-line voltage and winding rating must be positive')
    return hypot(p_mw, q_mvar) / (sqrt(3) * voltage_ll_kv * nominal_current_ka)

def endpoint_residuals(state: State, *, oil_limit_c: float, hotspot_limit_c: float):
    finite('oil_limit_c', oil_limit_c)
    finite('hotspot_limit_c', hotspot_limit_c)
    return dict(oil_excess_k=max(state.oil_c - oil_limit_c, 0), hotspot_excess_k=max(state.hotspot_c - hotspot_limit_c, 0))
