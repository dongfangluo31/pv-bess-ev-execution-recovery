# EN: Causal constant-persistence environment check ensuring terminal recovery planners only see current released (L, pv) point feeds.
# 中文：终端恢复规划器的因果常数持续性环境校验，确保规划器仅读取当前已发布点预测而不回溯读取未来序列。

import math

# EN: Verify packet issue/availability timestamps equal `now` and return constant `(L, pv)` over remaining seconds `T - now`.
# 中文：核验预测包签发与可用时间戳均等于当前时刻 `now`，并在剩余时段 `T - now` 上构造恒定 `(L, pv)` 轨迹。
def environment(forecast, now, T, L, pv):
    if forecast.get('method') != 'CURRENT_PUBLIC_POINT_PERSISTENCE' or forecast.get('forecast_issued_s') != now or forecast.get('available_s') != now:
        raise ValueError('Current released nominal point required, never backdated H96')
    if forecast.get('load_kw') != L or forecast.get('pv_kw') != pv or any((not math.isfinite(x) or x < 0 for x in (L, pv))):
        raise ValueError('Nominal point differs from public current packet')
    if forecast.get('support_is_certified') is not False:
        raise ValueError('Persistence is not certified future absorption')
    if forecast['horizon']['ev_point_kw'] != [L] * 96 or forecast['horizon']['pv_point_kw'] != [pv] * 96:
        raise ValueError('Additional forecast information forbidden')
    return [(L, pv)] * (T - now)
