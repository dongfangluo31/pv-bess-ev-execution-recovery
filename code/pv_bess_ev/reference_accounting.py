# EN: Terminal battery inventory valuation and single-channel REF ablation helpers.
# 中文：终端储能库存价值调整核算与单通道执行修正（仅撤回充电 / 仅增加放电）消融辅助函数。
# EN: Paired comparisons evaluate delta_stock_kwh = E_T(arm) - E_T(ref) against the uncorrected baseline arm (GS_F0 or F0).
# 中文：配对实验以未加执行修正的基准臂（GS_F0 或 F0）为对照计算终端库存差 delta_stock_kwh = E_T(arm) - E_T(ref)。

import math

# EN: Inventory-adjusted operating cost: C_adj = C_op - v * (E_final - E_initial); paired difference cancels shared E_initial.
# 中文：计及终端库存变化的等效运行成本：C_adj = C_op - v * (E_final - E_initial)；配对作差时共同初值自动抵消，基准对照为未修正臂终端电量。
def inventory_cost(cost, initial, final, value):
    return cost - value * (final - initial)

# EN: Restricted single-channel filter for mechanism ablation: CHARGE_ONLY cancels charging; DISCHARGE_ONLY adds discharge only when nominal_charge == 0.
# 中文：用于机理消融的受限单通道算子：CHARGE_ONLY 仅撤回充电；DISCHARGE_ONLY 仅在名义充电为零时动用可交付备用余量放电。
def channel_correction(channel, nominal_charge, common_charge, common_discharge, common_import, limit, deliverable_energy, reserve, power_limit, dt):
    values = (nominal_charge, common_charge, common_discharge, common_import, limit, deliverable_energy, reserve, power_limit, dt)
    if not all((math.isfinite(v) for v in values)) or dt <= 0:
        raise ValueError('Finite inputs and positive duration required')
    if channel not in ('CHARGE_ONLY', 'DISCHARGE_ONLY'):
        raise ValueError('Unknown restricted execution channel')
    if common_import <= limit + 1e-09:
        return (0.0, 0.0)
    excess = common_import - limit
    if channel == 'CHARGE_ONLY':
        return (min(common_charge, excess), 0.0)
    if nominal_charge > 0:
        return (0.0, 0.0)
    capacity = min(max(power_limit - common_discharge, 0.0), max(deliverable_energy - reserve - common_discharge * dt, 0.0) / dt)
    return (0.0, min(capacity, excess))

# EN: Mark published Dingle tariff-coefficient weighted import totals as dimensionless/non-poolable across CNY tables.
# 中文：标注公开 Dingle 电价系数加权购电指标的语义边界，禁止与人民币计价表直接跨币种混算。
def dingle_cost_alias(row):
    result = dict(row)
    for old, new in [('tariff_cost_EUR', 'published_price_coefficient_import_cost'), ('delta_tariff_cost_EUR', 'delta_published_price_coefficient_import_cost')]:
        if old in row:
            result[new] = row[old]
    result['effective_cost_semantics'] = 'published_price_coefficient_weighted_net_import'
    result['currency_scale_verified'] = False
    result['legacy_EUR_alias_only'] = True
    result['currency_pooling_allowed'] = False
    return result
