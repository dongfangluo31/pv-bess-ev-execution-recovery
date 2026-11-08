# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

def step_source_quality(segment, t):
    over = []
    for s in segment['source_support']:
        if not s['plant_estimate_overage_seconds']:
            continue
        seconds = max(0.0, min(t + 1, segment['end_s']) - max(t, s['source_reported_s'] + s['controller_age_cap_s']))
        if seconds:
            over.append(dict(connector_pk=s['connector_pk'], port_seconds=seconds))
    return dict(original_strict_whole_hold_valid=segment['original_strict_whole_hold_valid'], estimate_overage_ports=over, estimate_overage_port_seconds=sum((r['port_seconds'] for r in over)), source_is_offline_estimate=True, source_measured_exactness_claim=False)
