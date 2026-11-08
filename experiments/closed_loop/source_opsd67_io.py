# EN: Time-series packet loader, unit normalization, and causal release-timestamp validator.
# 中文：时序数据包读取、物理单位规整与因果发布时间戳校验模块。

def step_source_quality(segment, t):
    if not segment['start_s'] <= t < t + 1 <= segment['end_s']:
        raise ValueError('Step outside original support segment')
    r = segment['source_support']
    return dict(source_is_offline_estimate=True, source_measured_exactness_claim=False, allocation='UNIFORM_ORIGINAL_COUNTER_SUPPORT', fine_profile_qualified=True, original_full_profile_segment_id=r['segment_id'], native_support_indices=r['native_indices'], modeled_release_utc=r['modeled_release_utc'], demand_is_whole_house=True, EV_added_to_total_again=False)
