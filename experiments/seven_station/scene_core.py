from dataclasses import replace

def core(scale):
    from station_core.config import ExperimentConfig
    c = ExperimentConfig()
    return replace(c, battery=replace(c.battery, energy_kwh=c.battery.energy_kwh * scale, charge_limit_kw=c.battery.charge_limit_kw * scale, discharge_limit_kw=c.battery.discharge_limit_kw * scale), feeder_safe_kw=c.feeder_safe_kw * scale, pv_rated_kw=c.pv_rated_kw * scale)
