DT = 1 / 3600.0

def step(p, e, u, b, L, pv, r, alpha):
    z = p + alpha * max(-r, min(r, u - p))
    c = min(-z, b['charge_limit_kw'], max(0.0, (b['soc_max'] * b['energy_kwh'] - e) / (b['eta_charge'] * DT))) if z < 0 else 0.0
    d = min(z, b['discharge_limit_kw'], max(0.0, b['eta_discharge'] * (e - b['soc_min'] * b['energy_kwh']) / DT)) if z >= 0 else 0.0
    d = min(d, max(L - pv + c, 0.0))
    power = d - c
    return dict(p=power, e=e + b['eta_charge'] * c * DT - d * DT / b['eta_discharge'], grid=max(L - pv - power, 0.0), ramp_override=max(abs(power - p) - r, 0.0), projection=abs(power - z), charge=c, discharge=d)
