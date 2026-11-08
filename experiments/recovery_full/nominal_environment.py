import math

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
