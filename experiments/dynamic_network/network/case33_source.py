# EN: Distribution-network topology, AC power-flow coupling, and voltage-sensitivity helper.
# 中文：配电网拓扑映射、交流潮流耦合与节点电压灵敏度计算模块。

import json
import hashlib
from pathlib import Path
import numpy as np
STATIONS = ('S1_Bus', 'S2_ParkB', 'S3_Mall', 'S4_ExpwC', 'S5_Tech', 'S6_Indus', 'S7_Whole')
BUSES = dict(zip(STATIONS, (17, 32, 16, 31, 14, 30, 21)))
KIND = 'CASE33_VOLTAGE_ONLY'

def table(asset, name):
    value = json.loads(asset['_object'][name]['_object'])
    return [dict(index=int(i), **dict(zip(value['columns'], row))) for i, row in zip(value['index'], value['data'])]

def source_metadata(asset, case):
    buses = table(asset, 'bus')
    lines = table(asset, 'line')
    loads = table(asset, 'load')
    if len(buses) != 33 or len(lines) != 37 or set((r['index'] for r in loads)) != set(range(32)):
        raise ValueError('Original IEEE33 domain required')
    if case['bus_map'] != BUSES or case['station_order'] != list(STATIONS):
        raise ValueError('Frozen station mapping/order')
    if any((float(x['max_i_ka']) != 99999.0 for x in lines)):
        raise ValueError('Original unrated network source differs')
    if any((x['scaling'] != 1.0 or x['in_service'] is not True for x in loads)):
        raise ValueError('Original active unit-scale background')
    if table(asset, 'trafo') or table(asset, 'switch'):
        raise ValueError('No invented transformer/switch')
    return dict(kind=KIND, station_bus_map=dict(BUSES), native_background_loads=sorted([{k: r[k] for k in ('index', 'bus', 'p_mw', 'q_mvar')} for r in loads], key=lambda r: r['index']), base_mva=float(asset['_object']['sn_mva']), physical_bus_count=33, internal_bus_count=33, branch_count=37, current_rating_available=False, voltage_min=case['voltage_lower_pu'], voltage_max_by_bus=case['voltage_upper_by_bus'], thermal_scope='UNRATED_OPEN_99999_NOT_ENFORCED')

def dense(raw):
    a = np.zeros(raw['shape'], complex)
    if len({len(raw[k]) for k in ('row', 'col', 'real', 'imag')}) != 1:
        raise ValueError('Incomplete COO matrix')
    for i, j, re, im in zip(raw['row'], raw['col'], raw['real'], raw['imag']):
        a[i, j] += complex(re, im)
    return a

def validate_snapshot(snapshot, network, background_factor=None):
    bus = np.asarray(snapshot['bus'], float)
    if network['kind'] != KIND or snapshot['bus_lookup'] != list(range(33)) or bus.shape[0] != 33 or (len(snapshot['branch']) != 37):
        raise ValueError('IEEE33 lookup/partition')
    if snapshot['base_mva'] != network['base_mva'] or list(np.flatnonzero(bus[:, 1] == 3)) != [0] or list(np.flatnonzero(bus[:, 1] == 1)) != list(range(1, 33)):
        raise ValueError('Frozen IEEE33 base/PQ/ref')
    if snapshot['converged'] is not True:
        raise ValueError('An actual converged source snapshot is required')
    if set(snapshot['station_load_map']) != set(STATIONS) or set(snapshot['station_import_kw']) != set(STATIONS):
        raise ValueError('Complete station imports/mapping')
    loads = {r['index']: r for r in snapshot['net_load']}
    if set(loads) != set(range(39)):
        raise ValueError('All 32 background and seven station rows required')
    for j, s in enumerate(STATIONS):
        m = snapshot['station_load_map'][s]
        r = loads[32 + j]
        if m['bus'] != BUSES[s] or m['load_index'] != 32 + j or r['bus'] != BUSES[s] or (r['p_mw'] != snapshot['station_import_kw'][s] / 1000.0) or (r['q_mvar'] != 0.0):
            raise ValueError('Actual station position or kW/MW binding')
    if background_factor is not None:
        for source in network['native_background_loads']:
            r = loads[source['index']]
            if r['bus'] != source['bus'] or any((r[k] != source[k] * background_factor for k in ('p_mw', 'q_mvar'))):
                raise ValueError('Original unscaled background times declared factor')
    V = np.asarray(snapshot['V']['real']) + 1j * np.asarray(snapshot['V']['imag'])
    if V.shape != (33,) or np.any(abs(V) <= 0) or abs(abs(V[0]) - 1.0) > 1e-10:
        raise ValueError('Original slack and complete voltage')
    return (bus, V)

def assemble(snapshot, network):
    bus, V = validate_snapshot(snapshot, network)
    Y = dense(snapshot['Ybus'])
    I = Y @ V
    if Y.shape != (33, 33):
        raise ValueError('IEEE33 full Ybus')
    nodes = np.arange(1, 33)
    D = np.diag(V)
    U = np.diag(V / abs(V))
    da = 1j * D @ np.conj(np.diag(I) - Y @ D)
    dm = D @ np.conj(Y @ U) + np.diag(np.conj(I)) @ U
    J = np.block([[da[np.ix_(nodes, nodes)].real, dm[np.ix_(nodes, nodes)].real], [da[np.ix_(nodes, nodes)].imag, dm[np.ix_(nodes, nodes)].imag]])
    rhs = np.zeros((64, 8))
    base = network['base_mva']
    for j, s in enumerate(STATIONS):
        rhs[BUSES[s] - 1, j] = -1 / (1000 * base)
    for r in network['native_background_loads']:
        rhs[r['bus'] - 1, 7] -= r['p_mw'] / base
        rhs[32 + r['bus'] - 1, 7] -= r['q_mvar'] / base
    if not np.isfinite(J).all() or not np.isfinite(rhs).all():
        raise ValueError('Finite current equations')
    return (J, rhs)

def adapt_context(context, raw_asset_reference):
    if context.get('schema') != 'B15_CALENDAR_CONTEXT_V1':
        raise ValueError('Original calendar context required')
    result = dict(context, network_kind=KIND, network_model=raw_asset_reference, eight_rhs_source_network=raw_asset_reference)
    if context.get('eight_rhs_equation_contract') != 'CURRENT_7_STATION_PLUS_BACKGROUND_PQ':
        raise ValueError('Full eighth background column required')
    return result

def create_network(raw_asset, case):
    metadata = source_metadata(raw_asset, case)
    import pandapower as pp
    import pandapower.networks as networks
    from pandapower.networks.power_system_test_cases import _get_cases_path
    source = Path(_get_cases_path('case33bw.json')).resolve()
    expected = case['network_asset']
    if source != Path(expected['path']).resolve() or source.stat().st_size != expected['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest() != expected['sha256']:
        raise ValueError('Standard case33 constructor must read the same frozen asset')
    net = networks.case33bw()
    if list(net.bus.index) != list(range(33)) or list(net.line.index) != list(range(37)) or list(net.load.index) != list(range(32)):
        raise ValueError('Original sorted constructor index order required')
    base = net.load[['p_mw', 'q_mvar']].copy()
    station_loads = {s: int(pp.create_load(net, bus=BUSES[s], p_mw=0.0, q_mvar=0.0, name='network_replay_' + s)) for s in STATIONS}
    if station_loads != {s: 32 + i for i, s in enumerate(STATIONS)}:
        raise ValueError('Original station append indices')
    return (net, base, station_loads, metadata)

def validate_snapshot_fields(checked, snapshot, network):
    if not checked.get('valid'):
        raise ValueError('Valid AC power-flow snapshot required')
    _, V = validate_snapshot(snapshot, network)
    residuals = [max(network['voltage_min'] - abs(V[i]), abs(V[i]) - network['voltage_max_by_bus'][str(i)], 0.0) for i in range(1, 33)]
    return dict(AC_arithmetic_valid=True, strict_zero_voltage_residual=all((v == 0.0 for v in residuals)), voltage_within_1e10_numerical_tolerance=all((v <= 1e-10 for v in residuals)), positive_voltage_violation_bus_count=sum((v > 0.0 for v in residuals)), static_current_status='UNAVAILABLE_ORIGINAL_CASE33_99999', strict_static_current=None, required_endpoint_set='ALL_32_NONSLACK_VOLTAGE_BOTH_SIDES', network_kind=KIND)
