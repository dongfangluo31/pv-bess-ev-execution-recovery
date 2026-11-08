from copy import deepcopy
import time
import warnings
from snapshot_contract import require, safe, PF_OPTIONS, LP_OPTIONS, STATIONS, BUS_MAP, POWER_TOL, VOLTAGE_TOL
from scene_io import write, record

def make_callbacks(rd, state, attempt, run_id, max_ac_calls=34, max_lp_calls=34):
    import numpy as np
    import pandapower as pp
    import scipy.optimize as opt
    RUN_ID = run_id
    real_pf, real_lp = (pp.runpp, opt.linprog)

    def forbidden(*args, **kwargs):
        state['forbidden_optimizer_attempts'] += 1
        raise RuntimeError('Unaccounted optimizer/MILP entry forbidden')
    opt.milp = forbidden
    opt.minimize = forbidden
    pp.runpp = forbidden
    opt.linprog = forbidden

    def real_array(value):
        arr = np.asarray(value)
        if np.iscomplexobj(arr):
            require(np.all(arr.imag == 0), 'Unexpected complex field in raw real-valued MATPOWER table')
            arr = arr.real
        return arr.tolist()

    def complex_array(value):
        value = np.asarray(value, dtype=complex)
        return dict(real=value.real.tolist(), imag=value.imag.tolist())

    def ac_call(net, label, imports, station_loads):
        require(state['ac'] < max_ac_calls, 'AC hard call cap reached before invocation')
        state['ac'] += 1
        cid = state['ac']
        context = dict(**state['context'], label=label, station_import_kw=imports, options=PF_OPTIONS)
        attempt('AC', cid, context)
        error = None
        snapshot = None
        failure_state = None
        tick = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                real_pf(net, **deepcopy(PF_OPTIONS))
                write(rd / 'ac' / f'{cid:03d}_RETURN_KNOWN.json', dict(run_id=RUN_ID, call_id=cid, context=context, backend_returned=True, backend_seconds=time.perf_counter() - tick, snapshot_state='PENDING'))
                require(bool(net.converged), 'AC did not converge')
                ppc = net._ppc
                internal = ppc['internal']
                y = internal['Ybus'].tocoo()
                require(y.shape == (33, 33) and len(ppc['bus']) == 33, 'Unexpected active-bus indexing')
                volts = np.asarray(internal['V'], dtype=complex)
                require(len(volts) == 33, 'Missing complete complex voltage vector')
                bus = np.asarray(real_array(ppc['bus']), dtype=float)
                gen = np.asarray(real_array(ppc['gen']), dtype=float)
                specified = -(bus[:, 2] + 1j * bus[:, 3]) / float(ppc['baseMVA'])
                for row in gen:
                    if row[7] > 0:
                        specified[int(row[0])] += (row[1] + 1j * row[2]) / float(ppc['baseMVA'])
                calculated = volts * np.conj(internal['Ybus'] @ volts)
                line_start, line_end = net._pd2ppc_lookups['branch']['line']
                require(line_start == 0 and line_end == len(net.line) == len(ppc['branch']), 'Nontrivial line/branch mapping must not be silently dropped')
                snapshot = dict(base_mva=float(ppc['baseMVA']), bus=real_array(bus), gen=real_array(gen), branch=real_array(ppc['branch']), Ybus=dict(shape=list(y.shape), row=y.row.tolist(), col=y.col.tolist(), real=y.data.real.tolist(), imag=y.data.imag.tolist()), V=complex_array(volts), S_spec=complex_array(specified), S_calculated=complex_array(calculated), power_balance_residual_mva=complex_array((calculated - specified) * float(ppc['baseMVA'])), res_bus=[dict(bus=int(i), vm_pu=float(r['vm_pu']), va_degree=float(r['va_degree'])) for i, r in net.res_bus.iterrows()], station_import_kw=imports, station_load_map={profile: dict(load_index=int(station_loads[profile]), bus=BUS_MAP[profile]) for profile in STATIONS}, net_load=[dict(index=int(i), bus=int(r['bus']), p_mw=float(r['p_mw']), q_mvar=float(r['q_mvar']), in_service=bool(r['in_service']), scaling=float(r['scaling'])) for i, r in net.load.iterrows()], res_line=[dict(index=int(i), **{key: float(r[key]) for key in ('p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'i_from_ka', 'i_to_ka', 'i_ka', 'loading_percent')}) for i, r in net.res_line.iterrows()], line_current=[dict(index=int(line_start + i), i_ka=float(net.res_line.loc[index, 'i_ka'])) for i, index in enumerate(net.line.index)], line_to_ppc_branch=[dict(index=int(line_start + i), line_index=int(index), in_service=bool(net.line.loc[index, 'in_service'])) for i, index in enumerate(net.line.index)], bus_lookup=real_array(net._pd2ppc_lookups['bus']), converged=bool(net.converged))
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc))
                ppc = getattr(net, '_ppc', None)
                if isinstance(ppc, dict):
                    try:
                        failure_state = {k: real_array(ppc[k]) if k in ('bus', 'gen', 'branch') else safe(ppc.get(k)) for k in ('baseMVA', 'bus', 'gen', 'branch', 'success', 'iterations') if k in ppc}
                        current = ppc.get('internal', {})
                        if current.get('V') is not None:
                            failure_state['V'] = complex_array(current['V'])
                    except Exception as capture_error:
                        failure_state = dict(capture_error_type=type(capture_error).__name__, capture_error=str(capture_error), original_exception=error)
        elapsed = time.perf_counter() - tick
        raw_path = rd / 'ac' / f'{cid:03d}.json'
        write(raw_path, dict(run_id=RUN_ID, call_id=cid, context=context, solver_wall_seconds=elapsed, warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught], exception=error, snapshot=snapshot, raw_failure_state=failure_state))
        require(error is None and snapshot is not None, 'AC call or raw snapshot failed: ' + str(error))
        residual = snapshot['power_balance_residual_mva']
        mismatch = np.hypot(residual['real'], residual['imag'])
        require(np.all(np.isfinite(mismatch)) and np.max(mismatch) <= 1e-8, 'AC power-balance residual')
        return (snapshot, None, record(raw_path))

    def lp_call(label, active, matrix, rhs, caps):
        require(state['lp'] < max_lp_calls, 'LP hard call cap reached before invocation')
        state['lp'] += 1
        cid = state['lp']
        context = dict(**state['context'], label=label, active_row_indices=[int(v) for v in active], method='highs-ds', options=LP_OPTIONS)
        attempt('LP', cid, context)
        snapshot = dict(c=[1.0] * 7, A_ub=(-matrix[active]).tolist(), b_ub=rhs[active].tolist(), lower_bounds=[0.0] * 7, upper_bounds=caps.tolist(), method='highs-ds', options_before=deepcopy(LP_OPTIONS))
        error = None
        result = None
        tick = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                result = real_lp(np.ones(7), A_ub=-matrix[active], b_ub=rhs[active], bounds=[(0.0, float(v)) for v in caps], method='highs-ds', options=deepcopy(LP_OPTIONS))
                write(rd / 'lp' / f'{cid:03d}_RETURN_KNOWN.json', dict(run_id=RUN_ID, call_id=cid, context=context, backend_returned=True, backend_seconds=time.perf_counter() - tick, snapshot_state='PENDING'))
                snapshot['result'] = {key: getattr(result, key, None) for key in ('x', 'fun', 'status', 'success', 'message', 'nit', 'ineqlin', 'lower', 'upper')}
            except Exception as exc:
                error = dict(type=type(exc).__name__, message=str(exc))
                snapshot['result'] = None
        elapsed = time.perf_counter() - tick
        raw_path = rd / 'lp' / f'{cid:03d}.json'
        write(raw_path, dict(run_id=RUN_ID, call_id=cid, context=context, solver_wall_seconds=elapsed, snapshot=snapshot, warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught], exception=error))
        require(error is None and result is not None, 'LP exception retained: ' + str(error))
        require(int(result.status) == 0 and result.x is not None, 'Projection LP has no accepted optimum')
        x = np.asarray(result.x, dtype=float)
        require(np.isfinite(x).all() and np.min(x) >= -POWER_TOL and np.max(x - caps) <= POWER_TOL, 'Projection bounds')
        if len(snapshot['b_ub']):
            require(np.max(np.asarray(snapshot['A_ub']) @ x - np.asarray(snapshot['b_ub'])) <= VOLTAGE_TOL, 'Projection voltage residual')
        return (x, None, record(raw_path))
    return (ac_call, lp_call)
