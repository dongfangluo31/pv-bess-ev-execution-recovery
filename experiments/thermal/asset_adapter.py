import math
import time

EXPECTED = dict(sn_mva=.25, vn_hv_kv=10., vn_lv_kv=.4, vk_percent=4.,
                vkr_percent=1.2, pfe_kw=.6, i0_percent=.24, shift_degree=150.,
                vector_group="Dyn5", tap_neutral=0, tap_side="hv")


def make_net():
    import pandapower as pp
    net = pp.create_empty_network(sn_mva=1.)
    hv = pp.create_bus(net, vn_kv=10., name="INDEPENDENT_HV")
    lv = pp.create_bus(net, vn_kv=.4, name="INDEPENDENT_LV")
    pp.create_ext_grid(net, hv, vm_pu=1., va_degree=0.)
    pp.create_transformer(net, hv, lv, std_type="0.25 MVA 10/0.4 kV", tap_pos=0)
    for k, v in EXPECTED.items():
        if net.trafo.iloc[0][k] != v:
            raise ValueError("Installed standard type differs: "+k)
    pp.create_load(net, lv, p_mw=0., q_mvar=0., name="FIXED_SERVICE")
    pp.create_sgen(net, lv, p_mw=0., q_mvar=0., name="BESS_IDEAL_60S")
    if len(net.bus) != 2 or len(net.line) != 0 or len(net.trafo) != 1:
        raise ValueError("Electrical topology is not the frozen two-bus asset")
    return net


def complex_values(x):
    return [[float(complex(v).real), float(complex(v).imag)] for v in x]


def dense_complex_matrix(x):
    array = x.toarray() if hasattr(x, "toarray") else x
    return [complex_values(row) for row in array]


def run_ac(net, requested_kw, charge_kw, discharge_kw):
    import pandapower as pp
    net.load.at[0, "p_mw"] = requested_kw/1000
    net.load.at[0, "q_mvar"] = 0.
    net.sgen.at[0, "p_mw"] = (discharge_kw-charge_kw)/1000
    net.sgen.at[0, "q_mvar"] = 0.
    if list(net.bus.index) != [0,1] or net.bus.at[0,"vn_kv"] != 10. or net.bus.at[1,"vn_kv"] != .4:
        raise ValueError("Explicit voltage seed requires frozen HV0/LV1 bus indices")
    initial_vm_pu = [1.,1.]
    initial_va_degree = [0.,-150.]
    begun = time.perf_counter()
    pp.runpp(net, algorithm="nr", calculate_voltage_angles=True, init="auto",
             init_vm_pu=initial_vm_pu, init_va_degree=initial_va_degree,
             max_iteration=100, tolerance_mva=1e-10, trafo_model="t",
             trafo_loading="current", check_connectivity=True, numba=False,
             enforce_q_lims=False, voltage_depend_loads=False)
    if not net.converged:
        raise RuntimeError("Power flow returned without convergence")
    tr = {str(k): float(v) for k, v in net.res_trafo.iloc[0].items()}
    buses = [{str(k): float(v) for k, v in row.items()} for _, row in net.res_bus.iterrows()]
    ppc = net._ppc
    inner = ppc["internal"]
    v = inner["V"]
    irated = .25/(math.sqrt(3)*.4)
    lv_v = float(net.res_bus.at[1, "vm_pu"])*.4
    k_actual = math.hypot(tr["p_lv_mw"], tr["q_lv_mvar"])/(math.sqrt(3)*lv_v*irated)
    via_report = tr["i_lv_ka"]/irated
    finite = list(tr.values()) + [x for row in buses for x in row.values()]
    if not all(math.isfinite(x) for x in finite) or lv_v <= 0:
        raise ValueError("Nonfinite/nonphysical AC result")
    if abs(k_actual-via_report) > 1e-8:
        raise ValueError("LV same-terminal power/current mismatch")
    return dict(status="CONVERGED", wall_seconds=time.perf_counter()-begun,
                initial_bus_indices=[0,1],initial_vm_pu=initial_vm_pu,
                initial_va_degree=initial_va_degree,initialization_requires_pre_solve=False,
                iterations=int(ppc.get("iterations", -1)),
                terminal=tr, buses=buses, actual_lv_voltage_ll_kv=lv_v,
                rated_lv_current_ka=irated, actual_current_pu=k_actual,
                res_trafo_current_pu=via_report, current_crosscheck_error=abs(k_actual-via_report),
                baseMVA=float(ppc["baseMVA"]), complex_V=complex_values(v),
                Ybus=dense_complex_matrix(inner["Ybus"]),
                Yf=dense_complex_matrix(inner["Yf"]), Yt=dense_complex_matrix(inner["Yt"]),
                branch=[[float(complex(x).real) for x in row] for row in ppc["branch"]],
                ppc_bus=[[float(x) for x in row] for row in ppc["bus"]],
                bus_lookup=[int(x) for x in net._pd2ppc_lookups["bus"]],
                low_voltage_below_095=bool(net.res_bus.at[1, "vm_pu"] < .95),
                low_voltage_scope="Recorded condition; no additional hidden load shedding")
