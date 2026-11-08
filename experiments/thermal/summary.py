def summary(rows, common, case):
    initial = case["initial_thermal_state"]
    out = dict(completed_steps=len(rows), requested_kwh=0., delivered_service_kwh=0.,
        LV_grid_kwh=0., HV_grid_kwh=0., transformer_active_loss_kwh=0.,
        battery_discharge_kwh=0., battery_charge_kwh=0., throughput_kwh=0.,
        unmet_correction_kwh=0., free_shed_kwh=0., max_current_pu=0.,
        current_above_rated_seconds=0., current_above_1p5_seconds=0.,
        min_LV_voltage_pu=None, voltage_below_095_seconds=0.,
        hard_PCC_excess_kwh=0., cap_excess_kwh=0., final_SOC=.5,
        oil_peak_c=initial["oil_c"], hotspot_peak_c=initial["oil_c"]+initial["h1_k"]-initial["h2_k"],
        oil_exceed_seconds=0., hotspot_exceed_seconds=0.,
        oil_exceed_degree_minutes=0., hotspot_exceed_degree_minutes=0.,
        max_current_crosscheck_error=0., no_feasible_cap_steps=0,
        dynamic_peak_predictions=0, thermal_state_final=initial,
        last_thermal_exceed_end_s=None)
    for row in rows:
        a, ac, th = row["action"], row["AC"], row["thermal"]
        tr = ac["terminal"]
        k = ac["actual_current_pu"]
        out["requested_kwh"] += row["requested_kw"]/60
        out["delivered_service_kwh"] += a["delivered_service_kw"]/60
        out["LV_grid_kwh"] += a["grid_kw"]/60
        out["HV_grid_kwh"] += tr["p_hv_mw"]*1000/60
        out["transformer_active_loss_kwh"] += (tr["p_hv_mw"]+tr["p_lv_mw"])*1000/60
        for target, field in [("battery_discharge_kwh","discharge_kw"),("battery_charge_kwh","charge_kw"),
                              ("unmet_correction_kwh","unmet_correction_kw"),("free_shed_kwh","free_shed_kw")]:
            out[target] += a[field]/60
        out["throughput_kwh"] += (a["charge_kw"]+a["discharge_kw"])/60
        out["hard_PCC_excess_kwh"] += max(a["grid_kw"]-300,0.)/60
        out["cap_excess_kwh"] += max(a["grid_kw"]-row["grid_cap_kw"],0.)/60
        out["max_current_pu"] = max(out["max_current_pu"], k)
        out["current_above_rated_seconds"] += 60*int(k>1+1e-12)
        out["current_above_1p5_seconds"] += 60*int(k>1.5+1e-12)
        vm = ac["buses"][1]["vm_pu"]
        out["min_LV_voltage_pu"] = min(vm, out["min_LV_voltage_pu"] if out["min_LV_voltage_pu"] is not None else vm)
        out["voltage_below_095_seconds"] += 60*int(vm<.95)
        out["max_current_crosscheck_error"] = max(out["max_current_crosscheck_error"],ac["current_crosscheck_error"])
        out["no_feasible_cap_steps"] += int(row["cap"]["thermal_feasible"] is False)
        out["dynamic_peak_predictions"] += row["cap"]["prediction_count"]
        for kind in ("oil","hotspot"):
            stats=th["interval"]
            out[kind+"_peak_c"] = max(out[kind+"_peak_c"],stats[kind+"_peak_c"])
            for unit in ("seconds","degree_minutes"):
                out[kind+"_exceed_"+unit] += stats[kind]["exceed_"+unit]
            end=stats[kind]["last_exceed_end_offset_s"]
            if end is not None:
                out["last_thermal_exceed_end_s"]=max(out["last_thermal_exceed_end_s"] or 0,row["start_s"]+end)
        out["final_SOC"]=row["SOC_post"]
        out["thermal_state_final"]=th["thermal_post"]
    out["initial_stored_kwh"]=300.
    out["final_stored_kwh"]=600*out["final_SOC"]
    out["battery_energy_residual_kwh"]=(out["final_stored_kwh"]-300
        -.95*out["battery_charge_kwh"]+out["battery_discharge_kwh"]/.95)
    out["operating_cost_yuan"]=.6*out["HV_grid_kwh"]+.08*out["throughput_kwh"]
    for value in (.57,.60):
        out["inventory_adjusted_cost_"+format(value,".2f")+"_yuan"]=out["operating_cost_yuan"]+value*(300-out["final_stored_kwh"])
    for kind,limit in (("oil",105.),("hotspot",120.)):
        out[kind+"_positive_peak_residual_c"]=max(out[kind+"_peak_c"]-limit,0.)
    last=out["last_thermal_exceed_end_s"]
    out["thermal_recovery_status"]=("UNKNOWN_INCOMPLETE" if len(rows)!=360 else
        "NONE_REQUIRED" if last is None else "RIGHT_CENSORED" if last>=21600-1e-9
        else "RECOVERED_WITHIN_RECORDED_WINDOW")
    out["thermal_recovery_after_event_seconds"]=(max(0.,last-case["event_end_s"])
        if out["thermal_recovery_status"]=="RECOVERED_WITHIN_RECORDED_WINDOW" else
        0. if out["thermal_recovery_status"]=="NONE_REQUIRED" else None)
    return out
