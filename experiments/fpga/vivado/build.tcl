# EN: Non-project batch Vivado synthesis, place-and-route, timing/DRC verification, and bitstream script.
# 中文：Vivado 非工程批处理综合、布局布线、时序/DRC 设计检查与比特流生成脚本。
# EN: Usage: vivado -mode batch -source build.tcl -tclargs <fpga_root> <output_dir>
# 中文：用法：vivado -mode batch -source build.tcl -tclargs <fpga目录> <输出目录>


if {$argc != 2} {
    error "usage: vivado -mode batch -source build.tcl -tclargs <source_root> <output_dir>"
}

set source_root [file normalize [lindex $argv 0]]
set output_dir [file normalize [lindex $argv 1]]
file mkdir $output_dir

# EN: Target device: ALINX DaVinci Pro Xilinx Artix-7 XC7A100T-2FGG484 at 50 MHz.
# 中文：目标器件：ALINX 达芬奇 Pro 开发板 Xilinx Artix-7 XC7A100T-2FGG484（50 MHz 时钟）。
set part_name xc7a100tfgg484-2
set top_name ref_uart_top
set_param general.maxThreads 4

proc read_report {path} {
    set handle [open $path r]
    set contents [read $handle]
    close $handle
    return $contents
}

proc extract_count {contents pattern label} {
    if {![regexp -nocase -- $pattern $contents -> value]} {
        error "could not extract $label from generated report"
    }
    return [expr {int($value)}]
}

read_verilog -sv [file join $source_root rtl ref_filter_core.sv]
read_verilog -sv [file join $source_root rtl uart_rx.sv]
read_verilog -sv [file join $source_root rtl uart_tx.sv]
read_verilog -sv [file join $source_root rtl ref_uart_top.sv]
read_xdc [file join $source_root constraints davinci_pro_a7_100t.xdc]

synth_design -top $top_name -part $part_name -flatten_hierarchy rebuilt
write_checkpoint -force [file join $output_dir post_synth.dcp]
report_utilization -hierarchical -file [file join $output_dir utilization_post_synth.rpt]
report_timing_summary -delay_type max -report_unconstrained -max_paths 20 \
    -file [file join $output_dir timing_post_synth.rpt]
check_timing -verbose -file [file join $output_dir check_timing_post_synth.rpt]

opt_design
place_design
phys_opt_design
route_design

write_checkpoint -force [file join $output_dir post_route.dcp]
report_utilization -hierarchical -file [file join $output_dir utilization_post_route.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -max_paths 50 \
    -file [file join $output_dir timing_post_route.rpt]
report_clock_utilization -file [file join $output_dir clock_utilization.rpt]
report_route_status -file [file join $output_dir route_status.rpt]
report_drc -file [file join $output_dir drc.rpt]
report_methodology -file [file join $output_dir methodology.rpt]
check_timing -verbose -file [file join $output_dir check_timing_post_route.rpt]
report_power -file [file join $output_dir power.rpt]

set setup_path [get_timing_paths -setup -max_paths 1 -nworst 1]
if {[llength $setup_path] == 0} {
    error "no setup timing path found"
}
set hold_path [get_timing_paths -hold -max_paths 1 -nworst 1]
if {[llength $hold_path] == 0} {
    error "no hold timing path found"
}
set wns [get_property SLACK $setup_path]
set whs [get_property SLACK $hold_path]

set check_text [read_report [file join $output_dir check_timing_post_route.rpt]]
set route_text [read_report [file join $output_dir route_status.rpt]]
set drc_text [read_report [file join $output_dir drc.rpt]]
set methodology_text [read_report [file join $output_dir methodology.rpt]]
set no_clock_count [extract_count $check_text {checking no_clock \(([0-9]+)\)} no_clock]
set unconstrained_internal_count [extract_count $check_text {checking unconstrained_internal_endpoints \(([0-9]+)\)} unconstrained_internal_endpoints]
set route_error_count [extract_count $route_text {# of nets with routing errors[^\r\n]*:[[:space:]]*([0-9]+)[[:space:]]*:} route_errors]
set drc_violation_count [extract_count $drc_text {Violations found:[[:space:]]*([0-9]+)} drc_violations]
set methodology_violation_count [extract_count $methodology_text {Violations found:[[:space:]]*([0-9]+)} methodology_violations]
set lut_count [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
set ff_count [llength [get_cells -hier -filter {REF_NAME =~ FD*}]]
set bram_count [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
set dsp_count [llength [get_cells -hier -filter {REF_NAME =~ DSP*}]]

set summary [open [file join $output_dir implementation_summary.txt] w]
puts $summary "part=$part_name"
puts $summary "top=$top_name"
puts $summary "clock_mhz=50"
puts $summary "wns_ns=$wns"
puts $summary "whs_ns=$whs"
puts $summary "no_clock_count=$no_clock_count"
puts $summary "unconstrained_internal_endpoints=$unconstrained_internal_count"
puts $summary "external_io_delay_exceptions_disclosed=true"
puts $summary "route_errors=$route_error_count"
puts $summary "drc_violations=$drc_violation_count"
puts $summary "methodology_violations=$methodology_violation_count"
puts $summary "lut_primitives=$lut_count"
puts $summary "ff_primitives=$ff_count"
puts $summary "bram_primitives=$bram_count"
puts $summary "dsp_primitives=$dsp_count"
set check_failures {}
# EN: Implementation timing and DRC checks: require WNS >= 0, WHS >= 0, zero unconstrained internal endpoints, and zero DRC/route errors.
# 中文：硬件实现时序与 DRC 检查：要求建立/保持裕量非负、无未约束内部端点且路由与 DRC 零违例。
if {$wns < 0.0} {
    lappend check_failures "setup timing failed: WNS=$wns ns"
}
if {$whs < 0.0} {
    lappend check_failures "hold timing failed: WHS=$whs ns"
}
if {$no_clock_count != 0} {
    lappend check_failures "internal no-clock check failed: count=$no_clock_count"
}
if {$unconstrained_internal_count != 0} {
    lappend check_failures "unconstrained internal endpoints: count=$unconstrained_internal_count"
}
if {$route_error_count != 0} {
    lappend check_failures "route errors: count=$route_error_count"
}
if {$drc_violation_count != 0} {
    lappend check_failures "DRC violations: count=$drc_violation_count"
}
if {$methodology_violation_count != 0} {
    lappend check_failures "methodology violations: count=$methodology_violation_count"
}
set check_verdict FAIL
if {[llength $check_failures] == 0} {
    set check_verdict PASS
}
puts $summary "check_verdict=$check_verdict"
close $summary

if {[llength $check_failures] != 0} {
    puts "REF_VIVADO_FAIL [join $check_failures {; }]"
    error "implementation check failed: [join $check_failures {; }]"
}

write_bitstream -force [file join $output_dir ref_uart_top.bit]
puts "REF_VIVADO_PASS WNS_NS=$wns WHS_NS=$whs UNCONSTRAINED_INTERNAL=$unconstrained_internal_count ROUTE_ERRORS=$route_error_count DRC=$drc_violation_count METHODOLOGY=$methodology_violation_count LUT=$lut_count FF=$ff_count BRAM=$bram_count DSP=$dsp_count"
