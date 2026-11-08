# EN: Physical pinout and 50 MHz (20.0 ns) clock constraints for ALINX DaVinci Pro (XC7A100T-2FGG484).
# 中文：ALINX 达芬奇 Pro（XC7A100T-2FGG484）物理引脚分配与 50 MHz（20.0 ns 周期）主时钟约束。


set_property PACKAGE_PIN R4 [get_ports clk_50mhz]
set_property IOSTANDARD LVCMOS15 [get_ports clk_50mhz]
create_clock -name clk_50mhz -period 20.000 -waveform {0.000 10.000} [get_ports clk_50mhz]

set_property PACKAGE_PIN U7 [get_ports reset_n]
set_property IOSTANDARD LVCMOS15 [get_ports reset_n]
set_property PULLUP true [get_ports reset_n]

set_property PACKAGE_PIN E14 [get_ports uart_rx_i]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx_i]

set_property PACKAGE_PIN D17 [get_ports uart_tx_o]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx_o]
set_property DRIVE 8 [get_ports uart_tx_o]
set_property SLEW SLOW [get_ports uart_tx_o]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]

# EN: Asynchronous external reset and 115.2 kbaud UART I/O pads are synchronized inside RTL.
# 中文：异步复位与低速 UART 外部引脚已在 RTL 内部同步，此处声明异步伪路径。
set_false_path -from [get_ports reset_n]
set_false_path -from [get_ports uart_rx_i]
set_false_path -to [get_ports uart_tx_o]

set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
