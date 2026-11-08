`timescale 1ns/1ps

// EN: 8N1 UART byte serializer transmitting start bit (0), 8 LSB-first data bits, and stop bit (1).
// 中文：8N1 UART 字节发送器，依次输出起始位（0）、低位优先 8 数据位与停止位（1）。


module uart_tx #(
    parameter integer CLK_FREQ_HZ = 50_000_000,
    parameter integer BAUD_RATE   = 115_200
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic [7:0] data_i,
    input  logic       start_i,
    output logic       tx_o,
    output logic       busy_o
);

    localparam integer CLKS_PER_BIT = (CLK_FREQ_HZ + BAUD_RATE/2) / BAUD_RATE;
    localparam integer COUNT_W = (CLKS_PER_BIT <= 2) ? 1 : $clog2(CLKS_PER_BIT);

    logic [9:0] shift_register;
    logic [3:0] bit_index;
    logic [COUNT_W-1:0] clock_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shift_register <= 10'h3ff;
            bit_index      <= 4'd0;
            clock_count    <= '0;
            tx_o           <= 1'b1;
            busy_o         <= 1'b0;
        end else begin
            if (!busy_o) begin
                tx_o <= 1'b1;
                if (start_i) begin
                    // EN: Load 10-bit frame {stop=1, data[7:0], start=0} and drive start bit immediately.
                    // 中文：装载 10 位发送移位帧 {停止位=1, 数据位[7:0], 起始位=0} 并立即拉低起始位。
                    shift_register <= {1'b1, data_i, 1'b0};
                    bit_index   <= 4'd0;
                    clock_count <= '0;
                    tx_o        <= 1'b0;
                    busy_o      <= 1'b1;
                end
            end else if (clock_count == CLKS_PER_BIT-1) begin
                clock_count <= '0;
                if (bit_index == 4'd9) begin
                    tx_o   <= 1'b1;
                    busy_o <= 1'b0;
                end else begin
                    bit_index      <= bit_index + 1'b1;
                    shift_register <= {1'b1, shift_register[9:1]};
                    tx_o           <= shift_register[1];
                end
            end else begin
                clock_count <= clock_count + 1'b1;
            end
        end
    end

endmodule

