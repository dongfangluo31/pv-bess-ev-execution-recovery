`timescale 1ns/1ps

// EN: 8N1 UART byte receiver with 2-FF asynchronous input synchronizer and mid-bit sampling.
// 中文：带两级触发器异步输入同步与半位中心采样的 8N1 UART 字节接收器。


module uart_rx #(
    parameter integer CLK_FREQ_HZ = 50_000_000,
    parameter integer BAUD_RATE   = 115_200
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       rx_i,
    output logic [7:0] data_o,
    output logic       data_valid_o,
    output logic       framing_error_o
);

    localparam integer CLKS_PER_BIT = (CLK_FREQ_HZ + BAUD_RATE/2) / BAUD_RATE;
    localparam integer COUNT_W = (CLKS_PER_BIT <= 2) ? 1 : $clog2(CLKS_PER_BIT);

    typedef enum logic [1:0] {IDLE, START_BIT, DATA_BITS, STOP_BIT} state_t;
    state_t state;

    // EN: Two-stage synchronizer prevents metastability on external UART RX pin.
    // 中文：双触发器同步链消除外部串口 RX 引脚跨时钟域亚稳态。
    (* ASYNC_REG = "TRUE" *) logic sync_ff1;
    (* ASYNC_REG = "TRUE" *) logic sync_ff2;
    logic [COUNT_W-1:0] clock_count;
    logic [2:0] bit_index;
    logic [7:0] shift_register;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sync_ff1 <= 1'b1;
            sync_ff2 <= 1'b1;
        end else begin
            sync_ff1 <= rx_i;
            sync_ff2 <= sync_ff1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= IDLE;
            clock_count     <= '0;
            bit_index       <= 3'd0;
            shift_register  <= 8'd0;
            data_o          <= 8'd0;
            data_valid_o    <= 1'b0;
            framing_error_o <= 1'b0;
        end else begin
            data_valid_o    <= 1'b0;
            framing_error_o <= 1'b0;
            case (state)
                IDLE: begin
                    clock_count <= '0;
                    if (!sync_ff2)
                        state <= START_BIT;
                end

                START_BIT: begin
                    if (clock_count == (CLKS_PER_BIT/2)-1) begin
                        clock_count <= '0;
                        if (!sync_ff2) begin
                            bit_index <= 3'd0;
                            state <= DATA_BITS;
                        end else begin
                            state <= IDLE;
                        end
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                DATA_BITS: begin
                    if (clock_count == CLKS_PER_BIT-1) begin
                        clock_count <= '0;
                        shift_register[bit_index] <= sync_ff2;
                        if (bit_index == 3'd7)
                            state <= STOP_BIT;
                        else
                            bit_index <= bit_index + 1'b1;
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                STOP_BIT: begin
                    if (clock_count == CLKS_PER_BIT-1) begin
                        clock_count <= '0;
                        if (sync_ff2) begin
                            data_o       <= shift_register;
                            data_valid_o <= 1'b1;
                        end else begin
                            framing_error_o <= 1'b1;
                        end
                        state <= IDLE;
                    end else begin
                        clock_count <= clock_count + 1'b1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule

