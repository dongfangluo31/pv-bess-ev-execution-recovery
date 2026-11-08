`timescale 1ns/1ps

// EN: Top-level UART-framed HIL wrapper around ref_filter_core for Xilinx Artix-7 (50 MHz, 115200 8N1).
// 中文：面向 Xilinx Artix-7 的 ref_filter_core 顶层串口硬件在环（HIL）封装（50 MHz 主频，115200 8N1）。
// EN: Frame header 0x55 0xAA, version 0x01, 24-byte request (type 0x10) -> 32-byte response (type 0x90) with CRC-16/CCITT-FALSE.
// 中文：帧头 0x55 0xAA，协议版本 0x01，24 字节请求载荷（0x10）对应 32 字节响应载荷（0x90），带 CRC-16/CCITT-FALSE 校验。


module ref_uart_top #(
    parameter integer CLK_FREQ_HZ = 50_000_000,
    parameter integer BAUD_RATE   = 115_200
) (
    input  logic clk_50mhz,
    input  logic reset_n,
    input  logic uart_rx_i,
    output logic uart_tx_o
);

    localparam integer CLKS_PER_BIT = (CLK_FREQ_HZ + BAUD_RATE/2) / BAUD_RATE;
    localparam integer RX_TIMEOUT_CYCLES = CLKS_PER_BIT * 20;
    localparam integer TIMEOUT_W = $clog2(RX_TIMEOUT_CYCLES + 1);

    localparam logic [7:0] PROTOCOL_VERSION = 8'h01;
    localparam logic [7:0] REQUEST_TYPE     = 8'h10;
    localparam logic [7:0] RESPONSE_TYPE    = 8'h90;
    localparam logic [15:0] REQUEST_LENGTH  = 16'd24;
    localparam logic [15:0] RESPONSE_LENGTH = 16'd32;

    // EN: Byte-wise CRC-16/CCITT-FALSE update (polynomial 0x1021, initial value 0xFFFF).
    // 中文：逐字节更新 CRC-16/CCITT-FALSE 校验值（生成多项式 0x1021，初值 0xFFFF）。
    function automatic logic [15:0] crc16_next(
        input logic [15:0] crc_in,
        input logic [7:0] data
    );
        logic [15:0] crc_value;
        integer i;
        begin
            crc_value = crc_in ^ {data, 8'h00};
            for (i = 0; i < 8; i = i + 1) begin
                if (crc_value[15])
                    crc_value = {crc_value[14:0], 1'b0} ^ 16'h1021;
                else
                    crc_value = {crc_value[14:0], 1'b0};
            end
            crc16_next = crc_value;
        end
    endfunction

    logic [7:0] rx_data;
    logic       rx_data_valid;
    logic       rx_framing_error;
    logic [7:0] tx_data;
    logic       tx_start;
    logic       tx_busy;

    uart_rx #(
        .CLK_FREQ_HZ(CLK_FREQ_HZ),
        .BAUD_RATE(BAUD_RATE)
    ) receiver (
        .clk(clk_50mhz),
        .rst_n(reset_n),
        .rx_i(uart_rx_i),
        .data_o(rx_data),
        .data_valid_o(rx_data_valid),
        .framing_error_o(rx_framing_error)
    );

    uart_tx #(
        .CLK_FREQ_HZ(CLK_FREQ_HZ),
        .BAUD_RATE(BAUD_RATE)
    ) transmitter (
        .clk(clk_50mhz),
        .rst_n(reset_n),
        .data_i(tx_data),
        .start_i(tx_start),
        .tx_o(uart_tx_o),
        .busy_o(tx_busy)
    );

    typedef enum logic [3:0] {
        P_WAIT_55,
        P_WAIT_AA,
        P_VERSION,
        P_TYPE,
        P_LENGTH_0,
        P_LENGTH_1,
        P_SEQUENCE_0,
        P_SEQUENCE_1,
        P_SEQUENCE_2,
        P_SEQUENCE_3,
        P_PAYLOAD,
        P_CRC_0,
        P_CRC_1
    } parser_state_t;

    parser_state_t parser_state;
    logic [TIMEOUT_W-1:0] timeout_counter;
    logic [7:0] request_version;
    logic [7:0] request_type;
    logic [15:0] request_length;
    logic [31:0] request_sequence;
    logic [15:0] request_crc;
    logic [7:0] request_crc_low;
    logic [15:0] payload_index;
    logic [7:0] request_payload [0:23];
    logic launch_pending;

    logic core_in_valid;
    logic signed [31:0] core_nominal_import;
    logic signed [31:0] core_nominal_charge;
    logic signed [31:0] core_nominal_discharge;
    logic signed [31:0] core_max_reduce_charge;
    logic signed [31:0] core_max_add_discharge;
    logic signed [31:0] core_feeder_limit;
    logic [31:0] core_sequence;

    always_ff @(posedge clk_50mhz or negedge reset_n) begin
        if (!reset_n) begin
            parser_state             <= P_WAIT_55;
            timeout_counter          <= '0;
            request_version          <= 8'd0;
            request_type             <= 8'd0;
            request_length           <= 16'd0;
            request_sequence         <= 32'd0;
            request_crc              <= 16'hffff;
            request_crc_low          <= 8'd0;
            payload_index            <= 16'd0;
            launch_pending           <= 1'b0;
            core_in_valid            <= 1'b0;
            core_nominal_import      <= 32'sd0;
            core_nominal_charge      <= 32'sd0;
            core_nominal_discharge   <= 32'sd0;
            core_max_reduce_charge   <= 32'sd0;
            core_max_add_discharge   <= 32'sd0;
            core_feeder_limit        <= 32'sd0;
            core_sequence            <= 32'd0;
        end else begin
            core_in_valid <= 1'b0;

            // EN: Unpack six little-endian int32 Q16.16 words (import, charge, discharge, reduce_cap, add_cap, limit) into core.
            // 中文：将 6 个小端序 int32 Q16.16 参数解包送入修正核并拉高单拍 core_in_valid。
            if (launch_pending) begin
                core_nominal_import    <= $signed({request_payload[3], request_payload[2], request_payload[1], request_payload[0]});
                core_nominal_charge    <= $signed({request_payload[7], request_payload[6], request_payload[5], request_payload[4]});
                core_nominal_discharge <= $signed({request_payload[11], request_payload[10], request_payload[9], request_payload[8]});
                core_max_reduce_charge <= $signed({request_payload[15], request_payload[14], request_payload[13], request_payload[12]});
                core_max_add_discharge <= $signed({request_payload[19], request_payload[18], request_payload[17], request_payload[16]});
                core_feeder_limit      <= $signed({request_payload[23], request_payload[22], request_payload[21], request_payload[20]});
                core_sequence          <= request_sequence;
                core_in_valid          <= 1'b1;
                launch_pending         <= 1'b0;
            end

            if (rx_framing_error) begin
                parser_state    <= P_WAIT_55;
                timeout_counter <= '0;
            end else if (rx_data_valid) begin
                timeout_counter <= '0;
                case (parser_state)
                    P_WAIT_55: begin
                        if (rx_data == 8'h55)
                            parser_state <= P_WAIT_AA;
                    end

                    P_WAIT_AA: begin
                        if (rx_data == 8'haa)
                            parser_state <= P_VERSION;
                        else if (rx_data != 8'h55)
                            parser_state <= P_WAIT_55;
                    end

                    P_VERSION: begin
                        request_version <= rx_data;
                        request_crc     <= crc16_next(16'hffff, rx_data);
                        parser_state    <= P_TYPE;
                    end

                    P_TYPE: begin
                        request_type <= rx_data;
                        request_crc  <= crc16_next(request_crc, rx_data);
                        parser_state <= P_LENGTH_0;
                    end

                    P_LENGTH_0: begin
                        request_length[7:0] <= rx_data;
                        request_crc <= crc16_next(request_crc, rx_data);
                        parser_state <= P_LENGTH_1;
                    end

                    P_LENGTH_1: begin
                        request_length[15:8] <= rx_data;
                        request_crc <= crc16_next(request_crc, rx_data);
                        parser_state <= P_SEQUENCE_0;
                    end

                    P_SEQUENCE_0: begin
                        request_sequence[7:0] <= rx_data;
                        request_crc <= crc16_next(request_crc, rx_data);
                        parser_state <= P_SEQUENCE_1;
                    end

                    P_SEQUENCE_1: begin
                        request_sequence[15:8] <= rx_data;
                        request_crc <= crc16_next(request_crc, rx_data);
                        parser_state <= P_SEQUENCE_2;
                    end

                    P_SEQUENCE_2: begin
                        request_sequence[23:16] <= rx_data;
                        request_crc <= crc16_next(request_crc, rx_data);
                        parser_state <= P_SEQUENCE_3;
                    end

                    P_SEQUENCE_3: begin
                        request_sequence[31:24] <= rx_data;
                        request_crc <= crc16_next(request_crc, rx_data);
                        payload_index <= 16'd0;
                        parser_state <= (request_length == 16'd0) ? P_CRC_0 : P_PAYLOAD;
                    end

                    P_PAYLOAD: begin
                        if (payload_index < REQUEST_LENGTH)
                            request_payload[payload_index] <= rx_data;
                        request_crc <= crc16_next(request_crc, rx_data);
                        if (payload_index + 1'b1 >= request_length)
                            parser_state <= P_CRC_0;
                        else
                            payload_index <= payload_index + 1'b1;
                    end

                    P_CRC_0: begin
                        request_crc_low <= rx_data;
                        parser_state <= P_CRC_1;
                    end

                    P_CRC_1: begin
                        if ((request_crc == {rx_data, request_crc_low}) &&
                            (request_version == PROTOCOL_VERSION) &&
                            (request_type == REQUEST_TYPE) &&
                            (request_length == REQUEST_LENGTH))
                            launch_pending <= 1'b1;
                        parser_state <= P_WAIT_55;
                    end

                    default: parser_state <= P_WAIT_55;
                endcase
            end else if (parser_state != P_WAIT_55) begin
                if (timeout_counter == RX_TIMEOUT_CYCLES-1) begin
                    parser_state    <= P_WAIT_55;
                    timeout_counter <= '0;
                end else begin
                    timeout_counter <= timeout_counter + 1'b1;
                end
            end else begin
                timeout_counter <= '0;
            end
        end
    end

    logic core_out_valid;
    logic signed [31:0] core_reduce_charge;
    logic signed [31:0] core_add_discharge;
    logic signed [31:0] core_executed_charge;
    logic signed [31:0] core_executed_discharge;
    logic signed [31:0] core_executed_import;
    logic signed [31:0] core_remaining_overlimit;
    logic [31:0] core_status;

    ref_filter_core filter_core (
        .clk(clk_50mhz),
        .rst_n(reset_n),
        .in_valid(core_in_valid),
        .nominal_import_q(core_nominal_import),
        .nominal_charge_q(core_nominal_charge),
        .nominal_discharge_q(core_nominal_discharge),
        .max_reduce_charge_q(core_max_reduce_charge),
        .max_add_discharge_q(core_max_add_discharge),
        .feeder_limit_q(core_feeder_limit),
        .out_valid(core_out_valid),
        .reduce_charge_q(core_reduce_charge),
        .add_discharge_q(core_add_discharge),
        .executed_charge_q(core_executed_charge),
        .executed_discharge_q(core_executed_discharge),
        .executed_import_q(core_executed_import),
        .remaining_overlimit_q(core_remaining_overlimit),
        .status(core_status)
    );

    logic [31:0] response_sequence;
    logic [31:0] response_reduce_charge;
    logic [31:0] response_add_discharge;
    logic [31:0] response_executed_charge;
    logic [31:0] response_executed_discharge;
    logic [31:0] response_executed_import;
    logic [31:0] response_remaining_overlimit;
    logic [31:0] response_status;
    logic [31:0] response_latency;
    logic [15:0] response_crc;

    function automatic logic [7:0] word_byte(
        input logic [31:0] word,
        input logic [1:0] byte_index
    );
        begin
            case (byte_index)
                2'd0: word_byte = word[7:0];
                2'd1: word_byte = word[15:8];
                2'd2: word_byte = word[23:16];
                default: word_byte = word[31:24];
            endcase
        end
    endfunction

    function automatic logic [7:0] response_byte(input logic [5:0] index);
        begin
            case (index)
                6'd0: response_byte = 8'h55;
                6'd1: response_byte = 8'haa;
                6'd2: response_byte = PROTOCOL_VERSION;
                6'd3: response_byte = RESPONSE_TYPE;
                6'd4: response_byte = RESPONSE_LENGTH[7:0];
                6'd5: response_byte = RESPONSE_LENGTH[15:8];
                6'd6, 6'd7, 6'd8, 6'd9:
                    response_byte = word_byte(response_sequence, index - 6'd6);
                6'd10, 6'd11, 6'd12, 6'd13:
                    response_byte = word_byte(response_reduce_charge, index - 6'd10);
                6'd14, 6'd15, 6'd16, 6'd17:
                    response_byte = word_byte(response_add_discharge, index - 6'd14);
                6'd18, 6'd19, 6'd20, 6'd21:
                    response_byte = word_byte(response_executed_charge, index - 6'd18);
                6'd22, 6'd23, 6'd24, 6'd25:
                    response_byte = word_byte(response_executed_discharge, index - 6'd22);
                6'd26, 6'd27, 6'd28, 6'd29:
                    response_byte = word_byte(response_executed_import, index - 6'd26);
                6'd30, 6'd31, 6'd32, 6'd33:
                    response_byte = word_byte(response_remaining_overlimit, index - 6'd30);
                6'd34, 6'd35, 6'd36, 6'd37:
                    response_byte = word_byte(response_status, index - 6'd34);
                6'd38, 6'd39, 6'd40, 6'd41:
                    response_byte = word_byte(response_latency, index - 6'd38);
                6'd42: response_byte = response_crc[7:0];
                6'd43: response_byte = response_crc[15:8];
                default: response_byte = 8'd0;
            endcase
        end
    endfunction

    typedef enum logic [1:0] {TX_READY, TX_WAIT_BUSY, TX_WAIT_DONE} tx_state_t;
    tx_state_t tx_state;
    logic tx_active;
    logic [5:0] tx_index;

    always_ff @(posedge clk_50mhz or negedge reset_n) begin
        if (!reset_n) begin
            response_sequence            <= 32'd0;
            response_reduce_charge       <= 32'd0;
            response_add_discharge       <= 32'd0;
            response_executed_charge     <= 32'd0;
            response_executed_discharge  <= 32'd0;
            response_executed_import     <= 32'd0;
            response_remaining_overlimit <= 32'd0;
            response_status              <= 32'd0;
            response_latency             <= 32'd2;
            response_crc                 <= 16'hffff;
            tx_data                      <= 8'd0;
            tx_start                     <= 1'b0;
            tx_active                    <= 1'b0;
            tx_index                     <= 6'd0;
            tx_state                     <= TX_READY;
        end else begin
            tx_start <= 1'b0;

            if (core_out_valid && !tx_active) begin
                response_sequence            <= core_sequence;
                response_reduce_charge       <= core_reduce_charge;
                response_add_discharge       <= core_add_discharge;
                response_executed_charge     <= core_executed_charge;
                response_executed_discharge  <= core_executed_discharge;
                response_executed_import     <= core_executed_import;
                response_remaining_overlimit <= core_remaining_overlimit;
                response_status              <= core_status;
                response_latency             <= 32'd2;
                response_crc                 <= 16'hffff;
                tx_index                     <= 6'd0;
                tx_state                     <= TX_READY;
                tx_active                    <= 1'b1;
            end else if (tx_active) begin
                case (tx_state)
                    TX_READY: begin
                        if (!tx_busy) begin
                            tx_data  <= response_byte(tx_index);
                            tx_start <= 1'b1;
                            if ((tx_index >= 6'd2) && (tx_index <= 6'd41))
                                response_crc <= crc16_next(response_crc, response_byte(tx_index));
                            tx_state <= TX_WAIT_BUSY;
                        end
                    end

                    TX_WAIT_BUSY: begin
                        if (tx_busy)
                            tx_state <= TX_WAIT_DONE;
                    end

                    TX_WAIT_DONE: begin
                        if (!tx_busy) begin
                            if (tx_index == 6'd43) begin
                                tx_active <= 1'b0;
                                tx_state  <= TX_READY;
                            end else begin
                                tx_index <= tx_index + 1'b1;
                                tx_state <= TX_READY;
                            end
                        end
                    end

                    default: tx_state <= TX_READY;
                endcase
            end
        end
    end

endmodule
