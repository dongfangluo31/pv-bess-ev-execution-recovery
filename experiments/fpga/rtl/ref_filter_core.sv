`timescale 1ns/1ps

// EN: Two-stage pipelined Q16.16 fixed-point reserve-bounded execution filter (REF) core (50 MHz, 2-cycle latency).
// 中文：两级流水线 Q16.16 定点备用有界执行滤波器（REF）硬件核（50 MHz 时钟，固定 2 周期延迟）。
// EN: Power inputs/outputs use signed 32-bit Q16.16 kW (1 LSB = 2^-16 kW); charge/discharge/caps are non-negative.
// 中文：功率输入/输出采用有符号 32 位 Q16.16 千瓦格式（1 LSB = 2^-16 kW）；充放电与能力上限均为非负量。


module ref_filter_core (
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic                     in_valid,
    input  logic signed [31:0]       nominal_import_q,
    input  logic signed [31:0]       nominal_charge_q,
    input  logic signed [31:0]       nominal_discharge_q,
    input  logic signed [31:0]       max_reduce_charge_q,
    input  logic signed [31:0]       max_add_discharge_q,
    input  logic signed [31:0]       feeder_limit_q,
    output logic                     out_valid,
    output logic signed [31:0]       reduce_charge_q,
    output logic signed [31:0]       add_discharge_q,
    output logic signed [31:0]       executed_charge_q,
    output logic signed [31:0]       executed_discharge_q,
    output logic signed [31:0]       executed_import_q,
    output logic signed [31:0]       remaining_overlimit_q,
    output logic        [31:0]       status
);

    localparam logic signed [32:0] S32_MAX_W33 = 33'sd2147483647;
    localparam logic signed [32:0] S32_MIN_W33 = -33'sd2147483648;
    localparam logic signed [33:0] S32_MAX_W34 = 34'sd2147483647;
    localparam logic signed [33:0] S32_MIN_W34 = -34'sd2147483648;

    localparam integer STATUS_EXCESS              = 0;
    localparam integer STATUS_REDUCED_CHARGE      = 1;
    localparam integer STATUS_ADDED_DISCHARGE     = 2;
    localparam integer STATUS_REMAINING_OVERLIMIT = 3;
    localparam integer STATUS_INVALID_INPUT       = 4;
    localparam integer STATUS_SATURATED           = 5;
    localparam integer STATUS_SAFE_IDENTITY       = 6;
    localparam integer STATUS_FULLY_CLOSED        = 7;

    function automatic logic signed [31:0] sat33(
        input logic signed [32:0] value
    );
        begin
            if (value > S32_MAX_W33)
                sat33 = 32'sh7fff_ffff;
            else if (value < S32_MIN_W33)
                sat33 = 32'sh8000_0000;
            else
                sat33 = value[31:0];
        end
    endfunction

    function automatic logic signed [31:0] sat34(
        input logic signed [33:0] value
    );
        begin
            if (value > S32_MAX_W34)
                sat34 = 32'sh7fff_ffff;
            else if (value < S32_MIN_W34)
                sat34 = 32'sh8000_0000;
            else
                sat34 = value[31:0];
        end
    endfunction

    function automatic logic signed [31:0] min2_nonnegative(
        input logic signed [31:0] lhs,
        input logic signed [31:0] rhs
    );
        begin
            min2_nonnegative = ($signed(lhs) < $signed(rhs)) ? lhs : rhs;
        end
    endfunction

     
     
    logic signed [31:0] charge_nonnegative;
    logic signed [31:0] discharge_nonnegative;
    logic signed [31:0] reduce_cap_nonnegative;
    logic signed [31:0] add_cap_nonnegative;
    logic signed [31:0] limit_nonnegative;
    logic signed [32:0] excess_wide_s1;
    logic signed [31:0] excess_s1_comb;
    logic signed [31:0] reduce_s1_comb;
    logic signed [31:0] exec_charge_s1_comb;
    logic invalid_s1_comb;
    logic saturated_s1_comb;

    always_comb begin
        charge_nonnegative     = nominal_charge_q[31]    ? 32'sd0 : nominal_charge_q;
        discharge_nonnegative  = nominal_discharge_q[31] ? 32'sd0 : nominal_discharge_q;
        reduce_cap_nonnegative = max_reduce_charge_q[31] ? 32'sd0 : max_reduce_charge_q;
        add_cap_nonnegative    = max_add_discharge_q[31] ? 32'sd0 : max_add_discharge_q;
        limit_nonnegative      = feeder_limit_q[31]      ? 32'sd0 : feeder_limit_q;

        invalid_s1_comb = nominal_charge_q[31] || nominal_discharge_q[31] ||
                          max_reduce_charge_q[31] || max_add_discharge_q[31] ||
                          feeder_limit_q[31];

        // EN: Stage 1 (combinational): compute feeder import excursion above cap and cancel nominal charging first.
        // 中文：第一级组合逻辑：计算名义受电超出馈线限额的越限量，并优先撤回名义充电功率。
        excess_wide_s1 = $signed({nominal_import_q[31], nominal_import_q}) -
                         $signed({limit_nonnegative[31], limit_nonnegative});
        if (excess_wide_s1 <= 33'sd0)
            excess_s1_comb = 32'sd0;
        else
            excess_s1_comb = sat33(excess_wide_s1);
        saturated_s1_comb = (excess_wide_s1 > S32_MAX_W33) ||
                            (excess_wide_s1 < S32_MIN_W33);

        reduce_s1_comb = min2_nonnegative(
            excess_s1_comb,
            min2_nonnegative(charge_nonnegative, reduce_cap_nonnegative)
        );
        exec_charge_s1_comb = charge_nonnegative - reduce_s1_comb;
    end

    logic valid_s1;
    logic signed [31:0] nominal_import_s1;
    logic signed [31:0] discharge_s1;
    logic signed [31:0] add_cap_s1;
    logic signed [31:0] limit_s1;
    logic signed [31:0] excess_s1;
    logic signed [31:0] reduce_s1;
    logic signed [31:0] exec_charge_s1;
    logic invalid_s1;
    logic saturated_s1;

     
     
    logic signed [31:0] residual_s2_comb;
    logic signed [31:0] add_s2_comb;
    logic signed [32:0] discharge_wide_s2;
    logic signed [33:0] import_wide_s2;
    logic signed [31:0] exec_discharge_s2_comb;
    logic signed [31:0] exec_import_s2_comb;
    logic signed [32:0] remaining_wide_s2;
    logic signed [31:0] remaining_s2_comb;
    logic saturated_s2_comb;
    logic [31:0] status_s2_comb;

    always_comb begin
        // EN: Stage 2 (combinational): add reserve-bounded discharge only after nominal charging reaches zero.
        // 中文：第二级组合逻辑：仅在名义充电已完全撤回至零后，按可交付备用与变流器余量补充放电。
        residual_s2_comb = excess_s1 - reduce_s1;
        if (exec_charge_s1 == 32'sd0)
            add_s2_comb = min2_nonnegative(residual_s2_comb, add_cap_s1);
        else
            add_s2_comb = 32'sd0;

        discharge_wide_s2 = $signed({discharge_s1[31], discharge_s1}) +
                            $signed({add_s2_comb[31], add_s2_comb});
        exec_discharge_s2_comb = sat33(discharge_wide_s2);

        import_wide_s2 = $signed({{2{nominal_import_s1[31]}}, nominal_import_s1}) -
                         $signed({{2{reduce_s1[31]}}, reduce_s1}) -
                         $signed({{2{add_s2_comb[31]}}, add_s2_comb});
        exec_import_s2_comb = sat34(import_wide_s2);

        remaining_wide_s2 = $signed({exec_import_s2_comb[31], exec_import_s2_comb}) -
                            $signed({limit_s1[31], limit_s1});
        if (remaining_wide_s2 <= 33'sd0)
            remaining_s2_comb = 32'sd0;
        else
            remaining_s2_comb = sat33(remaining_wide_s2);

        saturated_s2_comb = saturated_s1 ||
                            (discharge_wide_s2 > S32_MAX_W33) ||
                            (discharge_wide_s2 < S32_MIN_W33) ||
                            (import_wide_s2 > S32_MAX_W34) ||
                            (import_wide_s2 < S32_MIN_W34) ||
                            (remaining_wide_s2 > S32_MAX_W33) ||
                            (remaining_wide_s2 < S32_MIN_W33);

        status_s2_comb = 32'd0;
        status_s2_comb[STATUS_EXCESS]              = (excess_s1 > 32'sd0);
        status_s2_comb[STATUS_REDUCED_CHARGE]      = (reduce_s1 > 32'sd0);
        status_s2_comb[STATUS_ADDED_DISCHARGE]     = (add_s2_comb > 32'sd0);
        status_s2_comb[STATUS_REMAINING_OVERLIMIT] = (remaining_s2_comb > 32'sd0);
        status_s2_comb[STATUS_INVALID_INPUT]       = invalid_s1;
        status_s2_comb[STATUS_SATURATED]           = saturated_s2_comb;
        status_s2_comb[STATUS_SAFE_IDENTITY]       = (excess_s1 == 32'sd0);
        status_s2_comb[STATUS_FULLY_CLOSED]        = (excess_s1 > 32'sd0) &&
                                                     (remaining_s2_comb == 32'sd0);
    end

    // EN: Register Stage 1 -> Stage 2 pipeline state; out_valid asserts two clock cycles after in_valid.
    // 中文：寄存第一级至第二级流水状态；out_valid 在 in_valid 有效后第 2 个时钟上升沿置位。
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_s1                  <= 1'b0;
            nominal_import_s1         <= 32'sd0;
            discharge_s1              <= 32'sd0;
            add_cap_s1                <= 32'sd0;
            limit_s1                  <= 32'sd0;
            excess_s1                 <= 32'sd0;
            reduce_s1                 <= 32'sd0;
            exec_charge_s1            <= 32'sd0;
            invalid_s1                <= 1'b0;
            saturated_s1              <= 1'b0;
            out_valid                 <= 1'b0;
            reduce_charge_q           <= 32'sd0;
            add_discharge_q           <= 32'sd0;
            executed_charge_q         <= 32'sd0;
            executed_discharge_q      <= 32'sd0;
            executed_import_q         <= 32'sd0;
            remaining_overlimit_q     <= 32'sd0;
            status                    <= 32'd0;
        end else begin
            valid_s1 <= in_valid;
            if (in_valid) begin
                nominal_import_s1 <= nominal_import_q;
                discharge_s1      <= discharge_nonnegative;
                add_cap_s1        <= add_cap_nonnegative;
                limit_s1          <= limit_nonnegative;
                excess_s1         <= excess_s1_comb;
                reduce_s1         <= reduce_s1_comb;
                exec_charge_s1    <= exec_charge_s1_comb;
                invalid_s1        <= invalid_s1_comb;
                saturated_s1      <= saturated_s1_comb;
            end

            out_valid <= valid_s1;
            if (valid_s1) begin
                reduce_charge_q       <= reduce_s1;
                add_discharge_q       <= add_s2_comb;
                executed_charge_q     <= exec_charge_s1;
                executed_discharge_q  <= exec_discharge_s2_comb;
                executed_import_q     <= exec_import_s2_comb;
                remaining_overlimit_q <= remaining_s2_comb;
                status                <= status_s2_comb;
            end
        end
    end

endmodule

