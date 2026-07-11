`include "fpu_pkg.vh"

module tb();
    // Inputs
    reg clk;
    reg reset_n;
    reg data_ready;
    reg [15:0] A;
    reg [15:0] B;
    reg [2:0] op;
    reg acc;

    // Outputs
    wire [15:0] accumulate_register;
    wire result_ready;
    wire flag_NAN;
    wire flag_overflow;
    wire flag_underflow;

    fpu_system dut
    (
        .clk(clk),
        .reset_n(reset_n),
        .data_ready(data_ready),
        .A(A),
        .B(B),
        .op(op),
        .acc(acc),
        .accumulate_register(accumulate_register),
        .result_ready(result_ready),
        .flag_NAN(flag_NAN),
        .flag_overflow(flag_overflow),
        .flag_underflow(flag_underflow)
    );

endmodule