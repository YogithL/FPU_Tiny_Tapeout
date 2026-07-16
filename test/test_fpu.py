import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
from enum import IntEnum
import numpy as np
import ml_dtypes

global acc_reg

class ALU_Ops(IntEnum):
    ADD = 0  
    SUB = 1  
    MUL = 2  
    DIV = 3  
    NEG = 4  
    ABS = 5  
    SLT = 6  
    NOP = 7  

def goldenModel(A, B, op, acc, acc_reg_val):
    A_bfloat = np.array([A], dtype=np.uint16).view(ml_dtypes.bfloat16)
    B_bfloat = np.array([B], dtype=np.uint16).view(ml_dtypes.bfloat16)
    
    flag_NAN = flag_overflow = flag_underflow = 0   

    if acc == 1:
        A_bfloat = np.array([acc_reg_val], dtype=np.uint16).view(ml_dtypes.bfloat16)

    if op == ALU_Ops.ADD:
        result_bfloat = A_bfloat + B_bfloat
    elif op == ALU_Ops.SUB:
        result_bfloat = A_bfloat - B_bfloat
    elif op == ALU_Ops.MUL:
        result_bfloat = A_bfloat * B_bfloat
    elif op == ALU_Ops.DIV:
        result_bfloat = A_bfloat / B_bfloat
    elif op == ALU_Ops.NEG:
        result_bfloat = -A_bfloat 
    elif op == ALU_Ops.ABS:
        result_bfloat = np.abs(A_bfloat)
    elif op == ALU_Ops.SLT:
        slt_val = 1 if (A_bfloat < B_bfloat)[0] else 0   
        result_bfloat = np.array([slt_val], dtype=np.uint16).view(ml_dtypes.bfloat16)
    elif op == ALU_Ops.NOP:
        result_bfloat = A_bfloat
    else:
        result_bfloat = np.array([0], dtype=np.uint16).view(ml_dtypes.bfloat16)

    result_int = int(result_bfloat.view(np.uint16)[0])
    exponent = (result_int >> 7) & 0b1111_1111
    mantissa = result_int & 0b0111_1111
    
    if exponent == 0b1111_1111: 
        if mantissa == 0:
            flag_overflow = 1
        else:
            flag_NAN = 1
    
    if exponent == 0:
        if mantissa != 0:
            flag_underflow = 1
        result_int = 0
    
    return result_int, flag_underflow, flag_overflow, flag_NAN


@cocotb.test()
async def test_project(dut):

    clock = Clock(dut.clk, 10, units="us")
    cocotb.start_soon(clock.start())

    #Resetting 
    dut.reset_n.value = 0
    
    dut.data_ready.value = 0
    dut.A.value = 0
    dut.B.value = 0
    dut.op.value = ALU_Ops.NOP
    dut.acc.value = 0

    await ClockCycles(dut.clk, 3)
    dut.reset_n.value = 1


    #Simple test
    await FallingEdge(dut.clk)

    val_A = 0x4000
    val_B = 0x4040
    val_op = ALU_Ops.MUL
    val_acc = 0

    dut.A.value = val_A
    dut.B.value = val_B
    dut.op.value = val_op
    dut.acc.value = val_acc
    dut.data_ready.value = 1

    await RisingEdge(dut.clk)
    exp_res, exp_uf, exp_of, exp_nan = goldenModel(
        val_A, val_B, val_op, val_acc, int(dut.accumulate_register.value)
    )

    hardware_value = int(dut.accumulate_register.value)
    
    assert hardware_value == exp_res, f"Math failed! Expected {hex(exp_res)}, Got {hex(hardware_value)}"
    
    dut._log.info(f"SUCCESS: Hardware matched Golden Model -> {hex(hardware_value)}")


