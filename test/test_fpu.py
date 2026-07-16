import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
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
    