import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, ReadOnly
from enum import IntEnum
import numpy as np
import ml_dtypes
import vsc

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


@vsc.randobj
class FPUTransaction:
    def __init__(self):
        self.val_A = vsc.rand_bit_t(16)
        self.val_B = vsc.rand_bit_t(16)
        self.val_op = vsc.rand_enum_t(ALU_Ops)
        self.acc = vsc.rand_bit_t(1)

    @vsc.constraint
    def val_A_rules(self):
        vsc.dist
        (
            self.val_A,
            [
                # 10 percent Zero
                vsc.weight(0x0000, 5),   
                vsc.weight(0x8000, 5),   
                
                # 10 percent Infin
                vsc.weight(0x7F80, 5),   
                vsc.weight(0xFF80, 5),   
                
                # 5 percent NAN
                vsc.weight(vsc.rng(0xFFC0, 0xFFFF), 5),
                
                # 5 percent 1
                vsc.weight(0x3F80, 5),
                
                # 70 percent Normal
                vsc.weight(vsc.rng(0x0001, 0x3F7F), 70) 
            ]
        ) 
    
    @vsc.constraint
    def val_B_rules(self):
        vsc.dist
        (
            self.val_B,
            [
                # 10 percent Zero
                vsc.weight(0x0000, 5),   
                vsc.weight(0x8000, 5),   
                
                # 10 percent Infin
                vsc.weight(0x7F80, 5),   
                vsc.weight(0xFF80, 5),   
                
                # 5 percent NAN
                vsc.weight(vsc.rng(0xFFC0, 0xFFFF), 5),
                
                # 5 percent 1
                vsc.weight(0x3F80, 5),
                
                # 70 percent Normal
                vsc.weight(vsc.rng(0x0001, 0x3F7F), 70) 
            ]
        )

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

    for _ in range(100):
        #Driving Stim
        ##########################################
        await FallingEdge(dut.clk)
        txn = FPUTransaction()
        txn.randomize()

        val_A = txn.val_A
        val_B = txn.val_B
        val_op = int(txn.val_op)
        val_acc = txn.acc

        dut.A.value = val_A
        dut.B.value = val_B
        dut.op.value = val_op
        dut.acc.value = val_acc
        dut.data_ready.value = 1
        ##########################################



        #Scoreboard
        ##########################################
        await RisingEdge(dut.clk)
        await ReadOnly()
        dut.data_ready.value = 0

        exp_res, exp_uf, exp_of, exp_nan = goldenModel(
            val_A, val_B, val_op, val_acc, int(dut.accumulate_register.value)
        )

        hardware_res = int(dut.accumulate_register.value)
        hardware_uf  = int(dut.fpu_core.flag_underflow) 
        hardware_of  = int(dut.fpu_core.flag_overflow)
        hardware_nan = int(dut.fpu_core.flag_NAN)       
        
        allTestsPassed = (
            (hardware_res == exp_res) and 
            (hardware_uf == exp_uf) and 
            (hardware_of == exp_of) and 
            (hardware_nan == exp_nan)
        )

        assert allTestsPassed, (
            f"Test Failed! \n"
            f"Inputs: A={hex(val_A)}, B={hex(val_B)}, OP={hex(val_op)}, ACC={hex(val_acc)}\n"
            f"Math: Exp {hex(exp_res)}, Got {hex(hardware_res)} \n"
            f"UF: Exp {exp_uf}, Got {hardware_uf} \n"
            f"OF: Exp {exp_of}, Got {hardware_of} \n"
            f"NAN: Exp {exp_nan}, Got {hardware_nan}"
        )
        
        dut._log.info(f"SUCCESS: Hardware matched Golden Model -> {hex(hardware_res)}")    ##########################################
        ##########################################
        


        #Reset
        ##########################################
        await FallingEdge(dut.clk) 
    
        dut.data_ready.value = 0
        ##########################################
