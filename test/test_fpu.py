import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, ReadOnly
from enum import IntEnum
import random 
import numpy as np
import ml_dtypes
import vsc

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
    A_val_int = acc_reg_val if acc == 1 else A
    B_val_int = B
    
    is_arithmetic = op in [ALU_Ops.ADD, ALU_Ops.SUB, ALU_Ops.MUL, ALU_Ops.DIV]
    
    if is_arithmetic:
        if ((A_val_int >> 7) & 0xFF) == 0: 
            A_val_int = A_val_int & 0x8000
            
        if ((B_val_int >> 7) & 0xFF) == 0: 
            B_val_int = B_val_int & 0x8000

    A_bfloat = np.array([A_val_int], dtype=np.uint16).view(ml_dtypes.bfloat16)
    B_bfloat = np.array([B_val_int], dtype=np.uint16).view(ml_dtypes.bfloat16)

    flag_NAN = flag_overflow = flag_underflow = 0   
    
    A_is_inf = (A_val_int & 0x7FFF) == 0x7F80
    B_is_inf = (B_val_int & 0x7FFF) == 0x7F80

    A_is_zero = (A_val_int & 0x7FFF) == 0
    B_is_zero = (B_val_int & 0x7FFF) == 0

    is_div_by_zero = (op == ALU_Ops.DIV) and B_is_zero

    if op == ALU_Ops.ADD:
        result_bfloat = A_bfloat + B_bfloat
    
    elif op == ALU_Ops.SUB:
        result_bfloat = A_bfloat - B_bfloat
    
    elif op == ALU_Ops.MUL:
        result_bfloat = A_bfloat * B_bfloat
    
    elif op == ALU_Ops.DIV:
        result_bfloat = A_bfloat / B_bfloat
    
    elif op == ALU_Ops.NEG:
        result_int = A_val_int ^ 0x8000
        return result_int, 0, 0, 0
    
    elif op == ALU_Ops.ABS:
        result_int = A_val_int & 0x7FFF
        return result_int, 0, 0, 0
    
    elif op == ALU_Ops.SLT:
        slt_val = 0x3F80 if (A_bfloat < B_bfloat)[0] else 0x0000
        result_bfloat = np.array([slt_val], dtype=np.uint16).view(ml_dtypes.bfloat16)
    
    elif op == ALU_Ops.NOP:
        result_bfloat = np.array([acc_reg_val], dtype=np.uint16).view(ml_dtypes.bfloat16)
    
    else:
        result_bfloat = np.array([0], dtype=np.uint16).view(ml_dtypes.bfloat16)
    
    result_int = int(result_bfloat.view(np.uint16)[0])
    exponent = (result_int >> 7) & 0b1111_1111
    mantissa = result_int & 0b0111_1111
    
    if exponent == 0b1111_1111: 
        if mantissa == 0:
            if not (A_is_inf or B_is_inf or is_div_by_zero):
                flag_overflow = 1
        else:
            if is_arithmetic:
                flag_NAN = 1
                result_int = 0x7FC0

    elif exponent == 0:
        if mantissa != 0:
            flag_underflow = 1
            result_int = result_int & 0x8000
        
        else:                
            if op == ALU_Ops.MUL and not A_is_zero and not B_is_zero:
                flag_underflow = 1
                
            elif op == ALU_Ops.DIV and not A_is_zero and not B_is_inf:
                flag_underflow = 1    
    
    if not is_arithmetic:
        flag_underflow = 0
        flag_overflow = 0
        flag_NAN = 0

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

def get_adversarial_B(A):
    return [
        # Equal
        A,
        # Sign flipped
        A ^ 0x8000,
        # Exponent +1
        (A + 0x0080) & 0x7F80,
        # Exponent -1
        (A - 0x0080) & 0x7F80,
        # Exponent +14
        (A + 0x0700) & 0x7F80,
        # Exponent -14
        (A - 0x0700) & 0x7F80,
        # Minimum positive normal
        0x0100,
        # Maximum positive normal
        0x7E7F,
        # Minimum negative normal
        0x8100,
        # Maximum negative normal
        0xFE7F,
        # Inverted mantissa
        A ^ 0x007F,
        # Mantissa all 1s
        A | 0x007F,
        # Mantissa all 0s
        A & 0x7F00,
        # +1.0
        0x3F80,
        # -1.0
        0xBF80,
        # Random finite normal
        random.randint(0x0100, 0x7E7F)
    ]

# CRT 
# @cocotb.test()
# async def test_project(dut):

#     clock = Clock(dut.clk, 10, unit = "us")
#     cocotb.start_soon(clock.start())

#     #Resetting 
#     ##########################################
#     dut.reset_n.value = 0
    
#     dut.data_ready.value = 0
#     dut.A.value = 0
#     dut.B.value = 0
#     dut.op.value = ALU_Ops.NOP
#     dut.acc.value = 0

#     await ClockCycles(dut.clk, 3)
#     dut.reset_n.value = 1
#     ##########################################


#     #Driving Stim
#     ##########################################
#     for i in range(10000):
#         await FallingEdge(dut.clk)
#         txn = FPUTransaction()
#         txn.randomize()

#         val_A = txn.val_A
#         val_B = txn.val_B
#         val_op = int(txn.val_op)
#         val_acc = txn.acc

#         dut.A.value = val_A
#         dut.B.value = val_B
#         dut.op.value = val_op
#         dut.acc.value = val_acc
#         dut.data_ready.value = 1

#         current_accumulate_register = int(dut.accumulate_register.value)
#     ##########################################


#         #Scoreboard
#         ##########################################
#         await RisingEdge(dut.clk)
#         await ReadOnly()

#         exp_res, exp_uf, exp_of, exp_nan = goldenModel(
#             val_A, val_B, val_op, val_acc, current_accumulate_register
#         )

#         hardware_res = int(dut.accumulate_register.value)
#         hardware_uf = int(dut.flag_underflow.value) 
#         hardware_of = int(dut.flag_overflow.value)
#         hardware_nan = int(dut.flag_NAN.value) 
        
#         if val_op == ALU_Ops.DIV:
#             hardware_res_accurate = (hardware_res == exp_res) or \
#                                     (hardware_res + 1 == exp_res) or \
#                                     (hardware_res - 1 == exp_res)
#         else:
#             hardware_res_accurate = (hardware_res == exp_res)
              
#         allTestsPassed = (
#             (hardware_res_accurate) and 
#             (hardware_uf == exp_uf) and 
#             (hardware_of == exp_of) and 
#             (hardware_nan == exp_nan)
#         )

#         assert allTestsPassed, (
#             f"Test Number {i} Failed! \n"
#             f"Inputs: A={hex(val_A)}, B={hex(val_B)}, OP={ALU_Ops(val_op).name}, ACC={hex(val_acc)}\n"
#             f"Math: Exp {hex(exp_res)}, Got {hex(hardware_res)} \n"
#             f"UF: Exp {exp_uf}, Got {hardware_uf} \n"
#             f"OF: Exp {exp_of}, Got {hardware_of} \n"
#             f"NAN: Exp {exp_nan}, Got {hardware_nan}"
#         )
        
#         dut._log.info(f"Test Number {i} Success!: Hardware matched Golden Model -> {hex(hardware_res)}")    ##########################################
#         ##########################################
        

#         #Reset
#         ##########################################
#         await FallingEdge(dut.clk) 
    
#         dut.data_ready.value = 0
#         ##########################################


#Logical sweep
# @cocotb.test()
# async def test_project(dut):

#     clock = Clock(dut.clk, 10, unit = "us")
#     cocotb.start_soon(clock.start())

#     #Resetting
#     dut.reset_n.value = 0
#     dut.data_ready.value = 0
#     dut.A.value = 0
#     dut.B.value = 0
#     dut.op.value = ALU_Ops.NOP
#     dut.acc.value = 0
#     await ClockCycles(dut.clk, 3)
#     dut.reset_n.value = 1

#     #Driving values
#     test_count = 0
#     error_count = 0
    
#     for exp_a in range(1, 255):
#         for mant_a in range(128):
#             val_A = (exp_a << 7) | mant_a
            
#             b_matrix = get_adversarial_B(val_A)
            
#             for val_op in [ALU_Ops.ADD, ALU_Ops.SUB, ALU_Ops.MUL, ALU_Ops.DIV, ALU_Ops.SLT]:
#                 for val_B in b_matrix:
                    
#                     await FallingEdge(dut.clk)
#                     dut.A.value = val_A
#                     dut.B.value = val_B
#                     dut.op.value = val_op
#                     dut.acc.value = 0
#                     dut.data_ready.value = 1
                    
#                     current_accumulate_register = int(dut.accumulate_register.value)

#                     await RisingEdge(dut.clk)
#                     await ReadOnly()

#                     exp_res, exp_uf, exp_of, exp_nan = goldenModel(
#                         val_A, val_B, val_op, 0, current_accumulate_register
#                     )

#                     hardware_res = int(dut.accumulate_register.value)
#                     hardware_uf = int(dut.flag_underflow.value) 
#                     hardware_of = int(dut.flag_overflow.value)
#                     hardware_nan = int(dut.flag_NAN.value) 
                    
#                     if val_op == ALU_Ops.DIV:
#                         hardware_res_accurate = (hardware_res == exp_res) or \
#                                                 (hardware_res + 1 == exp_res) or \
#                                                 (hardware_res - 1 == exp_res)
#                     else:
#                         hardware_res_accurate = (hardware_res == exp_res)
                            
#                     allTestsPassed = (
#                         (hardware_res_accurate) and 
#                         (hardware_uf == exp_uf) and 
#                         (hardware_of == exp_of) and 
#                         (hardware_nan == exp_nan)
#                     )

#                     if not allTestsPassed:
#                         dut._log.error(
#                             f"Test Number {test_count} Failed! \n"
#                             f"Inputs: A={hex(val_A)}, B={hex(val_B)}, OP={ALU_Ops(val_op).name}, ACC={hex(0)}\n"
#                             f"Math: Exp {hex(exp_res)}, Got {hex(hardware_res)} \n"
#                             f"UF: Exp {exp_uf}, Got {hardware_uf} \n"
#                             f"OF: Exp {exp_of}, Got {hardware_of} \n"
#                             f"NAN: Exp {exp_nan}, Got {hardware_nan}"
#                         )
#                         error_count += 1
#                     #else:
#                     #    dut._log.info(f"Test Number {test_count} Success!: Hardware matched Golden Model -> {hex(hardware_res)}")
                    
#                     test_count += 1
                    
#                     await FallingEdge(dut.clk)
#                     dut.data_ready.value = 0        
    
#     assert error_count == 0, f"Total of {error_count} tests failed out of {test_count}!"            


# Unit testing
@cocotb.test()
async def test_project(dut):

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # Resetting
    dut.reset_n.value = 0
    dut.data_ready.value = 0
    dut.A.value = 0
    dut.B.value = 0
    dut.op.value = ALU_Ops.NOP
    dut.acc.value = 0
    await ClockCycles(dut.clk, 3)
    dut.reset_n.value = 1

    test_vectors = [
        (0x0d00, 0x32ff, ALU_Ops.MUL),
        (0x195a, 0x2696, ALU_Ops.MUL),
        (0x1fff, 0x2000, ALU_Ops.MUL),
        (0x207f, 0x1f80, ALU_Ops.MUL),
        (0x3eff, 0x0100, ALU_Ops.MUL),
        (0x3eff, 0x8100, ALU_Ops.MUL),
    ]    

    test_count = 0
    error_count = 0

    for val_A, val_B, val_op in test_vectors:

        await FallingEdge(dut.clk)
        dut.A.value = val_A
        dut.B.value = val_B
        dut.op.value = val_op
        dut.acc.value = 0
        dut.data_ready.value = 1

        current_accumulate_register = int(dut.accumulate_register.value)

        await RisingEdge(dut.clk)
        await ReadOnly()

        exp_res, exp_uf, exp_of, exp_nan = goldenModel(
            val_A, val_B, val_op, 0, current_accumulate_register
        )

        hardware_res = int(dut.accumulate_register.value)
        hardware_uf = int(dut.flag_underflow.value)
        hardware_of = int(dut.flag_overflow.value)
        hardware_nan = int(dut.flag_NAN.value)

        if val_op == ALU_Ops.DIV:
            hardware_res_accurate = (
                hardware_res == exp_res or
                hardware_res + 1 == exp_res or
                hardware_res - 1 == exp_res
            )
        else:
            hardware_res_accurate = (hardware_res == exp_res)

        allTestsPassed = (
            hardware_res_accurate and
            hardware_uf == exp_uf and
            hardware_of == exp_of and
            hardware_nan == exp_nan
        )

        if not allTestsPassed:
            dut._log.error(
                f"Test Number {test_count} Failed!\n"
                f"Inputs: A={hex(val_A)}, B={hex(val_B)}, OP={ALU_Ops(val_op).name}, ACC={hex(0)}\n"
                f"Math: Exp {hex(exp_res)}, Got {hex(hardware_res)}\n"
                f"UF: Exp {exp_uf}, Got {hardware_uf}\n"
                f"OF: Exp {exp_of}, Got {hardware_of}\n"
                f"NAN: Exp {exp_nan}, Got {hardware_nan}"
            )
            error_count += 1

        test_count += 1

        await FallingEdge(dut.clk)
        dut.data_ready.value = 0

    assert error_count == 0