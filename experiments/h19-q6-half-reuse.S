/* H19: fixed Q6_K half-block schedule; exact original values and float order.
 * QL64B/QH32B loaded once per128 weights, reused by four known groups.
 * No DotProd/I8MM, no model expansion, no callee-saved registers touched. */
.arch armv8-a+crc
#ifndef LFM_PREFETCH
#define LFM_PREFETCH 1
#endif
.text
.macro H19_GROUP qla,qlb,shift,nibble
    .if \nibble
    ushr v6.16b,v\qla\().16b,#4
    ushr v7.16b,v\qlb\().16b,#4
    .else
    and v6.16b,v\qla\().16b,v26.16b
    and v7.16b,v\qlb\().16b,v26.16b
    .endif
    .if \shift
    ushr v29.16b,v4.16b,#\shift
    ushr v30.16b,v5.16b,#\shift
    and v29.16b,v29.16b,v27.16b
    and v30.16b,v30.16b,v27.16b
    .else
    and v29.16b,v4.16b,v27.16b
    and v30.16b,v5.16b,v27.16b
    .endif
    shl v29.16b,v29.16b,#4
    shl v30.16b,v30.16b,#4
    orr v6.16b,v6.16b,v29.16b
    orr v7.16b,v7.16b,v30.16b
    sub v6.16b,v6.16b,v23.16b
    sub v7.16b,v7.16b,v23.16b
    ldrsb w5,[x8],#1
    ldrsb w6,[x8],#1
    ld1 {v19.16b,v20.16b},[x7],#32
    smull v21.8h,v6.8b,v19.8b
    smlal2 v21.8h,v6.16b,v19.16b
    smull v22.8h,v7.8b,v20.8b
    smlal2 v22.8h,v7.16b,v20.16b
    saddlv s21,v21.8h
    saddlv s22,v22.8h
    fmov w11,s21
    fmov w12,s22
    madd w9,w11,w5,w9
    madd w9,w12,w6,w9
.endm
.balign 64
.global lfm_q6_q8_dot_h19
.type lfm_q6_q8_dot_h19,%function
lfm_q6_q8_dot_h19:
    fmov s28,wzr
    movi v26.16b,#15
    movi v27.16b,#3
    movi v23.16b,#32
1:  .if LFM_PREFETCH
    prfm pldl1strm,[x0,#416]
    .endif
    ldr h16,[x0,#208]
    fcvt s16,h16
    ldr s17,[x1]
    fmul s16,s16,s17
    mov w9,#0
    mov w3,#0
2:  add x4,x0,x3,lsl #6
    ld1 {v0.16b-v3.16b},[x4]
    add x5,x0,#128
    add x5,x5,x3,lsl #5
    ld1 {v4.16b,v5.16b},[x5]
    add x7,x1,#32
    add x7,x7,x3,lsl #7
    add x8,x0,#192
    add x8,x8,x3,lsl #3
    H19_GROUP 0,1,0,0
    H19_GROUP 2,3,2,0
    H19_GROUP 0,1,4,1
    H19_GROUP 2,3,6,1
    add w3,w3,#1
    cmp w3,#2
    b.lo 2b
    scvtf s18,w9
    fmadd s28,s16,s18,s28
    add x0,x0,#210
    add x1,x1,#320
    subs w2,w2,#256
    b.ne 1b
    fmov s0,s28
    ret
.size lfm_q6_q8_dot_h19,.-lfm_q6_q8_dot_h19
.section .note.GNU-stack,"",%progbits
