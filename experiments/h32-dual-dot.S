/* H32 feasibility: exact two-query dots sharing original Q4_K/Q6_K bytes.
 * void kernel(weight,q8A,q8B,outA,outB,columns): x0,x1,x2,x3,x4,w5.
 * Leaf AAPCS64: preserves x19-x28,d8-d15; no model expansion.
 * Q4 requires sums32 at Q8 spare header16..31 (same320B layout).
 * Each query retains original integer math and FP block order.
 */
.arch armv8-a+crc
#ifndef LFM_PREFETCH
#define LFM_PREFETCH 1
#endif
.text
.macro H32_Q4_GROUP high,group
    and w5,w16,#255
    and w6,w17,#255
    lsr x16,x16,#8
    lsr x17,x17,#8
    .if \high
    ushr v6.16b,v0.16b,#4
    ushr v7.16b,v1.16b,#4
    .else
    and v6.16b,v0.16b,v26.16b
    and v7.16b,v1.16b,v26.16b
    .endif
    ld1 {v2.16b,v3.16b},[x14],#32
    smull v4.8h,v6.8b,v2.8b
    smlal2 v4.8h,v6.16b,v2.16b
    smull v5.8h,v7.8b,v3.8b
    smlal2 v5.8h,v7.16b,v3.16b
    saddlp v4.4s,v4.8h
    saddlp v5.4s,v5.8h
    add v4.4s,v4.4s,v5.4s
    addv s4,v4.4s
    fmov w7,s4
    madd w9,w7,w5,w9
    ldrsh w8,[x1,#(16+2*\group)]
    madd w10,w8,w6,w10
    ld1 {v2.16b,v3.16b},[x15],#32
    smull v4.8h,v6.8b,v2.8b
    smlal2 v4.8h,v6.16b,v2.16b
    smull v5.8h,v7.8b,v3.8b
    smlal2 v5.8h,v7.16b,v3.16b
    saddlp v4.4s,v4.8h
    saddlp v5.4s,v5.8h
    add v4.4s,v4.4s,v5.4s
    addv s4,v4.4s
    fmov w7,s4
    madd w11,w7,w5,w11
    ldrsh w8,[x2,#(16+2*\group)]
    madd w12,w8,w6,w12
.endm
.balign 64
.global lfm_q4_q8_dual_h32
.type lfm_q4_q8_dual_h32,%function
lfm_q4_q8_dual_h32:
    mov x13,x3
    mov w3,w5
    fmov s28,wzr
    fmov s31,wzr
    movi v26.16b,#15
1:  .if LFM_PREFETCH
    prfm pldl1strm,[x0,#288]
    .endif
    ldr h16,[x0]
    ldr h17,[x0,#2]
    fcvt s16,h16
    fcvt s17,h17
    ldr s18,[x1]
    ldr s19,[x2]
    fmul s20,s16,s18
    fmul s21,s17,s18
    fmul s22,s16,s19
    fmul s23,s17,s19
    ldr w9,[x0,#4]
    ldr w10,[x0,#8]
    ldr w11,[x0,#12]
    and w16,w9,#0x3f3f3f3f
    and w17,w10,#0x3f3f3f3f
    lsr w7,w9,#2
    and w7,w7,#0x30303030
    and w8,w11,#0x0f0f0f0f
    orr w8,w8,w7
    orr x16,x16,x8,lsl #32
    lsr w7,w10,#2
    and w7,w7,#0x30303030
    lsr w8,w11,#4
    and w8,w8,#0x0f0f0f0f
    orr w8,w8,w7
    orr x17,x17,x8,lsl #32
    mov w9,#0
    mov w10,#0
    mov w11,#0
    mov w12,#0
    add x14,x1,#32
    add x15,x2,#32
    ldp q0,q1,[x0,#16]
    H32_Q4_GROUP 0,0
    H32_Q4_GROUP 1,1
    ldp q0,q1,[x0,#48]
    H32_Q4_GROUP 0,2
    H32_Q4_GROUP 1,3
    ldp q0,q1,[x0,#80]
    H32_Q4_GROUP 0,4
    H32_Q4_GROUP 1,5
    ldp q0,q1,[x0,#112]
    H32_Q4_GROUP 0,6
    H32_Q4_GROUP 1,7
    scvtf s16,w9
    scvtf s17,w10
    scvtf s18,w11
    scvtf s19,w12
    fmsub s28,s21,s17,s28
    fmadd s28,s20,s16,s28
    fmsub s31,s23,s19,s31
    fmadd s31,s22,s18,s31
    add x0,x0,#144
    add x1,x1,#320
    add x2,x2,#320
    subs w3,w3,#256
    b.ne 1b
    str s28,[x13]
    str s31,[x4]
    ret
.size lfm_q4_q8_dual_h32,.-lfm_q4_q8_dual_h32

.macro H32_Q6_GROUP qla,qlb,shift,nibble
    .if \shift == 6
    sshr v6.16b,v4.16b,#2
    sshr v7.16b,v5.16b,#2
    .else
    shl v6.16b,v4.16b,#(6-\shift)
    shl v7.16b,v5.16b,#(6-\shift)
    sshr v6.16b,v6.16b,#2
    sshr v7.16b,v7.16b,#2
    .endif
    .if \nibble
    ushr v29.16b,v\qla\().16b,#4
    ushr v30.16b,v\qlb\().16b,#4
    bit v6.16b,v29.16b,v26.16b
    bit v7.16b,v30.16b,v26.16b
    .else
    bit v6.16b,v\qla\().16b,v26.16b
    bit v7.16b,v\qlb\().16b,v26.16b
    .endif
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
    ld1 {v19.16b,v20.16b},[x14],#32
    smull v21.8h,v6.8b,v19.8b
    smlal2 v21.8h,v6.16b,v19.16b
    smull v22.8h,v7.8b,v20.8b
    smlal2 v22.8h,v7.16b,v20.16b
    saddlv s21,v21.8h
    saddlv s22,v22.8h
    fmov w11,s21
    fmov w12,s22
    madd w10,w11,w5,w10
    madd w10,w12,w6,w10
.endm
.balign 64
.global lfm_q6_q8_dual_h32
.type lfm_q6_q8_dual_h32,%function
lfm_q6_q8_dual_h32:
    mov x13,x3
    mov x15,x4
    mov w16,w5
    fmov s28,wzr
    fmov s31,wzr
    movi v26.16b,#15
    movi v23.16b,#0xaa
1:  .if LFM_PREFETCH
    prfm pldl1strm,[x0,#416]
    .endif
    ldr h16,[x0,#208]
    fcvt s16,h16
    ldr s17,[x2]
    ldr s18,[x1]
    fmul s17,s16,s17
    fmul s18,s16,s18
    mov w9,#0
    mov w10,#0
    mov w3,#0
2:  add x4,x0,x3,lsl #6
    ld1 {v0.16b-v3.16b},[x4]
    add x5,x0,#128
    add x5,x5,x3,lsl #5
    ld1 {v4.16b,v5.16b},[x5]
    eor v4.16b,v4.16b,v23.16b
    eor v5.16b,v5.16b,v23.16b
    add x7,x1,#32
    add x7,x7,x3,lsl #7
    add x14,x2,#32
    add x14,x14,x3,lsl #7
    add x8,x0,#192
    add x8,x8,x3,lsl #3
    H32_Q6_GROUP 0,1,0,0
    H32_Q6_GROUP 2,3,2,0
    H32_Q6_GROUP 0,1,4,1
    H32_Q6_GROUP 2,3,6,1
    add w3,w3,#1
    cmp w3,#2
    b.lo 2b
    scvtf s16,w9
    scvtf s19,w10
    fmadd s28,s18,s16,s28
    fmadd s31,s17,s19,s31
    add x0,x0,#210
    add x1,x1,#320
    add x2,x2,#320
    subs w16,w16,#256
    b.ne 1b
    str s28,[x13]
    str s31,[x15]
    ret
.size lfm_q6_q8_dual_h32,.-lfm_q6_q8_dual_h32
.section .note.GNU-stack,"",%progbits
