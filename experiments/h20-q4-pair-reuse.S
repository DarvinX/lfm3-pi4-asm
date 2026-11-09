/* H20: Q4_K fixed low/high pairs; unchanged integers and float block order. */
.arch armv8-a+crc
#ifndef LFM_PREFETCH
#define LFM_PREFETCH 1
#endif
.text
.macro H20_GROUP high
    and w5,w12,#255
    and w6,w13,#255
    lsr x12,x12,#8
    lsr x13,x13,#8
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
    ldrsh w8,[x15],#2
    ldrsh w11,[x15],#2
    add w8,w8,w11
    madd w10,w8,w6,w10
.endm
.balign 64
.global lfm_q4_q8_dot_h20
.type lfm_q4_q8_dot_h20,%function
lfm_q4_q8_dot_h20:
    fmov s28,wzr
    movi v26.16b,#15
1:  .if LFM_PREFETCH
    prfm pldl1strm,[x0,#288]
    .endif
    ldr h16,[x0]
    ldr h17,[x0,#2]
    fcvt s16,h16
    fcvt s17,h17
    ldr s18,[x1]
    fmul s16,s16,s18
    fmul s17,s17,s18
    mov w9,#0
    mov w10,#0
    ldr w4,[x0,#4]
    ldr w5,[x0,#8]
    ldr w6,[x0,#12]
    and w12,w4,#0x3f3f3f3f
    and w13,w5,#0x3f3f3f3f
    lsr w7,w4,#2
    and w7,w7,#0x30303030
    and w8,w6,#0x0f0f0f0f
    orr w8,w8,w7
    orr x12,x12,x8,lsl #32
    lsr w7,w5,#2
    and w7,w7,#0x30303030
    lsr w8,w6,#4
    and w8,w8,#0x0f0f0f0f
    orr w8,w8,w7
    orr x13,x13,x8,lsl #32
    add x4,x0,#16
    add x14,x1,#32
    add x15,x1,#288
    mov w3,#4
2:  ld1 {v0.16b,v1.16b},[x4],#32
    H20_GROUP 0
    H20_GROUP 1
    subs w3,w3,#1
    b.ne 2b
    scvtf s18,w9
    scvtf s19,w10
    fmsub s28,s17,s19,s28
    fmadd s28,s16,s18,s28
    add x0,x0,#144
    add x1,x1,#320
    subs w2,w2,#256
    b.ne 1b
    fmov s0,s28
    ret
.size lfm_q4_q8_dot_h20,.-lfm_q4_q8_dot_h20
.section .note.GNU-stack,"",%progbits
