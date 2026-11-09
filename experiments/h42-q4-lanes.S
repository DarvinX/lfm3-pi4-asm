/* H42: exact widening scale lanes/static groups, original packed weight bytes. */
/* H42 source-derived vector minimum dot, cached sums, early FMSUB. */
/* H24: H20 Q4 pairs consume shared precomputed32-element sums. */
.arch armv8-a+crc
#ifndef LFM_PREFETCH
#define LFM_PREFETCH 1
#endif
.text
/* All8 sum32*minimum terms fit signed32 and exact FP32 conversion. */
.macro H42_MIN8 offset
    fmov d23,x13
    uxtl v23.8h,v23.8b
    ldr q24,[x1,#\offset]
    smull v25.4s,v24.4h,v23.4h
    smlal2 v25.4s,v24.8h,v23.8h
    addv s25,v25.4s
    fmov w10,s25
.endm
.macro H42_GROUP high,scale
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
    add v4.8h,v4.8h,v5.8h
    smlal v22.4s,v4.4h,v8.h[\scale]
    smlal2 v22.4s,v4.8h,v8.h[\scale]
.endm
.balign 64
.global lfm_q4_q8_dot_h42
.type lfm_q4_q8_dot_h42,%function
lfm_q4_q8_dot_h42:
    str d8,[sp,#-16]! /* AAPCS64 preserves low d8; SP stays16B aligned */
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
    movi v22.4s,#0
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
    fmov d8,x12
    uxtl v8.8h,v8.8b /* exact positive6-bit scale lanes */
    H42_MIN8 16
    scvtf s19,w10
    fmsub s28,s17,s19,s28 /* same dependent order; latency hidden by integer loop */
    add x4,x0,#16
    add x14,x1,#32
    ld1 {v0.16b,v1.16b},[x4],#32
    H42_GROUP 0,0
    H42_GROUP 1,1
    ld1 {v0.16b,v1.16b},[x4],#32
    H42_GROUP 0,2
    H42_GROUP 1,3
    ld1 {v0.16b,v1.16b},[x4],#32
    H42_GROUP 0,4
    H42_GROUP 1,5
    ld1 {v0.16b,v1.16b},[x4],#32
    H42_GROUP 0,6
    H42_GROUP 1,7
    addv s22,v22.4s
    fmov w9,s22
    scvtf s18,w9
    fmadd s28,s16,s18,s28
    add x0,x0,#144
    add x1,x1,#320
    subs w2,w2,#256
    b.ne 1b
    fmov s0,s28
    ldr d8,[sp],#16
    ret
.size lfm_q4_q8_dot_h42,.-lfm_q4_q8_dot_h42

#ifdef H42_VALIDATE
.balign 64
.global lfm_h42_minimum_test
.type lfm_h42_minimum_test,%function
lfm_h42_minimum_test: /* x0 minima8 bytes,x1 sum32[8] int16 -> w0 */
    ldr x13,[x0]
    H42_MIN8 0
    mov w0,w10
    ret
.size lfm_h42_minimum_test,.-lfm_h42_minimum_test
#endif
.section .note.GNU-stack,"",%progbits
