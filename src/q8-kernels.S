/* Cortex-A72 integer dot products. No SDOT / I8MM instructions.
 * Activation block: 320 bytes, 64-byte aligned: float d at 0,
 * int8 qs[256] at 32, int16 sums[16] at 288. 30 FF blocks = 9600 B.
 * Weight layout remains the original GGUF Q4_K / Q6_K representation. */
.arch armv8-a+crc
.include "src/common.inc"
#ifndef LFM_PREFETCH
#define LFM_PREFETCH 1
#endif
.text
.balign 64
.macro Q8_PACK_FN name,native
.global \name
.type \name,%function
/* x0 FP32 input, x1 aligned block output, w2 elements (multiple of 256). */
\name:
    .if \native
    ldr s21,.Lq8magic
    dup v21.4s,v21.s[0]
    movi v22.4s,#0x7f,msl #16
    mvni v23.4s,#0x3f,msl #16
    movi v24.4s,#127
    .endif
1:  mov x3,x0
    mov w4,#256
    fmov s16,wzr
    fmov s17,wzr
2:  ldr s0,[x3],#4
    fabs s1,s0
    fcmp s1,s17
    b.le 3f
    fmov s16,s0
    fmov s17,s1
3:  subs w4,w4,#1
    b.ne 2b
    fcmp s17,#0.0
    b.eq 6f
    ldr s18,.Lminus127
    fdiv s18,s18,s16
    fmov s19,#1.0
    fdiv s19,s19,s18
    str s19,[x1]
    dup v20.4s,v18.s[0]
    mov x3,x0
    add x4,x1,#32
    add x5,x1,#288
    mov w6,#16
4:  ld1 {v0.4s-v3.4s},[x3],#64
    .if \native
    /* Native GCC fuses nearest_int(iscale*x)'s magic addition. A separate
     * FMUL+FCVTNS can disagree near half-integers despite matching ties. */
    mov v4.16b,v21.16b
    mov v5.16b,v21.16b
    mov v6.16b,v21.16b
    mov v7.16b,v21.16b
    fmla v4.4s,v0.4s,v20.4s
    fmla v5.4s,v1.4s,v20.4s
    fmla v6.4s,v2.4s,v20.4s
    fmla v7.4s,v3.4s,v20.4s
    and v0.16b,v4.16b,v22.16b
    and v1.16b,v5.16b,v22.16b
    and v2.16b,v6.16b,v22.16b
    and v3.16b,v7.16b,v22.16b
    add v0.4s,v0.4s,v23.4s
    add v1.4s,v1.4s,v23.4s
    add v2.4s,v2.4s,v23.4s
    add v3.4s,v3.4s,v23.4s
    smin v0.4s,v0.4s,v24.4s
    smin v1.4s,v1.4s,v24.4s
    smin v2.4s,v2.4s,v24.4s
    smin v3.4s,v3.4s,v24.4s
    .else
    fmul v0.4s,v0.4s,v20.4s
    fmul v1.4s,v1.4s,v20.4s
    fmul v2.4s,v2.4s,v20.4s
    fmul v3.4s,v3.4s,v20.4s
    fcvtns v0.4s,v0.4s
    fcvtns v1.4s,v1.4s
    fcvtns v2.4s,v2.4s
    fcvtns v3.4s,v3.4s
    .endif
    sqxtn v4.4h,v0.4s
    sqxtn2 v4.8h,v1.4s
    sqxtn v5.4h,v2.4s
    sqxtn2 v5.8h,v3.4s
    sqxtn v6.8b,v4.8h
    sqxtn2 v6.16b,v5.8h
    str q6,[x4],#16
    saddlv h7,v6.16b
    str h7,[x5],#2
    subs w6,w6,#1
    b.ne 4b
    b 8f
6:  str wzr,[x1]
    add x4,x1,#32
    add x5,x1,#288
    mov w6,#16
    movi v0.16b,#0
7:  str q0,[x4],#16
    strh wzr,[x5],#2
    subs w6,w6,#1
    b.ne 7b
8:  add x0,x0,#1024
    add x1,x1,#320
    subs w2,w2,#256
    b.ne 1b
    ret
.size \name,.-\name
.endm
Q8_PACK_FN lfm_q8_pack,0
.balign 64
Q8_PACK_FN lfm_q8_pack_native,1
.balign 4
.Lminus127: .float -127.0
.Lq8magic: .float 12582912.0

.balign 64
.global lfm_q4_q8_dot
.type lfm_q4_q8_dot,%function
/* x0 GGUF Q4_K row, x1 padded activation blocks, w2 n -> s0 dot */
lfm_q4_q8_dot:
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
    /* Expand all eight scale/min pairs once; keep them in registers.
     * Low four scales/mins occupy six bits; the tail packs the others. */
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
    mov w3,#0
2:  and w5,w12,#255
    and w6,w13,#255
    lsr x12,x12,#8
    lsr x13,x13,#8
    lsr w4,w3,#1
    add x4,x0,x4,lsl #5
    add x4,x4,#16
    ld1 {v0.16b,v1.16b},[x4]
    tbz w3,#0,5f
    ushr v0.16b,v0.16b,#4
    ushr v1.16b,v1.16b,#4
5:  and v0.16b,v0.16b,v26.16b
    and v1.16b,v1.16b,v26.16b
    add x7,x1,#32
    add x7,x7,x3,lsl #5
    ld1 {v2.16b,v3.16b},[x7]
    smull v4.8h,v0.8b,v2.8b
    smlal2 v4.8h,v0.16b,v2.16b
    smull v7.8h,v1.8b,v3.8b
    smlal2 v7.8h,v1.16b,v3.16b
    saddlp v4.4s,v4.8h
    saddlp v7.4s,v7.8h
    add v4.4s,v4.4s,v7.4s
    addv s4,v4.4s
    fmov w7,s4
    madd w9,w7,w5,w9
    add x7,x1,#288
    add x7,x7,x3,lsl #2
    ldrsh w8,[x7]
    ldrsh w11,[x7,#2]
    add w8,w8,w11
    madd w10,w8,w6,w10
    add w3,w3,#1
    cmp w3,#8
    b.lo 2b
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
.size lfm_q4_q8_dot,.-lfm_q4_q8_dot

.balign 64
.global lfm_q6_q8_dot
.type lfm_q6_q8_dot,%function
lfm_q6_q8_dot:
    fmov s28,wzr
    movi v26.16b,#15
    movi v27.16b,#3
1:  .if LFM_PREFETCH
    prfm pldl1strm,[x0,#416]
    .endif
    ldr h16,[x0,#208]
    fcvt s16,h16
    ldr s17,[x1]
    fmul s16,s16,s17
    mov w9,#0
    mov w3,#0
2:  lsr w4,w3,#2
    and w5,w3,#1
    lsl w4,w4,#6
    add w4,w4,w5,lsl #5
    add x4,x0,w4,uxtw
    ld1 {v0.16b,v1.16b},[x4]
    tbz w3,#1,3f
    ushr v0.16b,v0.16b,#4
    ushr v1.16b,v1.16b,#4
3:  and v0.16b,v0.16b,v26.16b
    and v1.16b,v1.16b,v26.16b
    lsr w4,w3,#2
    add x4,x0,x4,lsl #5
    add x4,x4,#128
    ld1 {v2.16b,v3.16b},[x4]
    and w4,w3,#3
    neg w4,w4,lsl #1
    dup v18.16b,w4
    ushl v2.16b,v2.16b,v18.16b
    ushl v3.16b,v3.16b,v18.16b
    and v2.16b,v2.16b,v27.16b
    and v3.16b,v3.16b,v27.16b
    shl v2.16b,v2.16b,#4
    shl v3.16b,v3.16b,#4
    orr v0.16b,v0.16b,v2.16b
    orr v1.16b,v1.16b,v3.16b
    movi v18.16b,#32
    sub v0.16b,v0.16b,v18.16b
    sub v1.16b,v1.16b,v18.16b
    add x4,x0,#192
    add x4,x4,x3,lsl #1
    ldrsb w5,[x4]
    ldrsb w6,[x4,#1]
    add x7,x1,#32
    add x7,x7,x3,lsl #5
    ld1 {v2.16b,v3.16b},[x7]
    smull v4.8h,v0.8b,v2.8b
    smlal2 v4.8h,v0.16b,v2.16b
    smull v7.8h,v1.8b,v3.8b
    smlal2 v7.8h,v1.16b,v3.16b
    saddlv s4,v4.8h
    saddlv s7,v7.8h
    fmov w7,s4
    fmov w8,s7
    madd w9,w7,w5,w9
    madd w9,w8,w6,w9
    add w3,w3,#1
    cmp w3,#8
    b.lo 2b
    scvtf s18,w9
    fmadd s28,s16,s18,s28
    add x0,x0,#210
    add x1,x1,#320
    subs w2,w2,#256
    b.ne 1b
    fmov s0,s28
    ret
.size lfm_q6_q8_dot,.-lfm_q6_q8_dot

.balign 64
.global lfm_rms_wide
.type lfm_rms_wide,%function
/* Match ggml: FP32 squares, FP64 accumulation/mean, FP32 normalization. */
lfm_rms_wide:
    mov x4,x1
    mov w5,w3
    movi v16.2d,#0
1:  ldr q1,[x4],#16
    fmul v1.4s,v1.4s,v1.4s
    fcvtl v2.2d,v1.2s
    fcvtl2 v3.2d,v1.4s
    fadd v16.2d,v16.2d,v2.2d
    fadd v16.2d,v16.2d,v3.2d
    subs w5,w5,#4
    b.ne 1b
    faddp d16,v16.2d
    ucvtf d17,w3
    fdiv d16,d16,d17
    fcvt s16,d16
    fadd s16,s16,s0
    fsqrt s16,s16
    fmov s17,#1.0
    fdiv s16,s17,s16
    dup v16.4s,v16.s[0]
2:  ldr q1,[x1],#16
    ldr q2,[x2],#16
    fmul v1.4s,v1.4s,v16.4s
    fmul v1.4s,v1.4s,v2.4s
    str q1,[x0],#16
    subs w3,w3,#4
    b.ne 2b
    ret
.size lfm_rms_wide,.-lfm_rms_wide
.include "src/native-math.inc"
.section .note.GNU-stack,"",%progbits
