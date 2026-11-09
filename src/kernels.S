/* AArch64 / Cortex-A72. AAPCS64. No dot-product or FP16 arithmetic extension. */
.arch armv8-a+crc
#ifndef LFM_PREFETCH
#define LFM_PREFETCH 1
#endif
.include "src/common.inc"
.text
.balign 64
.macro DOT_FN name,native
.global \name
.type \name,%function
/* float dot(const float *a, const float *b, unsigned n); n multiple of 4 */
\name:
    movi v16.4s,#0
    movi v17.4s,#0
    movi v18.4s,#0
    movi v19.4s,#0
1:  cmp w2,#16
    b.lo 2f
    ld1 {v0.4s-v3.4s},[x0],#64
    ld1 {v4.4s-v7.4s},[x1],#64
    fmla v16.4s,v0.4s,v4.4s
    fmla v17.4s,v1.4s,v5.4s
    fmla v18.4s,v2.4s,v6.4s
    fmla v19.4s,v3.4s,v7.4s
    sub w2,w2,#16
    b 1b
2:  cbz w2,4f
3:  ldr q0,[x0],#16
    ldr q1,[x1],#16
    fmla v16.4s,v0.4s,v1.4s
    subs w2,w2,#4
    b.ne 3b
4:  .if \native
    fadd v16.4s,v16.4s,v18.4s
    fadd v17.4s,v17.4s,v19.4s
    fadd v16.4s,v16.4s,v17.4s
    .else
    fadd v16.4s,v16.4s,v17.4s
    fadd v18.4s,v18.4s,v19.4s
    fadd v16.4s,v16.4s,v18.4s
    .endif
    faddp v16.4s,v16.4s,v16.4s
    faddp s0,v16.2s
    ret
.size \name,.-\name
.endm
DOT_FN lfm_dot,0
.balign 64
DOT_FN lfm_dot_native,1

/* Four output vectors from 16 quantized bytes; no intermediate weight array. */
.macro Q4_CONVERT src
    uxtl v2.8h,\src\().8b
    uxtl2 v3.8h,\src\().16b
    uxtl v4.4s,v2.4h
    uxtl2 v5.4s,v2.8h
    uxtl v6.4s,v3.4h
    uxtl2 v7.4s,v3.8h
    ucvtf v4.4s,v4.4s
    ucvtf v5.4s,v5.4s
    ucvtf v6.4s,v6.4s
    ucvtf v7.4s,v7.4s
    fmul v4.4s,v4.4s,v20.4s
    fmul v5.4s,v5.4s,v20.4s
    fmul v6.4s,v6.4s,v20.4s
    fmul v7.4s,v7.4s,v20.4s
    fsub v4.4s,v4.4s,v21.4s
    fsub v5.4s,v5.4s,v21.4s
    fsub v6.4s,v6.4s,v21.4s
    fsub v7.4s,v7.4s,v21.4s
.endm
.macro RESULT decode
.if \decode
    st1 {v4.4s-v7.4s},[x1],#64
.else
    ld1 {v22.4s-v25.4s},[x1],#64
    fmla v28.4s,v4.4s,v22.4s
    fmla v29.4s,v5.4s,v23.4s
    fmla v30.4s,v6.4s,v24.4s
    fmla v31.4s,v7.4s,v25.4s
.endif
.endm
.macro Q4_FUNCTION name,decode
.balign 64
.global \name
.type \name,%function
/* x0 = GGML blocks, x1 = vector (or output), w2 = elements, multiple of 256 */
\name:
    movi v28.4s,#0
    movi v29.4s,#0
    movi v30.4s,#0
    movi v31.4s,#0
    movi v26.16b,#15
.Lblock\@:
    .if LFM_PREFETCH
    prfm pldl1strm,[x0,#288]
    .endif
    ldr h16,[x0]
    ldr h17,[x0,#2]
    fcvt s16,h16
    fcvt s17,h17
    mov w3,#0
.Lgroup\@:
    cmp w3,#4
    b.hs .Lscalehi\@
    add x4,x0,#4
    ldrb w5,[x4,w3,uxtw]
    add x4,x0,#8
    ldrb w6,[x4,w3,uxtw]
    and w5,w5,#63
    and w6,w6,#63
    b .Lscaleend\@
.Lscalehi\@:
    sub w4,w3,#4
    add x7,x0,#4
    ldrb w5,[x7,w4,uxtw]
    add x7,x0,#8
    ldrb w6,[x7,w4,uxtw]
    add x7,x0,#12
    ldrb w7,[x7,w4,uxtw]
    lsr w5,w5,#6
    lsl w5,w5,#4
    and w8,w7,#15
    orr w5,w5,w8
    lsr w6,w6,#6
    lsl w6,w6,#4
    lsr w7,w7,#4
    orr w6,w6,w7
.Lscaleend\@:
    ucvtf s18,w5
    ucvtf s19,w6
    fmul s18,s18,s16
    fmul s19,s19,s17
    dup v20.4s,v18.s[0]
    dup v21.4s,v19.s[0]
    lsr w4,w3,#1
    add x4,x0,x4,lsl #5
    add x4,x4,#16
    ld1 {v0.16b,v1.16b},[x4]
    tbz w3,#0,.Llow\@
    ushr v0.16b,v0.16b,#4
    ushr v1.16b,v1.16b,#4
.Llow\@:
    and v0.16b,v0.16b,v26.16b
    and v1.16b,v1.16b,v26.16b
    Q4_CONVERT v0
    RESULT \decode
    Q4_CONVERT v1
    RESULT \decode
    add w3,w3,#1
    cmp w3,#8
    b.lo .Lgroup\@
    add x0,x0,#144
    subs w2,w2,#256
    b.ne .Lblock\@
    fadd v28.4s,v28.4s,v29.4s
    fadd v30.4s,v30.4s,v31.4s
    fadd v28.4s,v28.4s,v30.4s
    faddp v28.4s,v28.4s,v28.4s
    faddp s0,v28.2s
    ret
.size \name,.-\name
.endm
Q4_FUNCTION lfm_q4_dot,0
Q4_FUNCTION lfm_q4_unpack,1

.macro Q6_CONVERT src,scale
    sshll v2.8h,\src\().8b,#0
    sshll2 v3.8h,\src\().16b,#0
    sshll v4.4s,v2.4h,#0
    sshll2 v5.4s,v2.8h,#0
    sshll v6.4s,v3.4h,#0
    sshll2 v7.4s,v3.8h,#0
    scvtf v4.4s,v4.4s
    scvtf v5.4s,v5.4s
    scvtf v6.4s,v6.4s
    scvtf v7.4s,v7.4s
    fmul v4.4s,v4.4s,\scale\().4s
    fmul v5.4s,v5.4s,\scale\().4s
    fmul v6.4s,v6.4s,\scale\().4s
    fmul v7.4s,v7.4s,\scale\().4s
.endm
.macro Q6_FUNCTION name,decode
.balign 64
.global \name
.type \name,%function
\name:
    movi v28.4s,#0
    movi v29.4s,#0
    movi v30.4s,#0
    movi v31.4s,#0
    movi v26.16b,#15
    movi v27.16b,#3
.Lblock\@:
    .if LFM_PREFETCH
    prfm pldl1strm,[x0,#416]
    .endif
    ldr h16,[x0,#208]
    fcvt s16,h16
    mov w3,#0
.Lgroup\@:
    lsr w4,w3,#2
    and w5,w3,#1
    lsl w4,w4,#6
    add w4,w4,w5,lsl #5
    add x4,x0,w4,uxtw
    ld1 {v0.16b,v1.16b},[x4]
    tbz w3,#1,.Lql\@
    ushr v0.16b,v0.16b,#4
    ushr v1.16b,v1.16b,#4
.Lql\@:
    and v0.16b,v0.16b,v26.16b
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
    scvtf s18,w5
    scvtf s19,w6
    fmul s18,s18,s16
    fmul s19,s19,s16
    dup v20.4s,v18.s[0]
    dup v21.4s,v19.s[0]
    Q6_CONVERT v0,v20
    RESULT \decode
    Q6_CONVERT v1,v21
    RESULT \decode
    add w3,w3,#1
    cmp w3,#8
    b.lo .Lgroup\@
    add x0,x0,#210
    subs w2,w2,#256
    b.ne .Lblock\@
    fadd v28.4s,v28.4s,v29.4s
    fadd v30.4s,v30.4s,v31.4s
    fadd v28.4s,v28.4s,v30.4s
    faddp v28.4s,v28.4s,v28.4s
    faddp s0,v28.2s
    ret
.size \name,.-\name
.endm
Q6_FUNCTION lfm_q6_dot,0
Q6_FUNCTION lfm_q6_unpack,1

.balign 64
.global lfm_add
.type lfm_add,%function
lfm_add: /* x0 += x1, w2 elements, multiple of four */
    ldr q0,[x0]
    ldr q1,[x1],#16
    fadd v0.4s,v0.4s,v1.4s
    str q0,[x0],#16
    subs w2,w2,#4
    b.ne lfm_add
    ret
.size lfm_add,.-lfm_add

.global lfm_rms
.type lfm_rms,%function
/* x0 out, x1 input, x2 weights, w3 n, s0 epsilon. Aliasing out=input is valid. */
lfm_rms:
    mov x4,x1
    mov w5,w3
    movi v16.4s,#0
1:  ldr q1,[x4],#16
    fmla v16.4s,v1.4s,v1.4s
    subs w5,w5,#4
    b.ne 1b
    faddp v16.4s,v16.4s,v16.4s
    faddp s16,v16.2s
    ucvtf s17,w3
    fdiv s16,s16,s17
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
.size lfm_rms,.-lfm_rms

.global lfm_swiglu
.type lfm_swiglu,%function
lfm_swiglu: /* x0 gate *= sigmoid(gate)*up, x1 up, w2 n */
    ENTER
    mov x19,x0
    mov x20,x1
    mov w21,w2
1:  ldr s8,[x19]
    ldr s9,[x20],#4
    fneg s0,s8
    bl expf
    fmov s1,#1.0
    fadd s0,s0,s1
    fdiv s0,s8,s0
    fmul s0,s0,s9
    str s0,[x19],#4
    subs w21,w21,#1
    b.ne 1b
    LEAVE
.size lfm_swiglu,.-lfm_swiglu
.section .note.GNU-stack,"",%progbits
