/* Exact operation sequence of pinned ggml's Arm NEON ggml_v_expf/silu.
 * Source: ggml/src/ggml-cpu/vec.h at 8642541b255bfb5dc39cd351a178be09013a55d6.
 * Adapted to assembly; original MIT notice: references/ggml-LICENSE.
 * Coefficients load once for the model's 7680-element FFN, not per lane.
 */
.text
.balign 64
.global lfm_swiglu_native
.type lfm_swiglu_native,%function
lfm_swiglu_native: /* x0 in/out gate, x1 up, w2 n (multiple of four) */
    adrp x3,native_exp_constants
    add x3,x3,:lo12:native_exp_constants
    ld1 {v16.4s,v17.4s,v18.4s,v19.4s},[x3],#64
    ld1 {v20.4s,v21.4s,v22.4s,v23.4s},[x3],#64
    ld1 {v24.4s,v25.4s,v26.4s,v27.4s},[x3],#64
    ld1 {v28.4s,v29.4s},[x3]
1:  ldr q0,[x0]
    movi v1.4s,#0
    fsub v1.4s,v1.4s,v0.4s
    mov v2.16b,v16.16b
    fmla v2.4s,v1.4s,v17.4s  /* z */
    fsub v3.4s,v2.4s,v16.4s /* n */
    mov v4.16b,v1.16b
    fmls v4.4s,v3.4s,v18.4s
    fmls v4.4s,v3.4s,v19.4s /* b */
    shl v5.4s,v2.4s,#23      /* exponent bits */
    add v6.4s,v5.4s,v25.4s  /* k (integer add of float bit patterns) */
    facgt v7.4s,v3.4s,v26.4s
    fmul v30.4s,v4.4s,v4.4s /* u */
    mov v31.16b,v21.16b
    fmla v31.4s,v22.4s,v4.4s
    mov v2.16b,v23.16b
    fmla v2.4s,v24.4s,v4.4s
    fmla v31.4s,v2.4s,v30.4s
    fmul v2.4s,v20.4s,v4.4s
    fmla v2.4s,v31.4s,v30.4s /* j */
    addp d31,v7.2d
    fmov x4,d31
    cbnz x4,3f
    fmla v6.4s,v2.4s,v6.4s
2:  fadd v6.4s,v6.4s,v25.4s
    fdiv v0.4s,v0.4s,v6.4s
    ldr q1,[x1],#16
    fmul v0.4s,v0.4s,v1.4s
    str q0,[x0],#16
    subs w2,w2,#4
    b.ne 1b
    ret
3:  facgt v1.4s,v3.4s,v27.4s
    fcmle v31.4s,v3.4s,#0.0
    and v31.16b,v31.16b,v28.16b
    add v30.4s,v31.4s,v29.4s /* s1 */
    sub v5.4s,v5.4s,v31.4s  /* s2 */
    fmul v3.4s,v30.4s,v30.4s
    fmla v5.4s,v5.4s,v2.4s
    fmul v5.4s,v5.4s,v30.4s
    fmla v6.4s,v6.4s,v2.4s
    bsl v7.16b,v5.16b,v6.16b
    bsl v1.16b,v3.16b,v7.16b
    mov v6.16b,v1.16b
    b 2b
.size lfm_swiglu_native,.-lfm_swiglu_native

.section .rodata
.balign 64
native_exp_constants:
    .rept 4
    .word 0x4b400000  /* 0x1.8p23 */
    .endr
    .rept 4
    .word 0x3fb8aa3b  /* 0x1.715476p+0 */
    .endr
    .rept 4
    .word 0x3f317200  /* 0x1.62e4p-1 */
    .endr
    .rept 4
    .word 0x35bfbe8e  /* 0x1.7f7d1cp-20 */
    .endr
    .rept 4
    .word 0x3f7ffff6  /* 0x1.ffffecp-1 */
    .endr
    .rept 4
    .word 0x3efffedb  /* 0x1.fffdb6p-2 */
    .endr
    .rept 4
    .word 0x3e2aaf33  /* 0x1.555e66p-3 */
    .endr
    .rept 4
    .word 0x3d2b9f17  /* 0x1.573e2ep-5 */
    .endr
    .rept 4
    .word 0x3c072010  /* 0x1.0e4020p-7 */
    .endr
    .rept 4
    .word 0x3f800000  /* 1.0, also integer exponent bias bits */
    .endr
    .rept 4
    .word 0x42fc0000  /* 126.0 */
    .endr
    .rept 4
    .word 0x43400000  /* 192.0 */
    .endr
    .rept 4
    .word 0x82000000
    .endr
    .rept 4
    .word 0x7f000000
    .endr

.text
.balign 64
.global lfm_exp4_native
.type lfm_exp4_native,%function
lfm_exp4_native: /* v0 FP32x4 -> v0 exp, pinned native polynomial */
    adrp x3,native_exp_constants
    add x3,x3,:lo12:native_exp_constants
    ld1 {v16.4s,v17.4s,v18.4s,v19.4s},[x3],#64
    ld1 {v20.4s,v21.4s,v22.4s,v23.4s},[x3],#64
    ld1 {v24.4s,v25.4s,v26.4s,v27.4s},[x3],#64
    ld1 {v28.4s,v29.4s},[x3]
    mov v1.16b,v0.16b
    mov v2.16b,v16.16b
    fmla v2.4s,v1.4s,v17.4s  /* z */
    fsub v3.4s,v2.4s,v16.4s /* n */
    mov v4.16b,v1.16b
    fmls v4.4s,v3.4s,v18.4s
    fmls v4.4s,v3.4s,v19.4s /* b */
    shl v5.4s,v2.4s,#23      /* exponent bits */
    add v6.4s,v5.4s,v25.4s  /* k (integer add of float bit patterns) */
    facgt v7.4s,v3.4s,v26.4s
    fmul v30.4s,v4.4s,v4.4s /* u */
    mov v31.16b,v21.16b
    fmla v31.4s,v22.4s,v4.4s
    mov v2.16b,v23.16b
    fmla v2.4s,v24.4s,v4.4s
    fmla v31.4s,v2.4s,v30.4s
    fmul v2.4s,v20.4s,v4.4s
    fmla v2.4s,v31.4s,v30.4s /* j */
    addp d31,v7.2d
    fmov x4,d31
    cbnz x4,3f
    fmla v6.4s,v2.4s,v6.4s
    mov v0.16b,v6.16b
    ret
3:  facgt v1.4s,v3.4s,v27.4s
    fcmle v31.4s,v3.4s,#0.0
    and v31.16b,v31.16b,v28.16b
    add v30.4s,v31.4s,v29.4s /* s1 */
    sub v5.4s,v5.4s,v31.4s  /* s2 */
    fmul v3.4s,v30.4s,v30.4s
    fmla v5.4s,v5.4s,v2.4s
    fmul v5.4s,v5.4s,v30.4s
    fmla v6.4s,v6.4s,v2.4s
    bsl v7.16b,v5.16b,v6.16b
    bsl v1.16b,v3.16b,v7.16b
    mov v6.16b,v1.16b
    mov v0.16b,v6.16b
    ret
.size lfm_exp4_native,.-lfm_exp4_native

.balign 64
.global lfm_softmax_native
.type lfm_softmax_native,%function
lfm_softmax_native: /* x0 scores, w1 valid count, s0 max; buffer capacity rounds to4 */
    ENTER
    mov x19,x0
    mov w20,w1
    fmov s8,s0
    add w21,w20,#3
    and w21,w21,#0xfffffffc
    mov w22,w20
    ldr w0,=0xff800000
1:  cmp w22,w21
    b.hs 2f
    str w0,[x19,w22,uxtw #2]
    add w22,w22,#1
    b 1b
2:  fmov d9,xzr
    mov w22,#0
3:  add x23,x19,x22,lsl #2
    ldr q0,[x23]
    dup v1.4s,v8.s[0]
    fsub v0.4s,v0.4s,v1.4s
    bl lfm_exp4_native
    str q0,[x23]
    faddp v0.4s,v0.4s,v0.4s
    faddp s0,v0.2s
    fcvt d0,s0
    fadd d9,d9,d0
    add w22,w22,#4
    cmp w22,w21
    b.lo 3b
    fmov d0,#1.0
    fdiv d9,d0,d9
    fcvt s8,d9
    dup v1.4s,v8.s[0]
    mov w22,#0
4:  add x23,x19,x22,lsl #2
    ldr q0,[x23]
    fmul v0.4s,v0.4s,v1.4s
    str q0,[x23]
    add w22,w22,#4
    cmp w22,w21
    b.lo 4b
    LEAVE
.size lfm_softmax_native,.-lfm_softmax_native

/* Four output dimensions in parallel, matching ggml's dot reduction over
 * sequence positions. The sixteen accumulators correspond to positions
 * modulo16; absent/padded positions contribute zero. */
.macro VALUE_STEP accumulator
    ldr s1,[x0],#4
    dup v1.4s,v1.s[0]
    ldr q0,[x1]
    add x1,x1,x3
    fmla v\accumulator\().4s,v0.4s,v1.4s
    subs w2,w2,#1
    b.eq 2f
.endm
.balign 64
.global lfm_value_native
.type lfm_value_native,%function
lfm_value_native: /* x0 probs, x1 strided values, w2 count>0, x3 byte stride -> v0 */
    movi v16.4s,#0
    movi v17.4s,#0
    movi v18.4s,#0
    movi v19.4s,#0
    movi v20.4s,#0
    movi v21.4s,#0
    movi v22.4s,#0
    movi v23.4s,#0
    movi v24.4s,#0
    movi v25.4s,#0
    movi v26.4s,#0
    movi v27.4s,#0
    movi v28.4s,#0
    movi v29.4s,#0
    movi v30.4s,#0
    movi v31.4s,#0
1:  VALUE_STEP 16
    VALUE_STEP 17
    VALUE_STEP 18
    VALUE_STEP 19
    VALUE_STEP 20
    VALUE_STEP 21
    VALUE_STEP 22
    VALUE_STEP 23
    VALUE_STEP 24
    VALUE_STEP 25
    VALUE_STEP 26
    VALUE_STEP 27
    VALUE_STEP 28
    VALUE_STEP 29
    VALUE_STEP 30
    VALUE_STEP 31
    b 1b
2:  fadd v16.4s,v16.4s,v24.4s
    fadd v17.4s,v17.4s,v25.4s
    fadd v18.4s,v18.4s,v26.4s
    fadd v19.4s,v19.4s,v27.4s
    fadd v20.4s,v20.4s,v28.4s
    fadd v21.4s,v21.4s,v29.4s
    fadd v22.4s,v22.4s,v30.4s
    fadd v23.4s,v23.4s,v31.4s
    fadd v16.4s,v16.4s,v20.4s
    fadd v17.4s,v17.4s,v21.4s
    fadd v18.4s,v18.4s,v22.4s
    fadd v19.4s,v19.4s,v23.4s
    fadd v16.4s,v16.4s,v17.4s
    fadd v18.4s,v18.4s,v19.4s
    fadd v0.4s,v16.4s,v18.4s
    ret
.size lfm_value_native,.-lfm_value_native
