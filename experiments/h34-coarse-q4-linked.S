/* Runtime-link variant of native-screen H34 helper: constants follow RET in
 * read-only text so embedded tokenizer assets cannot exceed literal reach.
 * Arithmetic/instruction sequence is unchanged; original screen source frozen. */
/* H34 centered4-bit coarse dot. NOT original Q4_K quantization.
 * Each146B block:128B low/high code pairs,16 signed scales,original half d.
 * q_coarse=4*code-30, scale per16 inputs; same integer/FMA order as Q6.
 * This helper alone cannot select a token; original-head certificate/refine
 * remains mandatory and is not yet implemented.
 */
.arch armv8-a+crc
#ifndef LFM_PREFETCH
#define LFM_PREFETCH 1
#endif
.text
.macro H34_GROUP high
    .if \high
    ushr v6.16b,v0.16b,#4
    ushr v7.16b,v1.16b,#4
    .else
    and v6.16b,v0.16b,v26.16b
    and v7.16b,v1.16b,v26.16b
    .endif
    shl v6.16b,v6.16b,#2
    shl v7.16b,v7.16b,#2
    sub v6.16b,v6.16b,v27.16b
    sub v7.16b,v7.16b,v27.16b
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
.global lfm_q4_signed_q8_dot_h34
.type lfm_q4_signed_q8_dot_h34,%function
lfm_q4_signed_q8_dot_h34: /* x0coarse row,x1canonical Q8,w2columns ->s0 */
    fmov s28,wzr
    movi v26.16b,#15
    movi v27.16b,#30
1:  .if LFM_PREFETCH
    prfm pldl1strm,[x0,#288]
    .endif
    ldr h16,[x0,#144]
    fcvt s16,h16
    ldr s17,[x1]
    fmul s16,s16,s17
    mov w9,#0
    mov x4,x0
    add x8,x0,#128
    add x7,x1,#32
    mov w13,#4
2:  ld1 {v0.16b,v1.16b},[x4],#32
    H34_GROUP 0
    H34_GROUP 1
    subs w13,w13,#1
    b.ne 2b
    scvtf s18,w9
    fmadd s28,s16,s18,s28
    add x0,x0,#146
    add x1,x1,#320
    subs w2,w2,#256
    b.ne 1b
    fmov s0,s28
    ret
.size lfm_q4_signed_q8_dot_h34,.-lfm_q4_signed_q8_dot_h34

/* Fixed2048-column head: compute coarse score and residual bound together,
 * keeping each block's d hot. Void(row,q8,E8,A8,out_score,out_upper).
 * E/A must be verified upper norms, bound to the original checkpoint/query.
 * Unsafe arithmetic environments/data return score0,upper+inf: callers MUST
 * refine every such row. This function alone never chooses a token.
 */
.balign 64
.global lfm_coarse_score_upper_h34
.type lfm_coarse_score_upper_h34,%function
lfm_coarse_score_upper_h34:
    mov x14,x4
    mov x15,x5
    mrs x17,fpcr
    tst x17,#0x1c00000 /* RMode/FZ: RN with gradual underflow required */
    b.ne 9f
    fmov s28,wzr
    fmov d29,xzr
    fmov d30,xzr
    movi v26.16b,#15
    movi v27.16b,#30
    mov w16,#8
1:  .if LFM_PREFETCH
    prfm pldl1strm,[x0,#288]
    .endif
    ldr h16,[x0,#144]
    fcvt s16,h16
    ldr s17,[x1]
    fmul s16,s16,s17
    mov w9,#0
    mov x4,x0
    add x8,x0,#128
    add x7,x1,#32
    mov w13,#4
2:  ld1 {v0.16b,v1.16b},[x4],#32
    H34_GROUP 0
    H34_GROUP 1
    subs w13,w13,#1
    b.ne 2b
    scvtf s18,w9
    fmadd s28,s16,s18,s28
    fmov w17,s16
    and w17,w17,#0x7fffffff
    mov w5,#0x7f800000
    cmp w17,w5
    b.hs 9f
    fcvt d18,s16
    fabs d18,d18
    fadd d30,d30,d18
    ldr s23,[x2],#4
    ldr s24,[x3],#4
    fmov w17,s23
    tbnz w17,#31,9f
    cmp w17,w5
    b.hs 9f
    fmov w17,s24
    tbnz w17,#31,9f
    cmp w17,w5
    b.hs 9f
    fcvt d23,s23
    fcvt d24,s24
    fmul d25,d18,d23
    fmadd d29,d25,d24,d29
    add x0,x0,#146
    add x1,x1,#320
    subs w16,w16,#1
    b.ne 1b
    fmov w17,s28
    and w17,w17,#0x7fffffff
    mov w5,#0x7f800000
    cmp w17,w5
    b.hs 9f
    ldr d23,h34_sum_alpha_limit
    fcmp d30,d23
    b.hs 9f
    ldr d24,h34_bound_inflation
    ldr d25,h34_rounding_coefficient
    ldr d26,h34_underflow_slack
    fmul d29,d29,d24
    fmadd d29,d30,d25,d29
    fadd d29,d29,d26
    fcvt d31,s28
    fadd d0,d31,d29
    /* next_up64 of a finite signed sum, including either zero sign. */
    fcmp d0,#0.0
    b.eq 5f
    fmov x17,d0
    tbnz x17,#63,4f
    add x17,x17,#1
    b 6f
4:  sub x17,x17,#1
    b 6f
5:  mov x17,#1
6:  fmov d0,x17
    str s28,[x14]
    str d0,[x15]
    ret
9:  str wzr,[x14]
    ldr d0,h34_positive_infinity
    str d0,[x15]
    ret
.size lfm_coarse_score_upper_h34,.-lfm_coarse_score_upper_h34
.text
.balign 8
h34_sum_alpha_limit: .quad 0x4620000000000000 /* 2^99, conservative overflow cutoff */
h34_bound_inflation: .quad 0x3ff0000000001000 /* 1+2^-40 exactly */
h34_rounding_coefficient: .double 160.0
h34_underflow_slack: .double 0.001
h34_positive_infinity: .quad 0x7ff0000000000000
.section .note.GNU-stack,"",%progbits
