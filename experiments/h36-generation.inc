/* H36 scalar-default generation; optional explicit verified modes unchanged. */
/* Causal short-cycle drafts, always verified by original model logits.
 * LFM_SPEC=0 scalar,1 adaptive(default),2 last-ID probe,3 different-ID probe.
 * Only real emitted IDs enter the history ring. Never cache completions.
 */
.text
h33_generate: /* w0 next absolute position,w1 maximum new outputs */
    ENTER
    mov w20,w0
    mov w19,w1
    mov w23,#0 /* no pending verified second output */
    mov w24,#0 /* ordinary-text default: no drafting */
    ADDR x0,h33_spec_env
    bl getenv
    cbz x0,1f
    mov x1,#0
    mov w2,#10
    bl strtol
    cmp x0,#0
    b.lt h33_spec_error
    cmp x0,#3
    b.gt h33_spec_error
    mov w24,w0
1:  ADDR x0,use_q8
    ldr w0,[x0]
    cbnz w0,2f
    mov w24,#0
2:  ADDR x0,h33_active_mode
    str w24,[x0]
    bl argmax
    mov w21,w0
h33_emit_next:
    ldr w0,=EOS
    cmp w21,w0
    b.ne 3f
    ADDR x0,ignore_eos
    ldr w0,[x0]
    cbz w0,h33_generation_done
3:  ADDR x0,generated_count
    ldr w1,[x0]
    and w2,w1,#31
    ADDR x3,h33_history
    str w21,[x3,w2,uxtw #2]
    add w1,w1,#1
    str w1,[x0]
    mov w0,w21
    bl emit_token
    subs w19,w19,#1
    b.eq h33_generation_done
    cbz w23,4f
    mov w21,w22
    mov w23,#0
    b h33_emit_next
4:  cbz w24,h33_scalar_step
    cmp w19,#2
    b.lo h33_scalar_step
    cmp w24,#1
    b.ne 5f
    ADDR x0,generated_count
    ldr w0,[x0]
    bl h33_cycle_draft
    cbz w0,h33_scalar_step
    mov w25,w1
    b 7f
5:  mov w25,w21
    cmp w24,#3
    b.ne 7f
    add w25,w25,#1
    ldr w0,=VOCAB
    cmp w25,w0
    b.lo 7f
    sub w25,w25,w0
7:  mov w0,w21
    mov w1,w25
    mov w2,w20
    bl h33_forward_pair
    ADDR x0,h33_batches
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    ADDR x0,h33_evaluated
    ldr w1,[x0]
    add w1,w1,#2
    str w1,[x0]
    bl argmax
    mov w26,w0
    cmp w26,w25
    b.ne h33_reject
    bl h33_argmax_b
    mov w22,w0
    mov w23,#1
    mov w21,w26
    add w20,w20,#2
    ADDR x0,decode_steps
    ldr w1,[x0]
    add w1,w1,#2
    str w1,[x0]
    ADDR x0,h33_accepts
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    b h33_emit_next
h33_reject:
    bl h33_restore_conv
    mov w21,w26
    add w20,w20,#1
    ADDR x0,decode_steps
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    ADDR x0,h33_rejects
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    b h33_emit_next
h33_scalar_step:
    mov w0,w21
    mov w1,w20
    mov w2,#1
    bl forward
    add w20,w20,#1
    ADDR x0,decode_steps
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    ADDR x0,h33_evaluated
    ldr w1,[x0]
    add w1,w1,#1
    str w1,[x0]
    bl argmax
    mov w21,w0
    b h33_emit_next
h33_generation_done:
    LEAVE

h33_cycle_draft: /* count -> w0=valid,w1=proposal; only emitted history */
    ADDR x2,h33_history
    mov w3,#1
1:  mov w4,#3
    mul w4,w3,w4
    cmp w0,w4
    b.lo 5f
    mov w5,#0
    lsl w6,w3,#1
2:  sub w7,w0,#1
    sub w7,w7,w5
    and w8,w7,#31
    sub w7,w7,w3
    and w7,w7,#31
    ldr w9,[x2,w8,uxtw #2]
    ldr w10,[x2,w7,uxtw #2]
    cmp w9,w10
    b.ne 4f
    add w5,w5,#1
    cmp w5,w6
    b.lo 2b
    sub w7,w0,w3
    and w7,w7,#31
    ldr w1,[x2,w7,uxtw #2]
    mov w0,#1
    ret
4:  add w3,w3,#1
    cmp w3,#8
    b.ls 1b
5:  mov w0,#0
    ret

h33_argmax_b:
    ADDR x1,h33_logits_b
    b h33_argmax_common
h33_argmax_common: /* same first-maximum/finite semantics as scalar argmax */
    ldr s0,[x1],#4
    ldr w5,=0x7f800000
    fmov w4,s0
    and w4,w4,#0x7fffffff
    cmp w4,w5
    b.hs numeric_error
    mov w0,#0
    mov w2,#1
    ldr w3,=VOCAB
1:  ldr s1,[x1],#4
    fmov w4,s1
    and w4,w4,#0x7fffffff
    cmp w4,w5
    b.hs numeric_error
    fcmp s1,s0
    b.vs numeric_error
    b.le 2f
    fmov s0,s1
    mov w0,w2
2:  add w2,w2,#1
    cmp w2,w3
    b.lo 1b
    ret

h33_report:
    ENTER
    mov w0,#2
    ADDR x1,h33_report_format
    ADDR x2,h33_batches
    ldr w2,[x2]
    ADDR x3,h33_accepts
    ldr w3,[x3]
    ADDR x4,h33_rejects
    ldr w4,[x4]
    ADDR x5,h33_evaluated
    ldr w5,[x5]
    ADDR x6,h33_active_mode
    ldr w6,[x6]
    bl dprintf
    LEAVE
h33_spec_error:
    ADDR x0,h33_spec_error_text
    b fatal
.section .rodata
h33_spec_env: .asciz "LFM_SPEC"
h33_spec_error_text: .asciz "LFM_SPEC must be0..3 (2/3 diagnostic draft modes).\n"
h33_report_format: .asciz "H33 batches=%u accepts=%u rejects=%u evaluated_positions=%u mode=%u\n"
.bss
.balign 64
h33_history: .skip 32*4
h33_batches: .skip 4
h33_accepts: .skip 4
h33_rejects: .skip 4
h33_evaluated: .skip 4
h33_active_mode: .skip 4
.text
