/* Validation-only A/B graph and recurrent rejection probe. Never enabled in
 * measured/user application. Output float/state frames compared byte for byte.
 */
.ifdef H33_PAIR_PROBE
.text
h33_pair_probe: /* next position after ordinary prefill */
    ENTER
    mov w20,w0
    add w0,w20,#2
    cmp w0,#CTX
    b.hs context_error
    bl argmax
    mov w21,w0
    add w22,w21,#1
    ldr w0,=VOCAB
    cmp w22,w0
    b.lo 1f
    sub w22,w22,w0
1:
    .ifdef H33_PROBE_SINGLE
    mov w0,w21
    mov w1,w20
    mov w2,#1
    bl forward
    bl h33_save_all_conv
    .else
    mov w0,w21
    mov w1,w22
    mov w2,w20
    bl h33_forward_pair
    .endif
    mov w0,#1
    ADDR x1,logits
    ldr x2,=VOCAB*4
    bl write_all
    mov w0,#1
    ADDR x1,h33_conv_checkpoint
    ldr x2,=LAYERS*DIM*(CONV-1)*4
    bl write_all
    .ifdef H33_PROBE_SINGLE
    mov w0,w22
    add w1,w20,#1
    mov w2,#1
    bl forward
    ADDR x1,logits
    .else
    ADDR x1,h33_logits_b
    .endif
    mov w0,#1
    ldr x2,=VOCAB*4
    bl write_all
    bl h33_probe_dump_state
    bl h33_restore_conv
    add w0,w21,#2
    ldr w1,=VOCAB
    cmp w0,w1
    b.lo 2f
    sub w0,w0,w1
2:  add w1,w20,#1
    mov w2,#1
    bl forward
    mov w0,#1
    ADDR x1,logits
    ldr x2,=VOCAB*4
    bl write_all
    bl h33_probe_dump_state
    LEAVE
h33_save_all_conv:
    ENTER
    mov w19,#0
1:  ADDR x0,layer_types
    ldr w0,[x0,w19,uxtw #2]
    cbnz w0,2f
    mov w0,w19
    bl h33_save_conv_one
2:  add w19,w19,#1
    cmp w19,#LAYERS
    b.lo 1b
    LEAVE
h33_probe_dump_state:
    ENTER
    mov w0,#1
    ADDR x1,conv_states
    ldr x2,=LAYERS*DIM*(CONV-1)*4
    bl write_all
    mov w0,#1
    ADDR x1,key_cache
    ldr x2,=LAYERS*CTX*KW*4
    bl write_all
    mov w0,#1
    ADDR x1,value_cache
    ldr x2,=LAYERS*CTX*KW*4
    bl write_all
    LEAVE
.endif
