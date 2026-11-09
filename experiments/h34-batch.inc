/* H33 exact layerwise two-position graph. Ordinary graph remains frozen.
 * Uses original GEMV arithmetic separately for A/B; only weight decode shared.
 */
.text
.macro H33_MAT slot,src,dst
    add x0,x19,#(\slot*32)
    ADDR x1,\src
    ADDR x2,h33_\src\()_b
    ADDR x3,\dst
    ADDR x4,h33_\dst\()_b
    bl h33_matrix
.endm
.macro H33_NORM slot,src,dst,n
    NORM \slot,\src,\dst,\n
    NORM \slot,h33_\src\()_b,h33_\dst\()_b,\n
.endm
h33_forward_pair: /* w0=A token,w1=B proposal,w2=position A; always two logits */
    ENTER
    mov w20,w2
    mov w22,w1
    mov w23,w0
    bl h34_disable_for_pair
    mov w0,w23
    bl embedding_row
    mov w0,w22
    bl h33_embedding_b
    mov w0,w20
    bl make_rope
    add w0,w20,#1
    bl h33_make_rope_b
    mov w21,#0
1:  ADDR x19,layer_tensors
    mov x0,#576
    madd x19,x21,x0,x19
    H33_NORM 0,hidden,normalized,DIM
    ADDR x0,layer_types
    ldr w0,[x0,w21,uxtw #2]
    cbnz w0,2f
    H33_MAT 7,normalized,bcx
    mov x0,x19
    mov w1,w21
    mov w2,w20
    bl h33_convolution_a_tail
    mov w0,w21
    bl h33_save_conv_one
    mov x0,x19
    mov w1,w21
    add w2,w20,#1
    bl h33_convolution_b_tail
    H33_MAT 8,conv_out,operator_out
    b 3f
2:  H33_MAT 10,normalized,query
    H33_MAT 11,normalized,key
    H33_MAT 12,normalized,value
    H33_MAT 16,normalized,head_gates
    mov x0,x19
    mov w1,w21
    mov w2,w20
    bl h33_attention_a_tail
    mov x0,x19
    mov w1,w21
    add w2,w20,#1
    bl h33_attention_b_tail
    H33_MAT 13,attn_out,operator_out
3:  ldr x0,[x19,#64]
    cbz x0,4f
    H33_NORM 2,operator_out,operator_out,DIM
4:  ADDR x0,hidden
    ADDR x1,operator_out
    mov w2,#DIM
    bl lfm_add
    ADDR x0,h33_hidden_b
    ADDR x1,h33_operator_out_b
    mov w2,#DIM
    bl lfm_add
    H33_NORM 1,hidden,normalized,DIM
    H33_MAT 4,normalized,ff_gate
    H33_MAT 5,normalized,ff_up
    ADDR x0,ff_gate
    ADDR x1,ff_up
    mov w2,#FF
    bl lfm_swiglu_native
    ADDR x0,h33_ff_gate_b
    ADDR x1,h33_ff_up_b
    mov w2,#FF
    bl lfm_swiglu_native
    H33_MAT 6,ff_gate,operator_out
    ldr x0,[x19,#96]
    cbz x0,5f
    H33_NORM 3,operator_out,operator_out,DIM
5:  ADDR x0,hidden
    ADDR x1,operator_out
    mov w2,#DIM
    bl lfm_add
    ADDR x0,h33_hidden_b
    ADDR x1,h33_operator_out_b
    mov w2,#DIM
    bl lfm_add
    add w21,w21,#1
    cmp w21,#LAYERS
    b.lo 1b
    ADDR x0,final_norm
    ADDR x1,hidden
    ADDR x2,normalized
    mov w3,#DIM
    bl norm
    ADDR x0,final_norm
    ADDR x1,h33_hidden_b
    ADDR x2,h33_normalized_b
    mov w3,#DIM
    bl norm
    ADDR x0,output_tensor
    ADDR x1,normalized
    ADDR x2,h33_normalized_b
    ADDR x3,logits
    ADDR x4,h33_logits_b
    bl h33_matrix
    LEAVE

h33_save_conv_one: /* w0=physical layer; copy state after A, before B */
    mov w3,w0
    ADDR x0,h33_conv_checkpoint
    ADDR x1,conv_states
    ldr x2,=DIM*(CONV-1)*4
    madd x0,x3,x2,x0
    madd x1,x3,x2,x1
    b memcpy

h33_restore_conv:
    ENTER
    mov w19,#0
1:  ADDR x0,layer_types
    ldr w0,[x0,w19,uxtw #2]
    cbnz w0,2f
    ADDR x0,conv_states
    ADDR x1,h33_conv_checkpoint
    ldr x2,=DIM*(CONV-1)*4
    madd x0,x19,x2,x0
    madd x1,x19,x2,x1
    bl memcpy
2:  add w19,w19,#1
    cmp w19,#LAYERS
    b.lo 1b
    LEAVE

h33_matrix: /* descriptor,floatA,floatB,outA,outB */
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    mov x22,x3
    mov x23,x4
    mov x0,x20
    ADDR x1,activation_q8
    ldr w2,[x19,#8]
    bl lfm_q8_pack_h24
    mov x0,x21
    ADDR x1,h33_activation_q8_b
    ldr w2,[x19,#8]
    bl lfm_q8_pack_h24
    ADDR x0,job_descriptor
    str x19,[x0]
    ADDR x1,activation_q8
    str x1,[x0,#8]
    str x22,[x0,#16]
    str wzr,[x0,#24]
    ADDR x1,h33_activation_q8_b
    str x1,[x0,#32]
    str x23,[x0,#40]
    ADDR x0,h33_dual_active
    mov w1,#1
    str w1,[x0]
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w0,#1
    b.eq 4f
    ldr w0,[x19,#12]
    cmp w0,#64
    b.lo 4f
    mov w19,#1
1:  ADDR x0,worker_semaphores
    add x0,x0,x19,lsl #6
    bl sem_post
    add w19,w19,#1
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w19,w0
    b.lo 1b
    bl h33_matrix_work
    mov w19,#1
2:  ADDR x0,done_semaphore
    bl sem_wait_retry
    add w19,w19,#1
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w19,w0
    b.lo 2b
    b 5f
4:  bl h33_matrix_work
5:  ADDR x0,h33_dual_active
    str wzr,[x0]
    LEAVE

h33_matrix_work:
    ENTER
    ADDR x19,job_descriptor
    ldr x0,[x19]
    ldr w25,[x0,#8]
    ldr w26,[x0,#20]
    ldr w1,[x0,#16]
    cmp w1,#12
    b.ne 1f
    ADDR x28,lfm_q4_q8_dual_h32
    b 2f
1:  ADDR x28,lfm_q6_q8_dual_h32
2:  bl tensor_ptr
    mov x20,x0
    ldr x21,[x19,#8]
    ldr x22,[x19,#32]
    ldr x23,[x19,#16]
    ldr x24,[x19,#40]
3:  add x0,x19,#24
4:  ldaxr w27,[x0]
    add w1,w27,#ROW_TILE
    stlxr w2,w1,[x0]
    cbnz w2,4b
    ldr x0,[x19]
    ldr w1,[x0,#12]
    cmp w27,w1
    b.hs 8f
    madd x0,x27,x26,x20
    add x3,x23,x27,lsl #2
    add x4,x24,x27,lsl #2
    sub w1,w1,w27
    mov w27,#ROW_TILE
    cmp w1,w27
    csel w27,w1,w27,lo
    sub sp,sp,#32
    stp x0,x3,[sp]
    str x4,[sp,#16]
5:  ldp x0,x3,[sp]
    ldr x4,[sp,#16]
    mov x1,x21
    mov x2,x22
    mov w5,w25
    blr x28
    ldp x0,x3,[sp]
    ldr x4,[sp,#16]
    add x0,x0,x26
    add x3,x3,#4
    add x4,x4,#4
    stp x0,x3,[sp]
    str x4,[sp,#16]
    subs w27,w27,#1
    b.ne 5b
    add sp,sp,#32
    b 3b
8:  LEAVE

.bss
.balign 64
h33_dual_active: .skip 4
.balign 64
h33_activation_q8_b: .skip ((FF+255)/256)*320
h33_hidden_b: .skip DIM*4
h33_normalized_b: .skip DIM*4
h33_operator_out_b: .skip DIM*4
h33_bcx_b: .skip DIM*12
h33_conv_out_b: .skip DIM*4
h33_query_b: .skip DIM*4
h33_key_b: .skip KW*4
h33_value_b: .skip KW*4
h33_attn_out_b: .skip DIM*4
h33_head_gates_b: .skip HEADS*4
h33_ff_gate_b: .skip FF*4
h33_ff_up_b: .skip FF*4
h33_logits_b: .skip VOCAB*4
h33_cos_global_b: .skip HD*2
h33_sin_global_b: .skip HD*2
h33_cos_window_b: .skip HD*2
h33_sin_window_b: .skip HD*2
.balign 64
h33_conv_checkpoint: .skip LAYERS*DIM*(CONV-1)*4
.text
