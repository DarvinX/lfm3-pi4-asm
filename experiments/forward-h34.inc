.text
.macro TRACEBUF stage,src,n,layer=21,pos=20
    .ifdef TRACE_DETAIL
    sub sp,sp,#16
    mov w0,#\stage
    str w0,[sp]
    str w\layer,[sp,#4]
    ldr w0,=\n
    str w0,[sp,#8]
    str w\pos,[sp,#12]
    mov w0,#3
    mov x1,sp
    mov x2,#16
    bl write_all
    add sp,sp,#16
    mov w0,#3
    ADDR x1,\src
    ldr x2,=(\n*4)
    bl write_all
    .endif
.endm
.macro MAT slot,src,dst
    add x0,x19,#(\slot*32)
    ADDR x1,\src
    ADDR x2,\dst
    bl matrix
.endm
.macro NORM slot,src,dst,n
    add x0,x19,#(\slot*32)
    ADDR x1,\src
    ADDR x2,\dst
    mov w3,#\n
    bl norm
.endm

forward: /* w0 token, w1 absolute position, w2 produce logits */
    ENTER
    mov w20,w1
    mov w22,w2
    bl embedding_row
    mov w21,#-1
    TRACEBUF 0,hidden,DIM
    mov w0,w20
    bl make_rope
    mov w21,#0
1:  ADDR x19,layer_tensors
    mov x0,#576
    madd x19,x21,x0,x19
    NORM 0,hidden,normalized,DIM
    TRACEBUF 1,normalized,DIM
    ADDR x0,layer_types
    ldr w0,[x0,w21,uxtw #2]
    cbnz w0,2f
    mov x0,x19
    mov w1,w21
    mov w2,w20
    bl convolution
    b 3f
2:  mov x0,x19
    mov w1,w21
    mov w2,w20
    bl attention
3:  add x0,x19,#64
    ldr x0,[x0]
    cbz x0,4f
    NORM 2,operator_out,operator_out,DIM
4:  ADDR x0,hidden
    ADDR x1,operator_out
    mov w2,#DIM
    bl lfm_add
    TRACEBUF 2,hidden,DIM
    NORM 1,hidden,normalized,DIM
    TRACEBUF 3,normalized,DIM
    MAT 4,normalized,ff_gate
    TRACEBUF 4,ff_gate,FF
    MAT 5,normalized,ff_up
    TRACEBUF 5,ff_up,FF
    ADDR x0,ff_gate
    ADDR x1,ff_up
    mov w2,#FF
    ADDR x3,use_q8
    ldr w3,[x3]
    cbnz w3,7f
    bl lfm_swiglu
    b 8f
7:  bl lfm_swiglu_native
8:
    TRACEBUF 6,ff_gate,FF
    MAT 6,ff_gate,operator_out
    TRACEBUF 7,operator_out,DIM
    add x0,x19,#96
    ldr x0,[x0]
    cbz x0,5f
    NORM 3,operator_out,operator_out,DIM
5:  ADDR x0,hidden
    ADDR x1,operator_out
    mov w2,#DIM
    bl lfm_add
    TRACEBUF 8,hidden,DIM
    .ifdef TRACE
    mov w0,#3
    ADDR x1,hidden
    mov x2,#DIM*4
    bl write_all
    .endif
    add w21,w21,#1
    cmp w21,#LAYERS
    b.lo 1b
    cbz w22,6f
    ADDR x0,final_norm
    ADDR x1,hidden
    ADDR x2,normalized
    mov w3,#DIM
    bl norm
    TRACEBUF 9,normalized,DIM
    ADDR x0,output_tensor
    ADDR x1,normalized
    ADDR x2,logits
    bl h34_output
    TRACEBUF 10,logits,VOCAB
6:
    LEAVE

convolution: /* x0 layer table, w1 layer id */
    ENTER
    mov x19,x0
    mov w28,w1
    mov w27,w2
    ADDR x20,conv_states
    ldr x2,=DIM*(CONV-1)*4
    madd x20,x1,x2,x20
    MAT 7,normalized,bcx
    TRACEBUF 11,bcx,(DIM*3),28,27
    add x0,x19,#288
    bl tensor_ptr
    mov x21,x0
    ADDR x22,bcx
    ldr x0,=DIM*4
    add x23,x22,x0
    add x24,x23,x0
    ADDR x25,conv_out
    mov w26,#DIM
1:  ldr s0,[x22],#4
    ldr s1,[x24],#4
    fmul s8,s0,s1
    fmov s0,wzr
    mov w2,#0
2:  ldr s1,[x20,w2,uxtw #2]
    ldr s2,[x21,w2,uxtw #2]
    fmadd s0,s1,s2,s0
    add w2,w2,#1
    cmp w2,#CONV-1
    b.lo 2b
    ldr s2,[x21,#(CONV-1)*4]
    fmadd s0,s8,s2,s0
    ldr s1,[x23],#4
    fmul s0,s0,s1
    str s0,[x25],#4
    mov w2,#0
3:  add w3,w2,#1
    ldr s1,[x20,w3,uxtw #2]
    str s1,[x20,w2,uxtw #2]
    add w2,w2,#1
    cmp w2,#CONV-2
    b.lo 3b
    str s8,[x20,#(CONV-2)*4]
    add x20,x20,#(CONV-1)*4
    add x21,x21,#CONV*4
    subs w26,w26,#1
    b.ne 1b
    TRACEBUF 12,conv_out,DIM,28,27
    MAT 8,conv_out,operator_out
    TRACEBUF 13,operator_out,DIM,28,27
    LEAVE

make_rope: /* w0 position -> sin/cos arrays for global and window attention */
    ADDR x1,use_q8
    ldr w1,[x1]
    cbnz w1,make_rope_native
    ENTER
    mov w19,w0
    ADDR x0,rope_factors
    ldr x1,[x0]
    mov x21,#0
    cbz x1,1f
    bl tensor_ptr
    mov x21,x0
1:  mov w20,#0
2:  ADDR x23,cos_global
    ADDR x24,sin_global
    ADDR x0,rope_global
    cbz w20,3f
    ADDR x23,cos_window
    ADDR x24,sin_window
    ADDR x0,rope_window
3:  ldr s9,[x0]
    mov w22,#0
4:  fmov s0,s9
    ucvtf s1,w22
    fmov s2,#-2.0
    fmul s1,s1,s2
    mov w0,#HD
    ucvtf s2,w0
    fdiv s1,s1,s2
    bl powf
    cbnz w20,5f
    cbz x21,5f
    ldr s1,[x21,w22,uxtw #2]
    fdiv s0,s0,s1
5:  ucvtf s1,w19
    fmul s8,s0,s1
    fmov s0,s8
    bl cosf
    str s0,[x23,w22,uxtw #2]
    fmov s0,s8
    bl sinf
    str s0,[x24,w22,uxtw #2]
    add w22,w22,#1
    cmp w22,#HD/2
    b.lo 4b
    add w20,w20,#1
    cmp w20,#2
    b.lo 2b
    LEAVE

make_rope_native: /* Pinned native recurrence: theta = position; theta *= scale. */
    ENTER
    mov w19,w0
    ADDR x0,rope_factors
    ldr x1,[x0]
    mov x21,#0
    cbz x1,1f
    bl tensor_ptr
    mov x21,x0
1:  mov w20,#0
2:  ADDR x23,cos_global
    ADDR x24,sin_global
    ADDR x0,rope_global
    cbz w20,3f
    ADDR x23,cos_window
    ADDR x24,sin_window
    ADDR x0,rope_window
3:  ldr s0,[x0]
    fmov s1,#-2.0
    mov w0,#HD
    ucvtf s2,w0
    fdiv s1,s1,s2
    bl powf
    fmov s10,s0
    ucvtf s11,w19
    mov w22,#0
4:  fmov s0,s11
    cbnz w20,5f
    cbz x21,5f
    ldr s1,[x21,w22,uxtw #2]
    fdiv s0,s0,s1
5:  fmov s8,s0
    bl cosf
    str s0,[x23,w22,uxtw #2]
    fmov s0,s8
    bl sinf
    str s0,[x24,w22,uxtw #2]
    fmul s11,s11,s10
    add w22,w22,#1
    cmp w22,#HD/2
    b.lo 4b
    add w20,w20,#1
    cmp w20,#2
    b.lo 2b
    LEAVE

rotate: /* x0 heads, x1 cos, x2 sin, w3 head count; NeoX split halves */
    ADDR x8,use_q8
    ldr w8,[x8]
    cbnz w8,rotate_native
    mov x4,x1
    mov x5,x2
1:  mov x1,x4
    mov x2,x5
    add x6,x0,#HD*2
    mov w7,#HD/2
2:  ldr q0,[x0]
    ldr q1,[x6]
    ldr q2,[x1],#16
    ldr q3,[x2],#16
    fmul v4.4s,v0.4s,v2.4s
    fmls v4.4s,v1.4s,v3.4s
    fmul v5.4s,v1.4s,v2.4s
    fmla v5.4s,v0.4s,v3.4s
    str q4,[x0],#16
    str q5,[x6],#16
    subs w7,w7,#4
    b.ne 2b
    mov x0,x6
    subs w3,w3,#1
    b.ne 1b
    ret

rotate_native: /* x0 heads, x1 cos, x2 sin, w3 head count; NeoX split halves */
    mov x4,x1
    mov x5,x2
1:  mov x1,x4
    mov x2,x5
    add x6,x0,#HD*2
    mov w7,#HD/2
2:  ldr q0,[x0]
    ldr q1,[x6]
    ldr q2,[x1],#16
    ldr q3,[x2],#16
    fmul v4.4s,v1.4s,v3.4s
    fneg v4.4s,v4.4s
    fmla v4.4s,v0.4s,v2.4s
    fmul v5.4s,v1.4s,v2.4s
    fmla v5.4s,v0.4s,v3.4s
    str q4,[x0],#16
    str q5,[x6],#16
    subs w7,w7,#4
    b.ne 2b
    mov x0,x6
    subs w3,w3,#1
    b.ne 1b
    ret

attention: /* x0 layer table,w1 layer id,w2 position */
    ENTER
    mov x19,x0
    mov w20,w1
    mov w21,w2
    MAT 10,normalized,query
    MAT 11,normalized,key
    MAT 12,normalized,value
    MAT 16,normalized,head_gates
    mov w22,#0
1:  add x0,x19,#448
    ADDR x1,query
    mov x2,#HD*4
    madd x1,x22,x2,x1
    mov x2,x1
    mov w3,#HD
    bl norm
    add w22,w22,#1
    cmp w22,#HEADS
    b.lo 1b
    mov w22,#0
2:  add x0,x19,#480
    ADDR x1,key
    mov x2,#HD*4
    madd x1,x22,x2,x1
    mov x2,x1
    mov w3,#HD
    bl norm
    add w22,w22,#1
    cmp w22,#KV
    b.lo 2b
    TRACEBUF 15,query,DIM,20,21
    TRACEBUF 16,key,KW,20,21
    ADDR x0,layer_types
    ldr w0,[x0,w20,uxtw #2]
    mov w28,w0
    ADDR x0,query
    ADDR x1,cos_global
    ADDR x2,sin_global
    cmp w28,#2
    b.ne 3f
    ADDR x1,cos_window
    ADDR x2,sin_window
3:  mov w3,#HEADS
    bl rotate
    ADDR x0,key
    ADDR x1,cos_global
    ADDR x2,sin_global
    cmp w28,#2
    b.ne 4f
    ADDR x1,cos_window
    ADDR x2,sin_window
4:  mov w3,#KV
    bl rotate
    TRACEBUF 17,query,DIM,20,21
    TRACEBUF 18,key,KW,20,21
    ldr x0,=CTX*KW*4
    ADDR x22,key_cache
    madd x22,x20,x0,x22
    ADDR x23,value_cache
    madd x23,x20,x0,x23
    mov x2,#KW*4
    madd x0,x21,x2,x22
    ADDR x1,key
    bl memcpy
    mov x2,#KW*4
    madd x0,x21,x2,x23
    ADDR x1,value
    bl memcpy
    mov w26,#0
    cmp w28,#2
    b.ne 5f
    add w26,w21,#1
    subs w26,w26,#WINDOW
    csel w26,w26,wzr,ge
5:  mov w0,#HD
    ucvtf s0,w0
    fsqrt s0,s0
    fmov s1,#1.0
    fdiv s8,s1,s0
    mov w24,#0
6:  mov w0,#HEADS/KV
    udiv w27,w24,w0
    mov w0,#HD*4
    mul w27,w27,w0
    ADDR x25,query
    madd x25,x24,x0,x25
    mov w28,w26
    ldr w0,=0xff800000
    fmov s9,w0
7:  mov x0,x25
    mov x2,#KW*4
    madd x1,x28,x2,x22
    add x1,x1,x27
    mov w2,#HD
    ADDR x3,use_q8
    ldr w3,[x3]
    cbnz w3,22f
    bl lfm_dot
    b 23f
22: bl lfm_dot_native
23:
    fmul s0,s0,s8
    ADDR x0,scores
    str s0,[x0,w28,uxtw #2]
    fmax s9,s9,s0
    add w28,w28,#1
    cmp w28,w21
    b.ls 7b
    ADDR x0,use_q8
    ldr w0,[x0]
    cbz w0,24f
    ADDR x0,scores
    add x0,x0,x26,lsl #2
    sub w1,w21,w26
    add w1,w1,#1
    fmov s0,s9
    bl lfm_softmax_native
    b 20f
24:
    fmov s10,wzr
    mov w28,w26
8:  ADDR x0,scores
    ldr s0,[x0,w28,uxtw #2]
    fsub s0,s0,s9
    bl expf
    ADDR x0,scores
    str s0,[x0,w28,uxtw #2]
    fadd s10,s10,s0
    add w28,w28,#1
    cmp w28,w21
    b.ls 8b
20:
    ADDR x25,attn_out
    mov x0,#HD*4
    madd x25,x24,x0,x25
    ADDR x0,use_q8
    ldr w0,[x0]
    cbz w0,25f
    mov w28,#0
27: ADDR x0,scores
    add x0,x0,x26,lsl #2
    mov x3,#KW*4
    madd x1,x26,x3,x23
    add x1,x1,x27
    add x1,x1,x28,lsl #2
    sub w2,w21,w26
    add w2,w2,#1
    bl lfm_value_native
    add x0,x25,x28,lsl #2
    str q0,[x0]
    add w28,w28,#4
    cmp w28,#HD
    b.lo 27b
    b 26f
25:
    mov x0,x25
    mov w1,#0
    mov x2,#HD*4
    bl memset
    mov w28,w26
9:  ADDR x0,scores
    ldr s0,[x0,w28,uxtw #2]
    ADDR x3,use_q8
    ldr w3,[x3]
    cbnz w3,21f
    fdiv s0,s0,s10
21:
    dup v16.4s,v0.s[0]
    mov x2,#KW*4
    madd x0,x28,x2,x23
    add x0,x0,x27
    mov x1,x25
    mov w2,#HD
10: ldr q0,[x0],#16
    ldr q1,[x1]
    fmla v1.4s,v0.4s,v16.4s
    str q1,[x1],#16
    subs w2,w2,#4
    b.ne 10b
    add w28,w28,#1
    cmp w28,w21
    b.ls 9b
26:
    ADDR x0,head_gates
    ldr s0,[x0,w24,uxtw #2]
    fneg s0,s0
    bl expf
    fmov s1,#1.0
    fadd s0,s0,s1
    fdiv s0,s1,s0
    dup v16.4s,v0.s[0]
    mov x0,x25
    mov w1,#HD
11: ldr q0,[x0]
    fmul v0.4s,v0.4s,v16.4s
    str q0,[x0],#16
    subs w1,w1,#4
    b.ne 11b
    add w24,w24,#1
    cmp w24,#HEADS
    b.lo 6b
    TRACEBUF 19,attn_out,DIM,20,21
    MAT 13,attn_out,operator_out
    TRACEBUF 20,operator_out,DIM,20,21
    LEAVE
