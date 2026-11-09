/* Isolated H42 Q4 lane schedule; rest frozen H40. */
/* Isolated H40; row8, unchanged Q6/head/graph; sum32 query cache producer. */
/* H36 exact scalar-default ordinary runtime with first-output timing. */
/* H34v2 fixed-buffer tied-head reads; v1 is immutable. */
#ifdef H34_DRIVER_MAIN
#define main lfm_runtime_main
#endif
/* H33 isolated exact verified short-cycle draft runtime. */
/* H23 combines independently retained H19/Q6 and H20/Q4. */
/* Entire inference runtime is AArch64 assembly. libc handles OS/stdio;
 * libm supplies expf/powf/sinf/cosf. No C/C++ model, tokenizer or scheduler. */
.arch armv8-a+crc
.include "src/common.inc"
#ifndef MODEL_INDEX
#define MODEL_INDEX "build/model.inc"
#endif
#include MODEL_INDEX
#ifndef CONTEXT
#define CONTEXT 512
#endif
#ifndef LFM_ROW_TILE
#define LFM_ROW_TILE 8
#endif
.equ CTX,CONTEXT
.equ ROW_TILE,LFM_ROW_TILE
.if CTX < 32 || CTX > 4096
.error "CONTEXT must be in 32..4096"
.endif
.text
.global main
.type main,%function
main:
    ENTER
    cmp w0,#3
    b.lt usage
    mov w19,w0
    mov x20,x1
    ADDR x0,h36_request_start
    bl clock_stamp
    bl hardware_check
    ldr x0,[x20,#8]
    ADDR x1,tokenize_flag
    bl strcmp
    cbnz w0,1f
    ldr x0,[x20,#16]
    bl tokenize
    mov w21,w0
    mov w22,#0
0:  cmp w22,w21
    b.hs success
    ADDR x1,input_ids
    ldr w1,[x1,w22,uxtw #2]
    ADDR x0,id_format
    bl printf
    add w22,w22,#1
    b 0b
1:  mov w23,#0
    ldr x0,[x20,#8]
    ADDR x1,logits_flag
    bl strcmp
    cbnz w0,2f
    cmp w19,#4
    b.lt usage
    mov w23,#1
    add x20,x20,#8
    sub w19,w19,#1
2:  mov w24,#32
    cmp w19,#4
    b.gt usage
    cmp w19,#4
    b.lt 3f
    ldr x0,[x20,#24]
    mov x1,#0
    mov w2,#10
    bl strtol
    cmp x0,#0
    b.le usage
    cmp x0,#CTX
    b.hi usage
    mov w24,w0
3:  cmp w23,#0
    csel w24,wzr,w24,ne
    ldr x0,[x20,#16]
    bl tokenize
    mov w21,w0
    cbz w21,usage
    ADDR x0,prefill_count
    str w21,[x0]
    add w0,w21,w24
    cmp w0,#CTX
    b.hi context_error
    ADDR x0,ignore_eos_env
    bl getenv
    cmp x0,#0
    cset w1,ne
    ADDR x0,ignore_eos
    str w1,[x0]
    ADDR x0,q8_env
    bl getenv
    mov w1,#0
    cbz x0,8f
    ldrb w1,[x0]
    cmp w1,#48
    cset w1,ne
8:  ADDR x0,use_q8
    str w1,[x0]
    ADDR x0,load_start
    bl clock_stamp
    ldr x0,[x20,#8]
    bl map_model
    mov w0,w23
    bl h34_init
    bl init_pool
    bl h36_prepare
    ADDR x0,load_end
    bl clock_stamp
    mov w22,#0
4:  ADDR x0,input_ids
    ldr w0,[x0,w22,uxtw #2]
    mov w1,w22
    sub w2,w21,#1
    cmp w22,w2
    cset w2,eq
    bl forward
    add w22,w22,#1
    cmp w22,w21
    b.lo 4b
    ADDR x0,prefill_end
    bl clock_stamp
    cbz w23,5f
    mov w0,#1
    ADDR x1,logits
    ldr x2,=VOCAB*4
    bl write_all
    b success
5:  bl h36_begin_decode
    .ifdef H33_PAIR_PROBE
    mov w0,w22
    bl h33_pair_probe
    b success
    .endif
    mov w0,w22
    mov w1,w24
    bl h33_generate
    b success
success:
    ADDR x0,decode_end
    bl clock_stamp
    ADDR x0,report_env
    bl getenv
    cbz x0,7f
    ADDR x0,load_start
    ADDR x1,load_end
    bl seconds_between
    fmov d8,d0
    ADDR x0,load_end
    ADDR x1,prefill_end
    bl seconds_between
    fmov d9,d0
    ADDR x0,prefill_end
    ADDR x1,decode_end
    bl seconds_between
    fmov d10,d0
    mov w0,#0
    ADDR x1,usage_buffer
    bl getrusage
    cbnz w0,file_error
    mov w0,#2
    ADDR x1,report_format
    ADDR x2,prefill_count
    ldr w2,[x2]
    ADDR x3,generated_count
    ldr w3,[x3]
    ADDR x4,decode_steps
    ldr w4,[x4]
    ADDR x5,usage_buffer
    ldr x5,[x5,#32]
    fmov d0,d8
    fmov d1,d9
    fmov d2,d10
    bl dprintf
    bl h33_report
    bl h34_report
    bl h36_report
7:
    mov w0,#0
    LEAVE
usage:
    ADDR x0,usage_text
    b fatal
context_error:
    ADDR x0,context_text
    b fatal
.size main,.-main

clock_stamp: /* x0 -> struct timespec, Linux AArch64: two int64 values */
    ENTER
    mov x1,x0
    mov w0,#1 /* CLOCK_MONOTONIC */
    bl clock_gettime
    cbnz w0,file_error
    LEAVE

seconds_between: /* x0 start,x1 end -> d0 seconds */
    ldp x2,x3,[x0]
    ldp x4,x5,[x1]
    sub x4,x4,x2
    sub x5,x5,x3
    scvtf d0,x4
    scvtf d1,x5
    ADDR x2,nanosecond_scale
    ldr d2,[x2]
    fmadd d0,d1,d2,d0
    ret

hardware_check:
    ENTER
    mov w0,#16 /* AT_HWCAP */
    bl getauxval
    mov w1,#130 /* ASIMD | CRC32 */
    and w0,w0,w1
    cmp w0,w1
    b.ne hardware_error
    LEAVE
hardware_error:
    ADDR x0,hardware_text
    b fatal

fatal: /* x0 = static message; never returns */
    mov x1,x0
    mov w0,#2
    bl dprintf
    mov w0,#1
    bl exit

map_model:
    ENTER
    mov w1,#0
    bl open
    cmp w0,#0
    b.lt file_error
    mov w19,w0
    mov x1,#0
    mov w2,#2
    bl lseek
    ldr x1,=MODEL_BYTES
    cmp x0,x1
    b.ne checksum_error
    mov x0,#0
    ldr x1,=MODEL_BYTES
    mov w2,#1
    mov w3,#2
    mov w4,w19
    mov x5,#0
    bl mmap
    cmn x0,#1
    b.eq file_error
    ADDR x1,model_base
    str x0,[x1]
    mov x20,x0
    ADDR x1,h34_model_fd
    str w19,[x1]
    /* CRC32 over every byte binds the generated tensor index to this file. */
    mov x0,x20
    ldr x1,=MODEL_BYTES
    mov w2,#-1
1:  cmp x1,#8
    b.lo 2f
    ldr x3,[x0],#8
    crc32x w2,w2,x3
    sub x1,x1,#8
    b 1b
2:  cbz x1,4f
3:  ldrb w3,[x0],#1
    crc32b w2,w2,w3
    subs x1,x1,#1
    b.ne 3b
4:  mvn w2,w2
    ldr w3,=MODEL_CRC
    cmp w2,w3
    b.ne checksum_error
    LEAVE
file_error:
    ADDR x0,file_text
    b fatal
checksum_error:
    ADDR x0,checksum_text
    b fatal

write_all: /* w0 fd, x1 bytes, x2 length */
    ENTER
    mov w19,w0
    mov x20,x1
    mov x21,x2
1:  cbz x21,2f
    mov w0,w19
    mov x1,x20
    mov x2,x21
    bl write
    cmp x0,#0
    b.gt 3f
    bl __errno_location
    ldr w0,[x0]
    cmp w0,#4
    b.eq 1b
    b file_error
3:
    add x20,x20,x0
    sub x21,x21,x0
    b 1b
2:  LEAVE

emit_token:
    ENTER
    .ifdef TOKEN_TRACE
    sub sp,sp,#16
    str w0,[sp]
    mov w0,#4
    mov x1,sp
    mov x2,#4
    bl write_all
    ldr w0,[sp]
    add sp,sp,#16
    .endif
    ADDR x1,vocab_data
    add x1,x1,x0,lsl #4
    ldr x2,[x1]
    ldr w3,[x1,#8]
    ADDR x1,text_data
    add x1,x1,x2
    mov w0,#1
    mov w2,w3
    mov w19,w3
    bl write_all
    mov w0,w19
    bl h36_mark_first_output
    LEAVE

argmax:
    ADDR x1,logits
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
numeric_error:
    ADDR x0,numeric_text
    b fatal

/* descriptor: u64 offset; u32 cols,rows,type,row_bytes; u64 bytes */
tensor_ptr:
    ldr x0,[x0]
    ADDR x1,model_base
    ldr x1,[x1]
    add x0,x0,x1
    ret

matrix: /* x0 descriptor, x1 input, x2 output; optional persistent row workers */
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    ADDR x3,use_q8
    ldr w3,[x3]
    cbz w3,9f
    ldr w3,[x19,#16]
    cbz w3,9f
    mov x0,x20
    ADDR x1,activation_q8
    ldr w2,[x19,#8]
    bl lfm_q8_pack_h24
    ADDR x20,activation_q8
9:  mov x0,x19
    mov x1,x20
    mov x2,x21
    ADDR x3,pool_size
    ldr w3,[x3]
    cmp w3,#1
    b.hi 1f
0:
    bl matrix_serial
    LEAVE
1:  ldr w3,[x0,#12]
    cmp w3,#64
    b.lo 0b
    ADDR x3,job_descriptor
    str x0,[x3]
    str x1,[x3,#8]
    str x2,[x3,#16]
    str wzr,[x3,#24]
    mov w19,#1
2:  ADDR x0,worker_semaphores
    add x0,x0,x19,lsl #6
    bl sem_post
    add w19,w19,#1
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w19,w0
    b.lo 2b
    bl matrix_work
    mov w19,#1
3:  ADDR x0,done_semaphore
    bl sem_wait_retry
    add w19,w19,#1
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w19,w0
    b.lo 3b
    LEAVE

matrix_serial:
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    ldr w22,[x19,#12]
    ldr w23,[x19,#8]
    ldr w24,[x19,#16]
    ldr w25,[x19,#20]
    mov w0,w24
    bl select_dot
    mov x27,x0
    mov x0,x19
    bl tensor_ptr
    mov x26,x0
1:  mov x0,x26
    mov x1,x20
    mov w2,w23
    blr x27
    str s0,[x21],#4
    add x26,x26,x25
    subs w22,w22,#1
    b.ne 1b
    LEAVE

select_dot: /* w0 GGUF type -> x0 matching dot function */
    cbz w0,3f
    ADDR x1,use_q8
    ldr w1,[x1]
    cmp w0,#12
    b.ne 2f
    cbnz w1,4f
    ADDR x0,lfm_q4_dot
    ret
2:  cbnz w1,5f
    ADDR x0,lfm_q6_dot
    ret
3:  ADDR x0,lfm_dot
    ret
4:  ADDR x0,lfm_q4_q8_dot_h42
    ret
5:  ADDR x0,lfm_q6_q8_dot_h19
    ret

.include "experiments/workers-h34.inc"

embedding_row: /* tied Q6 via fixed pread buffer while coarse enabled */
    ENTER
    mov w19,w0
    ADDR x0,h34_enabled
    ldr w0,[x0]
    cbz w0,7f
    sub sp,sp,#1680
    mov w0,w19
    mov x1,sp
    bl h34_read_row
    mov x0,sp
    ADDR x1,hidden
    mov w2,#DIM
    bl lfm_q6_unpack
    add sp,sp,#1680
    LEAVE
7:
    ADDR x0,embedding
    ldr w20,[x0,#16]
    ldr w21,[x0,#20]
    bl tensor_ptr
    madd x0,x19,x21,x0
    ADDR x1,hidden
    mov w2,#DIM
    cbz w20,2f
    cmp w20,#12
    b.ne 1f
    bl lfm_q4_unpack
    b 3f
1:  bl lfm_q6_unpack
    b 3f
2:  mov x2,#DIM*4
    mov x3,x0
    mov x0,x1
    mov x1,x3
    bl memcpy
3:  LEAVE

norm: /* x0 descriptor, x1 input, x2 out, w3 count */
    ENTER
    mov x19,x1
    mov x20,x2
    mov w21,w3
    bl tensor_ptr
    mov x2,x0
    mov x0,x20
    mov x1,x19
    mov w3,w21
    ADDR x4,norm_eps
    ldr s0,[x4]
    ADDR x4,use_q8
    ldr w4,[x4]
    cbnz w4,1f
    bl lfm_rms
    LEAVE
1:  bl lfm_rms_wide
    LEAVE

.include "src/tokenizer.inc"
.include "experiments/forward-h34.inc"
.include "experiments/h33-tails.inc"
.include "experiments/h34-batch.inc"
.include "experiments/h36-generation.inc"
.include "experiments/h33-probe.inc"
.include "experiments/h34-controller-v2.inc"
.include "experiments/h36-latency.inc"

.section .rodata
usage_text: .asciz "Usage: lfm3-asm MODEL.gguf PROMPT [MAX_NEW=32]\n       lfm3-asm --tokenize PROMPT\n       lfm3-asm --logits MODEL.gguf PROMPT  (binary float32 stdout)\n"
context_text: .asciz "Prompt plus requested generation exceeds compiled context capacity.\n"
file_text: .asciz "File open, mmap, or write failed.\n"
checksum_text: .asciz "Model does not match the compiled checkpoint index (size/CRC32).\n"
numeric_text: .asciz "Non-finite model output.\n"
token_text: .asciz "Tokenizer input is invalid UTF-8 or exceeds context capacity.\n"
thread_text: .asciz "LFM_THREADS must be 1..4, and worker initialization must succeed.\n"
hardware_text: .asciz "This build requires ARM64 Advanced SIMD and CRC32 instructions.\n"
thread_env: .asciz "LFM_THREADS"
report_env: .asciz "LFM_REPORT"
ignore_eos_env: .asciz "LFM_BENCH_IGNORE_EOS"
q8_env: .asciz "LFM_Q8"
report_format: .asciz "prefill_tokens=%u generated_tokens=%u decode_steps=%u peak_rss_kib=%lu load_seconds=%.9f prefill_seconds=%.9f decode_seconds=%.9f\n"
.balign 8
nanosecond_scale: .double 0.000000001
tokenize_flag: .asciz "--tokenize"
logits_flag: .asciz "--logits"
id_format: .asciz "%u\n"
.bss
.balign 64
model_base: .skip 8
prefill_count: .skip 4
generated_count: .skip 4
decode_steps: .skip 4
ignore_eos: .skip 4
.balign 16
load_start: .skip 16
load_end: .skip 16
prefill_end: .skip 16
decode_end: .skip 16
usage_buffer: .skip 160
use_q8: .skip 4
.balign 64
activation_q8: .skip ((FF+255)/256)*320
input_ids: .skip CTX*4
hidden: .skip DIM*4
normalized: .skip DIM*4
operator_out: .skip DIM*4
bcx: .skip DIM*12
conv_out: .skip DIM*4
query: .skip DIM*4
key: .skip KW*4
value: .skip KW*4
attn_out: .skip DIM*4
head_gates: .skip HEADS*4
ff_gate: .skip FF*4
ff_up: .skip FF*4
logits: .skip VOCAB*4
scores: .skip CTX*4
cos_global: .skip HD*2
sin_global: .skip HD*2
cos_window: .skip HD*2
sin_window: .skip HD*2
.balign 64
conv_states: .skip LAYERS*DIM*(CONV-1)*4
.balign 64
key_cache: .skip LAYERS*CTX*KW*4
.balign 64
value_cache: .skip LAYERS*CTX*KW*4
.section .note.GNU-stack,"",%progbits
