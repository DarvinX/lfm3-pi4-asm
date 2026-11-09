/* H34v2 pread: bounds memory independently of file fault-around/candidate count. */
/* H34 experimental certified scalar head, all application/controller code ASM.
 * CRC-bound auxiliaries, fixed2048/128000 checkpoint. Full-logit/probe bypass.
 * First paired graph disables coarse permanently, dropping aux BEFORE originals.
 * This first integration has no dual-query coarse path or token-speed claim.
 */
.text
h34_init: /* w0=full-logit diagnostic; map_model CRC already passed */
    ENTER
    cbnz w0,9f
    .ifdef TRACE_DETAIL
    b 9f
    .endif
    .ifdef TRACE
    b 9f
    .endif
    .ifdef H33_PAIR_PROBE
    b 9f
    .endif
    ADDR x0,use_q8
    ldr w0,[x0]
    cbz w0,9f
    ADDR x0,h34_coarse_env
    bl getenv
    cbz x0,9f
    mov x19,x0
    ADDR x0,h34_norm_env
    bl getenv
    cbz x0,checksum_error
    mov x20,x0
    /* Full-model CRC has touched the original head. Release it FIRST. */
    bl h34_drop_original
    mov x0,x19
    ldr x1,=149504000
    ldr w2,=1828805539
    bl h34_map_aux
    ADDR x1,h34_coarse_base
    str x0,[x1]
    mov x0,x20
    ldr x1,=4096000
    ldr w2,=2620772692
    bl h34_map_aux
    ADDR x1,h34_norm_base
    str x0,[x1]
    ADDR x0,h34_enabled
    mov w1,#1
    str w1,[x0]
9:  ADDR x0,h34_enabled
    ldr w0,[x0]
    cbnz w0,8f
    /* Original mmap path needs no persistent descriptor in disabled modes. */
    ADDR x0,h34_model_fd
    ldr w0,[x0]
    bl close
    ADDR x0,h34_model_fd
    mov w1,#-1
    str w1,[x0]
8:  LEAVE

h34_map_aux: /* file,size,CRC32 -> mapped ptr; exact immutable data binding */
    ENTER
    mov x20,x1
    mov w21,w2
    mov w1,#0
    bl open
    cmp w0,#0
    b.lt file_error
    mov w19,w0
    mov x1,#0
    mov w2,#2
    bl lseek
    cmp x0,x20
    b.ne checksum_error
    mov x0,#0
    mov x1,x20
    mov w2,#1
    mov w3,#2
    mov w4,w19
    mov x5,#0
    bl mmap
    cmn x0,#1
    b.eq file_error
    mov x22,x0
    mov w0,w19
    bl close
    mov x0,x22
    mov x1,x20
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
    cmp w2,w21
    b.ne checksum_error
    mov x0,x22
    LEAVE

h34_drop_original:
    ENTER
    ADDR x0,model_base
    ldr x0,[x0]
    /* mmap base4KiB-aligned; ceil8255776=8257536,52499 interior pages. */
    ldr x1,=8257536
    add x0,x0,x1
    ldr x1,=215035904
    mov w2,#4 /* MADV_DONTNEED, process PTEs only; file cache stays warm */
    bl madvise
    cbnz w0,file_error
    LEAVE

h34_drop_aux:
    ENTER
    ADDR x0,h34_coarse_base
    ldr x0,[x0]
    ldr x1,=149504000
    mov w2,#4
    bl madvise
    cbnz w0,file_error
    ADDR x0,h34_norm_base
    ldr x0,[x0]
    ldr x1,=4096000
    mov w2,#4
    bl madvise
    cbnz w0,file_error
    LEAVE

h34_disable_for_pair:
    ENTER
    ADDR x0,h34_enabled
    ldr w1,[x0]
    cbz w1,9f
    str wzr,[x0]
    bl h34_drop_aux
    ADDR x0,h34_pair_disables
    ldr x1,[x0]
    add x1,x1,#1
    str x1,[x0]
9:  LEAVE

h34_read_row: /* w0 row,x1 destination1680B; no user model PTE access */
    ENTER
    ldr w2,=VOCAB
    cmp w0,w2
    b.hs numeric_error
    mov w19,w0
    mov x20,x1
    mov x21,#1680
    ldr x22,=8255776
    mov x0,#1680
    madd x22,x19,x0,x22
    ADDR x0,h34_model_fd
    ldr w23,[x0]
1:  mov w0,w23
    mov x1,x20
    mov x2,x21
    mov x3,x22
    bl pread
    cmp x0,#0
    b.gt 3f
    cbz x0,file_error
    bl __errno_location
    ldr w0,[x0]
    cmp w0,#4
    b.eq 1b
    b file_error
3:  add x20,x20,x0
    add x22,x22,x0
    subs x21,x21,x0
    b.ne 1b
    LEAVE

h34_activation_norms: /* canonical Q8 in activation_q8; return valid=1/0 */
    mrs x0,fpcr
    tst x0,#0x1c00000 /* RMode and FZ; proof requires RN/gradual underflow */
    b.ne 9f
    ADDR x0,activation_q8
    ADDR x1,h34_act_norms
    mov w2,#8
1:  add x3,x0,#32
    mov w4,#256
    mov w5,#0
2:  ldrsb w6,[x3],#1
    madd w5,w6,w6,w5
    subs w4,w4,#1
    b.ne 2b
    cbz w5,4f
    ucvtf s0,w5
    fsqrt s0,s0
    /* Positive finite normal sqrt; increment bits is next_up32. */
    fmov w6,s0
    add w6,w6,#1
    fmov s0,w6
    fcvt d1,s0
    fmul d1,d1,d1
    ucvtf d2,w5
    fcmp d1,d2
    b.mi 9f
    b 5f
4:  fmov s0,wzr
5:  str s0,[x1],#4
    add x0,x0,#320
    subs w2,w2,#1
    b.ne 1b
    mov w0,#1
    ret
9:  mov w0,#0
    ret

h34_output: /* output_tensor,float input,logits; only greedy-output mode */
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x2
    ADDR x0,h34_enabled
    ldr w0,[x0]
    cbz w0,8f
    mov x0,x20
    ADDR x1,activation_q8
    mov w2,#2048
    bl lfm_q8_pack_native
    bl h34_activation_norms
    cbz w0,7f
    ADDR x0,h34_job_output
    str x21,[x0]
    mov w0,#1
    ldr w1,=128000
    bl h34_launch
    bl argmax /* coarse proposal only; never a committed output */
    mov w22,w0
    sub sp,sp,#1680
    mov w0,w22
    mov x1,sp
    bl h34_read_row
    mov x0,sp
    ADDR x1,activation_q8
    mov w2,#2048
    bl lfm_q6_q8_dot_h19
    add sp,sp,#1680
    fmov s8,s0
    fmov w0,s8
    and w0,w0,#0x7fffffff
    mov w1,#0x7f800000
    cmp w0,w1
    b.hs 7f
    fcvt d8,s8
    ADDR x23,h34_uppers
    ADDR x24,h34_indices
    mov w25,#0
    mov w26,#0
    ldr w27,=128000
1:  ldr d0,[x23],#8
    cmp w25,w22
    b.eq 2f
    fcmp d0,d8
    b.mi 3f /* strict ordered upper<threshold; equality/unordered refine */
2:  cmp w26,#4096
    b.hs 7f
    str w25,[x24,w26,uxtw #2]
    add w26,w26,#1
3:  add w25,w25,#1
    cmp w25,w27
    b.lo 1b
    mov w0,#2
    mov w1,w26
    bl h34_launch
    /* Unrefined coarse<=upper<threshold, so original first-max is preserved. */
    bl h34_drop_original
    ADDR x0,h34_certified_heads
    ldr x1,[x0]
    add x1,x1,#1
    str x1,[x0]
    ADDR x0,h34_refined_rows
    ldr x1,[x0]
    add x1,x1,x26
    str x1,[x0]
    LEAVE
7:  ADDR x0,h34_enabled
    str wzr,[x0]
    bl h34_drop_aux
    ADDR x0,h34_fallbacks
    ldr x1,[x0]
    add x1,x1,#1
    str x1,[x0]
8:  mov x0,x19
    mov x1,x20
    mov x2,x21
    bl matrix
    LEAVE

h34_launch: /* mode1coarse/2indexed-refine,count; existing persistent pool */
    ENTER
    ADDR x2,h34_job_mode
    str w0,[x2]
    ADDR x2,h34_job_count
    str w1,[x2]
    ADDR x2,job_descriptor
    str wzr,[x2,#24]
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w0,#1
    b.eq 4f
    cmp w1,#64
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
    bl h34_work
    mov w19,#1
2:  ADDR x0,done_semaphore
    bl sem_wait_retry
    add w19,w19,#1
    ADDR x0,pool_size
    ldr w0,[x0]
    cmp w19,w0
    b.lo 2b
    b 5f
4:  bl h34_work
5:  ADDR x0,h34_job_mode
    str wzr,[x0]
    LEAVE

h34_work:
    ENTER
    ADDR x19,job_descriptor
    add x19,x19,#24
    ADDR x20,activation_q8
    ADDR x0,h34_job_output
    ldr x21,[x0]
    ADDR x0,h34_job_mode
    ldr w23,[x0]
    ADDR x0,h34_job_count
    ldr w24,[x0]
    cmp w23,#1
    b.ne 1f
    ADDR x0,h34_coarse_base
    ldr x22,[x0]
    ADDR x0,h34_norm_base
    ldr x26,[x0]
    b 2f
1:  ADDR x0,model_base
    ldr x22,[x0]
    ldr x0,=8255776
    add x22,x22,x0
    ADDR x26,h34_indices
2:  ldaxr w27,[x19]
    add w1,w27,#8
    stlxr w2,w1,[x19]
    cbnz w2,2b
    cmp w27,w24
    b.hs 9f
    add w28,w27,#8
    cmp w28,w24
    csel w28,w28,w24,lo
3:  cmp w23,#1
    b.ne 4f
    mov w0,#1168
    madd x0,x27,x0,x22
    mov x1,x20
    add x2,x26,x27,lsl #5
    ADDR x3,h34_act_norms
    add x4,x21,x27,lsl #2
    ADDR x5,h34_uppers
    add x5,x5,x27,lsl #3
    bl lfm_coarse_score_upper_h34
    b 5f
4:  ldr w9,[x26,w27,uxtw #2]
    sub sp,sp,#1696
    str w9,[sp,#1680]
    mov w0,w9
    mov x1,sp
    bl h34_read_row
    mov x0,sp
    mov x1,x20
    mov w2,#2048
    bl lfm_q6_q8_dot_h19
    ldr w9,[sp,#1680]
    add sp,sp,#1696
    str s0,[x21,w9,uxtw #2]
5:  add w27,w27,#1
    cmp w27,w28
    b.lo 3b
    b 2b
9:  LEAVE

h34_report:
    ENTER
    mov w0,#2
    ADDR x1,h34_report_format
    ADDR x2,h34_certified_heads
    ldr x2,[x2]
    ADDR x3,h34_refined_rows
    ldr x3,[x3]
    ADDR x4,h34_fallbacks
    ldr x4,[x4]
    ADDR x5,h34_pair_disables
    ldr x5,[x5]
    mov w6,#2
    bl dprintf
    LEAVE

.ifdef H34_HEAD_DRIVER
.global h34_test_setup,h34_test_winner,h34_test_scores,h34_test_stats,h34_test_pair_disable
.global h34_test_model,h34_test_aux,h34_test_norms
.global h34_test_stress,h34_test_embedding,h34_test_hidden
h34_test_setup: /* const char *model; env supplies exact auxiliary paths */
    ENTER
    ADDR x1,use_q8
    mov w2,#1
    str w2,[x1]
    bl map_model
    mov w0,#0
    bl h34_init
    bl init_pool
    LEAVE
h34_test_winner: /* float *input -> winner; real production head/controller */
    ENTER
    mov x1,x0
    ADDR x0,output_tensor
    ADDR x2,logits
    bl h34_output
    bl argmax
    LEAVE
h34_test_scores:
    ADDR x0,logits
    ret
h34_test_stats:
    ADDR x0,h34_certified_heads
    ret
h34_test_pair_disable:
    b h34_disable_for_pair
h34_test_model:
    ADDR x0,model_base
    ldr x0,[x0]
    ret
h34_test_aux:
    ADDR x0,h34_coarse_base
    ret
h34_test_stress: /* external test indices,count; NO page release in this hook */
    ENTER
    mov x20,x0
    mov w19,w1
    cmp w19,#4096
    b.hi numeric_error
    cbz w19,numeric_error
    ADDR x0,h34_indices
    mov x1,x20
    lsl x2,x19,#2
    bl memcpy
    ADDR x0,h34_job_output
    ADDR x1,logits
    str x1,[x0]
    mov w0,#2
    mov w1,w19
    bl h34_launch
    LEAVE
h34_test_embedding:
    b embedding_row
h34_test_hidden:
    ADDR x0,hidden
    ret
h34_test_norms:
    ADDR x0,h34_act_norms
    ret
.endif

.section .rodata
h34_coarse_env: .asciz "LFM_COARSE"
h34_norm_env: .asciz "LFM_RESIDUAL"
h34_report_format: .asciz "H34 heads=%llu refined_rows=%llu fallbacks=%llu paired_disables=%llu io=%u\n"
.bss
.balign 64
h34_enabled: .skip 4
h34_job_mode: .skip 4
h34_job_count: .skip 4
h34_model_fd: .skip 4
.balign 8
h34_coarse_base: .skip 8
h34_norm_base: .skip 8
h34_job_output: .skip 8
h34_certified_heads: .skip 8
h34_refined_rows: .skip 8
h34_fallbacks: .skip 8
h34_pair_disables: .skip 8
.balign 64
h34_act_norms: .skip 8*4
h34_indices: .skip 4096*4
h34_uppers: .skip 128000*8
.text
