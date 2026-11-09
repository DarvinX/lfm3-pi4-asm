/* Persistent POSIX workers. All scheduler logic is assembly. Each claim owns
 * eight independent output rows; LDAXR/STLXR is valid on Cortex-A72 (no LSE).
 * 64-byte semaphore slots exceed Linux AArch64 glibc's 32-byte sem_t size. */
.text
init_pool:
    ENTER
    mov w19,#1
    ADDR x0,thread_env
    bl getenv
    cbz x0,1f
    mov x1,#0
    mov w2,#10
    bl strtol
    cmp x0,#1
    b.lt thread_error
    cmp x0,#4
    b.gt thread_error
    mov w19,w0
1:  ADDR x0,pool_size
    str w19,[x0]
    cmp w19,#1
    b.eq 4f
    ADDR x0,done_semaphore
    mov w1,#0
    mov w2,#0
    bl sem_init
    cbnz w0,thread_error
    mov w20,#1
2:  ADDR x21,worker_semaphores
    add x21,x21,x20,lsl #6
    mov x0,x21
    mov w1,#0
    mov w2,#0
    bl sem_init
    cbnz w0,thread_error
    ADDR x0,worker_threads
    add x0,x0,x20,lsl #3
    mov x1,#0
    ADDR x2,worker_entry
    mov x3,x21
    bl pthread_create
    cbnz w0,thread_error
    add w20,w20,#1
    cmp w20,w19
    b.lo 2b
4:  LEAVE

sem_wait_retry:
    ENTER
    mov x19,x0
1:  mov x0,x19
    bl sem_wait
    cbz w0,2f
    bl __errno_location
    ldr w0,[x0]
    cmp w0,#4 /* EINTR */
    b.eq 1b
    b thread_error
2:  LEAVE

worker_entry:
    ENTER
    mov x19,x0
1:  mov x0,x19
    bl sem_wait_retry
    bl matrix_work
    ADDR x0,done_semaphore
    bl sem_post
    b 1b

matrix_work:
    ADDR x0,h34_job_mode
    ldr w0,[x0]
    cbnz w0,h34_work
    ADDR x0,h33_dual_active
    ldr w0,[x0]
    cbnz w0,h33_matrix_work
    ENTER
    ADDR x19,job_descriptor
    ldr x20,[x19]
    ldr x21,[x19,#8]
    ldr x22,[x19,#16]
    ldr w23,[x20,#12]
    ldr w24,[x20,#8]
    ldr w25,[x20,#16]
    ldr w26,[x20,#20]
    mov w0,w25
    bl select_dot
    mov x25,x0
    mov x0,x20
    bl tensor_ptr
    mov x20,x0
1:  add x0,x19,#24
2:  ldaxr w27,[x0]
    add w1,w27,#ROW_TILE
    stlxr w2,w1,[x0]
    cbnz w2,2b
    cmp w27,w23
    b.hs 8f
    sub w28,w23,w27
    mov w0,#ROW_TILE
    cmp w28,w0
    csel w28,w28,w0,lo
    madd x3,x27,x26,x20
    add x4,x22,x27,lsl #2
    /* x3/x4 are caller-saved; keep the row pointers in a local stack frame. */
    sub sp,sp,#16
    stp x3,x4,[sp]
3:  ldr x0,[sp]
    mov x1,x21
    mov w2,w24
    blr x25
    ldp x3,x4,[sp]
    str s0,[x4],#4
    add x3,x3,x26
    stp x3,x4,[sp]
    subs w28,w28,#1
    b.ne 3b
    add sp,sp,#16
    b 1b
8:  LEAVE
thread_error:
    ADDR x0,thread_text
    b fatal
.bss
.balign 64
pool_size: .skip 4
.balign 64
job_descriptor: .skip 64
worker_semaphores: .skip 4*64
done_semaphore: .skip 64
worker_threads: .skip 4*8
.text
