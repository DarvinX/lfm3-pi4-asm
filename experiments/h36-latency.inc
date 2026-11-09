/* Ordinary-text measurement and app-local decode-worker selection only.
 * Prefill pool already exists; reducing pool_size leaves extra workers asleep.
 * TTFT is request timestamp to first NONEMPTY token piece fully written.
 */
.text
h36_prepare:
    ENTER
    ADDR x0,pool_size
    ldr w19,[x0]
    ADDR x0,h36_prefill_threads
    str w19,[x0]
    mov w20,w19
    ADDR x0,h36_decode_env
    bl getenv
    cbz x0,1f
    mov x1,#0
    mov w2,#10
    bl strtol
    cmp x0,#1
    b.lt h36_thread_error
    cmp x0,x19
    b.gt h36_thread_error
    mov w20,w0
1:  ADDR x0,h36_decode_threads
    str w20,[x0]
    LEAVE

h36_begin_decode:
    ADDR x0,h36_decode_threads
    ldr w1,[x0]
    ADDR x0,pool_size
    str w1,[x0]
    ret

h36_mark_first_output: /* w0=nonempty piece bytes; called after write_all */
    cbz w0,9f
    ADDR x1,h36_first_recorded
    ldr w2,[x1]
    cbnz w2,9f
    ENTER
    ADDR x1,h36_first_piece_bytes
    str w0,[x1]
    ADDR x1,h36_first_recorded
    mov w2,#1
    str w2,[x1]
    ADDR x0,h36_first_output
    bl clock_stamp
    LEAVE
9:  ret

h36_report:
    ENTER
    ADDR x0,h36_first_recorded
    ldr w0,[x0]
    cbz w0,9f
    ADDR x0,h36_request_start
    ADDR x1,h36_first_output
    bl seconds_between
    fmov d8,d0
    ADDR x0,h36_request_start
    ADDR x1,decode_end
    bl seconds_between
    fmov d9,d0
    ADDR x0,load_end
    ADDR x1,h36_first_output
    bl seconds_between
    fmov d2,d0
    fmov d0,d8
    fmov d1,d9
    mov w0,#2
    ADDR x1,h36_report_format
    ADDR x2,h36_first_piece_bytes
    ldr w2,[x2]
    ADDR x3,h36_prefill_threads
    ldr w3,[x3]
    ADDR x4,h36_decode_threads
    ldr w4,[x4]
    bl dprintf
9:  LEAVE
h36_thread_error:
    ADDR x0,h36_thread_text
    b fatal
.section .rodata
h36_decode_env: .asciz "LFM_DECODE_THREADS"
h36_thread_text: .asciz "Decode workers must be1..prefill workers.\n"
h36_report_format: .asciz "LATENCY ttft_seconds=%.9f request_seconds=%.9f prefill_ttft_seconds=%.9f first_piece_bytes=%u prefill_threads=%u decode_threads=%u\n"
.bss
.balign 64
h36_request_start: .skip 16
h36_first_output: .skip 16
h36_first_recorded: .skip 4
h36_first_piece_bytes: .skip 4
h36_prefill_threads: .skip 4
h36_decode_threads: .skip 4
.text
