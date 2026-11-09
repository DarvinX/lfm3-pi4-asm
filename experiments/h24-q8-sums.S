/* Precompute eight exact32-element activation sums once per matrix.
 * Reuses bytes16..31 of the existing320-byte Q8 block; no extra allocation.
 */
.arch armv8-a
.text
.balign 64
.global lfm_q8_pack_h24
.type lfm_q8_pack_h24,%function
lfm_q8_pack_h24:
    stp x19,x20,[sp,#-32]!
    stp x29,x30,[sp,#16]
    mov x19,x1
    mov w20,w2
    bl lfm_q8_pack_native
1:  ldp q0,q1,[x19,#288]
    addp v2.8h,v0.8h,v1.8h
    str q2,[x19,#16]
    add x19,x19,#320
    subs w20,w20,#256
    b.ne 1b
    ldp x29,x30,[sp,#16]
    ldp x19,x20,[sp],#32
    ret
.size lfm_q8_pack_h24,.-lfm_q8_pack_h24
.section .note.GNU-stack,"",%progbits
