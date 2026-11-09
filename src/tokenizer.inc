/* Exact byte-BPE with the LFM2/Llama-3 pretokenization alternatives.
 * Unicode L/N/whitespace classifications are generated offline and embedded. */
.text
vocab_lookup: /* x0 bytes, x1 len -> w0 token or -1 */
    ENTER
    mov x19,x0
    mov x20,x1
    ldr x21,=14695981039346656037
    ldr x3,=1099511628211
    mov x4,#0
1:  cmp x4,x20
    b.hs 2f
    ldrb w5,[x19,x4]
    eor x21,x21,x5
    mul x21,x21,x3
    add x4,x4,#1
    b 1b
2:  and w22,w21,#LOOKUP_MASK
3:  ADDR x3,lookup_data
    add x3,x3,x22,lsl #4
    ldr w23,[x3,#8]
    cbz w23,6f
    ldr x4,[x3]
    cmp x4,x21
    b.ne 5f
    sub w23,w23,#1
    ADDR x3,vocab_data
    add x3,x3,x23,lsl #4
    ldr w4,[x3,#8]
    cmp x4,x20
    b.ne 5f
    ldr x4,[x3]
    ADDR x0,text_data
    add x0,x0,x4
    mov x1,x19
    mov x2,x20
    bl memcmp
    cbnz w0,5f
    mov w0,w23
    LEAVE
5:  add w22,w22,#1
    and w22,w22,#LOOKUP_MASK
    b 3b
6:  mov w0,#-1
    LEAVE

merge_lookup: /* w0 left,w1 right -> w0 rank or -1, w1 merged token */
    lsl x2,x0,#32
    orr x2,x2,x1
    add x2,x2,#1
    ldr w3,=0x9e3779b1
    mul w3,w0,w3
    eor w3,w3,w1
    and w3,w3,#MERGE_MASK
    ADDR x4,merges_data
1:  add x5,x4,x3,lsl #4
    ldr x6,[x5]
    cbz x6,3f
    cmp x6,x2
    b.eq 2f
    add w3,w3,#1
    and w3,w3,#MERGE_MASK
    b 1b
2:  ldr w0,[x5,#8]
    ldr w1,[x5,#12]
    ret
3:  mov w0,#-1
    ret

append_input:
    ADDR x1,input_count
    ldr w2,[x1]
    cmp w2,#CTX
    b.hs tokenizer_error
    ADDR x3,input_ids
    str w0,[x3,w2,uxtw #2]
    add w2,w2,#1
    str w2,[x1]
    ret

bpe_piece: /* x0 raw word, x1 byte length */
    ENTER
    mov x19,x0
    mov x20,x1
    bl vocab_lookup
    cmp w0,#-1
    b.eq 1f
    bl append_input
    LEAVE
1:  mov x21,#0
    ADDR x22,bpe_ids
    ADDR x23,bpe_next
    ADDR x24,bytes_data
2:  ldrb w0,[x19,x21]
    ldr w0,[x24,x0,lsl #2]
    str w0,[x22,x21,lsl #2]
    add w0,w21,#1
    str w0,[x23,x21,lsl #2]
    add x21,x21,#1
    cmp x21,x20
    b.lo 2b
    sub x0,x20,#1
    mov w1,#-1
    str w1,[x23,x0,lsl #2]
3:  mov w21,#0
    mov w24,#-1
    mov w25,#-1
4:  ldr w26,[x23,w21,uxtw #2]
    cmp w26,#-1
    b.eq 7f
    ldr w0,[x22,w21,uxtw #2]
    ldr w1,[x22,w26,uxtw #2]
    bl merge_lookup
    cmp w0,w24
    b.hs 5f
    mov w24,w0
    mov w25,w21
    mov w27,w1
5:  mov w21,w26
    b 4b
7:  cmp w25,#-1
    b.eq 8f
    str w27,[x22,w25,uxtw #2]
    ldr w0,[x23,w25,uxtw #2]
    ldr w0,[x23,w0,uxtw #2]
    str w0,[x23,w25,uxtw #2]
    b 3b
8:  mov w21,#0
9:  ldr w0,[x22,w21,uxtw #2]
    bl append_input
    ldr w21,[x23,w21,uxtw #2]
    cmp w21,#-1
    b.ne 9b
    LEAVE

find_special: /* x0 text,x1 length -> x0 prefix length, w1 id/-1, w2 special length */
    ENTER
    mov x19,x0
    mov x20,x1
    mov x21,x1
    mov w22,#-1
    mov w23,#0
    ADDR x24,special_ids
1:  ldr w25,[x24],#4
    cmp w25,#-1
    b.eq 3f
    ADDR x2,vocab_data
    add x2,x2,x25,lsl #4
    ldr x3,[x2]
    ldr w26,[x2,#8]
    cbz w26,1b
    ADDR x2,text_data
    add x2,x2,x3
    mov w3,w26
    mov x0,x19
    mov x1,x20
    bl memmem
    cbz x0,1b
    sub x0,x0,x19
    cmp x0,x21
    b.hs 1b
    mov x21,x0
    mov w22,w25
    mov w23,w26
    b 1b
3:  mov x0,x21
    mov w1,w22
    mov w2,w23
    LEAVE

classify: /* UTF-8 decode; x0 ptr,x1 remaining -> w0 flags,w1 consumed bytes */
    cbz x1,tokenizer_error
    ldrb w2,[x0]
    cmp w2,#128
    b.lo 7f
    cmp w2,#0xc2
    b.lo tokenizer_error
    cmp w2,#0xe0
    b.lo 1f
    cmp w2,#0xf0
    b.lo 2f
    cmp w2,#0xf5
    b.hs tokenizer_error
    mov w3,#4
    and w2,w2,#7
    b 3f
1:  mov w3,#2
    and w2,w2,#31
    b 3f
2:  mov w3,#3
    and w2,w2,#15
3:  cmp x1,x3
    b.lo tokenizer_error
    mov w4,#1
4:  ldrb w5,[x0,x4]
    and w6,w5,#0xc0
    cmp w6,#0x80
    b.ne tokenizer_error
    and w5,w5,#63
    lsl w2,w2,#6
    orr w2,w2,w5
    add w4,w4,#1
    cmp w4,w3
    b.lo 4b
    cmp w3,#2
    b.eq 6f
    cmp w2,#0x800
    b.lo tokenizer_error
    cmp w3,#3
    b.eq 5f
    cmp w2,#0x10000
    b.lo tokenizer_error
5:  ldr w5,=0x110000
    cmp w2,w5
    b.hs tokenizer_error
    ldr w5,=0xd800
    sub w5,w2,w5
    cmp w5,#0x800
    b.lo tokenizer_error
6:  mov w1,w3
    b 8f
7:  mov w1,#1
8:  ADDR x3,categories_data
    ldrb w0,[x3,w2,uxtw]
    ret

lex_length: /* x0 text,x1 remaining -> x0 first regex match length */
    ENTER
    mov x19,x0
    mov x20,x1
    /* Case-insensitive English contractions are the first alternative. */
    ldrb w2,[x19]
    cmp w2,#39
    b.ne 3f
    cmp x20,#2
    b.lo 3f
    ldrb w2,[x19,#1]
    orr w2,w2,#32
    cmp w2,#115
    b.eq 1f
    cmp w2,#116
    b.eq 1f
    cmp w2,#109
    b.eq 1f
    cmp w2,#100
    b.eq 1f
    cmp x20,#3
    b.lo 3f
    ldrb w3,[x19,#2]
    orr w3,w3,#32
    cmp w2,#114
    b.eq 2f
    cmp w2,#118
    b.eq 2f
    cmp w2,#108
    b.ne 3f
    cmp w3,#108
    b.ne 3f
    mov x0,#3
    LEAVE
1:  mov x0,#2
    LEAVE
2:  cmp w3,#101
    b.ne 3f
    mov x0,#3
    LEAVE
3:  mov x0,x19
    mov x1,x20
    bl classify
    mov w21,w0
    mov w22,w1
    mov x23,#0
    tbnz w21,#0,5f
    mov w2,#11
    tst w21,w2
    b.ne 6f
    cmp x20,x22
    b.ls 6f
    add x0,x19,x22
    sub x1,x20,x22
    bl classify
    tbz w0,#0,6f
    mov x23,x22
5:  cmp x23,x20
    b.hs lex_return
    add x0,x19,x23
    sub x1,x20,x23
    bl classify
    tbz w0,#0,lex_return
    add x23,x23,x1
    b 5b
6:  mov x23,#0
    tbz w21,#1,8f
    mov w24,#0
7:  cmp x23,x20
    b.hs lex_return
    cmp w24,#3
    b.hs lex_return
    add x0,x19,x23
    sub x1,x20,x23
    bl classify
    tbz w0,#1,lex_return
    add x23,x23,x1
    add w24,w24,#1
    b 7b
8:  tst w21,#7
    b.eq 10f
    ldrb w0,[x19]
    cmp w0,#32
    b.ne 12f
    cmp x20,#1
    b.ls 12f
    add x0,x19,#1
    sub x1,x20,#1
    bl classify
    tst w0,#7
    b.ne 12f
    mov x23,#1
10: cmp x23,x20
    b.hs lex_return
    add x0,x19,x23
    sub x1,x20,x23
    bl classify
    tst w0,#7
    b.ne 11f
    add x23,x23,x1
    b 10b
11: cmp x23,x20
    b.hs lex_return
    add x0,x19,x23
    sub x1,x20,x23
    bl classify
    tbz w0,#3,lex_return
    add x23,x23,x1
    b 11b
12: mov x23,#0
    mov x24,#0
    mov x25,#0
13: cmp x23,x20
    b.hs 15f
    add x0,x19,x23
    sub x1,x20,x23
    bl classify
    tbz w0,#2,14f
    mov x25,x23
    add x23,x23,x1
    tbz w0,#3,13b
    mov x24,x23
    b 13b
14: cbnz x24,16f
    cbz x25,lex_return
    mov x23,x25
    b lex_return
15: cbz x24,lex_return
16: mov x23,x24
lex_return:
    cbz x23,tokenizer_error
    mov x0,x23
    LEAVE

tokenize:
    ENTER
    mov x19,x0
    bl strlen
    cmp x0,#65536
    b.hs tokenizer_error
    mov x20,x0
    ADDR x1,input_count
    str wzr,[x1]
.if ADD_BOS
    ldr w0,=BOS
    bl append_input
.endif
1:  cbz x20,6f
    mov x0,x19
    mov x1,x20
    bl find_special
    mov x21,x0
    mov w25,w1
    mov w26,w2
2:  cbz x21,4f
    mov x0,x19
    mov x1,x21
    bl lex_length
    mov x22,x0
    mov x0,x19
    mov x1,x22
    bl bpe_piece
    add x19,x19,x22
    sub x20,x20,x22
    sub x21,x21,x22
    b 2b
4:  cmp w25,#-1
    b.eq 1b
    mov w0,w25
    bl append_input
    add x19,x19,x26
    sub x20,x20,x26
    b 1b
6:  ADDR x0,input_count
    ldr w0,[x0]
    LEAVE
tokenizer_error:
    ADDR x0,token_text
    b fatal
.bss
.balign 16
input_count: .skip 4
bpe_ids: .skip 65536*4
bpe_next: .skip 65536*4
.text
