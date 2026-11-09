#!/usr/bin/env python3
"""Compile a checkpoint-specific assembly index. No inference is performed here."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct
import unicodedata
import zlib

EXPECTED = "3793e1e6e2266919d6e015287221b41100175a98eddaba92e111aed3b05c93c4"
SLOTS = ["attn_norm", "ffn_norm", "post_attention_norm", "post_ffw_norm",
         "ffn_gate", "ffn_up", "ffn_down", "shortconv.in_proj", "shortconv.out_proj",
         "shortconv.conv", "attn_q", "attn_k", "attn_v", "attn_output",
         "attn_q_norm", "attn_k_norm", "attn_gate", "rope_freqs"]


class Reader:
    def __init__(self, path):
        self.file = open(path, "rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        if self.take(4) != b"GGUF" or self.num("I") != 3:
            raise ValueError("requires GGUF v3")
        nt, nk = self.num("Q"), self.num("Q")
        self.meta = {}
        for _ in range(nk):
            key = self.string().decode()
            self.meta[key] = self.value(self.num("I"))
        self.tensors = {}
        for _ in range(nt):
            name = self.string().decode()
            shape = [self.num("Q") for _ in range(self.num("I"))]
            kind, offset = self.num("I"), self.num("Q")
            self.tensors[name] = [offset, shape, kind]
        alignment = self.meta.get("general.alignment", 32)
        self.base = (self.pos + alignment - 1) // alignment * alignment

    def take(self, size):
        end = self.pos + size
        if end > len(self.data):
            raise ValueError("truncated GGUF")
        result = self.data[self.pos:end]
        self.pos = end
        return result

    def num(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.take(self.num("Q"))

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            sub, count = self.num("I"), self.num("Q")
            return [self.value(sub) for _ in range(count)]
        return self.num({0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                         6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[kind])


def fnv(data):
    h = 14695981039346656037
    for b in data:
        h = ((h ^ b) * 1099511628211) & 0xffffffffffffffff
    return h


def prepare(path, out, allow_fixture=False):
    out.mkdir(parents=True, exist_ok=True)
    r = Reader(path)
    sha = hashlib.sha256(r.data).hexdigest()
    if sha != EXPECTED and not allow_fixture:
        raise ValueError(f"checkpoint SHA256 mismatch: {sha}")
    m = r.meta
    if m["general.architecture"] != b"lfm3":
        raise ValueError("requires lfm3 architecture")
    get = lambda k: m["lfm3." + k]
    dim, ff, layers = get("embedding_length"), get("feed_forward_length"), get("block_count")
    heads, hd = get("attention.head_count"), get("attention.key_length")
    kvs, swas = get("attention.head_count_kv"), get("attention.sliding_window_pattern")
    kv = max(kvs)
    if heads * hd != dim or heads % kv or get("attention.value_length") != hd:
        raise ValueError("unsupported attention layout")
    if any(k not in (0, kv) for k in kvs) or hd % 2:
        raise ValueError("unsupported head dimensions")
    tokens = m["tokenizer.ggml.tokens"]
    types = m["tokenizer.ggml.token_type"]
    if m["tokenizer.ggml.model"] != b"gpt2" or m["tokenizer.ggml.pre"] != b"lfm2":
        raise ValueError("requires LFM2 GPT-2 byte BPE tokenizer")
    # GPT-2's reversible byte alphabet. Store original bytes, so the runtime
    # never needs to materialize an intermediate Unicode byte encoding.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs.copy()
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    decode = dict(zip(map(chr, cs), bs))
    raw = []
    for text, typ in zip(tokens, types):
        text = text.decode("utf-8")
        raw.append(bytes(decode[c] for c in text) if typ in (1, 6) else text.encode())
    ids = {t: i for i, t in enumerate(tokens)}
    raw_ids = {t: i for i, t in enumerate(raw) if types[i] in (1, 6)}
    byte_ids = [raw_ids[bytes([b])] for b in range(256)]
    vocab_blob, descriptors = bytearray(), bytearray()
    for text, typ in zip(raw, types):
        descriptors += struct.pack("<QII", len(vocab_blob), len(text), typ)
        vocab_blob += text
    table_size = 1
    while table_size < len(tokens) * 2:
        table_size *= 2
    table = bytearray(table_size * 16)
    for i, text in enumerate(raw):
        if types[i] not in (1, 6):
            continue
        h, slot = fnv(text), fnv(text) & (table_size - 1)
        while struct.unpack_from("<I", table, slot * 16 + 8)[0]:
            slot = (slot + 1) & (table_size - 1)
        struct.pack_into("<QII", table, slot * 16, h, i + 1, 0)
    merges = m["tokenizer.ggml.merges"]
    merge_size = 1
    while merge_size < len(merges) * 2:
        merge_size *= 2
    merge_table = bytearray(merge_size * 16)
    for rank, merge in enumerate(merges):
        a, b = merge.split(b" ", 1)
        if a not in ids or b not in ids or a + b not in ids:
            continue
        ia, ib, result = ids[a], ids[b], ids[a + b]
        key = ((ia << 32) | ib) + 1
        slot = ((ia * 0x9e3779b1) ^ ib) & (merge_size - 1)
        while struct.unpack_from("<Q", merge_table, slot * 16)[0]:
            slot = (slot + 1) & (merge_size - 1)
        struct.pack_into("<QII", merge_table, slot * 16, key, rank, result)
    categories = bytearray(0x110000)
    for cp in range(len(categories)):
        c = chr(cp)
        cat = unicodedata.category(c)
        categories[cp] = ((1 if cat.startswith("L") else 0) |
                          (2 if cat.startswith("N") else 0) |
                          (4 if c.isspace() else 0) | (8 if c in "\r\n" else 0))
    assets = {"vocab": descriptors, "text": vocab_blob, "lookup": table,
              "merges": merge_table, "categories": categories,
              "bytes": struct.pack("<256I", *byte_ids)}
    for name, data in assets.items():
        (out / (name + ".bin")).write_bytes(data)
    constants = {"DIM": dim, "FF": ff, "LAYERS": layers, "HEADS": heads,
                 "HD": hd, "KV": kv, "KW": kv * hd, "VOCAB": len(tokens),
                 "CONV": get("shortconv.l_cache"), "WINDOW": get("attention.sliding_window"),
                 "BOS": m["tokenizer.ggml.bos_token_id"], "EOS": m["tokenizer.ggml.eos_token_id"],
                 "LOOKUP_MASK": table_size - 1, "MERGE_MASK": merge_size - 1,
                 "MODEL_BYTES": len(r.data), "MODEL_CRC": zlib.crc32(r.data),
                 "ADD_BOS": int(m.get("tokenizer.ggml.add_bos_token", True))}
    lines = [f".equ {k}, {v}" for k, v in constants.items()]
    lines += [".section .rodata", ".balign 16"]
    for name in assets:
        lines += [f"{name}_data:", f'.incbin "{out / (name + ".bin")}"', ".balign 16"]
    def tensor(name):
        entry = r.tensors.get(name)
        if entry is None:
            return ".quad 0\n.word 0,0,0,0\n.quad 0"
        off, shape, kind = entry
        if len(shape) not in (1, 2) or kind not in (0, 12, 14):
            raise ValueError(f"unsupported tensor {name}: {shape}, {kind}")
        cols, rows = shape[0], shape[1] if len(shape) == 2 else 1
        if kind and cols % 256:
            raise ValueError("quantized row must be a multiple of 256")
        rowbytes = cols * 4 if kind == 0 else cols // 256 * {12: 144, 14: 210}[kind]
        absolute = off + r.base
        size = rowbytes * rows
        if absolute + size > len(r.data):
            raise ValueError("tensor outside model")
        return f".quad {absolute}\n.word {cols},{rows},{kind},{rowbytes}\n.quad {size}"
    lines += ["embedding:", tensor("token_embd.weight"), "final_norm:", tensor("output_norm.weight")]
    lines += ["output_tensor:", tensor("output.weight" if "output.weight" in r.tensors else "token_embd.weight")]
    lines += ["rope_factors:", tensor("rope_freqs.weight")]
    lines += ["layer_types:", ".word " + ",".join(str(0 if not k else (2 if s else 1)) for k, s in zip(kvs, swas))]
    lines += [".balign 16", "layer_tensors:"]
    for layer in range(layers):
        for slot in SLOTS:
            name = f"blk.{layer}.{slot}.weight"
            if slot == "rope_freqs" and name not in r.tensors:
                name = "rope_freqs.weight"
            lines.append(tensor(name))
    lines += ["norm_eps:", f".float {get('attention.layer_norm_rms_epsilon')!r}",
              "rope_global:", f".float {get('rope.freq_base')!r}",
              "rope_window:", f".float {get('rope.freq_base_swa')!r}",
              "special_ids:", ".word " + ",".join(str(i) for i, t in enumerate(types) if t in (3, 4)),
              ".word -1", ".text"]
    (out / "model.inc").write_text("\n".join(lines) + "\n")
    manifest = {**constants, "sha256": sha, "model": str(path.resolve()),
                "scope": "checkpoint-specific index; original GGUF weights are unchanged"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--out", type=Path, default=Path("build"))
    p.add_argument("--fixture", action="store_true", help="allow a synthetic checkpoint for tests")
    a = p.parse_args()
    prepare(a.model, a.out, a.fixture)
