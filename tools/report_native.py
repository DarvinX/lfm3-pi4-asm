#!/usr/bin/env python3
"""Summarize recorded native measurements without inventing missing results."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    summary = json.loads((a.directory/"summary.json").read_text())
    metadata = json.loads((a.directory/"metadata.json").read_text())
    rows = [r for r in csv.DictReader(open(a.directory/"results.csv")) if r["warmup"] == "False"]
    if len(rows) != 2*metadata["repeat"]:
        raise ValueError("expected all registered measured runs")
    labels = metadata.get("engine_labels", {"assembly":"assembly","llama.cpp":"llama.cpp"})
    candidate, reference = labels["assembly"], labels["llama.cpp"]
    engines = [candidate, reference]
    ratio = summary.get("reference_speed_relative_to_candidate", summary.get("llama_speed_relative_to_assembly"))
    lines = ["# Physical Raspberry Pi comparison", "", "Measured on Eve, a Raspberry Pi 4 Model B Rev 1.5 with 4 GB RAM, using native AArch64 executables. The model file is identical in both engines and was verified by SHA-256. These are physical-device measurements, not emulator timings.", "",
             "| Engine | Generation speed, median (range), tok/s | Peak process RSS, median (max), MiB | Peak anonymous PSS, MiB |", "|---|---:|---:|---:|"]
    for engine in engines:
        s = summary[engine]
        lines.append(f"| {engine} | {s['decode_tps_median']:.3f} ({s['decode_tps_min']:.3f}–{s['decode_tps_max']:.3f}) | {s['peak_rss_mib_median']:.1f} ({s['peak_rss_mib_max']:.1f}) | {s['peak_pss_anon_mib_max']:.1f} |")
    latency = all(row.get('ttft_seconds') for row in rows)
    timing_note = ('First-output latency is separately timestamped below.' if latency else
                   'A matched first-output latency comparison is unavailable in these runs.')
    lines += ['', 'The speed table above is decode-only: `(generated_tokens-1)/decode_seconds`, measured after prefill. It excludes model loading and prompt processing. '+timing_note, '',
              '| Engine | Prompt + output phase, median tok/s | Full fresh-process wall, median tok/s | Full wall median, seconds |',
              '|---|---:|---:|---:|']
    for engine in engines:
        subset = [row for row in rows if row['engine'] == engine]
        phase = [int(row['generated_tokens'])/(float(row['prefill_seconds'])+float(row['decode_seconds'])) for row in subset]
        whole = [int(row['generated_tokens'])/float(row['wall_seconds']) for row in subset]
        wall = [float(row['wall_seconds']) for row in subset]
        lines.append(f"| {engine} | {statistics.median(phase):.3f} | {statistics.median(whole):.3f} | {statistics.median(wall):.3f} |")
    lines += ['', 'Prompt + output includes first-token computation but excludes loading/tokenization. Full wall includes process launch, loading, tokenization, prompt processing, output and exit, with warm OS file caches. Parent exit observation can add up to100ms polling delay. It is not resident-server latency or cold-storage load time.']
    if latency:
        lines += ['', '| Engine | TTFT median (range), seconds | Model-ready to first output, median seconds | Main-request rate, median tok/s |',
                  '|---|---:|---:|---:|']
        for engine in engines:
            subset = [row for row in rows if row['engine'] == engine]
            first = [float(row['ttft_seconds']) for row in subset]
            ready = [float(row['prefill_ttft_seconds']) for row in subset]
            rate = [float(row['request_tokens_per_second']) for row in subset]
            lines.append(f'| {engine} | {statistics.median(first):.3f} ({min(first):.3f}–{max(first):.3f}) | {statistics.median(ready):.3f} | {statistics.median(rate):.3f} |')
        lines += ['', 'TTFT runs from request start in main to the first nonempty token piece fully written to stdout. Both adapters stream each piece. It includes tokenization, model loading, prefill, argmax and output, while excluding exec/loader time before main. Model-ready latency starts at load_end. Main-request rate includes loading/prompt/output, ending before cleanup; full-process wall above includes cleanup. These definitions are separate from decode throughput.']
    else:
        note = ('Do not treat load+prefill as exact TTFT: one or both roles lack independent first-output timestamps.'
                if metadata.get('reference_is_assembly') and not metadata.get('candidate_is_llama') else
                'Do not treat load+prefill as exact TTFT: the adapters place tokenization differently, and the original CPP adapter buffers stdout until its final flush.')
        lines += ['', note]
    if metadata.get('candidate_packed_q4'):
        packed = metadata['candidate_packed_q4']
        lines += ['', f"Candidate uses an exact offline Q4 layout ({packed['bytes']:,} bytes, SHA-256 `{packed['sha256']}`). Both load the same original checkpoint; candidate releases original Q4 pages before validating its packed mapping. RSS includes both mappings' resident pages."]
    if metadata.get('reference_packed_q4'):
        packed = metadata['reference_packed_q4']
        lines += ['', f"Assembly reference also uses its explicitly recorded packed-Q4 mapping ({packed['bytes']:,} bytes, SHA-256 `{packed['sha256']}`). Its RSS includes resident pages from both original and packed mappings."]
    if metadata.get('candidate_certified_head'):
        aux = metadata['candidate_certified_head']
        size = sum(v['bytes'] for v in aux.values())
        io = ('Enabled tied embeddings, proposal scores and refinements read original rows through fixed1680-byte pread buffers; copying/syscalls are included in decoding. This avoids candidate/prefix-dependent population of the original-head mapping. Original-head boundary PTEs are released after a certified selection.'
              if metadata.get('candidate_certified_head_io') == 'pread-v2' else
              'Original-head PTEs are released after a certified selection, and sparse original soft faults/page-release work are included in decoding. This v1 path does not bound peak mapped pages solely by raw row count because Linux can fault around.')
        lines += ['', f"Candidate uses {size:,} additional bytes of immutable auxiliary files for a coarse projection and conservative residual ceilings. It recomputes every possible winner with the original Q6 weights; approximate scores cannot directly commit a token. {io}",
                  '', 'RSS counts process-resident mapped pages. It excludes unreferenced OS file-cache pages: original and auxiliary file caches are warm in this experiment. A lower RSS here is not a measurement of lower total system file-cache residency. Full-score diagnostics bypass auxiliaries.']
    if metadata.get('reference_certified_head'):
        lines += ['', 'The assembly reference uses the same separately SHA-bound certified-head auxiliaries; its source-row I/O mode is '+metadata['reference_certified_head_io']+'.']
    if metadata.get('worker_counts_requested'):
        lines += ['', 'Prefill/decode worker counts: '+json.dumps(metadata['worker_counts_requested'])+'. Both roles retain the same allowed CPU affinity. Latency reports validate actual configured counts.']
    if metadata.get('expected_canonical', {}).get('token_ids_sha256'):
        lines += ['', 'Every full-length timed run additionally matches the original canonical token-ID and decoded-byte SHA-256 fingerprints, beyond matching the other role. These digests are checked by the test controller after execution and are never supplied to either inference child.']
    if (metadata.get('reference_is_assembly') and metadata['asm_sha256'] == metadata['llama_sha256']
            and len(set(metadata.get('smaps_sample_intervals_seconds', {}).values())) > 1):
        lines += ['', 'Both roles execute the identical binary. This is monitoring-overhead calibration, not an inference implementation speedup. Peak RSS comes from the same process-reported getrusage accounting in both roles.']
    winner = reference if ratio >= 1 else candidate
    factor = ratio if ratio >= 1 else 1/ratio
    c, r = summary[candidate], summary[reference]
    overlap = max(c["decode_tps_min"],r["decode_tps_min"]) <= min(c["decode_tps_max"],r["decode_tps_max"])
    caveat = " Ranges overlap; this does not establish a robust speed advantage." if overlap else ""
    identity = ("Raw token IDs match across both engines and replicas." if metadata.get("token_ids_checked") and metadata.get("require_identical_output")
                else "Generated sequences differ; this is not evidence of output equivalence." if summary["different_generated_sequences"]
                else "Decoded output bytes match; raw token IDs were not checked.")
    lines += ["", f"{winner} has a {factor:.3f}× higher median rate.{caveat} Memory values include resident model pages and application buffers, but exclude the OS and unrelated desktop processes; they describe this workload, not a minimum-RAM guarantee.", "",
              "## Controlled setup", "", f"- Model SHA-256: `{metadata['model_sha256']}`.",
              ("- Both roles are assembly builds; exact binary hashes, activation modes and arguments are in metadata.json." if metadata.get("reference_is_assembly") else "- Reference: llama.cpp commit `8642541b255bfb5dc39cd351a178be09013a55d6`, native GCC 14.2.0 Release CPU-only build and C API adapter."),
              f"- {metadata['threads']} workers; context {metadata['context']}; FP32 KV caches in both; flash attention off; serial-token prefill with only the final vocabulary projection.",
              f"- {metadata['max_new']} fixed output tokens; {metadata['max_new']-1} useful decode steps. The first output uses prefill logits. Rate is useful steps / decode seconds; sampling/output overhead is included. Extra speculative evaluations, when present, are reported separately.",
              f"- {metadata['repeat']} runs per engine, alternating paired order after warmups. Model file pages are warm; each process starts with fresh application state. Start temperature <=55 C, unchanged governor, matching CPU affinity.",
              "- Standard EOS stopping is disabled in both engines to keep the measured amount of work fixed.",
              f"- Memory: {metadata['memory_definition']}. Rollup intervals: {metadata.get('smaps_sample_intervals_seconds', {'both': .1})} seconds.",
              f"- Activation modes: {metadata['activations']}. {identity}", "",
              "## Phase times and individual runs", "", "| Engine | Run | Load / setup, s | Prefill, s | Decode, s | Decode tok/s | Peak RSS, MiB | Start / end temperature, C |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    current_throttling = []
    for r in rows:
        before, after = json.loads(r["before"]), json.loads(r["after"])
        current_throttling.append((r["engine"], r["replicate"], before["throttling"], after["throttling"]))
        lines.append(f"| {r['engine']} | {int(r['replicate'])+1} | {float(r['load_seconds']):.3f} | {float(r['prefill_seconds']):.3f} | {float(r['decode_seconds']):.3f} | {float(r['decode_tokens_per_second']):.3f} | {int(r['peak_rss_kib'])/1024:.1f} | {before['temperature_c']} / {after['temperature_c']} |")
    verification = [(r,json.loads(r.get('verification_work') or 'null')) for r in rows]
    verification = [(r,w) for r,w in verification if w is not None]
    if verification:
        lines += ['', '## Verified-draft work', '',
                  'These are actual target-model positions evaluated after prefill, including rejected proposals. Useful output rate above counts emitted tokens, not speculative positions.', '',
                  '| Engine | Run | Mode | Batches | Accepted | Rejected | Evaluated positions |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r,w in verification:
            lines.append(f"| {r['engine']} | {int(r['replicate'])+1} | {w['mode']} | {w['batches']} | {w['accepts']} | {w['rejects']} | {w['evaluated_positions']} |")
    heads = [(r,json.loads(r.get('certified_head_work') or 'null')) for r in rows]
    heads = [(r,w) for r,w in heads if w is not None]
    if heads:
        lines += ['', '## Certified-head work', '',
                  'Counts include the final prefill head. Refined rows are original-score calculations from bounded candidate lists, excluding the separate coarse-proposal threshold calculation. A fallback or first paired graph disables this initial scalar-only certificate path for the remaining completion.', '',
                  '| Engine | Run | Certified heads | Refined rows | Dense fallbacks | Paired disables |',
                  '|---|---:|---:|---:|---:|---:|']
        for r,w in heads:
            lines.append(f"| {r['engine']} | {int(r['replicate'])+1} | {w['heads']} | {w['refined_rows']} | {w['fallbacks']} | {w['paired_disables']} |")
    lines += ["", "## Memory and thermal observations", ""]
    for engine in engines:
        s = summary[engine]
        lines += [f"- {engine}: maximum measured process swap {s['peak_process_swap_mib']:.3f} MiB."]
    lines += ["- Throttling before/after each measured run:", "", "```text"]
    lines += [str(x) for x in current_throttling]
    lines += ["```", "", "Background desktop services were left running. The raw per-run environment records governor, clock, temperature, throttling, system memory and load. Load/setup includes checkpoint validation in the assembly engine; decode timing excludes that startup work.", "",
              "## Evidence", "", "- [Machine-readable summary](summary.json)", "- [Every measured and warmup run](results.csv)", "- [Exact prompt, binaries, model hash and protocol](metadata.json)", "- Per-run `.stdout` files contain generated token bytes; `.stderr` files contain engine diagnostics and phase/memory counters.", ""]
    (a.directory/"REPORT.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
