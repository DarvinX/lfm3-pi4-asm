#!/usr/bin/env python3
"""Sequential, paired native Pi benchmark: decode speed + measured memory."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import time
from pmu import PMU


def read(path):
    try:
        return Path(path).read_text().strip().replace("\x00", "")
    except OSError:
        return "unavailable"


def thermal():
    value = read("/sys/class/thermal/thermal_zone0/temp")
    return float(value)/1000 if value.isdigit() else None


def meminfo(path):
    result = {}
    try:
        for line in Path(path).read_text().splitlines():
            m = re.match(r"(\w+):\s+(\d+)\s+kB", line)
            if m:
                result[m[1]] = int(m[2])
    except OSError:
        pass
    return result


def snapshot():
    return {"temperature_c": thermal(), "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "frequency_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
            "throttling": subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout.strip(),
            "memory_kib": meminfo("/proc/meminfo"), "loadavg": read("/proc/loadavg")}


def bind_certified_head(coarse, residual):
    """Bind only the immutable auxiliary format qualified for this checkpoint."""
    aux = {}
    for key, path, expected, size in [
        ('coarse', coarse, 'f874a26cc0b1b7aeae36bf89572449723936e61af07d11b2e8d1bccc4da1899c', 149504000),
        ('residual', residual, 'ca6634db00411678c62803911f0e78705dac74e731bce346c81cf6aae564f281', 4096000)]:
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest != expected or path.stat().st_size != size:
            raise ValueError('certified-head immutable auxiliary binding failed')
        aux[key] = {'path': str(path.resolve()), 'bytes': size, 'sha256': digest}
    return aux


def parse_latency(stderr, prefill_threads, decode_threads, wall, required=False):
    lines = [line for line in stderr.splitlines() if line.startswith('LATENCY ')]
    if not lines:
        if required:
            raise ValueError('missing required first-output latency report')
        return None
    if len(lines) != 1:
        raise ValueError('ambiguous latency report')
    fields = dict(re.findall(r'(\w+)=([^\s]+)', lines[0]))
    result = {key: float(fields[key]) for key in
              ['ttft_seconds', 'request_seconds', 'prefill_ttft_seconds']}
    result.update({key: int(fields[key]) for key in
                   ['first_piece_bytes', 'prefill_threads', 'decode_threads']})
    if (not all(math.isfinite(result[key]) for key in
                ['ttft_seconds', 'request_seconds', 'prefill_ttft_seconds'])
            or not 0 < result['prefill_ttft_seconds'] <= result['ttft_seconds'] <= result['request_seconds']
            or result['request_seconds'] > wall + .005
            or result['first_piece_bytes'] <= 0
            or result['prefill_threads'] != prefill_threads
            or result['decode_threads'] != decode_threads):
        raise ValueError(f'invalid latency timestamps or actual worker counts: {result}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asm", type=Path, required=True)
    p.add_argument("--llama", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, choices=[1, 2, 3, 4], default=4)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--max-start-temperature", type=float, default=55)
    p.add_argument("--smaps-interval", type=float, default=1,
                   help="rollup interval;1s adopted after paired overhead calibration")
    p.add_argument("--reference-smaps-interval", type=float,
                   help="override reference rollup cadence for monitor-overhead calibration")
    p.add_argument("--candidate-name", default="assembly")
    p.add_argument("--reference-name", default="llama.cpp")
    p.add_argument("--reference-is-assembly", action="store_true")
    p.add_argument("--candidate-is-llama", action="store_true",
                   help="candidate uses the pinned C API adapter CLI/token-ID interface")
    p.add_argument("--candidate-decode-threads", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--reference-decode-threads", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--require-latency", action="store_true",
                   help="require actual first-nonempty-output timestamps and worker-count reports")
    p.add_argument("--candidate-q8", action="store_true")
    p.add_argument("--candidate-packed-q4", type=Path,
                   help="optional exact packed-Q4 file; candidate only, hash recorded")
    p.add_argument("--candidate-coarse-head", type=Path,
                   help="H34 exact-winner auxiliary projection, candidate only; original weights unchanged")
    p.add_argument("--candidate-residual-norms", type=Path,
                   help="H34 CRC-bound residual ceilings; required with candidate-coarse-head")
    p.add_argument("--candidate-certified-head-io", choices=['mmap-v1', 'pread-v2'], default='mmap-v1',
                   help="record and verify frozen H34 implementation's original-row I/O path")
    p.add_argument("--reference-coarse-head", type=Path)
    p.add_argument("--reference-residual-norms", type=Path)
    p.add_argument("--reference-certified-head-io", choices=['mmap-v1', 'pread-v2'], default='mmap-v1')
    p.add_argument("--reference-packed-q4", type=Path,
                   help="optional exact packed-Q4 file for an assembly reference; hash recorded")
    p.add_argument("--reference-q8", action="store_true")
    p.add_argument("--candidate-spec-mode", type=int, choices=[0, 1],
                   help="explicit scalar/adaptive verified-draft mode when supported")
    p.add_argument("--reference-spec-mode", type=int, choices=[0, 1])
    p.add_argument("--pmu", action="store_true")
    p.add_argument("--pmu-profile", choices=["memory", "control", "scheduling"], default="memory")
    p.add_argument("--require-identical-output", action="store_true")
    p.add_argument("--token-ids", action="store_true", help="candidate TOKEN_TRACE(fd4), reference LFM_TOKEN_IDS; compare raw IDs")
    p.add_argument("--expected-model-sha256", help="optional original checkpoint binding")
    p.add_argument("--expected-token-ids-sha256", help="optional original canonical digest; timed runs only")
    p.add_argument("--expected-decoded-sha256", help="optional original canonical text digest; timed runs only")
    p.add_argument("--prompt", default="<|im_start|>user\nWrite a long, detailed guide to growing tomatoes in a home garden, covering soil, planting, watering, pests, and harvesting.\n<|im_end|>\n<|im_start|>assistant\n")
    a = p.parse_args()
    for value in [a.expected_model_sha256, a.expected_token_ids_sha256, a.expected_decoded_sha256]:
        if value is not None and not re.fullmatch('[0-9a-f]{64}', value):
            p.error('expected SHA-256 values must be64 lowercase hex digits')
    if a.expected_token_ids_sha256 and not a.token_ids:
        p.error('expected token-ID digest requires token-ids')
    for role in ['candidate', 'reference']:
        coarse, residual = getattr(a, role+'_coarse_head'), getattr(a, role+'_residual_norms')
        if bool(coarse) != bool(residual):
            p.error(role+" coarse-head and residual-norms must be specified together")
        is_assembly = not a.candidate_is_llama if role == 'candidate' else a.reference_is_assembly
        if coarse and (not is_assembly or not getattr(a, role+'_q8')):
            p.error(role+" certified head requires an assembly role with explicit Q8")
        count = getattr(a, role+'_decode_threads')
        if count is not None and count > a.threads:
            p.error(role+" decode workers cannot exceed prefill workers")
    if a.candidate_is_llama and (a.candidate_packed_q4 or a.candidate_spec_mode is not None):
        p.error("assembly candidate options cannot be used with candidate-is-llama")
    if a.candidate_name == a.reference_name:
        p.error("candidate/reference labels must differ")
    if a.reference_packed_q4 and not a.reference_is_assembly:
        p.error("reference-packed-q4 requires reference-is-assembly")
    if a.reference_spec_mode is not None and not a.reference_is_assembly:
        p.error("reference-spec-mode requires reference-is-assembly")
    if platform.machine() != "aarch64" or "Raspberry Pi" not in read("/sys/firmware/devicetree/base/model"):
        p.error("this comparison requires a physical ARM64 Raspberry Pi")
    if not (8 <= a.tokens <= 128) or a.repeat < 3:
        p.error("tokens must be 8..128 and repeat >= 3")
    if not .1 <= a.smaps_interval <= 5 or (a.reference_smaps_interval is not None and not .1 <= a.reference_smaps_interval <= 5):
        p.error("smaps intervals must be0.1..5seconds; safety checks remain100ms")
    if (a.out/"metadata.json").exists():
        p.error("Use a fresh output directory to preserve prior evidence")
    a.out.mkdir(parents=True, exist_ok=True)
    lock = open(a.out/"benchmark.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if meminfo("/proc/meminfo").get("MemAvailable", 0) < 1536*1024:
        p.error("less than 1.5 GiB available memory")
    active = subprocess.check_output(["ps", "-eo", "comm="], text=True).splitlines()
    if any(n.strip() in {"cc1", "cc1plus", "llama-server", "qemu-aarch64", "bench-llama", "parity-llama", "latency-llama", "tile16-check"} or n.strip().startswith("lfm3-") for n in active):
        p.error("a build or another inference job is active")
    # Fixed affinity applies only to benchmark children, never desktop services.
    cores = sorted(os.sched_getaffinity(0))[:a.threads]
    def child_limits():
        os.sched_setaffinity(0, set(cores))
        resource.setrlimit(resource.RLIMIT_AS, (3*1024**3, 3*1024**3))
    env = {**os.environ, "LFM_THREADS": str(a.threads), "LFM_REPORT": "1", "LFM_BENCH_IGNORE_EOS": "1",
           "OMP_NUM_THREADS": str(a.threads), "OMP_WAIT_POLICY": "PASSIVE"}
    labels = {"assembly": a.candidate_name, "llama.cpp": a.reference_name}
    metadata = {"board": read("/sys/firmware/devicetree/base/model"), "uname": platform.uname()._asdict(),
                "benchmark_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "pmu_runner_sha256": hashlib.sha256(Path(__file__).with_name('pmu.py').read_bytes()).hexdigest(),
                "cpuinfo": read("/proc/cpuinfo"), "cpu_affinity": cores, "threads": a.threads,
                "model_sha256": hashlib.file_digest(open(a.model, "rb"), "sha256").hexdigest(),
                "asm_sha256": hashlib.sha256(a.asm.read_bytes()).hexdigest(),
                "llama_sha256": hashlib.sha256(a.llama.read_bytes()).hexdigest(),
                "context": 512, "kv_cache_dtype": "float32 in both engines", "flash_attention": "off",
                "prefill": "serial tokens; vocabulary projection only at final prompt token",
                "activations": "assembly float32; llama.cpp default Q8 activation quantization in quantized matvec",
                "prompt": a.prompt, "max_new": a.tokens, "repeat": a.repeat,
                "cache_state": "warm model file pages; fresh process and cache state per run",
                "decode_rate_definition": "(generated tokens - 1) / decode_seconds; first output uses prefill logits",
                "memory_definition": "getrusage peak RSS; smaps_rollup at registered role cadence;100ms VmRSS/VmSwap/MemAvailable sampling",
                "memory_sample_interval_seconds": .1,
                "smaps_sample_intervals_seconds": {a.candidate_name: a.smaps_interval,
                    a.reference_name: a.reference_smaps_interval if a.reference_smaps_interval is not None else a.smaps_interval},
                "monitor_cpu_definition": "parent process CPU during child execution; includes procfs sampling",
                "eos_policy": "ignore EOS in both for fixed-size throughput measurement", "before": snapshot()}
    metadata["engine_labels"] = labels
    if a.expected_model_sha256 and metadata['model_sha256'] != a.expected_model_sha256:
        p.error('checkpoint differs from expected original canonical binding')
    metadata['expected_canonical'] = {'model_sha256':a.expected_model_sha256,
        'token_ids_sha256':a.expected_token_ids_sha256, 'decoded_sha256':a.expected_decoded_sha256,
        'scope':'test-controller checks only; full-length timed runs; digests are never supplied to inference children'}
    metadata["candidate_q8"] = a.candidate_q8
    for role in ['candidate', 'reference']:
        coarse, residual = getattr(a, role+'_coarse_head'), getattr(a, role+'_residual_norms')
        if coarse:
            metadata[role+'_certified_head'] = bind_certified_head(coarse, residual)
            metadata[role+'_certified_head_io'] = getattr(a, role+'_certified_head_io')
    if a.candidate_packed_q4:
        with a.candidate_packed_q4.open('rb') as packed:
            packed_sha = hashlib.file_digest(packed, 'sha256').hexdigest()
        metadata['candidate_packed_q4'] = {'path': str(a.candidate_packed_q4.resolve()),
            'bytes': a.candidate_packed_q4.stat().st_size, 'sha256': packed_sha,
            'warm_state': 'candidate warmup touches packed file; original GGUF validated by both'}
    if a.reference_packed_q4:
        with a.reference_packed_q4.open('rb') as packed:
            packed_sha = hashlib.file_digest(packed, 'sha256').hexdigest()
        metadata['reference_packed_q4'] = {'path': str(a.reference_packed_q4.resolve()),
            'bytes': a.reference_packed_q4.stat().st_size, 'sha256': packed_sha,
            'warm_state': 'reference warmup touches packed file; original GGUF validated by both'}
    metadata["reference_q8"] = a.reference_q8 if a.reference_is_assembly else True
    metadata["reference_is_assembly"] = a.reference_is_assembly
    metadata["candidate_is_llama"] = a.candidate_is_llama
    metadata['require_latency'] = a.require_latency
    metadata['worker_counts_requested'] = {
        label: {'prefill': a.threads, 'decode': count or a.threads}
        for label, count in [(a.candidate_name, a.candidate_decode_threads),
                             (a.reference_name, a.reference_decode_threads)]}
    metadata['ttft_definition'] = ('main request start to first nonempty token piece fully written to stdout; '
        'includes tokenization, loading, prompt processing, argmax and output; excludes exec/loader before main')
    metadata['request_time_definition'] = 'main request start to completion of generated output; excludes cleanup'
    metadata['full_process_time_definition'] = ('parent launch to observed child exit; '
        'includes cleanup and up to100ms polling delay, separate from in-process request time')
    metadata["pmu_enabled"] = a.pmu
    metadata["pmu_profile"] = a.pmu_profile
    metadata["require_identical_output"] = a.require_identical_output
    metadata["token_ids_checked"] = a.token_ids
    metadata["verification_modes_requested"] = {a.candidate_name:a.candidate_spec_mode,
                                               a.reference_name:a.reference_spec_mode}
    metadata["decode_steps_definition"] = "useful generated-token steps after the first prefill output; actual speculative evaluations recorded separately"
    metadata["activations"] = {a.candidate_name: "Q8_K" if (a.candidate_q8 or a.candidate_is_llama) else "FP32",
                               a.reference_name: "Q8_K" if (a.reference_q8 or not a.reference_is_assembly) else "FP32"}
    (a.out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    rows = []
    outputs = {}
    token_outputs = {}
    prefill_counts = set()
    def measure(engine, replicate, warmup=False):
        deadline = time.monotonic()+240
        while thermal() is not None and thermal() > a.max_start_temperature:
            if time.monotonic() > deadline:
                raise RuntimeError("Pi did not cool to the registered start temperature")
            time.sleep(5)
        count = a.warmup if warmup else a.tokens
        executable = a.asm if engine == "assembly" else a.llama
        is_assembly = not a.candidate_is_llama if engine == 'assembly' else a.reference_is_assembly
        cmd = [str(executable.resolve()), str(a.model.resolve()), a.prompt, str(count)]
        if not is_assembly:
            cmd.append(str(a.threads))
        process_env = {**env, "LFM_Q8": "1" if (a.candidate_q8 if engine == "assembly" else a.reference_q8) else "0"}
        process_env.pop('LFM_Q4_PACKED', None)
        process_env.pop('LFM_SPEC', None)
        process_env.pop('LFM_COARSE', None)
        process_env.pop('LFM_RESIDUAL', None)
        process_env.pop('LFM_DECODE_THREADS', None)
        process_env.pop('LFM_TOKEN_IDS', None)
        role = 'candidate' if engine == 'assembly' else 'reference'
        coarse, residual = getattr(a, role+'_coarse_head'), getattr(a, role+'_residual_norms')
        decode_threads = getattr(a, role+'_decode_threads')
        if decode_threads is not None:
            process_env['LFM_DECODE_THREADS'] = str(decode_threads)
        if coarse:
            process_env['LFM_COARSE'] = str(coarse.resolve())
            process_env['LFM_RESIDUAL'] = str(residual.resolve())
        spec_mode = a.candidate_spec_mode if engine == 'assembly' else a.reference_spec_mode
        if spec_mode is not None:
            process_env['LFM_SPEC'] = str(spec_mode)
        if engine == 'assembly' and a.candidate_packed_q4:
            process_env['LFM_Q4_PACKED'] = str(a.candidate_packed_q4.resolve())
        if engine == 'llama.cpp' and a.reference_packed_q4:
            process_env['LFM_Q4_PACKED'] = str(a.reference_packed_q4.resolve())
        name = f"{engine.replace('.', '-')}-{replicate}" + ("-warmup" if warmup else "")
        id_path = (a.out/(name+".ids")).resolve()
        if a.token_ids:
            process_env["LFM_TOKEN_IDS"] = str(id_path)
            if is_assembly:
                cmd = ["bash", "-c", 'exec "$@" 4>"$LFM_TOKEN_IDS"', "bash", *cmd]
        before = snapshot()
        sample_peak = {}
        minimum_available = before["memory_kib"].get("MemAvailable", 0)
        start = time.monotonic()
        monitor_cpu_start = time.process_time()
        smaps_interval = a.smaps_interval if engine == 'assembly' else (
            a.reference_smaps_interval if a.reference_smaps_interval is not None else a.smaps_interval)
        next_smaps = 0.0
        with open(a.out/(name+".stdout"), "wb") as out, open(a.out/(name+".stderr"), "wb") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=process_env, preexec_fn=child_limits)
            pmu = PMU(proc.pid, a.pmu_profile) if a.pmu else None
            while proc.poll() is None:
                if time.monotonic()-start > a.timeout:
                    proc.kill()
                    proc.wait()
                    raise RuntimeError(f"{name} timed out")
                available = meminfo("/proc/meminfo").get("MemAvailable", 0)
                minimum_available = min(minimum_available, available)
                if available and available < 256*1024:
                    proc.kill()
                    proc.wait()
                    raise RuntimeError("memory pressure floor reached; benchmark terminated to protect the desktop")
                status = meminfo(f"/proc/{proc.pid}/status")
                sample = {'Rss': status.get('VmRSS', 0), 'Swap': status.get('VmSwap', 0)}
                if time.monotonic() >= next_smaps:
                    rollup = meminfo(f"/proc/{proc.pid}/smaps_rollup")
                    sample['Rss'] = max(sample['Rss'], rollup.get('Rss', 0))
                    sample['Swap'] = max(sample['Swap'], rollup.get('Swap', 0))
                    sample.update({k: v for k, v in rollup.items() if k not in ['Rss', 'Swap']})
                    next_smaps = time.monotonic()+smaps_interval
                for key, value in sample.items():
                    sample_peak[key] = max(sample_peak.get(key, 0), value)
                time.sleep(.1)
        wall = time.monotonic()-start
        monitor_cpu = time.process_time()-monitor_cpu_start
        counters = pmu.finish() if pmu else {}
        stderr = (a.out/(name+".stderr")).read_text(errors="replace")
        lines = [s for s in stderr.splitlines() if s.startswith("prefill_tokens=")]
        if proc.returncode or not lines:
            raise RuntimeError(f"{name} failed rc={proc.returncode}: {stderr[-2000:]}")
        fields = dict(re.findall(r"(\w+)=([0-9.]+)", lines[-1]))
        work_lines = [s for s in stderr.splitlines() if s.startswith('H33 batches=')]
        verification_work = None
        if work_lines:
            verification_work = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',work_lines[-1])}
            if (verification_work['batches'] != verification_work['accepts']+verification_work['rejects']
                    or verification_work['evaluated_positions'] != count-1+verification_work['rejects']
                    or verification_work['evaluated_positions'] < 2*verification_work['batches']):
                raise RuntimeError(f'{name}: inconsistent verification-work counters')
            if spec_mode is not None and verification_work['mode'] != spec_mode:
                raise RuntimeError(f'{name}: requested verification mode was not honored')
        head_lines = [s for s in stderr.splitlines() if s.startswith('H34 heads=')]
        head_work = {k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',head_lines[-1])} if head_lines else None
        if coarse:
            if (not head_work or head_work['heads']+head_work['fallbacks'] == 0
                    or head_work['refined_rows'] > 4096*head_work['heads']
                    or head_work['fallbacks']+head_work['paired_disables'] > 1):
                raise RuntimeError(f'{name}: certified-head setup/work counters invalid')
            if head_work.get('io', 1) != (2 if getattr(a, role+'_certified_head_io') == 'pread-v2' else 1):
                raise RuntimeError(f'{name}: certified-head I/O implementation mismatch')
            if (verification_work and not verification_work['batches'] and not head_work['fallbacks']
                    and head_work['heads'] != count):
                raise RuntimeError(f'{name}: scalar certified head count mismatch')
        for key in ["prefill_tokens", "generated_tokens", "decode_steps", "peak_rss_kib"]:
            fields[key] = int(fields[key])
        for key in ["load_seconds", "prefill_seconds", "decode_seconds"]:
            fields[key] = float(fields[key])
        if fields["generated_tokens"] != count or fields["decode_steps"] != count-1:
            raise RuntimeError(f"{name}: not the registered fixed decode workload: {fields}")
        prefill_counts.add(fields['prefill_tokens'])
        if len(prefill_counts) != 1 or fields['prefill_tokens'] < 1:
            raise RuntimeError(f'{name}: candidate/reference prompt-token counts differ')
        phase_total = sum(fields[key] for key in ['load_seconds', 'prefill_seconds', 'decode_seconds'])
        if (not all(math.isfinite(fields[key]) and fields[key] > 0 for key in
                    ['load_seconds', 'prefill_seconds', 'decode_seconds'])
                or phase_total > wall + .005):
            raise RuntimeError(f'{name}: invalid phase clocks')
        latency = parse_latency(stderr, a.threads, decode_threads or a.threads, wall, a.require_latency)
        if latency and (latency['request_seconds'] + .005 < phase_total
                        or latency['prefill_ttft_seconds'] + .000001 < fields['prefill_seconds']):
            raise RuntimeError(f'{name}: first-output/request clocks disagree with phase clocks')
        stdout = (a.out/(name+".stdout")).read_bytes()
        fingerprint = hashlib.sha256(stdout).hexdigest()
        if not warmup and a.expected_decoded_sha256 and fingerprint != a.expected_decoded_sha256:
            raise RuntimeError(f'{name}: decoded output differs from original canonical digest')
        token_fingerprint = ""
        if a.token_ids:
            ids = id_path.read_bytes()
            if len(ids) != 4*count:
                raise RuntimeError(f"{name}: expected {count} token IDs, got {len(ids)//4}")
            token_fingerprint = hashlib.sha256(ids).hexdigest()
            if not warmup and a.expected_token_ids_sha256 and token_fingerprint != a.expected_token_ids_sha256:
                raise RuntimeError(f'{name}: token IDs differ from original canonical digest')
            token_key = "warmup" if warmup else "measured"
            previous = token_outputs.setdefault(token_key, {})
            if engine in previous and previous[engine] != token_fingerprint:
                raise RuntimeError(f"{engine}: non-deterministic token IDs")
            previous[engine] = token_fingerprint
            if a.require_identical_output and len(previous) == 2 and len(set(previous.values())) != 1:
                raise RuntimeError("candidate/reference token IDs differ; performance verdict blocked")
        if not warmup and engine in outputs and fingerprint != outputs[engine]:
            raise RuntimeError(f"{engine}: non-deterministic generation across replicas")
        if not warmup:
            outputs[engine] = fingerprint
            if a.require_identical_output and len(outputs) == 2 and len(set(outputs.values())) != 1:
                raise RuntimeError("candidate/reference generated outputs differ; performance verdict blocked")
        row = {"engine": labels[engine], "replicate": replicate, "warmup": warmup, "threads": a.threads,
               **fields, "decode_tokens_per_second": fields["decode_steps"]/fields["decode_seconds"],
               "wall_seconds": wall, "sampled_peak_rss_kib": sample_peak.get("Rss", 0),
               "full_process_tokens_per_second": count/wall,
               "latency_work": json.dumps(latency),
               "ttft_seconds": latency['ttft_seconds'] if latency else None,
               "prefill_ttft_seconds": latency['prefill_ttft_seconds'] if latency else None,
               "request_seconds": latency['request_seconds'] if latency else None,
               "request_tokens_per_second": count/latency['request_seconds'] if latency else None,
               "verification_work": json.dumps(verification_work),
               "certified_head_work": json.dumps(head_work),
               "monitor_cpu_seconds": monitor_cpu,
               "monitor_fraction_of_one_core": monitor_cpu/wall,
               "smaps_interval_seconds": smaps_interval,
               "sampled_peak_pss_anon_kib": sample_peak.get("Pss_Anon", 0),
               "sampled_peak_pss_file_kib": sample_peak.get("Pss_File", 0),
               "sampled_peak_swap_kib": sample_peak.get("Swap", 0),
               "minimum_system_available_kib": minimum_available, "output_sha256": fingerprint, "token_ids_sha256": token_fingerprint,
               "before": json.dumps(before), "after": json.dumps(snapshot()), "argv": json.dumps(cmd), "pmu": json.dumps(counters)}
        csvpath = a.out/"results.csv"
        fresh = not csvpath.exists()
        with open(csvpath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if fresh:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        if not warmup:
            rows.append(row)
        print(json.dumps(row), flush=True)
    measure("assembly", -1, True)
    measure("llama.cpp", -1, True)
    for replicate in range(a.repeat):
        order = ["assembly", "llama.cpp"] if replicate % 2 == 0 else ["llama.cpp", "assembly"]
        for engine in order:
            measure(engine, replicate)
    import statistics
    summary = {}
    for engine in ["assembly", "llama.cpp"]:
        subset = [r for r in rows if r["engine"] == labels[engine]]
        rates = [r["decode_tokens_per_second"] for r in subset]
        memory = [r["peak_rss_kib"]/1024 for r in subset]
        summary[labels[engine]] = {"replicates": len(subset), "decode_tps_median": statistics.median(rates),
                           "decode_tps_min": min(rates), "decode_tps_max": max(rates),
                           "peak_rss_mib_median": statistics.median(memory), "peak_rss_mib_max": max(memory),
                           "peak_pss_anon_mib_max": max(r["sampled_peak_pss_anon_kib"] for r in subset)/1024,
                           "peak_process_swap_mib": max(r["sampled_peak_swap_kib"] for r in subset)/1024}
        for metric, key in [('full_process_tps', 'full_process_tokens_per_second'),
                            ('ttft_seconds', 'ttft_seconds'), ('prefill_ttft_seconds', 'prefill_ttft_seconds'),
                            ('request_seconds', 'request_seconds'), ('request_tps', 'request_tokens_per_second')]:
            values = [row[key] for row in subset if row[key] is not None]
            if len(values) == len(subset):
                for stat, fn in [('median', statistics.median), ('min', min), ('max', max)]:
                    summary[labels[engine]][metric+'_'+stat] = fn(values)
    summary["reference_speed_relative_to_candidate"] = summary[a.reference_name]["decode_tps_median"]/summary[a.candidate_name]["decode_tps_median"]
    if a.candidate_name == "assembly" and a.reference_name == "llama.cpp":
        summary["llama_speed_relative_to_assembly"] = summary["reference_speed_relative_to_candidate"]
    summary["different_generated_sequences"] = outputs["assembly"] != outputs["llama.cpp"]
    summary["after"] = snapshot()
    (a.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
