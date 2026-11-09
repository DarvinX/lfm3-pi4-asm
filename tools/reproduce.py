#!/usr/bin/env python3
"""Run the published H42 comparison with locally built binaries on a Pi 4."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--asm", type=Path, default=ROOT / "build/lfm3-pi4-trace")
    parser.add_argument("--llama", type=Path, default=ROOT / "build/llama/latency-llama")
    parser.add_argument("--out", type=Path, default=ROOT / "results/reproduction")
    parser.add_argument("--case", choices=("both", "guide", "weather"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    canonical = json.loads((ROOT / "benchmarks/canonical.json").read_text())
    cases = ("guide", "weather") if args.case == "both" else (args.case,)
    coarse = ROOT / "build/h34-coarse-v1/coarse-output.bin"
    residual = ROOT / "build/h34-coarse-v1/residual-norms.f32"
    outputs = [args.out.resolve() / case for case in cases]

    if not args.dry_run:
        for path in (args.model, args.asm, args.llama, coarse, residual):
            if not path.is_file():
                parser.error(f"missing prerequisite: {path}")
        for path in outputs:
            if path.exists():
                parser.error(f"choose a fresh output directory: {path}")

    environment = {key: value for key, value in os.environ.items() if not key.startswith("LFM_")}
    for case, output in zip(cases, outputs):
        expected = canonical["cases"][case]
        command = [
            sys.executable, str(ROOT / "tools/compare_native.py"),
            "--asm", str(args.asm.resolve()), "--llama", str(args.llama.resolve()),
            "--model", str(args.model.resolve()), "--out", str(output),
            "--candidate-name", "selected-h42", "--reference-name", "pinned-cpp-3t",
            "--threads", "4", "--candidate-decode-threads", "3", "--reference-decode-threads", "3",
            "--candidate-q8", "--candidate-spec-mode", "0",
            "--candidate-coarse-head", str(coarse), "--candidate-residual-norms", str(residual),
            "--candidate-certified-head-io", "pread-v2",
            "--tokens", "128", "--warmup", "8", "--repeat", "3", "--smaps-interval", "5",
            "--pmu", "--pmu-profile", "memory",
            "--require-identical-output", "--token-ids", "--require-latency",
            "--expected-model-sha256", canonical["model_sha256"],
            "--expected-token-ids-sha256", expected["token_ids_sha256"],
            "--expected-decoded-sha256", expected["decoded_sha256"],
            "--prompt", expected["prompt"],
        ]
        if args.dry_run:
            print(json.dumps({"case": case, "argv": command}))
        else:
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            subprocess.run([sys.executable, str(ROOT / "tools/report_native.py"), str(output)],
                           cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
