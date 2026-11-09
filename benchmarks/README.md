# Benchmark evidence

- `historical-timings.csv` preserves all 12 timed rows from the two physical Pi 4 workloads, with a selected set of columns. The measured assembly engine is `selected-h42`; the measured baseline is `pinned-cpp-3t`. Warmup rows were excluded from the headline.
- `canonical.json` binds the timed prompts to the original reference output hashes. Token hashes cover little-endian 32-bit token IDs, while decoded hashes cover the raw output bytes.
- `greedy-parity.json` preserves the prompts, counts, and matching output hashes from six historical correctness runs. Seeds do not affect greedy selection.
- `protocol.json` records the matched configuration and runner definitions. `source-provenance.json` records the frozen source commit, measured binary hashes, and hashes of the exported source files.

Calculate decode throughput from post-prefill evaluations divided by the decode interval, including sampling and output. The first generated token comes from prefill logits. Time to first token starts in main and ends when the first nonempty output piece has been written. Whole-process throughput includes startup and cleanup. The CSV reports process peak RSS in KiB from the engine's resource usage.

Run `python3 tools/plot_results.py` with Matplotlib installed to generate the benchmark chart, vocabulary diagram, and square social graphic. The charts read the archived CSV. The social graphic uses the rain explanation workload and an [AI-generated illustration of the board](pi4-board-illustration.png). `gpu-projection.json` records the separate resident GPU projection comparison discussed in the article.
