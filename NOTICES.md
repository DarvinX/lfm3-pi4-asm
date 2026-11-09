# Licensing and third-party notices

The original engine code is available under the [MIT License](LICENSE). The upstream notice below applies to the adapted ggml routines.

The Arm NEON exponential and SiLU operation sequence in `src/native-math.inc` is adapted from ggml at the pinned llama.cpp revision. The [original MIT notice](references/ggml-LICENSE) accompanies the adaptation. The benchmark obtains llama.cpp independently and links it through its public C API.

The model and model-derived data come from Liquid AI's [LFM3-1.7B GGUF repository](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF), which lists the LFM Open License v1.0. Readers obtain the checkpoint and generate the index, tokenizer tables, and vocabulary auxiliaries on their own machine. The release contains source and measured evidence.
