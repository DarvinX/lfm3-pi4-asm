CC = gcc
CPPFLAGS ?= -I.
ASFLAGS ?= -O3 -mcpu=cortex-a72
LDFLAGS ?= -static
LDLIBS = -lm -pthread

SOURCES = experiments/runtime-h42.S src/kernels.S src/q8-kernels.S \
          experiments/h19-q6-half-reuse.S experiments/h20-q4-pair-reuse.S \
          experiments/h24-q8-sums.S experiments/h32-dual-dot.S \
          experiments/h34-coarse-q4-linked.S experiments/h42-q4-lanes.S
INCLUDES = $(wildcard src/*.inc experiments/*.inc)
ASSETS = build/model.inc build/vocab.bin build/text.bin build/lookup.bin \
         build/merges.bin build/categories.bin build/bytes.bin
FLAGS = -DCONTEXT=512 -DMODEL_INDEX='"build/model.inc"'

.PHONY: all
all: build/lfm3-pi4 build/lfm3-pi4-trace

build/lfm3-pi4: $(SOURCES) $(INCLUDES) $(ASSETS) Makefile
	$(CC) $(CPPFLAGS) $(ASFLAGS) $(FLAGS) $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/lfm3-pi4-trace: $(SOURCES) $(INCLUDES) $(ASSETS) Makefile
	$(CC) $(CPPFLAGS) $(ASFLAGS) $(FLAGS) -Wa,--defsym,TOKEN_TRACE=1 $(SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

$(ASSETS):
	@echo 'Run python3 tools/prepare.py /path/to/LFM3-1.7B-Q4_K_M.gguf first.' >&2
	@false
