# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is SGLang

SGLang is a high-performance serving framework for large language models (LLMs) and multimodal models. The core runtime (SRT) is a multi-process system with three components communicating via ZMQ IPC: TokenizerManager (main process), Scheduler (subprocess), and DetokenizerManager (subprocess).

## Build & Install

```bash
# Install in development mode (Python >=3.10)
pip install -e "python[dev]"

# sgl-kernel (custom CUDA kernels, separate package)
cd sgl-kernel && pip install -e .

# Rust extensions are built automatically via setuptools-rust when SGLANG_BUILD_RUST_EXTS is set
```

## Running Tests

Tests use both pytest and unittest. The CI runner (`test/run_suite.py`) launches each test file via `python <filename> -f`.

```bash
# Single test file
python3 test/srt/<test_file>.py

# Single test function (unittest style)
python3 test/srt/<test_file>.py TestClass.test_method

# Using pytest directly
pytest test/srt/<test_file>.py -k test_name

# CI suite runner
python3 test/run_suite.py --hw cuda --suite base-a-test-1-gpu-small
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu

# sgl-kernel tests
cd sgl-kernel && pytest tests/

# Rust tests
cd rust && cargo test --workspace
```

## Linting

All linting runs through pre-commit. CI enforces this on every PR.

```bash
# Run all hooks on all files
SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure

# Run specific hooks
pre-commit run black --all-files
pre-commit run ruff --all-files
pre-commit run isort --all-files
pre-commit run codespell --all-files
pre-commit run clang-format --all-files

# Rust
cd rust && cargo clippy --workspace -- -D warnings && cargo fmt --check
```

Tools: black (formatting), ruff (F401/F821/UP037 only), isort (profile=black), codespell, clang-format (C++/CUDA), rustfmt + clippy (Rust).

## Architecture

### Request Flow

```
HTTP/API (FastAPI) → TokenizerManager → [DataParallelController →] Scheduler → TpModelWorker → ModelRunner
                                                                                                    ↓
HTTP stream ← TokenizerManager ← DetokenizerManager ← Scheduler ← forward pass output ←──────────┘
```

- **TokenizerManager** (`srt/managers/tokenizer_manager.py`): tokenizes requests, manages sessions/multimodal data, sends to scheduler via ZMQ
- **Scheduler** (`srt/managers/scheduler.py`): receives requests, manages waiting queue, builds `ScheduleBatch`, calls model worker, sends token IDs to detokenizer
- **DetokenizerManager** (`srt/managers/detokenizer_manager.py`): incremental detokenization, streams text back
- **DataParallelController** (`srt/managers/data_parallel_controller.py`): optional load-balancing across multiple schedulers

### Key Data Structures

- `ScheduleBatch` (CPU, scheduler-managed) → `ForwardBatch` (GPU tensors, model-runner-managed). See `srt/managers/schedule_batch.py` and `srt/model_executor/forward_batch_info.py`.
- `Req` (`srt/managers/schedule_batch.py`): individual request with token IDs, sampling params, radix cache state
- IPC messages: `srt/managers/io_struct.py` — all ZMQ message types between processes

### Entrypoints

- **CLI**: `python/sglang/cli/main.py` — `sglang serve`, `sglang generate`
- **HTTP Server**: `srt/entrypoints/http_server.py` — FastAPI with OpenAI/Anthropic/Ollama-compatible APIs
- **Engine (programmatic)**: `srt/entrypoints/engine.py` — `Engine` class, launches subprocesses internally
- **Server launch dispatcher**: `python/sglang/launch_server.py` — routes to HTTP/gRPC/Ray/encoder modes

### Model Execution

- **ModelRunner** (`srt/model_executor/model_runner.py`): manages model, attention backends, CUDA graphs, KV cache pools
- **TpModelWorker** (`srt/managers/tp_worker.py`): wraps ModelRunner, exposes forward passes to scheduler
- **CUDA graph runners**: `srt/model_executor/runner/` — capture and replay for decode/prefill

### Memory & Caching

- **RadixCache** (`srt/mem_cache/radix_cache.py`): radix tree-based prefix caching (RadixAttention)
- **Memory pools**: `srt/mem_cache/memory_pool.py`, `srt/mem_cache/unified_memory_pool.py`
- **Allocators**: `srt/mem_cache/allocator/` — multiple strategies (SWA, HiSparse, mmap)

### Models

170+ model architectures in `srt/models/` with a registry at `srt/models/registry.py`. Includes LLMs, vision-language models, audio models, encoder-only models, and reward/classification models.

### Other Key Subsystems

- **Speculative decoding**: `srt/speculative/` — EAGLE, n-gram, DFlash, MTP workers
- **Distributed**: `srt/distributed/` — tensor/pipeline/data/context parallelism
- **Disaggregation**: `srt/disaggregation/` — prefill-decode disaggregation (NIXL, Mooncake, Mori, Ascend)
- **LoRA**: `srt/lora/` — adapter management, loading, overlap, eviction
- **Constrained decoding**: `srt/constrained/` — grammar backends (xgrammar, outlines, llguidance)
- **Server config**: `srt/server_args.py` — all CLI flags (~400+ parameters)

### Frontend DSL

`python/sglang/lang/` — SGLang programming language with `gen()`, `select()`, `function()`, role markers. The interpreter (`lang/interpreter.py`) executes IR trees and communicates with backends. Backend adapters in `lang/backend/` connect to SRT, OpenAI, Anthropic, etc.

### Other Components

- **sgl-kernel** (`sgl-kernel/`): custom CUDA/C++ kernels, CMake-based build
- **Rust workspace** (`rust/`): gRPC server, multimodal processing, PyO3 Python bindings
- **sgl-model-gateway** (`sgl-model-gateway/`): Rust-based model gateway (axum, tonic, k8s)

## Code Conventions

See `.claude/rules/` for full rules. Key points:

- **Use `msgspec.Struct`**, not `@dataclass` — for all new data containers
- **Prefer stateless/immutable** — pure functions, frozen data, pass specific values not god objects
- **Functions under ~100 LOC**, files under ~2k LOC
- **Core functions read like pseudocode** — push detail into well-named helpers
- **No mixins** — use explicit composition or plain functions
- **Default to protected** (`_name`) — expose only what callers use
- **Keyword arguments** for 2+ arg functions
- **Environment variables**: define in `python/sglang/srt/environ.py` `Envs` class, access via `envs.SGLANG_FOO.get()`, never raw `os.environ`. See `.claude/skills/env-var-conventions/SKILL.md`.
- **Never mutate `ScheduleBatch` in place** — build new values and rebind. See `.claude/skills/schedule-batch-out-of-place-mutation.md`.
- **Test admission is strict** — only bug regression, derived property, or critical-path bookkeeping tests. See `.claude/rules/unit-test-admission.md`.
