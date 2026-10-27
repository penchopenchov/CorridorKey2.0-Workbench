# CLI Performance Implementation Plan

> Execute inline with superpowers:executing-plans and test-driven-development.

**Goal:** Reduce complete 4K export time while preserving model, temporal, color, HDR and encoding behavior.
**Architecture:** Float FFmpeg streaming replaces mandatory EXR preparation. A bounded reader/cache feeds the existing Hann pipeline; resident tensors avoid intermediate downloads. The CLI reuses one model per batch. Alternate numerical backends remain opt-in.
**Tech stack:** Existing Python, NumPy, OpenCV, PyTorch/CUDA and FFmpeg; no mandatory new dependencies.
**Spec:** `research/performance-2026-10-26/ASSESSMENT.md`, approved by the user with instruction to implement in its recommended order.

## Global constraints

- Preserve native resolution, window/stride, carried hints, RVM chronology, BF16 autocast and FP16 baton boundary.
- Preserve sRGB/Rec.709/linear selection, video range conversion, float EXRs and existing encoders/settings.
- Keep staged low-memory mode and a reference path for comparisons.
- Reuse the current clean checkout on a dedicated branch: assets/runtime are intentionally untracked and already installed here.
- No publishing, merging or pushing to upstream. Preserve Git backups and commit independent changes.

## Review focus

- Odd dimensions, final windows, short clips: no padding or frame-order regressions.
- Signed/HDR values and transfer knees: no unintended clamping or precision downgrade.
- Decoder/encoder failure and cancellation: no leaked processes or blocked queues.
- Sequential batch jobs with different memory modes: reset per-job mutations and recurrent state.
- Full hints and carried first-frame hints: retain temporal conditioning semantics.

## Tasks and verification

1. Streaming input: add `corridorkey/streaming.py` with float FFmpeg decoding and bounded windows; route launcher/CLI through explicit color/range options. First test streamed frames against the reference EXR decoder for all transfers/ranges and truncation errors, then implement and run unittest discovery. Keep reference preparation callable. Commit.
2. GPU preprocessing: add tensor preprocessing with preserved reflect padding and BF16 boundaries. First test signed/HDR/transfer-knee inputs against CPU results. Enable only where validated; keep reference CPU mode available. Benchmark preprocessing. Commit.
3. Resident inference: add bounded GPU baton/carry/input storage and reuse frozen features across overlapping windows. Preserve exact stage-2 FP16 quantization and chronological RVM state. First test accumulator/order/tail behavior and compare real-model raw predictions against the baseline. Keep memory fallback. Commit.
4. Output: quantize on GPU before host transfer, preserve uint16 truncation and channel order; fix queue error propagation and cleanup while keeping encoder settings. First test output arrays against the reference across clipping boundaries and injected writer errors. Benchmark and commit.
5. Batch reuse: expose reusable argument/config execution and run multiple launcher jobs in one inference worker, with cache leases and model reset. First test consecutive memory-mode settings and config isolation; run multiple actual jobs. Commit.
6. GPU backends and benchmarks: optional explicit attention/compile modes with eager default/fallback. First test unsupported options and baseline configuration; benchmark supported backends, document numerical differences, keep changes that preserve default quality. Run complete regression suite and whole-branch review, fix findings, commit measured report.

Each task follows failing behavior test → minimal implementation → passing suite → benchmark where relevant → commit. Benchmark golden-output capture runs before production edits; uninstrumented speed runs exclude golden capture I/O.

## Execution ledger

- Baseline: `6f987fa`; backup `codex/backup-before-implementation-2026-10-26`; implementation branch `codex/cli-performance`.
- Ruling: user explicitly authorized ordered implementation; proceed inline without another plan approval round.
- Pre-flight: decoder metadata feeds preprocessing; reader tensors feed Hann and prepasses; resident tensor/NumPy items share a stage-2 writer interface; CLI config is shared by launcher batch jobs.
