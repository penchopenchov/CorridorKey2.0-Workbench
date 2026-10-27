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
- Task 1 complete: `543306d`; float decoder matches reference EXRs in all tested transfer/range combinations; 4K streamed predictions matched 401,080,320 values exactly.
- Task 2 complete: `4db05bd`; optional GPU preprocessing implemented and tested. Ruling: keep CPU default because GPU arithmetic changes rounding; honor the user's quality constraint.
- Tasks 3–6 complete: `a030f86`; resident inference, async output, model reuse and explicit optional backends. Ruling: commit their shared runtime interfaces together, keeping a coherent runnable snapshot.
- Ruling: frozen feature caching changes batch shapes and predictions; leave opt-in (`cache_foundations=False`) after measured comparison, rather than silently change quality.
- Ruling: preserve NumPy's FP32 division rounding via double quotient then FP32/FP16 on GPU; direct GPU FP32 division failed exactness tests.
- Final background preparation: `e875865`; bounded prefetch preserves final 4K raw predictions exactly and reduces clean export time further.
- Review findings fixed with red/green tests: shared-reader eviction on RVM lookahead; streamed brightness/contrast; compiler construction and fallback backend selection; writer close error/process reaping; terminal decoder error propagation.
- Validation: 22 unit tests pass; actual-model HDR/hint/irregular-tail matrix passes; real launcher low-to-normal batch reuses exactly one model; final clean 4K runs 44.74/45.23 s versus original 86.24/85.41 s (same two video outputs, load excluded).
- Independent review: no outstanding findings after final lifecycle fixes. No production dependencies upgraded, no merge or push performed.

## Follow-up: repeated per-frame costs (2026-10-27)

- User requests additional small optimizations that accumulate over long clips. Keep the same quality constraint and benchmark actual benefits.
- User's subsequent branch instruction supersedes the original implementation-branch choice: work directly on the fork's default `master` branch. Previous work is already published there.
- Backup before edits: `codex/backup-before-micro-optimizations-2026-10-27` at `f952e61`.
- Measure frozen `f952e61` against candidate exact positive-only sRGB arithmetic, reused despill results, and a per-export GPU checker background. Leave signed EXR conversion and encoder settings intact.
- Validate CPU/CUDA transfer knees and signed zero, all renderer branches including HDR/linear cutout, real-model raw predictions, and staged low-memory placement. Run paired full exports and compare encoded files. Keep measured improvements only.
- Request independent read-only review, resolve findings, rerun verification, document results, commit and push fork `master`.
- Result: exact nonnegative transfer, shared despill and cached checker implemented. Alternating component benchmark: CPU preprocessing 864.34 → 713.76 ms; all-four render/download 139.65 → 117.03 ms medians.
- Full export: before 187.67/189.71 s, after 180.92/185.00 s, 48 frames at 4K with all four outputs; mean time reduced 3.04%. All encoded files have identical SHA-256 hashes. No encoder/model/temporal quality changes.
- Verification complete: 23 unit tests; 120 exact CUDA renderer cases; zero changed original raw values out of 401,080,320; actual-model low-memory/HDR/hint matrix passes. Independent review reports no actionable findings. Results and reproduction scripts are ready for publication directly to fork `master`.
