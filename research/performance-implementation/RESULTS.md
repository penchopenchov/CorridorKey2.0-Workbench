# CLI optimization results

Implementation date: October 26, 2026. Original application baseline: `2280434` (research snapshot `6f987fa`). Working branch: `codex/cli-performance`.

**October 27 update:** This report's exact-quality results describe the reference
attention configuration. The user subsequently requested a labeled cuDNN
speed/quality selection, default ON in the CLI, plus additional opt-ins. Those
settings can change predictions. For current defaults, tradeoffs, measurements
and compatibility checks, see [GPU_BACKENDS.md](GPU_BACKENDS.md). Turn cuDNN OFF
and leave advanced speed choices OFF to reproduce the earlier reference results.

## Result

On this RTX 3090 installation, the tested 12-frame 4K export fell from **85.82 seconds to 44.99 seconds**, excluding model loading: **47.6% less time, or 1.91× throughput** for this short workload. Adding the measured model-load cost gives approximately **102.10 seconds versus 61.28 seconds**. Those cold-equivalent figures add a common measured load cost; they are not independent cold-cache command measurements.

The final default pipeline matched all **401,080,320 raw BF16 prediction values** from the original 4K run, bit for bit. Both final encoded MP4 files also have identical SHA-256 hashes to the original outputs. Existing model weights, resolution, Hann settings, BF16 inference, FP16 baton rounding, transfer functions and encoder settings are retained by default.

This is a short synthetic benchmark, **not** a 24 fps result or a promise of the same percentage improvement on photographic footage, long clips, other codecs, four outputs, or other machines. Stage 2 still has substantial GPU computation. Real footage and long-clip validation remain useful before merging into a release branch.

## Benchmark conditions

- NVIDIA RTX 3090, 24 GiB, Windows WDDM, bundled Python 3.11 / PyTorch 2.9.1+cu128.
- 12-frame H.264 FFmpeg test pattern, 3840×2160, 24 fps; padded model shape 3840×2176.
- No supplied hint, carry enabled, Hann 8/4, Stage 2 batch 1, normal memory mode.
- Outputs: alpha and foreground. libx264 medium, CRF 12, yuv444p10le, two encoder threads. The guided launcher's four-output default was not used in this performance comparison.
- Baseline runtime/video-I/O code is loaded directly from Git `6f987fa` in a separate Python package namespace; it includes the original float EXR preparation and linearization rewrite. All model and encoding settings match.
- Model weights are loaded once and reused equally. Two original exports, two intermediate optimized exports, and two final prefetched exports were timed. Final speed runs have no raw-output capture, per-layer synchronization instrumentation, or comparison I/O.

| Path | Run 1 | Run 2 | Mean |
|---|---:|---:|---:|
| Original preparation + export | 86.24 s | 85.41 s | 85.82 s |
| Resident/async output, before preparation prefetch | 52.71 s | 52.38 s | 52.55 s |
| Final default with preparation prefetch | 44.74 s | 45.23 s | 44.99 s |

Original source preparation alone took 25.09–25.86 seconds. The final default eliminates that whole-clip stage. Peak allocated GPU memory increased from **12.41 GiB to 13.08 GiB**. The background CPU queue holds at most four prepared frames, in addition to the bounded Hann reader cache; long-clip RAM is not intended to grow with clip length. The auto-memory estimator reserves additional space for resident buffers, and staged low-memory inference remains available.

Raw timing records: [benchmark-results.json](benchmark-results.json), [final-benchmark.json](final-benchmark.json). Quality-capture timing includes substantial extra disk/CPU work and must not be used as the clean export benchmark.

## Changes kept as defaults

1. **Float streaming input in the guided launcher.** FFmpeg uses the same float RGB conversion, range handling and signed transfer equations as the former EXR path. Source video is no longer written, reread and rewritten as a whole-clip EXR sequence. Frame-sequence and full-hint preparation remain compatible. Direct `infer.py` keeps its legacy OpenCV decoder unless `--video_decoder ffmpeg` is selected explicitly.
2. **Bounded background CPU preparation.** A four-frame queue overlaps the exact reference color/asinh/padding arithmetic with GPU inference. Producer errors propagate, including errors after the final decoded frame; cancellation unblocks a full queue and terminates/waits for the decoder before closing its pipe.
3. **Resident temporal intermediates.** GPU batons, carry hints and pending input frames avoid CPU round trips in normal mode. Hann accumulation preserves operation order and FP16 quantization. GPU division uses an FP64 quotient rounded to FP32 before FP16 because ordinary CUDA FP32 division produced occasional differences from the NumPy reference.
4. **Chronological RVM without a whole-clip prepass in normal mode.** The original chunk boundaries, recurrent state, CPU normalization and FP16 rounding are preserved. Shared readers retain the current Hann overlap independently of RVM's lookahead, including irregular final starts. Low-memory mode retains the original staged prepasses/memmaps.
5. **Asynchronous output staging.** Video quantization runs on GPU with the same uint16 truncation. Output copies use pinned host buffers, a download stream, and completion events consumed by writer threads. Alpha travels as one channel and is expanded to identical RGB bytes on CPU. Encoders/pixel formats are unchanged. The pipe writer avoids a full `tobytes()` copy; failed writers cannot leave a producer blocked on a full queue, and encoder processes are reaped even if pipe close fails.
6. **Model reuse across jobs.** Guided and legacy interactive launchers call a reusable `infer.main(argv, model)` entrypoint. Job-specific state is recreated, config is copied, and low-memory chunk settings are reset when a normal job follows. Successful exports save `*_performance.json` with timing, effective temporal/memory settings, GPU and runtime details.

## Experiments kept opt-in to preserve quality

| Option | Evidence | Default decision |
|---|---|---|
| `--preprocess_device cuda` | Random signed/HDR 4K preprocessing: 55.84 ms versus roughly 810–853 ms for the observed 8/16-thread CPU settings. Maximum BF16 input difference 0.00390625. | CPU arithmetic remains default. |
| `--cache_foundations` | Changed batch shapes produced 153,900,153 changed raw 4K output values, max absolute difference 0.0234375. | Disabled; do not claim exact equivalence. |
| `--stage2_attention cudnn` | Earlier isolated Stage 2 benchmark: approximately 11% lower warm latency, with changed logits and a large first-call cost. | Reference backend selection remains default. |
| `--compile_stage2 default` | Triton is absent in this installation. Compiler initialization/execution failure paths are regression-tested, including preservation of explicit attention selection during fallback. | Disabled; unavailable compilation warns and uses eager execution. |
| FP16 instead of BF16; larger Stage 2 batches | Earlier measurements showed under 1% benefit here. | Precision and batch defaults unchanged. |
| CPU thread tuning | 1/2/4/8/16 threads were measured. Results were bit-exact, but there was no sufficiently compelling portable benefit to change the default. | Thread default unchanged. |

ONNX, ONNX Runtime, TensorRT and Torch-TensorRT are not installed. No untested engine exporter, mandatory dependency change, or alternate-precision engine was added. An engine backend remains a separately validated experiment, rather than an assumed speedup. The original [research assessment](../performance-2026-10-26/ASSESSMENT.md) contains primary-source documentation and the rationale.

### Additional VRAM and batching check

At the user's request, Stage 2 was benchmarked with batches 1, 2, 4 and 6 on the actual checkpoint at padded 4K, BF16 autocast. Each batch repeats the identical synthetic RGB/baton input so its predictions can be compared with batch 1. One initial call is excluded from the median of three subsequent synchronized calls. The other model weights were on CPU: these allocation figures are **isolated Stage 2 memory, not full-pipeline peaks**.

| Stage 2 batch | Warm median per frame | Peak allocated VRAM | Raw predictions versus batch 1 |
|---|---:|---:|---|
| 1 | 1,004 ms | 3.31 GiB | Reference |
| 2 | 1,013 ms | 5.67 GiB | Bit-exact in this test |
| 4 | 1,007 ms | 10.13 GiB | Changed; max absolute difference 0.03125 |
| 6 | 1,012 ms | 14.64 GiB | Changed; max absolute difference 0.03125 |

There was no measured throughput benefit from larger batches, despite substantially more memory use. Therefore the Stage 2 default remains 1. `infer.py --stage2_batch 4` already permits experimentation; it is not a demonstrated speed improvement and larger batches can change numerical rounding. The complete default pipeline's approximately 13 GiB allocation is not a hard VRAM limit. Stage 1 already processes temporal windows (8/4 in the 4K benchmark); increasing that window changes temporal context and can add computation, rather than serving as a quality-neutral speed switch.

GPU memory capacity and GPU execution throughput are different resources. NVIDIA likewise advises measuring batch sizes rather than assuming larger batches are always faster. [NVIDIA inference batching guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html#batching). Raw records: [batching-results.json](batching-results.json); reproduce with `runtime/python.exe research/performance-implementation/benchmark_batching.py`.

## Follow-up: small repeated costs

The next pass uses the previously shipped `f952e61` as its baseline, backed up
before edits at `codex/backup-before-micro-optimizations-2026-10-27`.

- Use the exact positive sRGB branch where inputs have already been clamped
  nonnegative. Keep the same threshold, operation order and signed-zero clamp;
  signed scene-linear EXR output retains the original conversion path.
- Compute the checker background in linear light on the GPU once per export,
  after Stage 2 has reached its rendering device. Do not allocate a checker
  at all for exports that do not request it.
- Share despill results between checker and cutout. Skip linear despill when
  the selected video cutout uses the separately computed sRGB result.

Complete-export comparison: 48 synthetic 3840×2160 frames at 24 fps, Hann 8/4,
all four video outputs, unchanged encoders/settings, shared loaded model,
before/after/after/before execution order. Loading and fixture generation are
excluded; decoding, inference, rendering, encoding and final encoder draining
are included.

| Version | Export times | Mean |
|---|---:|---:|
| Previous shipped pipeline | 187.67 / 189.71 s | 188.69 s |
| Additional small optimizations | 180.92 / 185.00 s | 182.96 s |

This is **3.04% less completed-export time** in this test, an additional gain
over the earlier pipeline improvements. All four encoded files have identical
SHA-256 hashes across all four runs. Peak allocated VRAM rises from about
13.08 to 13.18 GiB because the checker background remains resident.
Raw results: [micro-export-benchmark.json](micro-export-benchmark.json).

The final component benchmark alternates execution order on each iteration.
Preprocessing has eight samples per version; rendering has twelve and includes
GPU quantization, download and resolving the CPU frames, with encoding omitted.
All tested preprocessing tensors and rendered pixels match the frozen baseline.

| Isolated 4K component | Previous median | New median |
|---|---:|---:|
| CPU preprocessing | 864.34 ms/frame | 713.76 ms/frame |
| Alpha + foreground rendering/download | 105.13 ms/frame | 96.13 ms/frame |
| All four outputs' rendering/download | 139.65 ms/frame | 117.03 ms/frame |

Raw samples: [micro-benchmark.json](micro-benchmark.json). These correspond to
17.4% less preprocessing time and 16.2% less four-output rendering/download time.

These are two samples per version on a two-second looping test pattern,
not a guaranteed percentage for multi-minute camera footage. The per-frame
savings recur, but GPU inference and encoding still dominate the complete
export. Do not add isolated component percentages to the full-export gain.
Alpha/foreground-only rendering showed overlapping timings in the initial
sample; the final alternating sample's median improvement is not a separate
demonstration of alpha/foreground-only full-export acceleration. If checker and cutout are unnecessary,
`--outputs alpha fg` avoids two complete rendering/encoding jobs while
preserving the requested outputs' quality.

Reproduce the paired complete exports with
`runtime/python.exe research/performance-implementation/benchmark_micro_export.py`.
Run the component benchmark, renderer branch comparisons, original raw-logit
comparison, actual-model hint/HDR/low-memory matrix and unit tests sequentially
with `runtime/python.exe research/performance-implementation/verify_micro.py`.
`micro_reference.py` loads all affected baseline modules from Git rather than
silently comparing the optimized code against itself.

Final verification: 23 unit tests pass; all 120 CUDA renderer cases match
the pre-pass float/uint16 bits exactly; the original 4K raw-prediction golden
still has zero changed values out of 401,080,320. The actual-model hint/HDR
matrix passes, including a staged low-memory export followed by normal exports.
An independent code review found no actionable correctness issues. Records:
[micro-render-quality.json](micro-render-quality.json),
[micro-compare.json](micro-compare.json), [matrix-results.json](matrix-results.json).

## Quality and failure validation

- Float decoding matches the reference EXRs bit for bit for sRGB/Rec.709/linear and auto/full/limited input ranges.
- CPU preprocessing matches reference EXRs on signed/HDR values, transfer adjustment, and odd-size reflect padding. GPU preprocessing is explicitly tolerance-tested and not selected by default.
- Original-versus-final 4K raw predictions: **zero changed values** across 12 frames. [Final comparison](final-compare.json).
- Actual-model 13-frame, 101×73 signed/HDR sequence, padded to 128×128, Hann 8/4 with irregular starts 0/4/5: bit-exact resident-versus-reference predictions for no hint, first-frame hint, full hint, and full hint without carry. All four float EXR output types were generated. A staged low-memory job produced finite output before the normal jobs. [Matrix results](matrix-results.json).
- Actual guided launcher: two video jobs, low-memory then normal, **one model load and the same model instance**. [Batch results](launcher-batch-results.json).
- Output quantization matches NumPy reference values/channel ordering at clipping and quantization boundaries. Alpha's single-channel download resolves to identical RGB bytes. [Encoded comparison](encoded-equivalence.json).
- Tests cover overlapping/tail-reader retention, bounded queue cancellation, truncated decode, terminal decoder errors, encoder failure with a full queue, pipe-close failure with process reaping, config defaults, low-memory resets, and compiler construction/execution fallbacks.
- An independent reviewer examined the branch and reran focused error/retention regressions. All reported findings were fixed; the final review has no outstanding findings.

The small HDR matrix and synthetic 4K clip establish regression evidence, not absolute matte accuracy against ground-truth footage. No model retraining, temporal context reduction, lower-resolution inference or encoder-quality downgrade was used to achieve the default speedup.

## Reproduce

Run from the repository root, with GPU jobs sequentially:

```powershell
.\runtime\python.exe -m unittest discover -s tests -v
.\runtime\python.exe research/performance-implementation/validate_export.py capture
.\runtime\python.exe research/performance-implementation/final_validation.py
.\runtime\python.exe research/performance-implementation/validate_matrix.py
.\runtime\python.exe research/performance-implementation/validate_launcher_batch.py
.\runtime\python.exe research/performance-implementation/benchmark.py
.\runtime\python.exe research/performance-implementation/benchmark_preprocess.py
```

The capture command expects the original prepared EXRs generated by `research/performance-2026-10-26/export_baseline.py`; the source test pattern is generated by the command in the research assessment. Both capture and benchmark load the original runtime/video-I/O code from Git automatically, so a modified pipeline is not captured as its own reference. Saved raw goldens, videos, EXRs and caches are excluded from Git; scripts, metrics and this report are versioned. The implementation and these results are published on the fork's default `master` branch; `codex/cli-performance` is retained as the implementation snapshot.
