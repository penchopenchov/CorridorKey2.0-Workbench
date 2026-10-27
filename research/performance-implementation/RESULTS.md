# CLI optimization results

Implementation date: October 26, 2026. Original application baseline: `2280434` (research snapshot `6f987fa`). Working branch: `codex/cli-performance`.

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

The capture command expects the original prepared EXRs generated by `research/performance-2026-10-26/export_baseline.py`; the source test pattern is generated by the command in the research assessment. Both capture and benchmark load the original runtime/video-I/O code from Git automatically, so a modified pipeline is not captured as its own reference. Saved raw goldens, videos, EXRs and caches are excluded from Git; scripts, metrics and this report are versioned.
