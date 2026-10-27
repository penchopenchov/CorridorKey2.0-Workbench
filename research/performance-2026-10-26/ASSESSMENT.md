# CorridorKey CLI performance assessment

Research date: October 26, 2026 (America/Vancouver). Application baseline: `22804345c6574cd63d2d00c28eb33b7e1c035130`.

Target: shortest time to a completed export of 3840×2160, 24 fps footage. Treat the existing matte, foreground, temporal behavior, color interpretation, and HDR support as the quality baseline. This is a research assessment, not a claim that an optimized implementation has shipped.

## Recommendation

Keep Python as the orchestration layer. First remove unnecessary image-file passes, CPU preprocessing, duplicate feature extraction, and GPU-to-CPU-to-GPU intermediate transfers. Pursue faster attention and compiled execution as a separate measured workstream. Preserve a staged low-memory fallback.

Do not start with a C++ rewrite, a FastAPI server, forced FP16, arbitrary larger temporal windows, or indiscriminately removing memory-management calls. The model's GPU computation is itself expensive. Pipeline cleanup is well justified, but this investigation does not establish a route to quality-equivalent 4K/24 fps on one RTX 3090.

Before any production edits, a backup branch was created at the baseline: `codex/backup-before-performance-2026-10-26`. The research lives on `codex/performance-research-2026-10-26`. The configured fork is `origin`, `https://github.com/penchopenchov/CorridorKey2.0-Portable.git`; `upstream` points to the original project. No runtime dependencies were installed or upgraded for this study.

## Measurements on this installation

Hardware/runtime observed: RTX 3090 with 24 GiB, Windows WDDM, Python 3.11.15, PyTorch 2.9.1+cu128. BF16 is supported. Triton is absent. The installed PyTorch build reports that FlashAttention is not compiled in; forcing that backend fails. This is a statement about this installation, not every Windows build or future version.

### Isolated components

The real checkpoint was loaded. Inputs and batons for isolated Stage 2 tests are random tensors, not actual model features. Each configuration has one initial call and three subsequent samples. Timings synchronize CUDA; GPU events corroborate the initial Stage 2 wall times. These are diagnostic samples, not a statistically robust production benchmark or a matte-quality evaluation.

| Experiment | Observed result | Interpretation |
|---|---:|---|
| 4K CPU preprocessing, already-decoded 8-bit frame | 800 ms/frame warm median | Excludes video decoding and disk reads. CPU color/tensor preparation is expensive. |
| 1080p CPU preprocessing | 208 ms/frame warm median | Same preprocessing function and limitations. |
| 4K Stage 2, BF16 autocast, batch 1 | 1,058 ms/frame warm median | Excludes Stage 1, foundations, preprocessing, rendering, and encoding. |
| 4K Stage 2, batch 2 | 1,051 ms/frame warm median | Under 1% throughput improvement in this run; peak allocated memory rises from 3.25 to 5.38 GiB. |
| 1080p Stage 2, batch 1 | 233 ms/frame warm median | A smaller input alone does not demonstrate real-time throughput. |
| Separate same-process BF16 versus FP16 autocast comparison | 1,010 versus 1,003 ms/frame | Under 1% difference; not evidence of a useful FP16 speedup. |
| Stage 2 with a BF16 rather than FP32 baton | 972 versus 1,010 ms/frame | About 4% improvement, with changed logits; numerical/quality validation required. |
| Separate full Stage 2 attention-backend comparison | cuDNN 902 versus memory-efficient 1,016 ms/frame | About 11% lower latency (13% more throughput); first cuDNN call took 7.22 seconds and logits changed. |
| Full model load | 54.3 seconds in first probe; 16.7 seconds in later export | Startup varies substantially with warm caches and other effects. These two observations are not universal loading costs. |

Parameter counts are approximately 569 million for the foundations, 464 million for Stage 1, and 202 million for Stage 2. Parameters are stored in FP32; inference uses BF16 autocast. Thus it is incorrect to describe this runtime as default FP32 inference throughout, and equally incorrect to say all persistent weights and activations are BF16.

At 24 fps the processing budget is 41.67 ms per source frame. The initial Stage 2 measurement alone is approximately 25 times that budget. This is a baseline gap, not proof that every optimized backend must have the same limit.

### Stage 2 GPU profile

A separate warmed 4K Stage 2 forward was captured with `torch.profiler`. Its self GPU time totaled approximately 1.033 seconds. Selected operator costs:

| Operator category | Self GPU time | Share |
|---|---:|---:|
| Memory-efficient SDPA attention | 387 ms | 37.5% |
| `addmm` matrix operations | 190 ms | 18.4% |
| Native depthwise convolution | 103 ms | 9.9% |
| Tensor copies | 86 ms | 8.4% |
| Transposed convolution | 84 ms | 8.1% |
| Other cuDNN convolution | 33 ms | 3.2% |
| Fused RMSNorm | 33 ms | 3.2% |

See [stage2-profile.txt](stage2-profile.txt). Profiler tables also show CUDA kernels associated with these operators: those kernel rows duplicate attribution and must not be added to the operator rows. CPU time inside blocking operations includes waits; it must not be described as pure Python execution overhead.

The model already uses PyTorch SDPA. The measured attention path is memory-efficient attention, not the unfused math fallback. A faster attention implementation is worth testing, but even making this attention cost zero would yield only about 1.6× Stage 2 speedup for this profile. [PyTorch documents backend selection, constraints, and numerical differences.](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)

### Short export through the original preparation and inference functions

Generated a 12-frame, 3840×2160, 24 fps H.264 test-pattern clip. Ran the launcher's existing sRGB/automatic-range EXR preparation, then its underlying inference runtime with Hann 8/4, carry enabled, no supplied hint, normal memory mode, and `alpha fg` video outputs. Encoding used the existing libx264 medium / CRF 12 / 10-bit 4:4:4 settings and two threads per writer.

| Measurement | Seconds |
|---|---:|
| EXR preparation, including linearization rewrite | 25.32 |
| Model loading | 16.66 |
| Inference and final export | 62.18 |
| Total, excluding initial Python imports | 104.16 |

Within inference: RVM prepass took 11.48 seconds. The Hann inference/rendering loop took 41.41 seconds, including two Stage 1 forwards totaling 7.05 seconds, twelve Stage 2 forwards totaling 12.32 seconds, and GPU rendering plus output download totaling 1.64 seconds. These are nested timings: do not add parent and child measurements together. Other loop work includes source preparation, foundation inference, blending/transfers, and writer backpressure. The remaining inference time includes phase setup and final output draining; it is not a measurement of encoding alone.

Peak allocated VRAM was 12.41 GiB. Both resulting MP4s were independently decoded and verified to contain 12 frames at 3840×2160 and 24 fps. The prepared EXRs occupied about 80 MB on this highly compressible test pattern. That size is not representative of photographic footage.

The benchmark synchronizes selected boundaries and writes timing records, adding diagnostic overhead. The input is only half a second long, there are only two Hann windows, and startup is a large fraction of the result. Do not extrapolate 104 seconds linearly to a long shot or call it a production FPS benchmark. Also, the guided launcher defaults to four outputs; this run explicitly selected only two, so it understates the cost of that four-output workload.

Raw data: [measurements.json](measurements.json), [stage2-experiments.json](stage2-experiments.json), [export-baseline.json](export-baseline.json), [output-validation.json](output-validation.json), [backend-stage2.json](backend-stage2.json), [attention-probe.json](attention-probe.json).

### A tested alternative: the installed cuDNN attention backend

A standalone attention probe with BF16 tensors shaped `(135,16,256,80)` succeeded with both memory-efficient and cuDNN attention, with and without a boolean key mask. FlashAttention failed because it is absent from this build. The full Stage 2 comparison then forced each supported backend separately on identical synthetic RGB/baton inputs: cuDNN lowered the warm median from 1,015.6 to 901.9 ms/frame. This is a useful candidate available in the current environment, before changing frameworks or installing compiler dependencies.

The first cuDNN Stage 2 call took 7.22 seconds versus 1.39 seconds for memory-efficient attention, so warm speed alone is insufficient for short exports. A rough amortization of the extra first-call cost over the measured warm saving is about 51 frames, assuming those costs recur and the same shapes are reused; test full workloads rather than treating this as a guaranteed break-even. Outputs were finite but the maximum absolute raw-logit difference was 0.03125 on random inputs. This does not establish acceptable matte/foreground quality. Stage 1 shapes, grouped masks, tail windows, and real features still require separate backend validation and fallback handling. [The supported backend-selection API is documented by PyTorch.](https://docs.pytorch.org/docs/2.9/generated/torch.nn.attention.sdpa_kernel.html)

## What the code actually does

### The launcher and direct CLI have different input paths

[The Windows launcher](../../corridorkey_launcher.py#L288) decodes the entire clip to ZIP16-compressed float EXRs. For non-linear input it then reads every EXR, applies the selected transfer function on CPU, and rewrites every EXR. Inference starts afterward and reads those frames again. A hint video can introduce another preparation sequence.

[Direct `infer.py` video input](../../corridorkey/video_io.py#L341) uses OpenCV sequential readers rather than making this EXR sequence. However, it creates 8-bit BGR frames and its preprocessing assumes sRGB. Simply routing all launcher inputs through it would change bit-depth and color behavior for some media. Replace the launcher cache with a streaming decoder that honors the existing color/range/precision contract; do not silently trade quality for speed.

### Whole-clip passes and overlap still cost work

[RVM runs a chronological prepass](../../corridorkey/runtime.py#L305) and stores features in an FP16 memmap even in normal memory mode. This is Google's recurrent representation model in this repository, not the similarly named Robust Video Matting package. Its recurrent state must remain chronological.

Normal mode then runs MoGe/C-RADIO inside each overlapping Hann window, re-extracting features for overlapping frames. Low-memory mode instead precomputes those features to disk and later performs Stage 2 in a separate phase. Normal mode already invokes Stage 2 as batons become final; it does not wait for a full-clip baton file.

With long clips and half-overlapping windows, most frames contribute to two Stage 1 windows. Cache per-frame frozen features once and reuse them, but preserve the trained temporal inference and Hann blend. Removing overlap or reducing context changes the prediction process and needs quality comparison.

### Large intermediates leave the GPU unnecessarily in normal mode

[Stage 1 blending](../../corridorkey/runtime.py#L539) converts batons to FP32, weights them, downloads them, accumulates them in NumPy, then converts finalized frames to FP16. [Stage 2 uploads](../../corridorkey/runtime.py#L883) those batons again and casts them to FP32. Carried alpha hints also make a GPU-to-CPU-to-GPU trip. Original model input frames are uploaded again for Stage 2.

At padded 4K (2176×3840), a 512-channel /16 baton is 63.75 MiB in FP32 or 31.875 MiB in FP16, per frame. At steady half-overlap, there are roughly two FP32 baton downloads per source frame and one FP16 upload, plus input and output transfers. These volumes justify investigation, but neither source inspection nor low GPU utilization alone proves that the PCIe link is saturated.

Keep a bounded GPU ring of pending inputs, carry hints, frozen features, and FP32 blend accumulators. Retire frames once no later window can contribute. Preserve the existing FP16 baton quantization before Stage 2 when validating equivalence; replacing it with BF16 is a separate change. Re-measure peak memory and retain a CPU/disk fallback.

### Output is partly asynchronous, with synchronous downloads remaining

[Both output writer classes](../../corridorkey/video_io.py#L607) already use bounded background queues. Rendering/composition largely runs on GPU, but [the video conversion helper](../../corridorkey/runtime.py#L623) downloads FP32 BGR and only then multiplies and casts it to uint16 on CPU. Alpha is expanded to three channels before download for video output.

For native 4K, each FP32 RGB output download is about 94.9 MiB/frame. Quantizing on GPU before download would halve that transfer size. A monochrome alpha path could reduce it further. Preserve clipping, truncation/rounding behavior, color metadata, bit depth, and frame ordering. Use pinned output buffers with completion events before CPU encoders access them. The present FFmpeg stdin worker also makes a `tobytes()` copy that is worth measuring after larger costs are reduced.

## Assessment of the supplied blueprint

| Proposal | Verdict for this CLI | Better application here |
|---|---|---|
| Keep model weights loaded | Implement for multi-clip runs | One Python process per batch with one loaded model. The launcher currently launches a fresh inference child for every job. A server is unnecessary. |
| Never call `del` or `empty_cache()` | Reject the blanket rule | Local tensor cleanup is normal. `empty_cache()` here occurs at phase/clip boundaries, not every frame. It does not unload live weights. Preserve deliberate low-memory staging and measure whether boundary cache releases matter. |
| Static tensors avoid repeated OS GPU allocations | Partly correct explanation; selective implementation | PyTorch already caches allocations. Preallocate bounded transfer/ring buffers and graph inputs, not every model intermediate. `inplace=True` is not a general inference optimization switch. |
| Async CPU decode, pinned memory | Implement CPU fallback | The prepasses already have one-worker lookahead; the main Hann readers are synchronous. Use reusable pinned buffers, a bounded producer queue, nonblocking copies, and explicit stream/event ownership. |
| Shared memory means zero-copy GPU input | Incorrect | Shared CPU memory avoids some process copies, not PCIe transfer. Prefer one process with threads for this CLI unless profiling justifies multiprocessing. |
| Native host VRAM/C++ plugin wrapper | Out of scope | There is no editor host framebuffer in a standalone CLI. NVIDIA's Python codec bindings can expose GPU decoder surfaces directly. |
| Force FP16 | Do not make default | BF16 autocast is already active. The measured FP16 change is negligible and changes logits. Keep BF16 unless a specific backend's measured gain passes quality/HDR validation. |
| `torch.compile(max-autotune)` | Prototype, do not blindly enable | Triton is absent from this installation. Start with isolated fixed-shape Stage 2 or repeated blocks, compare compile modes, account for cold-start cost, and detect unsupported environments. |
| TensorRT | Worth a contained backend experiment | Start with fixed-shape Stage 2. Preserve PyTorch fallback, compare complete export time, and account for engine build/cache/version/GPU compatibility. Exportability and speedups are unproven. |
| ONNX Runtime CUDA | Conditional alternative | CUDA provider alone does not guarantee faster execution. Use GPU I/O binding, inspect execution-provider assignment and copies, and verify operator/shape/precision support. |
| CUDA streams | Use for demonstrated overlap | Overlap transfer with inference and CPU encoding. Stage 1 and Stage 2 share the same GPU; simultaneous heavy compute can contend and increase memory use rather than improve throughput. |
| Temporal batches of 3–5 frames | Already temporal; benchmark rather than assume | Stage 1 already uses multi-frame Hann windows; Stage 2's batch knob already exists. Batch 2 brought under 1% improvement here. Do not arbitrarily replace temporal context or reset RVM state. |
| Profile with Nsight / PyTorch | Strongly agree; partially completed | PyTorch GPU profiling and component timing are recorded here. Use Nsight Systems next for whole-pipeline transfer/decoder/encoder timelines. |
| Persistent FastAPI backend / IPC | Unnecessary for current goal | Keep a CLI batch runner in one process. Add a service only if a future product needs independent clients or cross-run persistence. |

PyTorch's caching allocator reuses released blocks; `empty_cache()` releases unused cached blocks rather than live tensors. [PyTorch 2.9 CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html). Pinned memory is useful but pinning ordinary tensors in the main loop can itself block; correct overlap requires buffer and stream synchronization. [PyTorch transfer tutorial](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

FP16 has a smaller exponent range and can overflow. The HDR/asinh and normalization paths must be checked rather than globally cast. [NVIDIA accuracy guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html). The compile modes have different memory, warm-up, and graph constraints. [PyTorch 2.9 compile API](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html).

## Recommended implementation order

1. **Record reproducible export baselines and fix input preparation.** Add machine-readable stage timing and configuration logs; replace whole-clip EXR preparation/rewrite with streaming decode and GPU color/asinh/pad/resize preprocessing. Provide an FFmpeg CPU decoder with reusable pinned staging first; optionally use PyNvVideoCodec for supported codecs. Preserve an explicit cache option for workflows that benefit from reuse. This is a high-confidence target because both redundant passes and actual preparation costs are demonstrated.
2. **Bounded resident inference pipeline.** Compute chronological RVM features once for newly arriving frames, cache overlapping MoGe/C-RADIO features, blend and retain batons/hints on GPU, and reuse model inputs for Stage 2. Preserve numerical boundaries, ordering, window starts, tail handling, and low-memory fallback. This reduces redundant work without shrinking the model or temporal context.
3. **Reduce output conversion and encoder costs.** Quantize video outputs on GPU, add a monochrome alpha path, and stage downloads asynchronously. Benchmark encoder threads/presets and output selections. Only generate requested outputs. Evaluate an optional NVENC backend with its own codec-appropriate options and capability checks.
4. **Reuse models across a batch.** Remove per-clip interpreter/model initialization. Reset all clip-specific recurrent states, caches, metadata, and memory-mode settings between clips. Account for `configure_low_vram_inference()` mutating modules, which currently persists within a model instance. This particularly helps folders of short shots.
5. **Attack the GPU profile.** Start by validating the measured cuDNN attention improvement on actual model inputs and masks; investigate depthwise-convolution layout/kernel choices, transposed convolution, repeated casts/copies, and stable masks/RoPE metadata. Then benchmark compiled Stage 2 / repeated blocks and a TensorRT prototype. Compare a separate compatible Linux/WSL or updated Windows environment if it exposes useful attention/compiler backends, without replacing the working portable runtime in place. Keep each experiment independent so its benefit and numerical effects can be attributed.
6. **Only then consider quality-changing speed modes.** Reduced temporal overlap/context, smaller model inputs with a redesigned upsampling/refinement path, quantization, or student-model distillation need separate quality gates. The current renderer explicitly rejects non-native `model_scale`; reducing that flag is not presently a working fast-export mode.

For throughput, the existing automatic VRAM selector is not a speed tuner: it prefers the largest context that fits, including a larger low-memory window over a smaller normal one. On long 4K input it selects normal 8/4 for modeled budgets of 20–23 GiB, but low-memory 24/12 at 24 GiB. Add a distinct throughput policy based on measured candidates and a declared minimum temporal context; do not conflate maximum context, minimum VRAM, and fastest export. Recalibrate the VRAM estimator after adding persistent rings.

### Decoder and encoder constraints

PyNvVideoCodec supports Windows and Ampere and provides device-memory frames plus DLPack interoperability. That offers a Python-accessible route to GPU decoding without a bespoke C++ wrapper. It still needs demuxing, synchronization, format conversion, buffer lifetime management, and codec support checks; zero-copy interchange does not eliminate all GPU work. [NVIDIA decoder reference](https://docs.nvidia.com/video-technologies/pynvvideocodec/pynvc-api-reference/decoder.html), [system requirements](https://docs.nvidia.com/video-technologies/pynvvideocodec/read-me/system-requirements-common.html).

NVENC is not a drop-in substitute for every existing output. The RTX 3090 cannot encode the present 10-bit H.264 path in hardware; high-bit-depth H.264 requires Blackwell in the cited SDK. HEVC 10-bit is a candidate subject to per-device format capabilities and output requirements. ProRes 4444 with alpha and float EXR output need their own paths. Merely replacing `libx264` with `h264_nvenc` also leaves incompatible pixel-format/rate-control/preset assumptions in the writer. [NVENC programming guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-video-encoder-api-prog-guide/index.html).

ONNX Runtime must bind device inputs/outputs to avoid undoing gains with transfers. [I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html), [CUDA execution provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). TensorRT engine portability is constrained by platform, runtime versions, and GPU architecture or compatibility mode; build/cache keys need to include the relevant model, shape, precision, runtime, and device information. [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).

## Acceptance criteria and remaining evidence

Use actual representative 4K/24 fps clips for performance and quality acceptance: hair/fine detail, motion blur, transparent material, occlusion, difficult green spill, and HDR/negative linear values where applicable. Include no hint, first-frame hint, full hint, carry/no-carry, odd dimensions, and short/final windows.

Measure cold full-command time, warm batch throughput, time until the last encoded output is finalized, steady-state ms/source-frame, peak allocated/reserved VRAM, peak RAM, temporary bytes, queue stalls, and transfer/compute overlap. Record hardware, versions, source codec/bit depth/color/range, window settings, output formats, and encoder settings. Include long clips and batches: the present half-second synthetic run cannot replace them.

Compare raw alpha/foreground against the current baseline before video compression: error distributions, edge detail, temporal flicker/identity stability, and NaN/Inf counts. Require bitwise matching where the same arithmetic is preserved; define tolerances explicitly where backend fusion or reduction ordering changes results. Validate timestamps/frame counts and color transforms, not only a still image. Real ground-truth mattes are needed for absolute quality claims; a baseline comparison only measures regressions relative to existing behavior.

For backend experiments, include build/compile time and calculate break-even clip length: setup seconds divided by steady per-frame seconds saved. Reject an optimization that increases total export time for the workload it is intended to improve. Use profiler runs for attribution and clean uninstrumented runs for final speed claims. [Nsight Systems supports CUDA/NVTX/PyTorch timeline analysis.](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

There is no commercial implementation benchmark, internal architecture, or matched-output comparison in this study. Matching commercial speed cannot be promised from the supplied blueprint. The strongest supported conclusion is that this fork has removable pipeline costs and meaningful GPU-kernel costs; both must be addressed, and exact-quality 4K/24 fps remains unproven.

## Reproduction

Run from the repository root in the bundled environment. Run GPU benchmarks sequentially, with no other GPU workload. The export benchmark expects the generated source clip described below.

```powershell
.\runtime\python.exe research/performance-2026-10-26/measure.py
.\runtime\python.exe research/performance-2026-10-26/profile_stage2.py
.\runtime\python.exe research/performance-2026-10-26/attention_probe.py
.\runtime\python.exe research/performance-2026-10-26/backend_stage2.py
.\ffmpeg.exe -hide_banner -loglevel error -y -f lavfi -i "testsrc2=size=3840x2160:rate=24" -frames:v 12 -c:v libx264 -preset ultrafast -crf 18 research/performance-2026-10-26/synthetic4k.mp4
.\runtime\python.exe research/performance-2026-10-26/export_baseline.py
```

Generated videos, EXRs, caches, and console logs are excluded from Git. Scripts, measured summaries, and the profiler table are retained. No application inference code was changed for this research.
