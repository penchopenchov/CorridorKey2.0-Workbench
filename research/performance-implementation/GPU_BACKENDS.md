# GPU inference speed/quality choices

Measured October 27, 2026 on the portable Windows installation: RTX 3090
24 GiB, PyTorch 2.9.1+cu128, native 3840×2160 footage padded to 2176×3840.
Backup before edits: `codex/backup-before-gpu-backends-2026-10-27` at `9984b58`.

The user explicitly requested cuDNN as a selectable **default ON** tradeoff,
with a warning that quality may degrade, and other tradeoffs as opt-ins.
These options change numerical predictions; they are not exact-quality upgrades.
No model weights, native resolution, temporal window, BF16 autocast, FP16 baton
boundary, transfer interpretation or encoder settings were reduced.

## What ships

The guided launcher asks **Faster cuDNN attention: ON/OFF**. ON is the default,
as is `infer.py --stage2_attention cudnn`. OFF / `--stage2_attention default`
uses the previous reference policy. The library `InferenceConfig` retains its
reference default so regression tools must explicitly select numerical tradeoffs.
The legacy interactive launcher also offers ON/OFF and native convolutions.

Additional guided advanced choices default OFF:

| Choice | Direct flag | Tradeoff / scope |
|---|---|---|
| GPU video preprocessing | `--preprocess_device cuda` | Different arithmetic/rounding, more VRAM; prepared image sequences retain CPU preprocessing |
| Native PyTorch Stage 2 convolutions | `--stage2_native_convolutions` | Different convolution kernels and predictions; hardware-dependent speed |
| Foundation-feature reuse | `--cache_foundations` | Different batch shapes/predictions; normal VRAM mode only; no demonstrated export gain in this short test |

Each selection warns **quality may degrade**. Feature reuse is explicitly
labeled experimental with no measured export gain in the short test.
Choices are included in launcher JSON, settings text and performance JSON.
All speed options OFF restores the previous reference behavior.

cuDNN attention is preferred with supported Flash/Efficient/Math kernels still
enabled as fallbacks. Backend priority and cuDNN convolution flags are scoped
to Stage 2 and restored on exit, including compiler fallback. Native convolution
selection does not globally disable cuDNN for the foundations or Stage 1.
No dependencies were installed or upgraded.

## Actual-pipeline inference and clean exports

The same 12-frame 4K/24 fps synthetic source, Hann 8/4, carry enabled and no
supplied hint was used for every configuration. Outputs are alpha + foreground
with the unchanged encoders. One model instance is reused. Numerical checks
and their synchronization/comparison work run separately from clean exports.
Stage 2 timing below is the warm median of the last eleven real-pipeline frames.

| Setting, other options OFF | Warm Stage 2 per frame | Clean warm export | Peak allocated VRAM |
|---|---:|---:|---:|
| Reference | 1,002.02 ms | 43.48 / 43.52 s | 13.08 GiB |
| cuDNN attention | 891.11 ms | 42.26 s | 13.08 GiB |
| GPU video preprocessing | 1,003.70 ms | 41.45 s | 13.82 GiB |
| Native Stage 2 convolutions | 897.93 ms | 42.07 s | 13.08 GiB |
| Foundation-feature reuse | 1,008.51 ms | 43.46 s | 13.19 GiB |
| All four speed choices | 794.54 ms | 38.30 s | 13.94 GiB |

Against the reference bookend mean of 43.50 s, clean completed-export time was
2.9% lower with cuDNN, 4.7% lower with GPU preprocessing, 3.3% lower with native
convolutions, and 12.0% lower with all options together. Feature reuse's 0.1%
difference is below timing variation; it is not a demonstrated improvement.
These percentages cannot be added and do not establish multi-minute production
throughput. There is one clean sample per option and two reference bookends.

These clean timings use warmed kernels and exclude model loading. First cuDNN
use took 7.85 s for one real frame versus roughly 1 s for reference: about 6.9 s
of extra first-use work. Therefore very short cold exports can be slower with
the default ON choice. At the measured warm saving of about 111 ms per frame,
roughly 62 frames would amortize that Stage 2 setup difference, assuming the same
shape/kernel behavior; this is an estimate, not a guaranteed whole-export break-even.

Raw records and reproduction:
[speed-choice-results.json](speed-choice-results.json),
`runtime/python.exe research/performance-implementation/benchmark_speed_choices.py`.

## How much do predictions change?

The comparison is against the original raw 4K prediction golden, not a new
reference generated from the faster backend. The reference configuration still
matches all 401,080,320 numerical values. Alpha is compared after its [0,1]
clamp; foreground is compared after scene-linear decode and sRGB conversion.
Differences are measured before video encoding, avoiding compression masking.

| Setting | Mean absolute alpha difference | Mean absolute sRGB channel difference |
|---|---:|---:|
| cuDNN attention | 0.000259 | 0.000286 |
| GPU preprocessing | 0.000271 | 0.000340 |
| Native convolutions | 0.000557 | 0.000337 |
| Foundation reuse | 0.000167 | 0.000193 |
| All options | 0.000564 | 0.000388 |

For cuDNN alone, the maximum clamped-alpha difference was 0.013671875, and
0.00156% of alpha pixels differed by more than 1/255. Average foreground change
was about 0.073 of an 8-bit channel level; the maximum was about 7.13 levels,
and 1.09% of color-channel samples differed by more than one level. These are
numerical differences, **not percentages of quality lost**. All options together
had 0.00908% of alpha pixels differ by more than 1/255. All outputs were finite.

The source is a synthetic test pattern, not representative camera footage with
hair, motion blur and difficult greenscreen edges. No such local clip was supplied.
These results cannot establish that the differences are invisible, beneficial
or harmless to temporal stability. The warning and reference OFF choice are
intentional. PyTorch also documents that different fused SDPA kernels can
produce different floating-point results. [PyTorch SDPA documentation](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Other backend experiments

Fresh processes isolate each checkpoint test so backend/kernel caches do not
contaminate the next candidate. Each uses the same seeded synthetic RGB/baton,
one initialization and seven timed Stage 2 calls, plus a changed-input check.
Other model weights are on CPU, so these are **isolated Stage 2 memory figures**.

| Candidate | Warm median | Outcome |
|---|---:|---|
| Eager/default | 1,002.86 ms | Reference |
| Forced memory-efficient SDPA | 1,023.03 ms | Exact, no speed benefit |
| Forced cuDNN SDPA | 948.12 ms | Changed predictions; faster, variable timing |
| Correct cuDNN convolution autotuning | 996.90 ms | Exact in this probe, below-noise gain; no option shipped |
| Native PyTorch convolutions | 892.72 ms | Changed predictions; shipped as opt-in |

The first autotuning harness inadvertently used `cudnn.flags(benchmark=True)`,
whose default `enabled=False` actually disabled cuDNN convolutions. This was
identified from the installed source and corrected: tuning now preserves other
flags, and native convolutions are measured and named explicitly. The aggregate
was rebuilt from corrected per-backend records after independent review.
Records: [gpu-backends.json](gpu-backends.json).

CUDA Graph capture initially failed at CPU-tuple-to-CUDA index creation inside
grouped attention. A research-only bounded index cache made capture work, with
exact outputs on nine changing inputs. Alternating timings, including input
copies and an independent output clone, were 1,002.02 ms reference, 999.81 ms
cached, and 998.06 ms graph replay. Below 1% with substantial overlapping
variation did not justify a production graph adapter, its capture setup, memory
or address-lifetime complexity. The cache and graph adapter were discarded;
the reproducible spike remains in
[benchmark_gpu_replay.py](benchmark_gpu_replay.py) and
[gpu-replay-results.json](gpu-replay-results.json).
PyTorch requires stable addresses/shapes and forbids synchronizing host work
during graph capture. [CUDA Graph guidance](https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-graphs).

FlashAttention is not compiled into this installed torch build. Triton,
ONNX/ONNX Runtime, TensorRT and Torch-TensorRT are absent; no unavailable engine
is presented as benchmarked or supported. Compilation remains an explicit
existing experiment with eager fallback, rather than a default dependency change.

## CLI compatibility and verification

- Real menu tests cover default ON/warning, OFF, advanced opt-ins and Back navigation.
- Guided command -> real argparse -> runtime tests cover all source transfers/ranges,
  first/full/no hints, trust/carry, outputs, video/EXR precision, cutout color,
  despill, temporal/memory choices, frame limits and sequence FPS.
- Seven actual guided jobs share exactly one loaded model: reference, default
  cuDNN, full hints with all options and linear cutout, low-VRAM video with all
  options, first-hint HDR float EXRs, full authoritative HDR half EXRs, then reference.
  All four output types and frame counts pass; EXR channels/dimensions are checked.
- Reference video outputs before and after the option/memory transitions are
  byte-for-byte identical, including alpha, foreground, cutout and checker.
- Unit tests exercise actual unsupported-head-dimension fallback and restoring
  reference predictions after cuDNN/native choices. Global flags are restored.
- An independent read-only review found one stale aggregate research record;
  it was corrected. No production correctness findings were reported.

Actual job records: [speed-launcher-validation.json](speed-launcher-validation.json).
Run GPU commands sequentially:

```powershell
.\runtime\python.exe -m unittest discover -s tests -v
.\runtime\python.exe research/performance-implementation/benchmark_gpu_backends.py
.\runtime\python.exe research/performance-implementation/benchmark_gpu_replay.py
.\runtime\python.exe research/performance-implementation/benchmark_speed_choices.py
.\runtime\python.exe research/performance-implementation/validate_speed_launcher.py
```
