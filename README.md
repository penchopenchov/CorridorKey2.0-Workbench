# CorridorKey2.0 Workbench

A Windows-focused CorridorKey fork for guided video keying, explicit color handling, artist-supplied mattes, HDR exports, and faster batch processing.

Built on [CorridorKey 2.0](https://github.com/nikopueringer/CorridorKey2.0), created by Zachary Harris with Niko and Corridor Digital. This fork improves the launchers, packaging, export pipeline, and GPU execution controls while retaining the upstream model weights.

**[Quick start](#quick-start)** · **[What this fork adds](#what-this-fork-adds)** · **[Speed and quality](#speed-and-quality)** · **[Benchmarks](#measured-performance)** · **[CLI examples](#command-line-examples)** · **[Documentation](#documentation)**

> **Current GPU default:** Faster cuDNN attention is **ON**. It can change matte edges and foreground color: **quality may degrade**. Choose **OFF** and leave advanced speed options OFF for the reference numerical behavior. First use at a resolution can add setup time.

## What this fork adds

| Area | Improvements |
|---|---|
| Guided workflow | Windows drag-and-drop launcher for videos, image sequences, and batches; Back navigation; remembered answers within setup; final Start / Back / Cancel review |
| Source color | Explicit sRGB, inverse BT.709, or linear Rec.709 input; automatic, limited, or full video levels; high-precision float video decoding |
| Matte guidance | No matte, first-frame mask, or full matte video/sequence; adjustable trust; authoritative full-matte mode; companion-file discovery with confirmation |
| Exports | Alpha, foreground RGB, straight RGBA cutout, checker preview; ProRes 4444 cutout; scene-linear half/full-float EXR sequences; independent video cutout color and despill |
| Throughput | Float streaming, bounded background preparation, GPU-resident temporal intermediates, chronological RVM processing, asynchronous output transfers, and model reuse across clips |
| GPU choices | Default-ON cuDNN attention; opt-in GPU video preprocessing and native Stage 2 convolutions; experimental foundation-feature reuse; reference attention remains selectable |
| Memory | Automatic per-clip VRAM selection, manual Hann windows, staged low-VRAM mode, and configurable temporary storage |
| Progress | Clip/stage progress, finalized-frame counts, elapsed time, stage ETA, loading/finalization heartbeats, and allocated/reserved PyTorch VRAM |
| Recovery | Owned job caches cleaned after success, failure, or cancellation; abandoned-cache recovery; active-job leases; encoder/decoder errors propagated without leaving full queues blocked |
| Portability | Pinned Windows setup, optional bundled Python runtime, app-local Microsoft runtime DLLs, relative paths, local model assets, and offline operation after installation |
| Reproducibility | Export settings and performance records, regression tests, measured backend comparisons, and documented numerical differences |

The work in this fork was developed with AI assistance. Model authorship and the original checkpoint remain credited to upstream.

## Quick start

### Already have the complete portable folder?

1. Keep the application source, `runtime/`, `weights/`, `vfms/`, `third_party/`, and `ffmpeg.exe` together.
2. Double-click **`CorridorKey_DRAG_CLIPS_HERE.bat`**, or drop a clip/folder onto it.
3. Choose source color and video levels, matte guidance, outputs, memory, and GPU speed/quality options.
4. Review the settings and select **Start**.

The BAT prefers `runtime/python.exe` and falls back to the local `venv`. A fully installed folder can be copied to another compatible Windows x64 NVIDIA machine without installing system Python, pip, or a separate CUDA toolkit. A compatible NVIDIA driver is still required.

**A GitHub source clone does not contain the portable runtime, FFmpeg, or downloaded model weights.** Those large assets are excluded from Git.

### Installing from source on Windows

You need Git, Python 3.11 for Windows x64, an NVIDIA GPU/driver compatible with the bundled CUDA PyTorch, and internet access for setup. The script prints installation guidance if Python 3.11 is missing.

```powershell
git clone https://github.com/penchopenchov/CorridorKey2.0-Workbench.git
cd CorridorKey2.0-Workbench
.\setup_portable.bat
```

The setup script finds Python 3.11, creates `venv`, installs **PyTorch 2.9.1 / torchvision 0.24.1 with CUDA 12.8**, installs dependencies, and downloads the model assets. It checks for FFmpeg and copies a discovered executable beside the app. If FFmpeg is missing, follow the script's installation instructions before processing video.

The optional portable build uses [`tools/make_portable.ps1`](tools/make_portable.ps1) and requires the matching app-local Microsoft C++ redistributable files. If its prerequisites are unavailable, the installed venv can still run the BAT. See [the portable guide](PORTABLE_README.txt) for building and copying a complete installation.

Tested here with an **RTX 3090, 24 GiB VRAM, Windows, and the pinned runtime**. Memory requirements depend on resolution, temporal context, outputs, and enabled options; the test machine's figures are not minimum requirements for every configuration.

### Prefer a terminal?

Run the same guided interface with the bundled interpreter:

```powershell
.\runtime\python.exe corridorkey_launcher.py "D:\Footage\shot.mov"
```

For a venv installation, replace `.\runtime\python.exe` with `.\venv\Scripts\python.exe`.

The Linux `.sh` launcher and `run_interactive.py` use the legacy interactive interface. The bundled portable runtime is Windows-specific; its relocation checks do not establish compatibility with another OS or every GPU.

## The guided workflow

### Source color and levels

| Selection | Use when the source RGB is encoded as |
|---|---|
| **sRGB** | Standard sRGB, including its piecewise transfer curve |
| **Rec.709** | Nonlinear RGB using the BT.709 OETF |
| **Linear Rec.709 / gamma 1.0** | Linear-light Rec.709 RGB, including ProRes exported that way |

**ProRes 4444 is a codec, not a transfer function.** Select linear only when the footage was actually exported with linear RGB values. Source interpretation and output cutout color are separate settings.

Video levels can use metadata automatically or be overridden to limited/full. BT.709 transfer decoding is not a generic gamma-2.4 / BT.1886 setting. These paths assume Rec.709/sRGB primaries and a BT.709 YUV matrix; convert ACEScg, camera log, Rec.2020, PQ/HLG, and other unsupported color systems upstream.

Normal guided video input streams float frames from FFmpeg. Image sequences are prepared with their selected transfer and passed as linear frames. There is no mandatory whole-video EXR preparation pass for ordinary source video.

Video FPS is probed and forwarded. Variable-frame-rate timestamp preservation is not implemented; use an appropriate constant-FPS source.

### Matte guidance

| Mode | Behavior |
|---|---|
| **No matte** | The model chooses the foreground; companion masks are ignored |
| **First-frame matte** | A mask identifies the subject initially; carried predictions can guide later overlapping windows |
| **Full matte** | An aligned matte video or frame sequence guides the shot |
| **Authoritative full matte** | Disables carried predictions so supplied masks are used on every pass |

White means subject, black means background, and gray means partial coverage. Matte data is not gamma-decoded. An accepted first-frame image can use its alpha channel. A full matte movie must visibly contain the grayscale mask; source-video transparency is not automatically treated as a supplied matte.

The guided trust default is **0.75** for rough guidance; **0.95** is available for a clean mask. Direct `infer.py` defaults explicit hint trust to 0.95. A rough hint guides subject selection and does not guarantee repaired edges.

Companions such as `shotFrameHint.png` and `shot_AlphaHint.mp4` are discovered with confirmation. Full matte movies must match FPS, frame alignment, and coverage of the selected source. See [the After Effects matte recipe](README_ADVANCED.txt) for exporting a separate roto movie.

### Controls and batches

- Type **`b`** or **`back`** to revisit a setup question. Enter retains the displayed answer; `-` restores the default for optional output/cache folders.
- Drop a video, frame-sequence folder, or folder of clips. Duplicate inputs and colliding batch destinations are checked.
- Settings apply to the batch; group sources with the same transfer and processing choices.
- One loaded model is reused across jobs. Recurrent state and job-specific buffers are recreated, and normal-memory settings are reset after low-VRAM jobs.
- Image sequences support EXR, PNG, JPG, and JPEG in filename order. Use zero-padded numbering and supply the sequence FPS.

## Outputs and compositing

The guided launcher selects all four outputs initially. Direct `infer.py` defaults to **alpha + foreground**.

| Output | Video mode | EXR mode |
|---|---|---|
| `alpha` | Grayscale coverage preview in MP4 | Single-channel coverage, clamped to [0, 1] |
| `fg` | Foreground RGB MP4 | Scene-linear RGB |
| `cutout` | ProRes 4444 MOV with straight RGB + alpha | Straight scene-linear RGBA |
| `checker` | Despilled checkerboard composite MP4 | Scene-linear checker composite |

Video cutout color can be **sRGB** or **linear Rec.709**, independently of the input transfer. Import the cutout as **Straight / Unmatted** alpha when your editor asks. Despill defaults to 0.5 and applies to cutout/checker; the foreground export is available separately.

Video outputs are clipped/quantized for encoding. For scene-linear HDR, use **EXR sequences**: signed color values and values above 1 are retained within the selected format's range. Half float is the default; full float is available. Half output is bounded to the finite float16 range. A float32 file does not increase the underlying model's inference precision.

Only export what you need: each selected output adds rendering and encoding work. Audio is not copied into these exports; retain the original audio in your editing timeline.

## Speed and quality

Pipeline improvements and alternate GPU arithmetic are different decisions. Streaming, bounded preparation, resident buffers, model reuse, and output staging were checked against the original reference predictions. Faster numerical backends can change those predictions.

| Setting | Current default | What to expect |
|---|---|---|
| cuDNN attention | **ON** | Faster warm Stage 2 on the tested GPU; matte/foreground quality may degrade; extra first-use setup |
| GPU video preprocessing | OFF | Faster preparation in the measured workflow, different rounding, extra VRAM; image sequences retain CPU preprocessing |
| Native Stage 2 convolutions | OFF | Faster kernels on this RTX 3090, with different predictions; hardware-dependent |
| Foundation-feature reuse | OFF | Different batch shapes/predictions; normal VRAM mode only; no demonstrated export gain in the short test |
| Stage 2 batch size | 1 | Larger batches use more VRAM; tested larger sizes did not improve throughput and some changed predictions |
| Compilation | OFF | Separate explicit experiment; requires a compatible compiler environment and falls back to eager if unavailable |

The guided advanced menu labels the tradeoffs and leaves them OFF. Backend priority and convolution flags are scoped to Stage 2 and restored afterward. Supported attention kernels remain enabled as fallbacks when cuDNN cannot handle a shape.

**To retain the previous reference results:** turn cuDNN **OFF** and leave all advanced speed choices **OFF**. The CLI equivalent is `--stage2_attention default` with the other experimental flags omitted. Choices are recorded with each export.

### Measured performance

RTX 3090 / Windows / PyTorch 2.9.1+cu128. These are synthetic 4K measurements, not a real-time guarantee.

| Separate comparison | Before → after | Result |
|---|---|---|
| Original whole-clip preparation/export → optimized reference pipeline; 12 frames, alpha + foreground, model load excluded | 85.82 → 44.99 s | **47.6% less time**, with identical tested raw predictions and encoded files |
| Further exact color/rendering cleanup; 48 frames, all four outputs | 188.69 → 182.96 s | **3.0% less time**, identical encoded files across repeated runs |
| Reference → all optional speed choices; 12 frames, alpha + foreground, kernels warm | 43.50 → 38.30 s | **12.0% less time**, with changed numerical predictions |

The workloads differ: **do not add or multiply these percentages**. The first two comparisons cover quality-preserving pipeline changes in their tested reference configurations. The last comparison enables explicit speed/quality tradeoffs.

In the real-pipeline GPU comparison, warm Stage 2 fell from **1,002 ms/frame** to **891 ms** with cuDNN, **898 ms** with native convolutions, and **795 ms** with all speed choices. Whole-export gains were smaller because decoding, temporal inference, foundations, rendering, and encoding still contribute.

First cuDNN use added about **6.9 seconds** of setup relative to reference in that test, so a very short cold export can be slower. No representative camera clip was supplied for perceptual evaluation; synthetic numerical differences do not prove that hair, edges, motion blur, or temporal stability are unaffected.

Read the [reference pipeline report](research/performance-implementation/RESULTS.md) and [GPU backend report](research/performance-implementation/GPU_BACKENDS.md) for raw records, numerical differences, memory figures, methods, rejected experiments, and reproduction commands.

## Memory, temporary storage, and recovery

Start with **Auto VRAM**. It selects temporal/memory settings per clip from resolution and available GPU memory. Manual mode exposes staged low-VRAM inference and Hann chunk/stride.

- Larger temporal windows provide more context and require more memory. A smaller window changes temporal context; it is not a quality-neutral speed switch.
- Low-VRAM mode stages foundation features and other intermediates through temporary files. It can process cases that do not fit the normal resident path, with more I/O.
- The normal video path uses bounded frame queues/windows rather than retaining the entire clip in RAM.
- CPU preparation-thread selection applies where temporary EXR encoding is needed, including full-matte preparation. It is not a global cap on inference or encoder threads.
- Temporary job caches are removed on success, ordinary failure, or cancellation. A forced close can leave owned caches for recovery at the next launch.
- Recovery skips active jobs protected by OS leases and does not sweep source files, exports, models, or arbitrary unregistered folders.
- Failed deletion reports its retained path for a later retry. Choose a cache drive with sufficient free space, especially for low-VRAM staging and prepared sequences/hints.

## Progress and saved settings

The terminal reports clip/stage, finalized frames, elapsed stage time, and stage ETA. Loading and finalization use elapsed heartbeats when a reliable percentage is unavailable. Rendering reaching 100% does not mean encoding has finished; wait for output finalization.

VRAM is shown as **allocated / reserved** PyTorch memory for this worker. Reserved includes the reusable allocator cache; do not add the two figures. Driver overhead, other applications, and allocations outside PyTorch are excluded.

Each job saves:

- `launcher_settings.json` for guided selections, source/matte, FPS, and the executed command.
- `<clip>_settings.txt` for inference/export settings.
- `<clip>_performance.json` for completed-export time, effective memory/window settings, enabled backend options, peak allocated VRAM, GPU, and PyTorch version.

## Command-line examples

Run from the installed repository folder. These PowerShell examples use the bundled interpreter and explicit float-video input. Paths are examples; select the transfer that matches your footage.

### Guided session

```powershell
.\runtime\python.exe corridorkey_launcher.py "D:\Footage\shot.mov"
```

### Linear Rec.709 video with a transparent cutout

```powershell
.\runtime\python.exe infer.py --input "D:\Footage\linear_shot.mov" --output_dir "D:\Renders\shot" --ffmpeg_bin ".\ffmpeg.exe" --video_decoder ffmpeg --input_color linear --input_range auto --outputs alpha fg cutout
```

This retains the CLI's default-ON cuDNN tradeoff. Add `--cutout_linear` if the **video output cutout** should also use linear RGB.

### First-frame guidance with reference attention

```powershell
.\runtime\python.exe infer.py --input "D:\Footage\shot.mp4" --output_dir "D:\Renders\guided" --ffmpeg_bin ".\ffmpeg.exe" --video_decoder ffmpeg --input_color srgb --hint_first_frame "D:\Mattes\shotFrameHint.png" --hint_quality 0.75 --stage2_attention default --outputs alpha fg cutout
```

### Authoritative full matte

```powershell
.\runtime\python.exe infer.py --input "D:\Footage\shot.mov" --output_dir "D:\Renders\full_hint" --ffmpeg_bin ".\ffmpeg.exe" --video_decoder ffmpeg --input_color rec709 --hint_video "D:\Mattes\shot_AlphaHint.mp4" --hint_quality 0.95 --no_carry_hint --outputs alpha fg cutout
```

### Linear EXR plates to HDR EXR output

```powershell
.\runtime\python.exe infer.py --input "D:\Plates\linear_frames" --output_dir "D:\Renders\hdr" --frame_dir_linear --frame_dir_fps 24 --output_format exr --exr_dtype float --outputs alpha fg cutout --stage2_attention default
```

EXR output is always scene-linear. `--cutout_linear` is unnecessary for EXR output.

### Short memory-constrained test

```powershell
.\runtime\python.exe infer.py --input "D:\Footage\shot.mp4" --output_dir "D:\Renders\test" --ffmpeg_bin ".\ffmpeg.exe" --video_decoder ffmpeg --input_color srgb --num_frames 48 --low_vram --hann_chunk 8 --hann_stride 4 --temp_dir "D:\CorridorKeyCache"
```

The 8/4 window here is an explicit lower-context example, not a universal best setting.

### Try the optional speed tradeoffs

```powershell
.\runtime\python.exe infer.py --input "D:\Footage\shot.mp4" --output_dir "D:\Renders\speed_test" --ffmpeg_bin ".\ffmpeg.exe" --video_decoder ffmpeg --input_color srgb --stage2_attention cudnn --preprocess_device cuda --stage2_native_convolutions --outputs alpha fg
```

Quality may degrade. Add `--cache_foundations` only if you want to test that additional experimental choice; it was effectively neutral in the short export benchmark.

**Direct-CLI distinction:** without `--video_decoder ffmpeg`, `infer.py` retains its legacy OpenCV video-input path. Do not expect `--input_color linear` alone to turn that legacy reader into the guided float-video path.

See all flags with `.\runtime\python.exe infer.py --help`. For the original staged-intermediate comparison, use `--reference_pipeline --stage2_attention default` with advanced speed options omitted.

## Validation and troubleshooting

The latest GPU-selection pass completed **32 tests** and **seven actual guided jobs**, covering Back navigation, source selections, hints, all outputs, half/full-float HDR EXRs, low-VRAM transitions, and one-model batch reuse. Reference exports were byte-identical before and after switching speed/memory options back OFF. Validation on this installation does not establish compatibility with every driver, GPU, or input.

```powershell
.\runtime\python.exe -m unittest discover -s tests -v
.\runtime\python.exe scripts\smoke_test_runtime.py --load_model
```

| Symptom | Check |
|---|---|
| CUDA out of memory | Close other GPU-heavy applications; try staged low-VRAM mode and a smaller temporal window; reconsider optional GPU preprocessing/batch size |
| Wrong brightness or color | Source transfer and full/limited levels; codec names do not establish source color |
| Matte guides the wrong subject or drifts | Mask polarity, alignment, duration, confidence, and carry/authoritative choice |
| Exports differ from older reference runs | Turn cuDNN OFF and leave advanced speed options OFF |
| Short clip takes longer with cuDNN | First-use kernel setup may outweigh warm per-frame savings |
| Missing Python, FFmpeg, or models | A source clone excludes installed assets; complete setup or restore the full installed folder |
| EXR support error | Use the matching updated BAT/launcher; EXR support is enabled before OpenCV loads |
| Processing shows 100% but has not finished | Wait for encoder finalization and the completion message |

## Documentation

| Guide | Contents |
|---|---|
| [README.txt](README.txt) | Beginner portable workflow |
| [README_ADVANCED.txt](README_ADVANCED.txt) | Detailed color, matte/AE, export, memory, cache, and CLI notes |
| [PORTABLE_README.txt](PORTABLE_README.txt) | Installed-folder layout, copying, and portable rebuild requirements |
| [Changelog](changelog.txt) | Local workflow/packaging history and GPU selections |
| [Performance results](research/performance-implementation/RESULTS.md) | Reference-preserving pipeline changes and exact-output checks |
| [GPU backends](research/performance-implementation/GPU_BACKENDS.md) | Speed/quality choices, first-use cost, numerical differences, and compatibility evidence |
| [Initial assessment](research/performance-2026-10-26/ASSESSMENT.md) | Profiling, architecture research, and optimization decisions |

## Credits and licensing

- **CorridorKey 2.0:** Zachary Harris, Niko, and Corridor Digital. [Upstream repository](https://github.com/nikopueringer/CorridorKey2.0) · [Official CorridorKey tools](https://corridorkey.com).
- **C-RADIO:** NVIDIA. **MoGe-2:** Microsoft, with vendored DINOv2 components. **RVM:** Google DeepMind. See [third-party notices](THIRD_PARTY_NOTICES.md).
- **This fork:** guided CLI, portable Windows packaging, workflow controls, reliability fixes, and measured inference/export optimizations.

This project uses the **Corridor Key Licence**, including its additional terms. Read [LICENSE](LICENSE) for usage, redistribution, attribution, and other requirements. Third-party components retain their own licenses.
