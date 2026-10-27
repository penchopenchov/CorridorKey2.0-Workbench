CORRIDORKEY 2.0 PORTABLE - ADVANCED GUIDE
=======================================

1. PACKAGE LAYOUT AND PORTABILITY

Run CorridorKey_DRAG_CLIPS_HERE.bat from this folder. It prefers the bundled
runtime/python.exe and resolves application paths from the BAT's own location.
The package can be moved or renamed with its contents kept together.

runtime/        CPython 3.11.15, standard library, installed packages, CUDA
                PyTorch, and app-local Microsoft C++ redistributable DLLs.
weights/        CorridorKey checkpoint and foundation normalization statistics.
vfms/           C-RADIO snapshot, MoGe checkpoint and RVM checkpoint.
corridorkey/    Upstream model, inference runtime and media writers.
third_party/    Vendored MoGe source and its licensing material.
ffmpeg.exe      Local FFmpeg encoder/decoder executable.

The payload omits the original venv, Git history, cached downloads, previous
renders and temporary test jobs. Its developer handoff contains text source
changes needed for the explicitly requested fork workflow, not a second runtime.

runtime/python311._pth uses relative entries for the runtime, Lib, DLLs,
Lib/site-packages and application root. Isolated Python ignores registry/user
Python paths and external PYTHONHOME/PYTHONPATH settings. Required MSVCP140 and
VCRUNTIME140 DLLs are bundled next to python.exe. NVIDIA's GPU driver and the
Windows operating system remain external platform requirements.

The bundled CUDA runtime is 12.8, with torch 2.9.1+cu128 and torchvision
0.24.1+cu128. A separate CUDA toolkit installation is unnecessary. The target
needs compatible Windows x64 and a supported NVIDIA GPU/driver. Test a short
clip on the other PC; same-machine relocation verification cannot prove that
another PC has a compatible driver or sufficient VRAM.

2. SOURCE COLOR - INCLUDING LINEAR REC.709 PRORES 4444

The input-color menu explicitly selects the source transfer:

1 / sRGB:
  Standard piecewise inverse sRGB transfer, including the linear toe.
2 / Rec.709:
  Inverse BT.709 OETF. This is not a gamma-2.4 / BT.1886 display-EOTF setting.
3 / Linear Rec.709:
  RGB already represents linear-light values with Rec.709 primaries; gamma 1.0.
  Use this for footage actually exported as linear Rec.709 ProRes 4444.

The container/codec is independent of this choice. ProRes can carry linear or
nonlinear values, and an MP4's filename does not establish its transfer.
These modes assume Rec.709/sRGB primaries and a BT.709 YUV matrix for video.
Convert ACEScg, camera log, Rec.2020/PQ/HLG and other unsupported source color
systems upstream. No automatic general color-management/OCIO engine was added.

Video data levels are separate: Auto uses decoded range metadata; Limited and
Full override the range. Incorrect levels can crush blacks or alter contrast.

Precision path:
- OpenCV probes metadata only; the launcher does not feed its 8-bit video
  decode into the model for this guided workflow.
- FFmpeg streams float RGB video while handling the matrix/range. Normal source
  videos do not require whole-clip temporary EXRs; hint/sequence preparation can.
- Transfer is held linear on both sides of zscale during that decode, avoiding
  zimg's different display-EOTF interpretation of a BT.709 transfer tag.
- Explicit sRGB or inverse BT.709 equations are applied once to float RGB.
- Linear Rec.709 gets no further gamma conversion.
- infer.py receives explicit video color/range settings for that stream.
  Prepared frame sequences receive --frame_dir_linear and the source FPS, so
  already linear frame data is not gamma-decoded a second time.
- Native dimensions and selected leading-frame ordering are retained. Source
  fractional FPS is probed and forwarded. Variable-frame-rate timestamp
  preservation has not been added; use an appropriate constant-FPS source.

For image sequences, numbered EXR/PNG/JPG/JPEG files are sorted by filename.
Use zero-padded frame numbers. Both a flat frame folder and an Input subfolder
are supported. Supply sequence FPS explicitly. Integer images are normalized
from their dtype; linear EXR frames can be used directly.

3. MATTE / HINT WORKFLOW

No matte:
  Model selects the foreground. Companion mask files are ignored.
First-frame matte:
  A mask image or frame folder supplies guidance for the first selected frame.
  White is foreground, black is background; an image alpha channel is accepted
  by the upstream first-frame loader.
Full matte:
  An aligned grayscale movie or frame sequence supplies guidance across time.
  Matte values are coverage data, so the matte decode applies no gamma change.
  A video matte must match source FPS and have enough selected frames.

Matte confidence defaults to 0.75 for rough roto. A clean artist mask can use
0.95. During portable launcher setup, type b or back to revisit the previous
question. Enter keeps a previous answer; - restores the default on optional
output/cache folder questions. The final review offers Start, Back and Cancel.
Authoritative mode passes --no_carry_hint and uses the supplied mask on every
pass. Leave it off initially when you want carried predictions to help refine
a rough mask. A rough mask can steer systematic selection errors; inspect the
result rather than assuming every imperfect edge will be repaired.

Companions: shotFrameHint.png and shot_AlphaHint.mov/mp4. The launcher asks
before accepting discovered companions. Otherwise drop the actual mask path.
Structured frame input can have an AlphaHint companion folder.

AFTER EFFECTS RECIPE FOR A FULL ROUGH ROTO MOVIE

1. Match the original frame size, FPS, start and duration. Do the rough roto.
2. Put Shift Channels after Roto Brush / roto effects. If necessary precompose
   the isolated subject at the full source frame size first.
3. Take Red, Green and Blue From Alpha; Take Alpha From Full On. The result
   should visibly be white subject / black background, with gray coverage.
4. Export RGB as high-quality H.264 MP4 or ProRes. For coverage values, avoid
   an output color/gamma conversion; use Preserve RGB in AE's Output Module
   color settings when available. Keep source and matte timing aligned.
5. Choose Full matte in the launcher and give it that separate matte movie.

A source MOV's embedded alpha is not automatically used as a matte. A full
matte movie is read from its grayscale picture, so exporting only a normal
color cutout with alpha does not supply the intended grayscale guidance.

4. EXPORTS AND OUTPUT COLOR

alpha:   Single-channel coverage preview or EXR matte, coverage clamped [0,1].
fg:      Predicted foreground RGB. Video foreground has no embedded alpha.
checker: Foreground over baked checkerboard for inspecting the key.
cutout:  Transparent foreground, despilled, straight/unpremultiplied RGB+alpha.

Video mode uses the upstream writers: ProRes 4444 MOV for cutout and H.264
previews for other outputs. Video cutout can be straight sRGB or straight
linear Rec.709 independently of source color. Video values are clipped [0,1].
Preview outputs remain display-coded; selecting a linear cutout does not make
every preview linear. Interpret the ProRes alpha as Straight/Unmatted in the
compositor. Retain original audio separately in the editing timeline.

EXR mode is always scene-linear. Cutout is RGBA; choose half float or float32.
It preserves the runtime's unclipped float RGB, subject to model precision and
the selected EXR dtype. Half output is bounded to finite float16 range.
Despill (0..1, default 0.5) applies to checker/cutout as in the upstream runtime.

5. MEMORY, TEMPORARY SPACE, BATCHES AND LOGS

Auto memory selection is the recommended starting point. Manual mode exposes
low-VRAM staging and Hann chunk/stride. On a 16 GB card, 4K commonly needs
low-VRAM mode and smaller windows. If memory fails, close other GPU apps and
try 8/4, 4/2 or 1/1. Smaller temporal windows sacrifice context.

Normal source video streams float frames through bounded buffers. Prepared
image sequences, hints and low-VRAM feature staging can still consume substantial
temporary space. Choose a fast cache drive with enough room. Leading-frame
limits reduce decode/model work. Job caches are cleaned after success, failure
or cancellation. Output files remain at the chosen destination.

GPU speed/quality selections:
- Faster cuDNN attention defaults ON in the guided launcher and direct CLI.
  Quality may degrade: matte edges and foreground color can change. First use
  at a resolution adds setup time; short clips may be slower. OFF uses the
  reference attention policy. Other supported kernels remain fallback options.
- Advanced options default OFF. GPU video preprocessing changes rounding and
  uses more VRAM; image sequences retain CPU preprocessing. Foundation-feature
  reuse changes batch shapes and applies only in normal VRAM mode. Native
  Stage 2 convolutions can improve speed on some GPUs but change predictions.
- For previous numerical results, turn cuDNN OFF and leave all advanced speed
  options OFF. Color/range, hints, temporal windows, output formats and encoder
  settings remain independent choices. Back navigation and batch model reuse
  remain supported. Compare test exports before accepting a tradeoff.
- The settings text file and performance JSON record the selected options.
  The library InferenceConfig keeps reference defaults for regression tools;
  the CLI explicitly supplies its user-facing cuDNN default.

Direct flags: --stage2_attention cudnn (CLI default), or
--stage2_attention default for reference attention; optional
--preprocess_device cuda, --cache_foundations, --stage2_native_convolutions.

CPU preparation threads default to Auto (FFmpeg's thread count 0). Manual
accepts 1 through the detected logical CPU count; Conservative retains the
previous 2-thread setting. The selection applies where temporary EXR encoding
is required, including full-matte videos; normal source video streams instead.
It is shown in the settings review and
saved as preparation_threads in launcher_settings.json (0 means Auto).

This controls the EXR encoder, not a hard limit on the process's total CPU
threads. Source decoder and filter threading remain automatic. Image-sequence
conversion, the later transfer-conversion pass, GPU inference and final-output
encoders are unchanged. Float precision, ZIP compression and color equations
are unchanged. More threads may help CPU-bound EXR encoding, at the cost of
additional RAM; disk throughput or other processing can limit the speedup.
Use Manual or Conservative if Auto uses more resources than desired.

The fork owner confirmed a significant real-world speedup in Decoding source
with Auto on an AMD Ryzen 7 3800X (16 logical threads), compared with the old
two-thread setting. This is a CPU preparation optimization; no numeric speedup
factor or inference-stage speedup is claimed.

For a batch, input color/settings apply to all selected clips. Run sources
with different transfers separately. Companion matte PNGs do not turn a video
batch into an image sequence. Missing explicit masks reprompt; the launcher
does not silently process the clip without the selected matte.

The final review lists source color/levels, matte, outputs/color, frame count,
cache and destinations. Existing nonempty output folders are noted, and
colliding batch destinations are rejected. launcher_settings.json records
source/matte, selected settings, FPS and the executed model command.

6. COMMAND LINE

PORTABLE CACHE OWNERSHIP AND RECOVERY

The guided launcher creates a unique job_<uuid> folder in the selected cache
root. Ownership records and small lock files live in .corridorkey-jobs beside
the job folders, so an interrupted/partial deletion remains recoverable.
The default root is %TEMP%\corridorkey_launcher.

Each job holds an exclusive OS lease. Decoder and inference subprocesses
inherit that lease, keeping the cache protected even if their launcher dies.
Recovery must acquire the same exclusive lease before deleting a registered
job folder. Active jobs, unknown folders, invalid ownership records and linked
job directories are never swept. Cleanup is scoped to temporary caches owned
by this launcher, not the full Windows Temp folder or output/model folders.

Cleanup runs after each job, including ordinary errors and cancellation. It
briefly retries transient deletion failures and warns with the retained path
if it still cannot remove files. Recovery checks the default root at startup
and a selected custom root before processing. A forced termination may leave
files until the next launch after all processing children have exited. The
launcher does not install a nightly scheduled task.

This management applies to the portable launcher. Direct infer.py retains
its existing feature-file cleanup behavior. Old unregistered cache folders
require separate inspection; they are not automatically assumed safe to delete.

PROGRESS REPORTING

The guided launcher and direct infer.py both report stage progress. The
launcher covers media metadata, source/matte decoding, transfer conversion,
frame-sequence preparation and temporary-cache cleanup. The inference worker
covers Python/model loading, RVM, optional low-VRAM foundation precomputation,
temporal inference, rendering and output finalization.

The line shows clip index, stage, percentage, completed/total source frames,
elapsed stage time and stage ETA. ETA uses the average measured stage rate;
it is an estimate, not a prediction of the whole job. Unknown-length stages
keep an elapsed heartbeat with "estimating" instead of invented percentages.
Frame totals from video metadata are reconciled with actual decoded files.

Normal mode interleaves temporal inference and stage-2 rendering; its counter
advances when unique frames have been finalized and submitted to the writers,
not for each overlapping window or each selected output. Low-VRAM mode reports
temporal inference and rendering separately. Writers may still be encoding
after rendering reaches 100%; output finalization waits for them to finish.

Interactive terminals update in place, normally four times per second.
Redirected stderr gets line-based stage updates about every five seconds,
plus stage start/end. Existing diagnostic output remains visible. Interrupted
or failing stages retain their actual counts and are labeled canceled/failed.

GPU stages also show VRAM as allocated/reserved GiB for this inference worker.
Allocated is memory occupied by live PyTorch tensors. Reserved is the total
PyTorch allocator footprint, including allocated tensors and reusable cached
blocks; do not add the two numbers. They exclude CUDA context/driver overhead,
allocations outside the PyTorch allocator and other applications. This works
when Windows WDDM makes nvidia-smi per-process memory unavailable. Counters
are read without launching monitoring subprocesses or synchronizing the GPU.

Narrow consoles use a compact layout: [clip/total], frames with an f suffix,
t for elapsed stage time, and VRAM<allocated>/<reserved>GiB. Optional window
details are omitted so frame counts, ETA and memory remain visible.

From this portable folder in PowerShell:

  .\runtime\python.exe corridorkey_launcher.py "D:\Footage\shot.mov"
  .\runtime\python.exe infer.py --help

Direct infer.py retains upstream behavior. Its video reader treats input as
display-coded. Use the guided launcher for linear ProRes / explicit transfer
handling. --frame_dir_linear is for an actual frame directory, not a switch
that makes the upstream OpenCV video reader a linear high-bit-depth reader.

Direct scene-linear EXR sequence example:

  .\runtime\python.exe infer.py --input "D:\Footage\linear_frames" --frame_dir_linear --frame_dir_fps 24 --outputs alpha fg cutout --output_format exr --output_dir "D:\Renders\shot" --ffmpeg_bin ".\ffmpeg.exe"

7. EXR STARTUP FIX AND VERIFICATION

The launcher and BAT explicitly enable OPENCV_IO_ENABLE_OPENEXR=1 before
OpenCV imports. Enabling it only after probing video metadata was too late.
This package includes that fix. Fresh-process regressions cover sRGB/Rec.709
MP4 decoding with the variable missing or inherited as 0, without relying on
the main test process's EXR setting.

Details and dated regression history are in changelog.txt. The installation
has been checked with bundled Python and actual model processing. Historical
synthetic tests are not a quality benchmark on production footage or a second
physical PC. To validate model loading, run:
  runtime\python.exe scripts\smoke_test_runtime.py --load_model

8. FORK AND REBUILDING

The repository root contains the launcher, setup/build scripts and this
documentation, alongside the ignored installed runtime and model payloads.
There is no nested portable installation or separate development checkout.
Legacy regression tests and agent handoff copies have been removed.

Source Git commits exclude runtime, model binaries, FFmpeg, caches and output
media. A source-only clone is not itself a ready-installed portable release.
Follow upstream setup, then tools/make_portable.ps1 in a source checkout to
build Python plus dependencies and app-local Microsoft CRT files. See its
-SourceEnvironment, -PythonHome and -VCRedistDirectory options. Package the
runtime, models, source directories, FFmpeg and documentation together.

Licenses and notices travel with the payload. Use LICENSE and
THIRD_PARTY_NOTICES.md for the upstream terms and foundation-model notices.
