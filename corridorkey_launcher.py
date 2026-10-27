"""Guided Windows launcher with explicit input color and matte selection."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from cli_progress import Progress, progress_job
from cli_menu import SetupMenu
from cli_cache import JobCache, cache_popen, cleanup_abandoned

# Video metadata probing imports OpenCV before decoded float EXRs are read.
# OpenCV caches this option when it loads, so configure it at process startup.
# EXR support is required by this launcher's high-precision preparation path.
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from run_interactive import (
    clean_path, find_first_frame_hint,
    print_header,
)

ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {'.mov', '.mp4', '.mkv', '.avi', '.webm', '.mxf'}
IMAGE_EXTENSIONS = {'.exr', '.png', '.jpg', '.jpeg'}
COLOR_LABELS = {'srgb': 'sRGB', 'rec709': 'Rec.709 (BT.709 transfer)',
                'linear': 'Linear Rec.709 (gamma 1.0)'}


@dataclass
class Settings:
    input_color: str = 'srgb'
    input_range: str = 'auto'
    hint_mode: str = 'none'
    hint_quality: float = 0.75
    authoritative: bool = False
    output_format: str = 'video'
    exr_dtype: str = 'half'
    outputs: tuple[str, ...] = ('alpha', 'fg', 'cutout', 'checker')
    cutout_linear: bool = False
    despill_strength: float = 0.5
    low_vram: bool = False
    hann_chunk: int | None = None
    hann_stride: int | None = None
    num_frames: int = -1
    sequence_fps: float = 24.0
    output_root: Path | None = None
    cache_root: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / 'corridorkey_launcher')
    preparation_threads: int = 0  # FFmpeg's automatic EXR encoder thread selection.
    stage2_attention: str = 'cudnn'
    stage2_native_convolutions: bool = False
    preprocess_device: str = 'cpu'
    cache_foundations: bool = False


@dataclass
class Job:
    source: Path
    input_path: Path
    hint: Path | None = None


@dataclass
class PreparedMedia:
    path: Path
    fps: float


def image_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def sequence_folder(folder):
    folder = Path(folder)
    for child in folder.iterdir():
        if child.is_dir() and child.name.lower() == 'input' and image_files(child):
            return child
    # A batch folder can contain companion mask PNGs beside its source clips.
    if image_files(folder) and not any(p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS for p in folder.iterdir()):
        return folder
    return None


def collect_jobs(paths):
    jobs = []
    for path in paths:
        path = Path(path).resolve()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            jobs.append(Job(path, path))
        elif path.is_dir():
            sequence = sequence_folder(path)
            if sequence:
                jobs.append(Job(path, sequence))
            else:
                children = [p for p in sorted(path.iterdir())
                            if (p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                                and '_alphahint' not in p.stem.lower())
                            or (p.is_dir() and sequence_folder(p))]
                jobs.extend(collect_jobs(children))
        else:
            raise ValueError(f'Unsupported or missing input: {path}')
    if not jobs:
        raise ValueError('No supported clips or image sequences found.')
    # Avoid processing the same clip twice when both it and its folder were dropped.
    return list({job.input_path: job for job in jobs}.values())


def bounded_float(text, lower=0.0, upper=1.0):
    try:
        value = float(text)
        return value if lower <= value <= upper else None
    except ValueError:
        return None


def prompt_settings(*, has_sequences, menu=None):
    if menu is None:
        return SetupMenu().run(lambda current: prompt_settings(has_sequences=has_sequences, menu=current))
    settings = Settings()
    menu.note('\nInput color - how the SOURCE RGB values are encoded')
    color = menu.choices('input_color', 'Choose source color:', [
        'sRGB (standard images / sRGB-encoded video)',
        'Rec.709 (BT.709 transfer, non-linear)',
        'Linear Rec.709 / gamma 1.0 (including linear ProRes 4444)',
    ])
    settings.input_color = ('srgb', 'rec709', 'linear')[color - 1]
    menu.note('ProRes 4444 is a codec; choose the transfer used when the clip was exported.')
    levels = menu.choices('input_range', 'Video data levels (ignored for image sequences):', [
        'Auto - use decoded video range metadata', 'Limited / video / TV levels', 'Full / data / PC levels',
    ])
    settings.input_range = ('auto', 'limited', 'full')[levels - 1]
    if has_sequences:
        settings.sequence_fps = float(menu.text('sequence_fps', 'Image sequence frame rate', default='24',
                                                   validator=lambda x: bounded_float(x, 0.001, 1000)))
    menu.note('\nMatte / subject guidance')
    matte = menu.choices('hint_mode', 'What matte do you have?', [
        'No matte - let the model choose the foreground',
        'First-frame matte - a mask image for the first frame',
        'Full matte - a matching matte video or image sequence',
    ])
    settings.hint_mode = ('none', 'first', 'full')[matte - 1]
    if settings.hint_mode != 'none':
        settings.hint_quality = float(menu.text('hint_quality', 'Matte confidence / trust: 0.95 clean / 0.75 rough',
                                                default='0.75', validator=bounded_float))
        if settings.hint_mode == 'full':
            settings.authoritative = menu.confirm('authoritative', 'Use the supplied matte on every frame (disable carried predictions)?', default=False)
    menu.note('\nExports')
    output = menu.choices('output_format', 'Output format:', [
        'Video - ProRes 4444 cutout with alpha, plus H.264 previews',
        'EXR sequences - scene-linear HDR, cutout includes RGBA',
    ])
    settings.output_format = ('video', 'exr')[output - 1]
    if settings.output_format == 'exr':
        precision = menu.choices('exr_dtype', 'EXR precision:', ['Half float (16-bit)', 'Full float (32-bit)'])
        settings.exr_dtype = ('half', 'float')[precision - 1]
    names = ('alpha', 'fg', 'cutout', 'checker')
    def outputs(answer):
        try:
            indices = list(dict.fromkeys(int(n) for n in answer.replace(',', ' ').split()))
            if not indices or any(n < 1 or n > 4 for n in indices):
                raise ValueError
            return tuple(names[n - 1] for n in indices)
        except ValueError:
            raise ValueError('Choose only output numbers 1 through 4.')
    settings.outputs = menu.text('outputs', 'Outputs: 1) Alpha matte  2) Foreground RGB  3) Transparent cutout  4) Checker preview\nEnter numbers separated by spaces.',
                                 default='1 2 3 4', validator=outputs)
    if settings.output_format == 'video' and 'cutout' in settings.outputs:
        color = menu.choices('cutout_linear', 'OUTPUT cutout color (independent of source color):', [
            'Straight sRGB RGB + linear alpha (standard editing)',
            'Straight linear Rec.709 RGB + linear alpha',
        ])
        settings.cutout_linear = color == 2
    if {'checker', 'cutout'}.intersection(settings.outputs):
        settings.despill_strength = float(menu.text('despill_strength', 'Despill strength [0 = none, 1 = full]',
                                                       default='0.5', validator=bounded_float))
    memory = menu.choices('memory', 'GPU memory:', ['Auto - choose settings from free VRAM', 'Manual - low VRAM / temporal window'])
    if memory == 2:
        settings.low_vram = menu.confirm('low_vram', 'Enable low-VRAM mode?', default=True)
        window = menu.choices('hann_window', 'Select Hann Window Presets (Temporal Context):', [
            'High VRAM (48 chunk, 24 stride) - Recommended for standard GPUs',
            'Medium VRAM (24 chunk, 12 stride) - Lower peak memory',
            'Low VRAM (8 chunk, 4 stride) - Minimal memory, fast startup',
            'Max Continuity (80 chunk, 40 stride) - Requires high workstation VRAM',
            'Custom Size',
        ])
        if window == 5:
            def positive_int(text):
                value = int(text)
                return value if value > 0 else None
            settings.hann_chunk = menu.text('hann_chunk', 'Enter Hann Chunk (positive integer)', default='48', validator=positive_int)
            settings.hann_stride = menu.text('hann_stride', 'Enter Hann Stride (positive integer)', default='24', validator=positive_int)
        else:
            settings.hann_chunk, settings.hann_stride = {1: (48, 24), 2: (24, 12), 3: (8, 4), 4: (80, 40)}[window]
    menu.note('\nGPU speed / quality')
    attention=menu.choices('stage2_attention','Faster cuDNN attention:',[
        'ON - faster on long clips; quality may degrade',
        'OFF - reference PyTorch attention, preserves prior results',
    ])
    settings.stage2_attention='cudnn' if attention==1 else 'default'
    menu.note('cuDNN may change matte edges and foreground. First use at a resolution can add startup time.')
    if menu.confirm('advanced_speed','Show additional experimental speed options (quality may degrade)?',default=False):
        settings.preprocess_device=('cuda' if menu.confirm('gpu_preprocess',
            'GPU video preprocessing? Faster preparation, different rounding; quality may degrade. Image sequences still use CPU.',default=False) else 'cpu')
        settings.cache_foundations=menu.confirm('cache_foundations',
            'Reuse foundation features? No measured export gain in our short test; quality may degrade. Normal VRAM mode only.',default=False)
        settings.stage2_native_convolutions=menu.confirm('stage2_native_convolutions',
            'Native PyTorch convolutions for Stage 2? Hardware-dependent speedup; quality may degrade.',default=False)
    logical_cpus = os.cpu_count() or 1
    preparation = menu.choices('preparation', 'CPU preparation threads:', [
        f'Auto - let FFmpeg choose ({logical_cpus} logical CPUs detected)',
        'Manual - choose a thread count',
        'Conservative - 2 EXR encoder threads (previous setting)',
    ])
    if preparation == 2:
        def thread_count(text):
            try:
                value = int(text)
                return value if 1 <= value <= logical_cpus else None
            except ValueError:
                return None
        settings.preparation_threads = int(menu.text(
            'preparation_threads', f'EXR encoding threads [1-{logical_cpus}]', default=str(min(8, logical_cpus)),
            validator=thread_count))
    elif preparation == 3:
        settings.preparation_threads = 2
    menu.note('Controls temporary video-to-EXR encoding. More threads can increase RAM use.')
    def frame_limit(text):
        if text.lower() == 'all':
            return -1
        try:
            value = int(text)
            return value if value > 0 else None
        except ValueError:
            return None
    limit = menu.text('num_frames', 'Process first N frames, or all', default='all', validator=frame_limit)
    settings.num_frames = -1 if str(limit).lower() == 'all' else int(limit)
    destination = menu.text('output_root', 'Output parent folder [default: beside each source]', default='')
    settings.output_root = Path(clean_path(destination)) if destination else None
    menu.note('High-precision decoding uses temporary float EXRs, removed after each job.')
    cache = menu.text('cache_root', f'Temporary cache folder [default: {settings.cache_root}]', default='')
    if cache:
        settings.cache_root = Path(clean_path(cache))
    return settings


def resolve_hint(job, mode, menu=None):
    if menu is None:
        return SetupMenu().run(lambda current: resolve_hint(job, mode, menu=current))
    if mode == 'none':
        return None
    if mode == 'first':
        candidate = find_first_frame_hint(job.source)
    else:
        candidate = next((p for p in sorted(job.source.parent.iterdir())
                          if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                          and p.stem.lower().startswith(job.source.stem.lower() + '_alphahint')), None)
    if mode == 'full' and job.source.is_dir():
        candidate = next((p for p in job.source.iterdir() if p.is_dir() and p.name.lower() == 'alphahint'), candidate)
    if candidate and (mode == 'full' or candidate.suffix.lower() in IMAGE_EXTENSIONS):
        if menu.confirm(('companion', job.source, mode, candidate), f'Use companion matte for {job.source.name}: {candidate}?', default=True):
            return Path(candidate)
    def matte_path(answer):
        path = Path(clean_path(answer))
        valid_dir = path.is_dir() and bool(image_files(path))
        valid_file = path.is_file() and path.suffix.lower() in (
            IMAGE_EXTENSIONS if mode == 'first' else VIDEO_EXTENSIONS)
        if valid_dir or valid_file:
            return path
        raise ValueError('Matte path is missing or unsupported. Supply an existing mask; Ctrl+C cancels.')
    kind = 'first-frame matte image / folder' if mode == 'first' else 'full matte video / frame folder'
    return menu.text(('matte_path', job.source, mode), f'Drag the {kind} for {job.source.name} here', validator=matte_path)


def decode_transfer(values, color):
    import numpy as np
    if color == 'linear':
        return values
    magnitude = np.abs(values)
    if color == 'srgb':
        decoded = np.where(magnitude <= 0.04045, magnitude / 12.92, ((magnitude + 0.055) / 1.055) ** 2.4)
    elif color == 'rec709':
        decoded = np.where(magnitude < 0.081, magnitude / 4.5, ((magnitude + 0.099) / 1.099) ** (1 / 0.45))
    else:
        raise ValueError(f'Unknown input color: {color}')
    return np.sign(values) * decoded


def video_metadata(path):
    import cv2
    reader = cv2.VideoCapture(str(path))
    try:
        fps = reader.get(cv2.CAP_PROP_FPS)
        count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if not reader.isOpened() or not 0 < fps < 1000:
            raise ValueError(f'Cannot determine video frame rate: {path}')
        return fps, count
    finally:
        reader.release()


def decode_video(source, destination, color, levels, num_frames, ffmpeg, *, matte=False, threads=0):
    with Progress('Reading media metadata', unit=None):
        fps, frame_count = video_metadata(source)
    total = min(frame_count, num_frames) if frame_count > 0 and num_frames > 0 else max(frame_count, num_frames, 0)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if matte:
        # A matte is coverage, so no gamma/transfer conversion is applied.
        filters = 'format=grayf32le'
    else:
        # Decode code values to float RGB without an EOTF. zimg interprets BT.709
        # through a display EOTF, which differs from the inverse OETF selected in
        # our menu. Use the same explicit transfer equations as image sequences.
        filters = 'zscale=matrixin=709:primariesin=709:transferin=linear:matrix=gbr:primaries=709:transfer=linear:range=full'
        if levels != 'auto':
            filters += f':rangein={levels}'
        filters += ',format=gbrpf32le'
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-nostats',
               '-progress', 'pipe:1', '-stats_period', '0.5', '-y', '-i', str(source),
               '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', filters, '-fps_mode', 'passthrough']
    if num_frames > 0:
        command += ['-frames:v', str(num_frames)]
    command += ['-c:v', 'exr', '-compression', 'zip16', '-threads', str(threads),
                '-start_number', '0', str(destination / 'frame_%08d.exr')]
    with Progress('Decoding matte' if matte else 'Decoding source', total) as progress, tempfile.TemporaryFile() as error_log:
        process = cache_popen(command, stdout=subprocess.PIPE, stderr=error_log,
                                   text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                key, _, value = line.strip().partition('=')
                if key == 'frame':
                    progress.update(int(value))
            result = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
        if result:
            error_log.seek(max(0, error_log.seek(0, 2) - 16384))
            error = error_log.read().decode('utf-8', errors='replace').strip()
            raise RuntimeError(f'Could not decode {source}:\n{error}')
        frames = image_files(destination)
        if not frames:
            raise RuntimeError(f'Video produced no decoded frames: {source}')
        # Metadata frame counts can be approximate; finish with the actual count.
        progress.update(len(frames), total=len(frames))
    if not matte and color != 'linear':
        import cv2
        with Progress('Linearizing source', len(frames)) as progress:
            for index, path in enumerate(frames, 1):
                values = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if values is None or not cv2.imwrite(str(path), decode_transfer(values, color)):
                    raise RuntimeError(f'Could not linearize decoded frame: {path}')
                progress.update(index)
    return PreparedMedia(destination, fps)


def decode_sequence(source, destination, color, fps, num_frames):
    import cv2
    import numpy as np
    frames = image_files(source)
    if num_frames > 0:
        frames = frames[:num_frames]
    if not frames:
        raise ValueError(f'No supported frames: {source}')
    with Progress('Preparing frame sequence', len(frames)) as progress:
        if color == 'linear' and all(p.suffix.lower() == '.exr' for p in frames):
            progress.update(len(frames))
            return PreparedMedia(Path(source), fps)
        destination.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(frames):
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f'Cannot read image: {path}')
            values = raw.astype(np.float32)
            if np.issubdtype(raw.dtype, np.integer):
                values /= np.iinfo(raw.dtype).max
            if values.ndim == 2:
                values = np.repeat(values[:, :, None], 3, axis=2)
            values = decode_transfer(values[:, :, :3], color)
            if not cv2.imwrite(str(destination / f'frame_{index:08d}.exr'), values):
                raise RuntimeError(f'Could not write decoded image: {path}')
            progress.update(index + 1)
    return PreparedMedia(destination, fps)


@contextmanager
def prepare_job(job, settings, ffmpeg):
    with JobCache(settings.cache_root) as temporary:
        work = temporary.path / job.source.stem
        work.mkdir()
        print(f'Decoding {job.source.name} as {COLOR_LABELS[settings.input_color]} ...', flush=True)
        if job.input_path.is_dir():
            prepared = decode_sequence(job.input_path, work / 'input', settings.input_color,
                                       settings.sequence_fps, settings.num_frames)
        else:
            fps, _ = video_metadata(job.input_path)
            prepared = PreparedMedia(job.input_path, fps)
        hint = job.hint
        if settings.hint_mode == 'full' and hint:
            if hint.is_file():
                matte_fps, _ = video_metadata(hint)
                if abs(matte_fps - prepared.fps) > 0.001:
                    raise ValueError(f'Matte FPS {matte_fps:g} does not match source FPS {prepared.fps:g}.')
                hint = decode_video(hint, work / 'matte', 'linear', 'auto', settings.num_frames,
                                    ffmpeg, matte=True, threads=settings.preparation_threads).path
            source_count = (len(image_files(prepared.path)) if prepared.path.is_dir()
                            else video_metadata(prepared.path)[1])
            if settings.num_frames > 0:
                source_count = min(source_count, settings.num_frames)
            if len(image_files(hint)) < source_count:
                raise ValueError('The full matte has fewer frames than the selected source segment.')
        yield prepared, hint, work / 'features'


def inference_command(prepared, hint, output_dir, settings, ffmpeg, feature_cache=None):
    command = [sys.executable, str(ROOT / 'infer.py'), '--input', str(prepared.path),
               '--output_dir', str(output_dir), '--frame_dir_fps', str(prepared.fps),
               '--ffmpeg_bin', str(ffmpeg), '--output_format', settings.output_format,
               '--exr_dtype', settings.exr_dtype, '--despill_strength', str(settings.despill_strength),
               '--stage2_attention',settings.stage2_attention,
               '--preprocess_device',settings.preprocess_device,
               '--outputs', *settings.outputs]
    if settings.cache_foundations:
        command.append('--cache_foundations')
    if settings.stage2_native_convolutions:
        command.append('--stage2_native_convolutions')
    if prepared.path.is_dir():
        command += ['--frame_dir_linear']
    else:
        command += ['--video_decoder', 'ffmpeg', '--input_color', settings.input_color,
                    '--input_range', settings.input_range]
    if hint and settings.hint_mode != 'none':
        command += ['--hint_first_frame' if settings.hint_mode == 'first' else '--hint_video', str(hint),
                    '--hint_quality', str(settings.hint_quality)]
        if settings.authoritative:
            command.append('--no_carry_hint')
    if settings.cutout_linear:
        command.append('--cutout_linear')
    if settings.low_vram:
        command.append('--low_vram')
    if feature_cache:
        command += ['--temp_dir', str(feature_cache)]
    if settings.hann_chunk is not None:
        command += ['--hann_chunk', str(settings.hann_chunk), '--hann_stride', str(settings.hann_stride)]
    if settings.num_frames > 0:
        command += ['--num_frames', str(settings.num_frames)]
    return command


def output_folder(job, settings):
    parent = settings.output_root or job.source.parent
    return parent / f'out_{job.source.stem}'


def run_job(job, settings, ffmpeg, *, model=None):
    destination = output_folder(job, settings)
    destination.mkdir(parents=True, exist_ok=True)
    with prepare_job(job, settings, ffmpeg) as (prepared, hint, feature_cache):
        command = inference_command(prepared, hint, destination, settings, ffmpeg, feature_cache)
        record = {'source': str(job.source), 'matte': str(job.hint) if job.hint else None,
                  'settings': asdict(settings), 'fps': prepared.fps, 'command': command}
        (destination / 'launcher_settings.json').write_text(json.dumps(record, indent=2, default=str) + '\n', encoding='utf-8')
        print(f'Processing -> {destination}', flush=True)
        from infer import main as infer_main
        model = infer_main(command[2:], model=model)
    print(f'Finished: {destination}', flush=True)
    return model


def prompt_run(paths, menu):
    if paths:
        jobs = collect_jobs([Path(clean_path(path)) for path in paths])
    else:
        jobs = menu.text('source', 'Drag a video, frame folder, or folder of clips here',
                         validator=lambda value: collect_jobs([Path(clean_path(value))]))
    settings = prompt_settings(has_sequences=any(job.input_path.is_dir() for job in jobs), menu=menu)
    for job in jobs:
        job.hint = resolve_hint(job, settings.hint_mode, menu=menu)
    menu.note('\nReview settings')
    menu.note(f'Input color: {COLOR_LABELS[settings.input_color]} | Video levels: {settings.input_range}')
    menu.note(f'Matte: {settings.hint_mode} | Confidence: {settings.hint_quality:g} | Output: {settings.output_format} | Files: {", ".join(settings.outputs)}')
    menu.note('Cutout color: ' + ('Linear Rec.709' if settings.output_format == 'exr' or settings.cutout_linear else 'sRGB'))
    menu.note(f'GPU attention: {settings.stage2_attention} | GPU preprocessing: {settings.preprocess_device} | Feature reuse: {settings.cache_foundations} | Native convolutions: {settings.stage2_native_convolutions}')
    if settings.stage2_attention=='cudnn' or settings.preprocess_device=='cuda' or settings.cache_foundations or settings.stage2_native_convolutions:
        menu.note('Speed/quality tradeoffs enabled: quality may degrade. Turn OFF cuDNN and leave advanced options OFF for reference results.')
    thread_label = 'Auto (FFmpeg chooses)' if settings.preparation_threads == 0 else str(settings.preparation_threads)
    menu.note(f'CPU preparation threads: {thread_label}')
    menu.note(f'Frames: {settings.num_frames if settings.num_frames > 0 else "all"} | Cache: {settings.cache_root}')
    menu.note('Video input streams as float frames; frame-sequence and low-VRAM caches use temporary disk space.')
    for index, job in enumerate(jobs, 1):
        menu.note(f'{index}. {job.source} -> {output_folder(job, settings)}')
        if job.hint:
            menu.note(f'   Matte: {job.hint}')
    destinations = [output_folder(job, settings).resolve() for job in jobs]
    if len(set(destinations)) != len(destinations):
        raise ValueError('Two inputs map to the same output folder. Process them separately or rename them.')
    if any(folder.exists() and any(folder.iterdir()) for folder in destinations):
        menu.note('Existing output folders may contain files that this run will overwrite.')
    return (jobs, settings) if menu.review() else None


def recover_caches(cache_root):
    with Progress('Checking abandoned caches', unit=None):
        result = cleanup_abandoned(cache_root)
    if result.removed:
        print(f'Removed {result.removed} abandoned temporary job cache(s).')
    if result.active:
        print(f'Kept {result.active} active temporary job cache(s).')


def main(argv=None):
    if os.name == 'nt':
        os.system('')
    print_header('CorridorKey 2.0 - guided color and matte setup')
    try:
        paths = sys.argv[1:] if argv is None else argv
        if not all(path.strip() for path in paths):
            raise ValueError('No input selected.')
        ffmpeg = ROOT / 'ffmpeg.exe' if (ROOT / 'ffmpeg.exe').is_file() else shutil.which('ffmpeg')
        if not ffmpeg:
            raise ValueError('FFmpeg is missing. Restore ffmpeg.exe in the installation folder.')
        default_cache = Settings().cache_root
        recover_caches(default_cache)
        print('Setup: type b or back to revisit the previous question. Enter keeps the displayed default.')
        configured = SetupMenu().run(lambda menu: prompt_run(paths, menu))
        if configured is None:
            print('Canceled. No clips processed.')
            return 0
        jobs, settings = configured
        if settings.cache_root.resolve() != default_cache.resolve():
            recover_caches(settings.cache_root)
        model = None
        for index, job in enumerate(jobs, 1):
            print_header(f'Clip {index}/{len(jobs)}: {job.source.name}')
            with progress_job(f'{index}/{len(jobs)}'):
                model = run_job(job, settings, ffmpeg, model=model)
        print_header('All selected clips completed')
        return 0
    except KeyboardInterrupt:
        print('\nCanceled by user. Check the cache-cleanup messages above for any retained files.')
        return 130
    except (ValueError, RuntimeError, OSError) as error:
        print(f'\nERROR: {error}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
