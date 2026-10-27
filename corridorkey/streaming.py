"""Float video decoding without intermediate image files.

The filters and transfer equations deliberately match the guided launcher's
reference EXR path. CPU preprocessing preserves its rounding boundaries.
"""
from __future__ import annotations

import subprocess
import tempfile
import queue
import threading

import numpy as np
import torch

from cli_cache import cache_popen
from . import color_utils as cu
from .video_io import center_pad_to_shape, resize_hwc_lanczos, apply_linear_adjust


def decode_transfer(values, color):
    if color == 'linear':
        return values
    magnitude = np.abs(values)
    if color == 'srgb':
        decoded = np.where(magnitude <= .04045, magnitude / 12.92,
                           ((magnitude + .055) / 1.055) ** 2.4)
    elif color == 'rec709':
        decoded = np.where(magnitude < .081, magnitude / 4.5,
                           ((magnitude + .099) / 1.099) ** (1 / .45))
    else:
        raise ValueError(f'Unknown input color: {color}')
    return np.sign(values) * decoded


class FloatVideoDecoder:
    def __init__(self, source, *, width, height, count, ffmpeg_bin='ffmpeg',
                 color='srgb', levels='auto', start=0):
        if color not in ('srgb', 'rec709', 'linear') or levels not in ('auto', 'full', 'limited'):
            raise ValueError('Unsupported input transfer or range')
        self.width, self.height, self.count = int(width), int(height), int(count)
        self.color = color
        self._errors = tempfile.TemporaryFile()
        filters = 'zscale=matrixin=709:primariesin=709:transferin=linear:matrix=gbr:primaries=709:transfer=linear:range=full'
        if levels != 'auto':
            filters += f':rangein={levels}'
        filters += ',format=gbrpf32le'
        if start:
            filters += f',trim=start_frame={int(start)}'
        command = [str(ffmpeg_bin), '-hide_banner', '-loglevel', 'error', '-nostdin',
                   '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
                   '-vf', filters, '-fps_mode', 'passthrough', '-frames:v', str(self.count),
                   '-c:v', 'rawvideo', '-threads', '1', '-pix_fmt', 'gbrpf32le',
                   '-f', 'rawvideo', 'pipe:1']
        try:
            self.proc = cache_popen(command, stdout=subprocess.PIPE, stderr=self._errors)
        except BaseException:
            self._errors.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __iter__(self):
        size = self.width * self.height * 3 * 4
        for index in range(self.count):
            data = self.proc.stdout.read(size)
            if len(data) != size:
                self._errors.seek(0)
                error = self._errors.read().decode('utf-8', errors='replace')[-4096:]
                raise RuntimeError(f'Float decoder failed at frame {index}/{self.count}: {error}')
            gbr = np.frombuffer(data, dtype='<f4').reshape(3, self.height, self.width)
            rgb = np.moveaxis(gbr[[2, 0, 1]], 0, -1)
            if index + 1 == self.count:
                rc = self.proc.wait()
                if rc:
                    raise RuntimeError(f'Float decoder failed with exit code {rc}')
            yield decode_transfer(rgb, self.color)
        rc = self.proc.wait()
        if rc:
            raise RuntimeError(f'Float decoder failed with exit code {rc}')

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        self._errors.close()


def preprocess_linear_frame(linear, meta, *, need_model):
    if meta.get('preprocess_device') == 'cuda':
        return preprocess_linear_frame_cuda(linear, meta, need_model=need_model)
    linear = apply_linear_adjust(linear, meta)
    display = np.clip(cu.linear_to_srgb_nonnegative(np.maximum(linear, 0.0)), 0., 1.)
    h, w = meta['shape_hw']
    hw = meta['model_native_hw']
    display = resize_hwc_lanczos(display, hw)
    display, _ = center_pad_to_shape(np.clip(display, 0., 1.), h, w)
    display_t = torch.from_numpy(display.transpose(2, 0, 1)).to(torch.bfloat16)
    if not need_model:
        return None, display_t, display_t
    linear = resize_hwc_lanczos(linear, hw)
    linear, _ = center_pad_to_shape(linear, h, w)
    model = torch.from_numpy(cu.linear_to_asinh(linear).transpose(2, 0, 1)).to(torch.bfloat16)
    return model, display_t, display_t


def preprocess_linear_frame_cuda(linear, meta, *, need_model):
    """Optional GPU arithmetic; CPU mode remains the exact reference default."""
    if tuple(linear.shape[:2]) != tuple(meta['model_native_hw']):
        raise ValueError('GPU preprocessing requires native model_scale=1.0')
    host = torch.from_numpy(np.ascontiguousarray(linear.transpose(2, 0, 1))).pin_memory()
    x = host.to('cuda', non_blocking=True)
    gain = float(meta.get('linear_brightness', 1.0))
    contrast = float(meta.get('linear_contrast', 1.0))
    if abs(gain - 1) > 1e-6 or abs(contrast - 1) > 1e-6:
        pivot = float(meta.get('linear_contrast_pivot', .18))
        x = gain * (pivot + contrast * (x - pivot))
    h, w = x.shape[-2:]
    oh, ow = meta['shape_hw']
    # Reflect-101 works even when padding exceeds a tiny source dimension.
    def indices(size, target):
        if size == 1:
            return torch.zeros(target, device=x.device, dtype=torch.long)
        offsets = torch.arange(target, device=x.device) - (target - size) // 2
        offsets = offsets.remainder(2 * (size - 1))
        return torch.minimum(offsets, 2 * (size - 1) - offsets)
    x = x.index_select(1, indices(h, oh)).index_select(2, indices(w, ow))
    display = cu.linear_to_vfm_sdr_torch(x).to(torch.bfloat16)
    model = cu.linear_to_asinh(x).to(torch.bfloat16) if need_model else None
    return model, display, display


def open_decoder(source, meta):
    height, width = meta['source_hw']
    return FloatVideoDecoder(source, width=width, height=height,
                             count=meta['num_frames'], start=meta['start_frame'],
                             ffmpeg_bin=meta['ffmpeg_bin'], color=meta['input_color'],
                             levels=meta['input_range'])


class PrefetchedFrames:
    """Bounded CPU preparation ahead of the consumer, with explicit shutdown."""
    def __init__(self, decoder, meta, *, size=4):
        self.decoder, self.meta = decoder, meta
        self.q = queue.Queue(maxsize=max(1, size))
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def _worker(self):
        try:
            for frame in self.decoder:
                if self.stop.is_set():
                    break
                views = preprocess_linear_frame(frame, self.meta, need_model=True)
                if not self._put(views):
                    break
        except Exception as exc:
            if not self.stop.is_set():
                self.error = exc
        finally:
            self._put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is None:
            if self.error is not None:
                raise self.error
            raise StopIteration
        return item

    def close(self):
        self.stop.set()
        self.decoder.close()  # Break a worker blocked in the decoder pipe.
        self.thread.join()

    def finish(self):
        try:
            next(self)
        except StopIteration:
            return
        raise RuntimeError('Float decoder produced more frames than requested')


class StreamingVideoWindowReader:
    def __init__(self, source, meta):
        self.decoder = open_decoder(source, meta)
        self.frames = iter(self.decoder)
        self.prefetch = (PrefetchedFrames(self.decoder, meta, size=meta.get('prefetch_frames', 4))
                         if meta.get('preprocess_device', 'cpu') == 'cpu' else None)
        self.meta = meta
        self.cache = {}
        self.next_t = 0

    def window(self, t0, t1):
        for old in [t for t in self.cache if t < getattr(self, 'retain_from', t0)]:
            del self.cache[old]
        while self.next_t < t1:
            self.cache[self.next_t] = (next(self.prefetch) if self.prefetch is not None else
                                      preprocess_linear_frame(next(self.frames), self.meta, need_model=True))
            self.next_t += 1
            if self.next_t == self.meta['num_frames'] and self.prefetch is not None:
                self.prefetch.finish()
        model = torch.stack([self.cache[t][0] for t in range(t0, t1)])
        display = torch.stack([self.cache[t][1] for t in range(t0, t1)])
        return model, display, display

    def close(self):
        if self.prefetch is not None:
            self.prefetch.close()
        else:
            self.decoder.close()


def iter_stream_chunks(source, meta, *, chunk_size, need_model):
    with open_decoder(source, meta) as decoder:
        frames = iter(decoder)
        for t0 in range(0, meta['num_frames'], chunk_size):
            items = [preprocess_linear_frame(next(frames), meta, need_model=need_model)
                     for _ in range(min(chunk_size, meta['num_frames'] - t0))]
            yield t0, (torch.stack([x[0] for x in items]) if need_model else None), torch.stack([x[1] for x in items])
