"""Float video decoding without intermediate image files.

The filters and transfer equations deliberately match the guided launcher's
reference EXR path. CPU preprocessing preserves its rounding boundaries.
"""
from __future__ import annotations

import subprocess
import tempfile

import numpy as np
import torch

from cli_cache import cache_popen
from . import color_utils as cu
from .video_io import center_pad_to_shape, resize_hwc_lanczos


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
            yield decode_transfer(rgb, self.color)
        rc = self.proc.wait()
        if rc:
            raise RuntimeError(f'Float decoder failed with exit code {rc}')

    def close(self):
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._errors.close()


def preprocess_linear_frame(linear, meta, *, need_model):
    display = np.clip(cu.linear_to_srgb(np.maximum(linear, 0.0)), 0., 1.)
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


def open_decoder(source, meta):
    height, width = meta['source_hw']
    return FloatVideoDecoder(source, width=width, height=height,
                             count=meta['num_frames'], start=meta['start_frame'],
                             ffmpeg_bin=meta['ffmpeg_bin'], color=meta['input_color'],
                             levels=meta['input_range'])


class StreamingVideoWindowReader:
    def __init__(self, source, meta):
        self.decoder = open_decoder(source, meta)
        self.frames = iter(self.decoder)
        self.meta = meta
        self.cache = {}
        self.next_t = 0

    def window(self, t0, t1):
        for old in [t for t in self.cache if t < t0]:
            del self.cache[old]
        while self.next_t < t1:
            self.cache[self.next_t] = preprocess_linear_frame(next(self.frames), self.meta, need_model=True)
            self.next_t += 1
        model = torch.stack([self.cache[t][0] for t in range(t0, t1)])
        display = torch.stack([self.cache[t][1] for t in range(t0, t1)])
        return model, display, display

    def close(self):
        self.decoder.close()


def iter_stream_chunks(source, meta, *, chunk_size, need_model):
    with open_decoder(source, meta) as decoder:
        frames = iter(decoder)
        for t0 in range(0, meta['num_frames'], chunk_size):
            items = [preprocess_linear_frame(next(frames), meta, need_model=need_model)
                     for _ in range(min(chunk_size, meta['num_frames'] - t0))]
            yield t0, (torch.stack([x[0] for x in items]) if need_model else None), torch.stack([x[1] for x in items])
