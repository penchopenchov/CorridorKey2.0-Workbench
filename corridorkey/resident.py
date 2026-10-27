"""Bounded intermediates for normal-memory inference."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class BatonAccumulator:
    def __init__(self):
        self.values = {}
        self.weights = {}

    def __len__(self):
        return len(self.values)

    def add(self, t0, baton, weights):
        weighted = baton.float() * weights.to(baton.device).view(-1, 1, 1, 1)
        for j, value in enumerate(weighted):
            k = t0 + j
            if k in self.values:
                self.values[k].add_(value)
                self.weights[k] += float(weights[j])
            else:
                self.values[k] = value.clone()
                self.weights[k] = float(weights[j])

    def flush_before(self, limit):
        for k in sorted(k for k in self.values if k < limit):
            divisor = float(np.float32(max(self.weights.pop(k), 1e-6)))
            # CUDA float32 division uses an approximate reciprocal. Computing
            # the quotient in double then rounding to float32 matches NumPy's
            # float32 division before the reference FP16 boundary.
            value = (self.values.pop(k).double() / divisor).float()
            # This is intentionally FP16, matching NumPy's previous boundary.
            yield k, value.to(torch.float16)


class FoundationCache:
    def __init__(self):
        self.frames = {}

    def window(self, t0, display, extract):
        for old in [k for k in self.frames if k < t0]:
            del self.frames[old]
        missing = [j for j in range(len(display)) if t0 + j not in self.frames]
        if missing:
            features = extract(display[missing])
            for offset, j in enumerate(missing):
                self.frames[t0 + j] = {key: value[offset].clone() for key, value in features.items()}
        keys = self.frames[t0].keys()
        return {key: torch.stack([self.frames[t0 + j][key] for j in range(len(display))]) for key in keys}


class ContinuousRVM:
    """Retain reference prepass chunking/state, without a full-clip disk pass."""
    def __init__(self, model, reader, meta, chunk):
        self.model, self.reader, self.meta = model, reader, meta
        self.chunk = chunk
        self.next_t = 0
        self.state = None
        self.features = {}
        self.mean = model.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
        self.std = model.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)

    def window(self, t0, t1):
        rvm = self.model.foundations.rvm
        device = next(rvm.parameters()).device
        H, W = self.meta['shape_hw']
        while self.next_t < t1:
            end = min(self.next_t + self.chunk, self.meta['num_frames'])
            _, display, _ = self.reader.window(self.next_t, end)
            T = end - self.next_t
            rgb = display.unsqueeze(0).to(device, non_blocking=True).float()
            video = F.interpolate(rgb.reshape(T, 3, H, W), size=(H // 4, W // 4),
                                  mode='bilinear', align_corners=False)
            video = video.reshape(1, T, 3, H // 4, W // 4).clamp(0., 1.)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                feats, self.state = rvm(video, state=self.state)
            # Keep the reference CPU normalization/FP16 rounding exactly.
            normalized = (feats.float().cpu() - self.mean) / self.std
            quantized = torch.from_numpy(normalized[0].numpy().astype(np.float16)).to(device)
            for j in range(T):
                self.features[self.next_t + j] = quantized[j].clone()
            self.next_t = end
        for old in [k for k in self.features if k < t0]:
            del self.features[old]
        return torch.stack([self.features[k] for k in range(t0, t1)]).unsqueeze(0).float()
