# GPU Backend Validation Implementation Plan

> Execute inline with superpowers:executing-plans. Use one independent read-only review at the end.

**Goal:** Reduce GPU inference time without changing the accepted model predictions, color pipeline or CLI selections.
**Architecture:** Benchmark the current eager/BF16 Stage 2 against available SDPA implementations, cuDNN convolution tuning, and native CUDA Graph replay before adding production behavior. Prefer replay of existing kernels over numerical changes. Keep only measured improvements that pass exact regression comparisons.
**Tech stack:** Installed PyTorch 2.9.1+cu128, CUDA/cuDNN, RTX 3090, existing portable Python/FFmpeg. No replacement runtime or dependency upgrade is assumed.
**Spec:** User's explicit instruction to benchmark faster GPU backends, retain only quality-preserving improvements, back up first and push validated work directly to fork `master`. Existing CLI-selection behavior is part of the compatibility contract.

## Constraints and review focus

- Backup `codex/backup-before-gpu-backends-2026-10-27` preserves `9984b58` before edits.
- Keep native resolution, BF16 autocast, FP16 baton boundary, Hann windows, hints, RVM chronology, model weights and encoder settings.
- Reference/OFF must retain the original predictions. For subsequently user-authorized speed/quality choices, quantify numerical differences and warn explicitly; finite output alone is not a quality-equivalence claim.
- Graph replay requires stable addresses/shapes, fresh input copies, bounded memory, independent output lifetime and invalidation across placement/config changes. Low-VRAM mode and model reuse must remain safe.
- All guided selections must still reach `InferenceConfig`: color/range, hint mode/trust/carry, outputs, video/EXR and EXR precision, cutout color, despill, memory settings, frame count and sequence FPS.
- Do not install an alternate backend merely to claim support. Document unavailable candidates and primary-source constraints.

## Tasks

- [x] Benchmark actual-checkpoint 4K Stage 2: eager/default, forced efficient, cuDNN SDPA, cuDNN tuning, CUDA Graph replay; record first-use cost, repeated timings, memory and exactness on changing inputs.
- [x] Following the revised user instruction, add default-ON cuDNN ON/OFF selection and default-OFF numerical tradeoffs. Write failing configuration/restoration/menu tests, implement scoped policies, and run the full suite. Discard unhelpful graph/metadata-cache production changes.
- [x] Measure clean completed exports separately from setup/numerical checks. Compare reference with original raw goldens; check encoded output restoration and low-memory/hint/HDR cases.
- [x] Validate guided selections through the real parser/runtime boundary and run actual guided jobs; support legacy interactive cuDNN/native selections too.
- [ ] Independent review; resolve findings, fresh verification, document results, commit and push fork `master`, verify remote SHA.

## Decisions

- Ruling: user explicitly authorized the research/implementation and publication, with previously established quality constraints. Proceed without another design-approval round.
- Ruling: exact predictions are the acceptance gate because there is no ground-truth camera-footage quality corpus that would justify accepting alternate rounding as equivalent matte quality.
- Updated user instruction supersedes that gate for explicit speed/quality choices: add a guided cuDNN ON/OFF selection, default ON, warning that quality may degrade. OFF preserves reference behavior. Other measured tradeoffs may be offered as labeled opt-ins, default OFF. Do not describe these numerical changes as quality-preserving.
- CUDA Graph root cause: grouped attention creates CUDA integer indices from CPU tuples during capture. An experimental bounded device-index cache fixes capture, and nine changing-input comparisons remain exact. However, medians are 1002.02 ms before, 999.81 ms cached, 998.06 ms replay (including copies/output clone). This is below timing variation. Ruling: discard the production cache/graph adapter; preserve only the reproducible research spike.
- Convolution benchmark correction: `torch.backends.cudnn.flags(benchmark=True)` defaults `enabled=False`; the first harness unintentionally tested native convolutions. The corrected tuning context preserves enabled/deterministic/precision settings. Corrected autotuning is exact in this probe but 996.90 vs 1002.86 ms is below noise; do not retain a tuning option. Benchmark native convolutions explicitly before deciding whether that separate tradeoff is useful.
- Native convolutions independently measured 892.72 ms versus 1002.86 ms; retained as opt-in. Real-pipeline warm Stage 2: reference1002.02 ms, cuDNN891.11 ms, native897.93 ms, combined794.54 ms.
- Clean warm 12-frame exports: reference43.48/43.52 s; cuDNN42.26; GPU preprocessing41.45; native42.07; feature reuse43.46 (neutral); all38.30. Original reference matches401,080,320 numerical values. Tradeoff alpha/foreground differences are quantified in GPU_BACKENDS.md, with synthetic-input limitations.
- CLI compatibility: 32 unit tests pass; seven actual guided jobs share one model, covering all outputs, low-memory video, HDR float/half EXRs, first/full hints and options. All four reference encoded files are identical before and after those job/configuration transitions.
- Independent review: stale aggregate evidence corrected from per-backend records; follow-up reports no remaining actionable findings. Documentation and validated changes are ready for Git publication to fork master.
