"""Concept direction and injection steering."""

from __future__ import annotations

import numpy as np
import torch

from unet import get_target_block


def concept_direction_from_plane_sign(
    inject_out: torch.Tensor,
    plane_sign: np.ndarray,
) -> torch.Tensor:
    pos = np.where(plane_sign == 1)[0]
    neg = np.where(plane_sign == 0)[0]
    act_pos = inject_out[pos].mean(dim=0)
    act_neg = inject_out[neg].mean(dim=0)
    return act_pos - act_neg


def concept_direction_from_region_mask(
    inject_out: torch.Tensor,
    mask_region: np.ndarray,
) -> torch.Tensor:
    pos = np.where(mask_region)[0]
    neg = np.where(~mask_region)[0]
    act_pos = inject_out[pos].mean(dim=0) if len(pos) > 0 else torch.zeros_like(inject_out[0])
    act_neg = inject_out[neg].mean(dim=0) if len(neg) > 0 else torch.zeros_like(inject_out[0])
    return act_pos - act_neg


def _steering_delta_chw(d_np, step_idx, c, h, w):
    d = np.asarray(d_np, dtype=np.float32)
    if d.shape == (c, h, w):
        return d
    if d.ndim == 4 and d.shape[1:] == (c, h, w):
        return d[min(step_idx, d.shape[0] - 1)]
    raise ValueError(f"concept_direction {d.shape}, expected ({c},{h},{w})")


def make_forward_steering_hook(direction, alpha, num_steps):
    d_np = np.asarray(direction, dtype=np.float32)
    step = [0]

    def _hook(_module, _inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        si = step[0]
        step[0] += 1
        _, c, h, w = out.shape
        dt = torch.as_tensor(
            _steering_delta_chw(d_np, si, c, h, w),
            device=out.device,
            dtype=out.dtype,
        )
        alpha_t = alpha * (1.0 - si / max(num_steps - 1, 1))
        new_out = out.clone()
        if new_out.shape[0] >= 2:
            new_out[1] = out[1] + alpha_t * dt
        else:
            new_out = out + alpha_t * dt
        return (new_out,) + tuple(output[1:]) if isinstance(output, tuple) else new_out

    return _hook


def generate_with_concept_direction(
    pipe,
    cfg,
    device: str,
    prompt: str,
    seed: int,
    direction,
    alpha: float,
    num_steps: int | None = None,
):
    n_steps = num_steps or cfg.num_inference_steps
    pipe.scheduler.set_timesteps(n_steps, device=device)
    actual_steps = len(pipe.scheduler.timesteps)
    blk = get_target_block(pipe.unet, cfg.inject_block)
    blk._forward_hooks.clear()
    handle = blk.register_forward_hook(
        make_forward_steering_hook(direction, alpha, actual_steps)
    )
    try:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=cfg.negative_prompt,
                num_inference_steps=n_steps,
                guidance_scale=cfg.guidance_scale,
                generator=gen,
            )
    finally:
        handle.remove()
    return result.images[0] if hasattr(result, "images") else result["images"][0]
