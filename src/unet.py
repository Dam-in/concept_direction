"""UNet block navigation and forward hooks."""

from __future__ import annotations

import torch


def get_target_block(unet, target_block_name: str):
    parts = target_block_name.split(".")

    def resolve(obj, remaining):
        if not remaining:
            return obj
        part, *rest = remaining
        child = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return resolve(child, rest)

    return resolve(unet, parts)


def tensor_cond_batch_slice(t: torch.Tensor) -> torch.Tensor:
    if t.dim() >= 1 and t.shape[0] >= 2:
        return t[1]
    return t[0]


def capture_activation(x):
    if isinstance(x, torch.Tensor):
        return tensor_cond_batch_slice(x).clone().detach().cpu()
    if isinstance(x, (tuple, list)):
        tensors = [
            tensor_cond_batch_slice(t).clone().detach().cpu()
            for t in x
            if isinstance(t, torch.Tensor)
        ]
        if len(tensors) == 1:
            return tensors[0]
        return tensors
    return x


def make_forward_activation_hook_at_step(capture_step: int):
    state = {"forward_i": -1, "value": None}

    def hook(module, input, output):
        state["forward_i"] += 1
        if state["forward_i"] != capture_step:
            return
        state["value"] = {
            "input": capture_activation(input),
            "output": capture_activation(output),
        }

    return hook, state


def make_inject_hook(capture_step: int):
    state = {"forward_i": -1, "value": None}

    def hook(module, input, output):
        state["forward_i"] += 1
        if state["forward_i"] != capture_step:
            return
        out = output[0] if isinstance(output, tuple) else output
        state["value"] = tensor_cond_batch_slice(out).clone().detach().cpu()

    return hook, state
