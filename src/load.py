"""Load saved images and activations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import DamConfig
from paths import resolve_pdir


def load_images_and_ff_in(cfg: DamConfig):
    images, ff_in_list, prompt_index = [], [], []

    for p_idx, prompt in enumerate(cfg.prompts):
        pdir = resolve_pdir(cfg, p_idx, prompt)
        n = cfg.num_images_per_prompt
        images.extend([Image.open(pdir / "images" / f"{i:04d}.png") for i in range(n)])
        act = torch.load(
            pdir / cfg.target_block.replace(".", "_") / "activations.pt",
            weights_only=True,
        )
        try:
            ff_in_list.append(act["inputs"][:n].cpu().float())
        except Exception:
            ff_in_list.append(act["ff_in"][:n].cpu().float())
        prompt_index.extend([p_idx] * n)

    ff_in = torch.cat(ff_in_list, dim=0)
    prompt_index = np.array(prompt_index, dtype=np.int32)
    return images, ff_in, prompt_index


def load_inject_out(cfg: DamConfig) -> torch.Tensor:
    inject_list = []
    for p_idx, prompt in enumerate(cfg.prompts):
        pdir = resolve_pdir(cfg, p_idx, prompt)
        n = cfg.num_images_per_prompt
        data = torch.load(
            pdir / cfg.inject_block.replace(".", "_") / "activations.pt",
            weights_only=True,
        )
        inject_list.append(data["outputs"][:n].cpu().float())
    return torch.cat(inject_list, dim=0)


def save_cache(cfg: DamConfig, name: str, **tensors) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.cache_dir / f"{name}.pt"
    torch.save(tensors, path)
    print(f"cache saved → {path}")
    return path


def load_cache(cfg: DamConfig, name: str) -> dict:
    path = cfg.cache_dir / f"{name}.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"cache not found: {path}\nRun the previous analysis notebook first."
        )
    data = torch.load(path, weights_only=False)
    print(f"cache loaded ← {path}  keys={list(data.keys())}")
    return data
