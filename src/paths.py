"""Data directory path helpers."""

from __future__ import annotations

from pathlib import Path

from config import DamConfig


def prompt_dir_name(prompt_idx: int, prompt: str) -> str:
    slug = prompt.lower().replace(" ", "_").replace("/", "_")[:40]
    return f"prompt_{prompt_idx:02d}_{slug}"


def resolve_pdir(cfg: DamConfig, prompt_idx: int, prompt: str) -> Path:
    """
    데이터 로드 경로 자동 탐색 (for 호환)
    Layout A (run_concept.py):  data/{concept}/prompt_xx_.../images/
    Layout B (notebook multi):  data/{concept}/{block}/prompt_xx_.../images/
    """
    name = prompt_dir_name(prompt_idx, prompt)
    for cand in (cfg.data_root_concept / cfg.block_tag / name, cfg.data_root_concept / name):
        if (cand / "images").is_dir():
            return cand
    raise FileNotFoundError(
        "prompt dir not found for %r. tried:\n  %s\n  %s"
        % (name, cfg.data_root_concept / cfg.block_tag / name, cfg.data_root_concept / name)
    )
