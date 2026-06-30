"""Notebook bootstrap: add src to path and load config."""

from __future__ import annotations

import sys
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    start = start or Path.cwd().resolve()
    for p in [start, *start.parents]:
        if (p / "src" / "config.py").is_file() and (p / "config").is_dir():
            return p
    raise FileNotFoundError("forGithub project root not found (expected src/config.py + config/)")


def setup_notebook(config_name: str = "dog.yaml"):
    root = find_project_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from config import expose_legacy_vars, load_config
    from paths import prompt_dir_name as _prompt_dir_name
    from paths import resolve_pdir as _resolve_pdir
    from unet import get_target_block

    cfg_path = root / "config" / config_name
    cfg = load_config(cfg_path)
    legacy = expose_legacy_vars(cfg)

    def prompt_dir_name(prompt_idx: int, prompt: str) -> str:
        return _prompt_dir_name(prompt_idx, prompt)

    def resolve_pdir(prompt_idx: int, prompt: str):
        return _resolve_pdir(cfg, prompt_idx, prompt)

    legacy["prompt_dir_name"] = prompt_dir_name
    legacy["resolve_pdir"] = resolve_pdir
    legacy["get_target_block"] = get_target_block
    return root, cfg, legacy
