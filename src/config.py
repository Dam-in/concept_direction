"""YAML config loader with derived paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DamConfig:
    concept: str
    prompts: list[str]
    num_images_per_prompt: int = 100
    seed_base: int = 1000
    target_block: str = "up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0"
    inject_block: str = "up_blocks.1"
    num_inference_steps: int = 50
    capture_step: int = 10
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    eta: float = 0.0
    data_root: Path = field(default_factory=lambda: Path("/project/dam/data"))

    concept_slug: str = field(init=False)
    block_tag: str = field(init=False)
    data_root_concept: Path = field(init=False)
    save_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.concept_slug = self.concept.replace(" ", "_")
        self.prompts = [p.format(concept=self.concept) for p in self.prompts]
        self.block_tag = self.inject_block.replace(".", "_")
        self.data_root_concept = self.data_root / self.concept_slug
        self.save_dir = self.data_root_concept / self.block_tag
        self.cache_dir = self.data_root_concept / self.block_tag / "_cache"

    @property
    def num_prompts(self) -> int:
        return len(self.prompts)

    @property
    def num_images_total(self) -> int:
        return self.num_prompts * self.num_images_per_prompt

    @property
    def prompt(self) -> str:
        return self.prompts[0]


def load_config(path: str | Path) -> DamConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    if "data_root" in raw:
        raw["data_root"] = Path(raw["data_root"])
    return DamConfig(**raw)


def expose_legacy_vars(cfg: DamConfig) -> dict[str, Any]:
    """Notebook 호환용 전역 변수 dict."""
    return {
        "CONCEPT": cfg.concept_slug,
        "PROMPTS": cfg.prompts,
        "PROMPT": cfg.prompt,
        "NUM_PROMPTS": cfg.num_prompts,
        "NUM_IMAGES_PER_PROMPT": cfg.num_images_per_prompt,
        "NUM_IMAGES_TOTAL": cfg.num_images_total,
        "SEED_BASE": cfg.seed_base,
        "TARGET_BLOCK": cfg.target_block,
        "INJECT_BLOCK": cfg.inject_block,
        "DATA_ROOT": cfg.data_root_concept,
        "BLOCK_TAG": cfg.block_tag,
        "SAVE_DIR": cfg.save_dir,
        "NUM_INFERENCE_STEPS": cfg.num_inference_steps,
        "CAPTURE_STEP": cfg.capture_step,
        "GUIDANCE_SCALE": cfg.guidance_scale,
        "NEGATIVE_PROMPT": cfg.negative_prompt,
        "ETA": cfg.eta,
    }
