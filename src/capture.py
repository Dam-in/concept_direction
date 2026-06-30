"""Activation capture and inference loop."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from config import DamConfig
from paths import prompt_dir_name
from unet import (
    get_target_block,
    make_forward_activation_hook_at_step,
    make_inject_hook,
)


def save_capture_run(
    cfg: DamConfig,
    images,
    activations_per_image,
    block_name: str,
    save_dir: Path,
    *,
    meta: dict | None = None,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    images_dir = save_dir / "images"
    images_dir.mkdir(exist_ok=True)
    block_dir = save_dir / block_name.replace(".", "_")
    block_dir.mkdir(exist_ok=True)
    payload_meta = {
        "negative_prompt": cfg.negative_prompt,
        "seed_base": cfg.seed_base,
        "num_images": len(images),
        "target_block": cfg.target_block,
        "num_inference_steps": cfg.num_inference_steps,
        "capture_step": cfg.capture_step,
        "guidance_scale": cfg.guidance_scale,
        **(meta or {}),
    }
    ff_in_list, outputs_list = [], []
    for i, act in enumerate(activations_per_image):
        if act is None:
            raise ValueError(f"activations_per_image[{i}] is None")
        ff_in_list.append(act["input"])
        outputs_list.append(act["output"])
        images[i].save(images_dir / f"{i:04d}.png")
    ff_in_stacked = torch.stack(ff_in_list)
    outputs_stacked = torch.stack(outputs_list)
    torch.save(
        {"inputs": ff_in_stacked, "outputs": outputs_stacked},
        block_dir / "activations.pt",
    )
    with open(block_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(payload_meta, f, indent=2, ensure_ascii=False)
    print(f"saved → {save_dir}")
    print(f"  images: {len(images)} | activations: {ff_in_stacked.shape}")
    return save_dir


def run_capture_loop(pipe, cfg: DamConfig, device: str) -> Path:
    """Generate images and save FFN + inject-block activations."""
    generator = torch.Generator(device="cpu")
    ff_block = get_target_block(pipe.unet, cfg.target_block)
    inject_block = get_target_block(pipe.unet, cfg.inject_block)

    ff_hook, ff_state = make_forward_activation_hook_at_step(cfg.capture_step)
    ff_handle = ff_block.register_forward_hook(ff_hook)
    inject_hook, inject_state = make_inject_hook(cfg.capture_step)
    inject_handle = inject_block.register_forward_hook(inject_hook)

    save_root = cfg.data_root_concept
    save_root.mkdir(parents=True, exist_ok=True)

    for p_idx, prompt in enumerate(cfg.prompts):
        images, ff_acts, inject_acts = [], [], []
        pdir = save_root / prompt_dir_name(p_idx, prompt)

        for i in tqdm(range(cfg.num_images_per_prompt), desc=f"prompt {p_idx}: {prompt[:40]}..."):
            seed = cfg.seed_base + p_idx * cfg.num_images_per_prompt + i
            generator.manual_seed(seed)

            ff_state["forward_i"] = -1
            ff_state["value"] = None
            inject_state["forward_i"] = -1
            inject_state["value"] = None

            result = pipe(
                prompt=prompt,
                negative_prompt=cfg.negative_prompt,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                num_images_per_prompt=1,
                generator=generator,
                output_type="pil",
            )
            img = result["images"][0] if isinstance(result, dict) else result.images[0]
            images.append(img)
            ff_acts.append(ff_state["value"])
            inject_acts.append(inject_state["value"])

        save_capture_run(
            cfg,
            images,
            ff_acts,
            cfg.target_block.replace(".", "_"),
            pdir,
            meta={"prompt": prompt, "prompt_idx": p_idx},
        )

        inject_dir = pdir / cfg.inject_block.replace(".", "_")
        inject_dir.mkdir(exist_ok=True)
        inject_stacked = torch.stack(inject_acts)
        inject_meta = {
            "negative_prompt": cfg.negative_prompt,
            "seed_base": cfg.seed_base,
            "num_images": len(images),
            "inject_block": cfg.inject_block,
            "capture_step": cfg.capture_step,
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "prompt": prompt,
            "prompt_idx": p_idx,
        }
        torch.save({"outputs": inject_stacked}, inject_dir / "activations.pt")
        with open(inject_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(inject_meta, f, indent=2, ensure_ascii=False)
        print(f"  inject_block acts: {tuple(inject_stacked.shape)} → {inject_dir}")

    ff_handle.remove()
    inject_handle.remove()

    with open(save_root / "prompts.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompts": cfg.prompts,
                "num_images_per_prompt": cfg.num_images_per_prompt,
                "seed_base": cfg.seed_base,
                "target_block": cfg.target_block,
                "inject_block": cfg.inject_block,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"done: {cfg.num_prompts} prompts × {cfg.num_images_per_prompt} images → {save_root}")
    return save_root
