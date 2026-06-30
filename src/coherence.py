"""DINO embeddings, hyperplane codes, and coherence metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def compute_dino_embeds(images, device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
    dino.eval()
    if device != "cpu":
        dino = dino.to(device)

    tfm = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeds = []
    n = len(images)
    for i in range(0, n, batch_size):
        batch = torch.stack([tfm(img) for img in images[i : i + batch_size]])
        if device != "cpu":
            batch = batch.to(device)
        with torch.no_grad():
            feats = dino(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())

    return torch.cat(embeds).numpy()


def hyperplane_code(binary_np: np.ndarray, planes: list[int]) -> np.ndarray:
    cols = binary_np[:, planes]
    return (cols * (1 << np.arange(cols.shape[1]))).sum(axis=1).astype(int)


def gate_features(ff_in: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor):
    gate_img = F.gelu(ff_in @ w2.T + b2).mean(dim=1).cpu().numpy()
    pca_gate10 = PCA(n_components=10, random_state=42).fit_transform(gate_img)
    gate_feat = pca_gate10 / (np.linalg.norm(pca_gate10, axis=1, keepdims=True) + 1e-8)
    return gate_img, gate_feat


def gap_in_space(embeds, labels, min_size: int) -> float:
    labels = np.asarray(labels)
    keep = [l for l in np.unique(labels) if (labels == l).sum() > min_size]
    mask = np.isin(labels, keep)
    e, lab = embeds[mask], labels[mask]
    if len(np.unique(lab)) < 2 or len(lab) < 4:
        return -np.inf
    sim = e @ e.T
    iu = np.triu_indices(len(lab), k=1)
    same = lab[iu[0]] == lab[iu[1]]
    pair = sim[iu]
    return float(pair[same].mean() - pair[~same].mean())


def gap_filtered(embeds, labels, min_size: int = 5):
    labels = np.asarray(labels)
    keep = [l for l in np.unique(labels) if (labels == l).sum() > min_size]
    mask = np.isin(labels, keep)
    e, lab = embeds[mask], labels[mask]
    if len(np.unique(lab)) < 2 or len(lab) < 4:
        return dict(
            intra=np.nan,
            inter=np.nan,
            gap=np.nan,
            sil=np.nan,
            n_used=int(mask.sum()),
            n_groups=len(keep),
        )
    sim = e @ e.T
    iu = np.triu_indices(len(lab), k=1)
    same = lab[iu[0]] == lab[iu[1]]
    pair = sim[iu]
    intra, inter = float(pair[same].mean()), float(pair[~same].mean())
    try:
        sil = float(silhouette_score(e, lab))
    except Exception:
        sil = np.nan
    return dict(
        intra=intra,
        inter=inter,
        gap=intra - inter,
        sil=sil,
        n_used=int(mask.sum()),
        n_groups=len(keep),
    )


def select_planes_by_coherence(
    binary_np: np.ndarray,
    cand_idx: list[int],
    gate_feat: np.ndarray,
    *,
    max_planes: int = 100,
    min_region: int = 5,
    balance_min: int = 12,
    n_img: int | None = None,
) -> list[int]:
    n_img = n_img or binary_np.shape[0]
    col_pos = binary_np.sum(axis=0)
    balanced = [
        j
        for j in cand_idx
        if min(int(col_pos[j]), n_img - int(col_pos[j])) >= balance_min
    ]
    sel = []
    for step in range(max_planes):
        best_j, best_g = None, -np.inf
        for j in balanced:
            if j in sel:
                continue
            g = gap_in_space(gate_feat, hyperplane_code(binary_np, sel + [j]), min_region)
            if g > best_g:
                best_g, best_j = g, j
        sel.append(best_j)
        print(f"plane {step + 1}: neuron #{best_j}  gate-gap={best_g:+.4f}")
    print("선택된 평면(순서대로):", sel)
    return sel


def multi_resolution_comparison(
    binary_np: np.ndarray,
    sel: list[int],
    gate_img: np.ndarray,
    dino_embeds: np.ndarray,
    *,
    max_planes: int = 100,
    min_region: int = 5,
):
    n_img = len(dino_embeds)
    rng_mr = np.random.default_rng(42)
    records = []
    print(f"\n{'P':>2} {'K':>3}  {'method':18s}  {'gap':>8}  {'sil':>7}  groups   imgs")
    for p in range(1, max_planes + 1):
        k = 2 ** p
        feat_p = StandardScaler().fit_transform(gate_img[:, sel[:p]])
        labelings = {
            "Hyperplane (ours)": hyperplane_code(binary_np, sel[:p]),
            "K-means (gate)": KMeans(k, random_state=42, n_init=10).fit_predict(feat_p),
            "K-means (DINOv2)": KMeans(k, random_state=42, n_init=10).fit_predict(dino_embeds),
            "Random": rng_mr.integers(0, k, size=n_img),
        }
        for name, lab in labelings.items():
            r = gap_filtered(dino_embeds, lab, min_region)
            r.update(P=p, K=k, method=name, gap_per_bit=r["gap"] / p)
            records.append(r)
            print(
                f"{p:>2} {k:>3}  {name:18s}  {r['gap']:+.4f}  {r['sil']:+.4f}  "
                f"{r['n_groups']:>2}/{k:<2}   {r['n_used']:>3}/{n_img}"
            )

    for p in range(1, max_planes + 1):
        k = 2 ** p
        g_oracle = next(r["gap"] for r in records if r["method"] == "K-means (DINOv2)" and r["K"] == k)
        for r in records:
            if r["K"] == k:
                r["oracle_frac"] = 100.0 * r["gap"] / g_oracle if g_oracle and g_oracle > 0 else np.nan

    return records
