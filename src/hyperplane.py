"""FFN weight extraction and hyperplane computations."""

from __future__ import annotations

import torch

from unet import get_target_block


def extract_w2_b2(pipe, target_block: str):
    ff = get_target_block(pipe.unet, target_block)
    ff_state = ff.state_dict()
    w_geglu = ff_state["proj.weight"].cpu().float()
    n_half = w_geglu.shape[0] // 2
    w2 = w_geglu[n_half:]
    b_geglu = ff_state["proj.bias"].cpu().float()
    b2 = b_geglu[n_half:]
    return w2, b2


def compute_top_k_neurons(ff_in: torch.Tensor, w2: torch.Tensor, k: int = 100):
    n, seq_len, dim = ff_in.shape
    side = int(seq_len ** 0.5)
    assert side * side == seq_len

    h_3d = ff_in.reshape(n, side, side, dim)
    mean_map = h_3d.mean(dim=-1).reshape(n, seq_len)
    neuron_acts = ff_in @ w2.T
    similarity = (neuron_acts * mean_map.unsqueeze(-1)).sum(dim=1)
    similarity_mean = similarity.mean(dim=0)
    top_k_idx = similarity_mean.topk(k).indices
    return top_k_idx, side


def compute_signed_dist(ff_in: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor, top_k_idx=None):
    w = w2[top_k_idx] if top_k_idx is not None else w2
    b = b2[top_k_idx] if top_k_idx is not None else b2
    signed_dist = ff_in @ w.T + b
    return signed_dist, w, b


def compute_binary(ff_in: torch.Tensor, w2: torch.Tensor, b2: torch.Tensor):
    h_mean = ff_in.mean(dim=1)
    pre_acts = h_mean @ w2.T + b2
    binary = (pre_acts > 0).int()
    return binary, pre_acts, h_mean


def candidate_neurons(binary: torch.Tensor, k_cand: int = 500):
    bin_var = binary.float().var(dim=0)
    return bin_var.topk(k_cand).indices.tolist()
