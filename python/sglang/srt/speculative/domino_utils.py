# Adapted from jianuo-huang/sglang, feat/domino-tensor-parallel,
# commit be8c97b28790a2555bcff3b30ceac9c743708964 (TP=1 rollout).
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def domino_greedy_rollout(
    *,
    draft_hidden: torch.Tensor,
    verified_ids: torch.Tensor,
    target_embedding: nn.Module,
    lm_head_weight: torch.Tensor,
    prefix_gru: nn.GRU,
    embed_proj: nn.Sequential,
    vocab_size: int,
    shift_label: bool,
    candidate_pool_size: int = 2048,
) -> torch.Tensor:
    """Author-native serving layout: block16 gives anchor + 15 proposals.

    With shifted labels, proposals read hidden positions 0:15; all 16 draft
    positions still participate in noncausal attention. The HF spec_generate
    path instead predicts 16 proposals from the same 16 hidden positions.
    """
    batch_size, block_size, hidden_size = draft_hidden.shape
    num_proposals = block_size - 1
    start = 0 if shift_label else 1
    z = draft_hidden[:, start : start + num_proposals, :]
    weight = lm_head_weight[:vocab_size]
    logits_input = (
        z.to(weight.dtype)
        .transpose(0, 1)
        .contiguous()
        .view(num_proposals * batch_size, hidden_size)
    )
    base_logits = F.linear(logits_input, weight).view(
        num_proposals, batch_size, vocab_size
    )
    first_ids = base_logits[0].argmax(dim=-1)
    proposals = [first_ids]
    if num_proposals == 1:
        return first_ids[:, None]

    candidate_ids = None
    if 0 < candidate_pool_size < vocab_size:
        feedback_logits = base_logits[1:]
        candidate_ids = torch.topk(
            feedback_logits.amax(dim=0),
            k=candidate_pool_size,
            dim=-1,
            sorted=False,
        ).indices.contiguous()
        candidate_base = torch.gather(
            feedback_logits.transpose(0, 1),
            2,
            candidate_ids[:, None, :].expand(-1, num_proposals - 1, -1),
        ).transpose(0, 1)
        candidate_weight = F.embedding(candidate_ids, embed_proj[2].weight)

    prefix_ids = torch.stack((verified_ids, first_ids), dim=1)
    _, gru_hidden = prefix_gru(target_embedding(prefix_ids))
    for index in range(1, num_proposals):
        correction_hidden = embed_proj[1](
            embed_proj[0](torch.cat((z[:, index, :], gru_hidden[0]), dim=-1))
        )
        if candidate_ids is None:
            correction = embed_proj[2](correction_hidden)
            next_ids = (base_logits[index] + correction).argmax(dim=-1)
        else:
            correction = torch.bmm(
                candidate_weight, correction_hidden.unsqueeze(-1)
            ).squeeze(-1)
            candidate_position = (candidate_base[index - 1] + correction).argmax(dim=-1)
            next_ids = torch.gather(
                candidate_ids, 1, candidate_position[:, None]
            ).squeeze(1)
        proposals.append(next_ids)
        if index + 1 < num_proposals:
            gru_hidden = torch.ops.aten.gru_cell.default(
                target_embedding(next_ids),
                gru_hidden[0],
                prefix_gru.weight_ih_l0,
                prefix_gru.weight_hh_l0,
                None,
                None,
            )[None]
    return torch.stack(proposals, dim=1)
