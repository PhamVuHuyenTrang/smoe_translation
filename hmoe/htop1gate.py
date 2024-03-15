# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

from fairseq.utils import print_r0
from typing import Callable, Dict, Tuple, Optional

import math
import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn.functional as F
from .htop2gate import one_hot, entropy, one_hot_group_mask
import logging

logger = logging.getLogger(__name__)


gumbel_map: Dict[torch.device, Callable] = {}

# logging
# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 0.25

# logging
SAMPLE_FRACTION = 0.2

def htop1gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    capacity_factor=1.0,
    eval_mode=False,
    moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    batch_prioritized_routing=False,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top2Gating on logits."""
    metadata = {}
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()

    gates = logits
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    num_2nd_experts = num_experts//4
    if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
        capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
    else:
        # capacity = 2S/E
        capacity = int(2 * math.ceil(num_tokens / num_2nd_experts) * capacity_factor)

    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    mask1 = one_hot(indices1_s, num_classes=num_experts, unsqueeze_indices=True)
    if input_mask is not None and input_mask.any():
        nonpadding = ~ input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    # for logging (percent of tokens routed to each expert)
    expert1_hist = 100 * torch.histc((indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = torch.sort(expert1_hist, dim=0, descending=True).values + torch.finfo(torch.float32).tiny

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()


    gates1_s = (gates * mask1).sum(dim=1)

    # Compute locations in capacity buffer
    if batch_prioritized_routing:
        importance_scores = -1 * gates.max(dim=1)[0]
        sorted_mask1 = mask1[importance_scores.argsort(dim=0)]
        sorted_cumsum1 = (torch.cumsum(sorted_mask1, dim=0) - 1) * sorted_mask1
        importance_sorted_locations1 =  sorted_cumsum1[importance_scores.argsort(dim=0).argsort(dim=0)]
        locations1 = importance_sorted_locations1
    else:
        locations1 = torch.cumsum(mask1, dim=0) - 1

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    
    l_aux = torch.sum(me * ce)/num_2nd_experts
    l_aux = l_aux * num_experts * num_experts
    # Remove locations outside capacity from mask
    mask1 = mask1 * torch.lt(locations1, capacity)
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    # locations1_sc = num_tokens * capacity
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1), locations1_sc.to(gates1.dtype).unsqueeze(1)
    )
    dispatch_mask = combine1_sec.bool()
    if use_fp32:
        return l_aux, combine1_sec.to(orig_dtype), dispatch_mask, metadata
    else:
        return l_aux, combine1_sec, dispatch_mask, metadata


class HTop1Gate(torch.nn.Module):

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        input_noise_type=None,
        capacity_factor=1.0,
        moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
        batch_prioritized_routing=False,
        
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()
        self.wg1 = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.wg2 = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.use_fp32 = use_fp32
        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction
        self.batch_prioritized_routing=batch_prioritized_routing
        self.expert_group_size=num_experts//4
        
    def forward(self, input1: torch.Tensor=None, input2: torch.Tensor=None, mask: Optional[torch.Tensor] = None, has_tutel=False, logits:torch.Tensor=None, ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore
        if logits is None:        
            if input1 == None:
                orig_dtype = input2.dtype
                logits = self.wg2(input2).float()
                common_mlp_mask = torch.zeros((logits.shape),device=logits.device)
                common_mlp_mask[:,0] = 1
                logits_with_mask = F.softmax(logits.masked_fill(common_mlp_mask.bool(), -1e9), dim=1)
                
            elif input1.dtype != self.wg1.weight.dtype:
                input1 = input1.to(dtype=self.wg1.weight.dtype)
            else:
                group_logits = self.wg1(input1).float()
                #mask expert 0 
                common_mlp_mask = torch.zeros((group_logits.shape),device=group_logits.device)
                common_mlp_mask[:,0] = 1
                group_gates = F.softmax(group_logits.masked_fill(common_mlp_mask.bool(), -1e9), dim=1)
                group_index = torch.topk(group_gates, self.expert_group_size, dim=1, sorted=False)
                group_mask = one_hot_group_mask(group_index[1], group_gates.shape[1], unsqueeze_indices=True)
                if mask is not None and mask.any():
                    nonpadding = ~ mask
                    group_mask = group_mask * nonpadding.unsqueeze(-1).to(group_mask.dtype)
                # Compute l_aux for level one gating
                num_experts = group_gates.shape[1]
                me = torch.mean(group_gates, dim=0)
                ce = torch.mean(group_mask.to(group_gates.dtype), dim=0)
                l_aux_1st = torch.mean(me * ce)
                l_aux_1st = l_aux_1st * num_experts * num_experts

                logits = self.wg2(input2)

                orig_dtype = logits.dtype
                logits = logits.float()
                logits_with_mask = F.softmax(logits.masked_fill(~group_mask.bool(), -1e9), dim=1) 

        l_aux_2nd, combine_weights, dispatch_mask, metadata = htop1gating(
            logits_with_mask,
            mask,
            use_fp32=self.use_fp32,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            capacity_factor=self.capacity_factor,
            batch_prioritized_routing=self.batch_prioritized_routing,
        )
        l_aux = l_aux_2nd
        return l_aux, combine_weights.to(orig_dtype), dispatch_mask, metadata
