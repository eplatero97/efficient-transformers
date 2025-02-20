# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from QEfficient.transformers.cache_utils import QEffDynamicCache

def filter_hidden_states(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    num_logits_to_keep: Optional[int] = None,
) -> torch.Tensor:
    """
    Filter hidden states based on whether this is a TLM SpD model

    ``Mandatory`` Args:
        :hidden_states (torch.Tensor): Hidden states tensor.
        :position_ids (torch.Tensor): Position ids tensor.
    ``Optional`` Args:
        :num_logits_to_keep (int, optional): Number of speculative tokens, specified only for TLM SpD model

    Returns:
        :torch.Tensor: Filtered hidden states.
    """
    batch_size = position_ids.size(0)
    batch_indices = torch.arange(batch_size)
    # Cast to INT32 to avoid issue while running in ONNXRT
    logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)

    if num_logits_to_keep is None:
        # return the last logit
        return hidden_states[batch_indices.view(-1, 1), logit_index]

    # gather approach
    num_logits_to_keep = num_logits_to_keep.shape[0]
    lower_idx = torch.where(logit_index < num_logits_to_keep, 0, logit_index + 1 - num_logits_to_keep).view(
        -1, 1
    )  # shape: [bsz, 1]
    spec_idx = torch.arange(num_logits_to_keep).view(1, -1)  # shape: [1, k]
    indices = torch.add(lower_idx, spec_idx).unsqueeze(2)  # shape: [bsz, k, 1]
    indices = indices.repeat(1, 1, hidden_states.size(-1))  # shape: [bsz, ,k, d_model]
    hidden_states = torch.gather(hidden_states, dim=1, index=indices)  # shape: [bsz, k, d_model]
    return hidden_states

def select_best_candidates(spec_candidates: torch.Tensor, target_candidates: torch.Tensor):
    bsz = spec_candidates.size(0)
    num_speculative_tokens = spec_candidates.size(2)-1
    batch_indices = torch.arange(bsz)
    posterior_mask = spec_candidates[:, :, 1:] == target_candidates[:, :, :num_speculative_tokens] # shape: [decode_batch_size, leaf_nodes, num_speculative_tokens]
    #candidates_accept_length = torch.cumprod(posterior_mask, dim=2).sum(dim=2) # cumprod op not supported by onnx, shape: [decode_batch_size, leaf_nodes]
    #min_indices = torch.argmin(posterior_mask.to(torch.int64), dim=2) # not supported by qaic optimization, shape: [decode_batch_size, leaf_nodes]
    #min_indices[(min_indices==0) & (posterior_mask.sum(dim=2)==num_speculative_tokens)] = num_speculative_tokens
    #candidates_accept_length = min_indices
    cumsum_mask = posterior_mask.to(torch.int64).cumsum(dim=2) == (torch.arange(num_speculative_tokens, dtype=torch.int64)+1)
    candidates_accept_length = cumsum_mask.sum(dim=2)
    num_tokens_selected = candidates_accept_length.max(dim=1).values + 1 # shape: [deode_batch_size]
    best_path_indices = torch.argmax(candidates_accept_length, dim=1) # shape: [decode_batch_size]
    best_tlm_paths = target_candidates[batch_indices, best_path_indices, :] # target_tokens, shape: [decode_batch_size, num_speculative_tokens+1]
    return best_tlm_paths, best_path_indices, num_tokens_selected, 

def generate_sequential_and_dispersed_pids(position_ids, retrieve_best_indices):
    bsz, n_nodes = position_ids.size()
    num_speculative_tokens = retrieve_best_indices.size(1) - 1 
    root_pids = position_ids[:, 0] # shape: [decode_batch_size]
    tree_seq_pids = torch.arange(1, n_nodes, dtype=torch.int32).view(1,-1).repeat(bsz, 1) + root_pids # shape: [decode_batch_size, n_nodes-1]
    aligned_pids = tree_seq_pids[:, :num_speculative_tokens] # ctx_len indices that will be updated, shape: [decode_batch_size, num_speculative_tokens]
    no_root_best_indices = retrieve_best_indices[:, 1:] - 1 # retrieve only non-root indices, shape: [decode_batch_size, num_speculative_tokens]
    dispersed_pids = torch.gather(tree_seq_pids, 1, no_root_best_indices) # ctx_len indices that will be extracted, shape: [decode_batch_size, num_speculative_tokens]
    return aligned_pids, dispersed_pids

def tlm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    batch_index: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: Optional[torch.LongTensor] = None,
    retrieve_indices: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    print(f"{position_ids=}")

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        batch_index=batch_index,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = filter_hidden_states(outputs[0], position_ids, num_logits_to_keep)
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)
    logits = logits.float()
    topk_logits: Optional[int] = getattr(self, "topk_logits", None)
    if topk_logits:
        if topk_logits == 1:
            logits: torch.LongTensor = logits.argmax(dim=2, keepdim=True)
        else:
            logits: torch.LongTensor = logits.topk(k=topk_logits, dim=2).indices
    if retrieve_indices is not None:
        assert hasattr(position_ids, "is_tree") and position_ids.is_tree, f"position ids has no `is_tree` attribute or it's False"
        if isinstance(past_key_values, (list, tuple)):
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)
        if topk_logits is None:
            # extract candidates
            target_tree_tokens = logits.argmax(dim=2, keepdim=True)
        else:
            target_tree_tokens = logits[:, :, 0:1]
        target_candidates = target_tree_tokens[:, retrieve_indices, 0] # shape: [bsz, *retrieve_indices.size()]
        spec_candidates = input_ids[:, retrieve_indices] # shape: [bsz, *retrieve_indices.size()]
        print(f"{target_tree_tokens=}")
        print(f"{target_candidates=}")
        print(f"{spec_candidates=}")
        # select best candidate
        (best_tlm_paths, # shape: [bsz, num_speculative_tokens+1]
            best_path_indices, # shape: [bsz]
            num_tokens_selected, # shape: [bsz]
            ) = select_best_candidates(spec_candidates, target_candidates)
        retrieve_best_indices = retrieve_indices[best_path_indices] # shape: [bsz, num_speculative_tokens+1]
        print(f"{num_tokens_selected=}")
        print(f"{best_path_indices=}")
        print(f"{retrieve_best_indices=}")
        # generate sequential and dispersed pids
        #position_ids = position_ids.to(torch.int32)
        #retrieve_best_indices = retrieve_best_indices.to(torch.int32) # gather expects dtype int64 for index
        aligned_pids, dispersed_pids = generate_sequential_and_dispersed_pids(position_ids, retrieve_best_indices)
        print(f"{aligned_pids=}")
        print(f"{dispersed_pids=}")
        #aligned_pids = aligned_pids.to(torch.int32)
        #dispersed_pids = dispersed_pids.to(torch.int32)
        # align kv$
        cache_kwargs = {"batch_index": batch_index}
        for decode_layer_idx in range(len(self.model.layers)):
            past_key_values.align(
                dispersed_pids, 
                aligned_pids, 
                decode_layer_idx,
                cache_kwargs)
        # prepare output variables
        logits = torch.cat([best_tlm_paths, num_tokens_selected.view(-1,1), best_path_indices.view(-1,1)], dim=1)
        past_key_values = past_key_values.to_legacy_cache()

    return CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
