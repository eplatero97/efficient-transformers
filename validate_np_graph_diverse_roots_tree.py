# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pprint import pprint
from time import perf_counter
from typing import List, Optional, Tuple, Union
import pickle

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoConfig

from QEfficient.utils._utils import get_padding_shape_from_config
from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.utils.constants import Constants
from QEfficient.utils.tree_attn_utils import create_4d_causal_mask, generate_medusa_buffers, generate_candidates, TOPK

RS_SUFFIX = "_RetainedState"

@dataclass
class PerfMetrics:
    """
    Holds all performance metrics

    Args:
        :mean_ttft (float): Average TLM+DLM TTFT.
        :batch_ttft (float): Total TLM+DLM Batch TTFT.
        :decode_throughput (float): Decode throughput.
        :e2e_throughput (float): E2E throughput.
        :mean_num_accepted_tokens (float): Average number of accepted tokens.
        :max_gen_len (int): Max generation length.
        :generated_tokens_per_prompt (List[int]): Total generated tokens per prompt.
        :freq (List[List[int]]): Frequency matrix that records chosen path and number of accepted tokens
    """

    mean_ttft: float
    batch_ttft: float
    decode_throughput: float
    e2e_throughput: float
    mean_num_accepted_tokens: float
    max_gen_len: int
    generated_tokens_per_prompt: List[int]
    freq: List[List[int]]


@dataclass
class CloudAI100ExecInfo:
    """
    Holds all the information about Cloud AI 100 execution

    Args:
        :prompts (List[str]): Prompts to perfrom inferencing on.
        :batch_size (int): Batch size of the QPC compilation.
        :generated_texts (Union[List[List[str]], List[str]]): Generated text(s).
        :generated_ids (Union[List[np.ndarray], np.ndarray]): Generated IDs.
        :perf_metrics (PerfMetrics): Performance metrics.
        :num_speculative_tokens (int): Number of speculative tokens.
        :prefill_seq_len (int): Prefill sequence length.
        :ctx_len (int): Context length.
        :prefill_bsz (int): Prefill batch size.
        :draft_model_name (str): Draft model name.
        :target_model_name (str): Target model name.
        :full_batch_size (Optional[int]): Full batch size.
    """

    prompts: List[str]
    batch_size: int
    generated_texts: Union[List[str], List[List[str]]]
    generated_ids: Union[List[np.ndarray], np.ndarray]
    perf_metrics: PerfMetrics
    num_speculative_tokens: int
    ctx_len: int
    draft_model_name: str
    target_model_name: str
    iterations: int # total number of iterations

    def __repr__(self):
        return (
            f"Avg TLM+DLM TTFT = {round(self.perf_metrics.mean_ttft, 2)}\n"
            f"Total TLM+DLM Batch TTFT = {round(self.perf_metrics.batch_ttft, 2)}\n"
            f"Decode Throughput = {round(self.perf_metrics.decode_throughput, 2)}\n"
            f"E2E Throughput = {round(self.perf_metrics.e2e_throughput, 2)}\n"
            f"Avg number of accepted tokens = {round(self.perf_metrics.mean_num_accepted_tokens, 2)}\n"
            f"Max generation len = {self.perf_metrics.max_gen_len}\n"
            f"Total Generated Tokens per Prompt: = {self.perf_metrics.generated_tokens_per_prompt}"
        )


def align_tlm_kvs(
    tlm_outputs: dict, 
    tree_pids: np.ndarray, 
    retrieve_best_indices: np.ndarray,
    ) -> None:
    """align KV$

    Args:
        tlm_outputs (dict): outputs from tlm tree inputs
        tree_pids (np.ndarray): input tree position ids, shape: [1, num_nodes]
        retrieve_best_indices (np.ndarray): tree indices of best candidates, shape: [decode_batch_size, num_speculative_tokens+1]
    """
    # generate sequential and dispersed pids
    decode_batch_size, n_nodes = tree_pids.shape
    num_speculative_tokens = retrieve_best_indices.shape[1] - 1 
    root_pids = tree_pids[:, 0] # shape: [decode_batch_size]
    tree_seq_pids = np.arange(1, n_nodes).reshape(1,-1).repeat(decode_batch_size, axis=0) + root_pids # shape: [decode_batch_size, n_nodes-1]
    aligned_pids = tree_seq_pids[:, :num_speculative_tokens] # ctx_len indices that will be updated, shape: [decode_batch_size, num_speculative_tokens]
    no_root_best_indices = retrieve_best_indices[:, 1:] - 1 # retrieve only non-root indices, shape: [decode_batch_size, num_speculative_tokens]
    dispersed_pids = np.take_along_axis(tree_seq_pids, no_root_best_indices, axis=1) # ctx_len indices that will be extracted, shape: [decode_batch_size, num_speculative_tokens+1]
    # expand pids to kv shape
    dispersed_pids_exp = dispersed_pids[:, np.newaxis, :, np.newaxis]
    aligned_pids_exp = aligned_pids[:, np.newaxis, :, np.newaxis]
    # align kv$
    pkvs = tlm_outputs["past_key_values"]
    new_pkvs = []
    for pkv in pkvs:
        past_key, past_value = pkv
        # align key
        selected_kv = np.take_along_axis(past_key, dispersed_pids_exp, axis=2) # shape: [decode_batch_size, n_heads, num_speculative_tokens+1, d_model]
        np.put_along_axis(past_key, aligned_pids_exp, selected_kv, axis=2) # shape: [decode_batch_size, n_heads, num_speculative_tokens+1, d_model]
        # align value
        selected_kv = np.take_along_axis(past_value, dispersed_pids_exp, axis=2) # shape: [decode_batch_size, n_heads, num_speculative_tokens+1, d_model]
        np.put_along_axis(past_value, aligned_pids_exp, selected_kv, axis=2) # shape: [decode_batch_size, n_heads, num_speculative_tokens+1, d_model]
        # store
        new_pkvs.append((past_key, past_value))
    tlm_outputs["past_key_values"] = tuple(new_pkvs)


def prepare_decode_dlm_inputs(
    target_tokens: np.ndarray, 
    target_pids: np.ndarray, 
    valid_greedy_spec_tokens: np.ndarray, 
    valid_best_path_indices: np.ndarray, 
    greedy_candidate_idx: int, 
    valid_num_tokens_selected: np.ndarray, 
    valid_batch_indices: np.ndarray,
    debug=False,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """prepare decode dlm inputs for next iteration

    Args:
        target_tokens (np.ndarray): best candidate target model predictions, shape: [decode_batch_size, num_speculative_tokens+1]
        target_pids (np.ndarray): best candidate target model predictions position ids, shape: [decode_batch_size, num_speculative_tokens+1]
        valid_greedy_spec_tokens (np.ndarray): greedy speculated tokens, shape: [decode_batch_size, num_speculative_tokens]
        valid_best_path_indices (np.ndarray): best path indices of target tree input, shape: [decode_batch_size, num_speculative_tokens+1]
        greedy_candidate_idx (int): greedy candidate path idx
        valid_num_tokens_selected (np.ndarray): number of accepted target tokens, shape: [num_valid_batch_indices]
        valid_batch_indices (np.ndarray): geometric boolean array whose idx determines whether corresponding batch idx is valid, shape: [decode_batch_size]

    Returns:
        Tuple[np.ndarray, np.ndarray]: tuple containing speculative input and positional ids of next iteration
    """
    # debug
    if debug:
        print(f"{target_tokens=}")
        print(f"{target_pids=}")
        print(f"{valid_greedy_spec_tokens=}")
        print(f"{valid_best_path_indices=}")
        print(f"{greedy_candidate_idx=}")
        print(f"{valid_num_tokens_selected=}")
    # determine whether greedy candidate was chosen
    decode_batch_size, target_len = target_tokens.shape # shape: [decode_batch_size, num_speculative_tokens+1]
    all_are_greedy_candidate = (greedy_candidate_idx == valid_best_path_indices).all()
    max_num_tokens_selected = valid_num_tokens_selected.max()
    if all_are_greedy_candidate:
        min_num_tokens_selected = valid_num_tokens_selected.min()
        if min_num_tokens_selected == target_len:
            # all tokens were accepted, pack last two tokens
            dlm_iids = target_tokens[:, -2:]
            dlm_pids = target_pids[:, -2:]
            return dlm_iids, dlm_pids
        # TODO: generalize to bsz>1
        accept_lengths = valid_num_tokens_selected-1
        #min_num_tokens_selected = max_num_tokens_selected-1 # it 4 bug
        min_num_tokens_selected = accept_lengths.min() # it 4 bug
    else:
        # determine number of accepted tokens
        posterior_mask = target_tokens[valid_batch_indices, :-1] == valid_greedy_spec_tokens[valid_batch_indices] # shape: [num_valid_batch_indices, num_speculative_tokens]
        accept_lengths = np.cumprod(posterior_mask, axis=1).sum(axis=1) # shape: [num_valid_batch_indices]
        min_num_tokens_selected = accept_lengths.min()
    assert min_num_tokens_selected >= 0
    # TODO: generalize to bsz>1
    dlm_iids = target_tokens[:, min_num_tokens_selected:max_num_tokens_selected] 
    # create left padded positional ids
    pids_mask = np.ones((decode_batch_size, target_len), dtype=np.int64)
    #pids_mask[valid_batch_indices, valid_num_tokens_selected-1] = 0 # TODO: something wrong here
    pids_mask[valid_batch_indices, accept_lengths] = 0 # TODO: something wrong here
    pids_mask = np.cumprod(pids_mask, axis=1)
    #dlm_pids = np.where(pids_mask, -1, target_pids)[:, min_num_tokens_selected:max_num_tokens_selected+1]
    dlm_pids = np.where(pids_mask, -1, target_pids)[:, min_num_tokens_selected:max_num_tokens_selected]
    assert dlm_iids.shape == dlm_pids.shape
    if debug:
        print(f"{dlm_iids=}")
        print(f"{dlm_pids=}")
    return dlm_iids, dlm_pids

def mv_retainedstate_buffers_to_decode_input(tlm_outputs: dict, tlm_precode_inputs: dict) -> None:
    """mv RS_SUFFIX keys from tlm_outputs to inputs

    Args:
        tlm_outputs (dict): output of decode run 
        tlm_precode_inputs (dict): input to next decode run
    """
    for key in tlm_outputs:
        if not key.endswith(RS_SUFFIX):
            continue
        pkv_key = key.split(RS_SUFFIX)[0]
        tlm_precode_inputs[pkv_key] = tlm_outputs[key]



def run_prefill_on_draft_and_target(
    tlm: torch.nn,
    dlm: torch.nn,
    inputs: dict,
):

    cast_pt(inputs)
    target_pkvs = inputs.pop("target_past_key_values")
    draft_pkvs = inputs.pop("draft_past_key_values")
    inputs["past_key_values"] = target_pkvs
    tlm_outputs = tlm(**inputs)
    inputs["past_key_values"] = draft_pkvs
    del inputs["attention_mask"]
    dlm_outputs = dlm(**inputs)
    cast_np(tlm_outputs)
    cast_np(dlm_outputs)
    return tlm_outputs, dlm_outputs


def get_padded_input_len(input_len: int, prefill_seq_len: int, ctx_len: int):
    """return padded input length (must be factor of `prefill_seq_len`)

    Args:
        input_len (int): prompt length
        prefill_seq_len (int): prefill sequence length
        ctx_len (int): context length

    Returns:
        input_len_padded (int): padded input length
    """
    num_chunks = -(input_len // -prefill_seq_len)  # ceil divide without float
    input_len_padded = num_chunks * prefill_seq_len  # Convert input_len to a multiple of prefill_seq_len
    assert (
        input_len_padded <= ctx_len
    ), "input_len rounded to nearest prefill_seq_len multiple should be less than ctx_len"
    return input_len_padded


def split_dlm_bonus_token_inputs(dlm_decode_inputs: dict) -> List[dict]:
    """split input into seq_len=1 inputs

    Args:
        dlm_decode_inputs (dict): inputs contain column-wise dimension (axis=1) equal to 2.
    """
    decode_batch_size, seq_len = dlm_decode_inputs["input_ids"].shape
    batch_indices = np.arange(decode_batch_size).reshape(-1,1)
    iids = np.hsplit(dlm_decode_inputs["input_ids"], seq_len)
    pids = np.hsplit(dlm_decode_inputs["position_ids"], seq_len)
    inputs = []
    for iid, pid in zip(iids, pids):
        input = dict(
            input_ids=iid,
            position_ids=pid,
            batch_index=batch_indices
        )
        inputs.append(input)

    return inputs

def speculate_tokens(dlm: torch.nn, dlm_decode_inputs: dict, valid_batch_indices: np.ndarray, num_speculative_tokens: int, spec_decode_nst_topk: np.ndarray):
    input_len = dlm_decode_inputs["input_ids"].shape[1]
    #dlm_decode_inputs["num_logits_to_keep"] = np.zeros((input_len,1), dtype=np.int64)
    start_idx = 0
    topks: List[List[int]] = 
    cast_pt(dlm_decode_inputs)
    causal_mask = dlm_decode_inputs["attention_mask"]
    for i in range(num_speculative_tokens):
        topk: List[int] = topks[i]
        end_idx: int = cumsum_nodes[i]
        if i+1 != num_speculative_tokens:
            dlm_decode_inputs["attention_mask"] = causal_mask.copy()[:, :, end_idx:] = True
        else:
            dlm_decode_inputs["attention_mask"] = causal_mask
        dlm_outputs = dlm(**dlm_decode_inputs)
        dlm_logits = dlm_outputs["logits"][:, start_idx:end_idx, :] # shape: [decode_batch_size, 1, TOPK] # TOPK indices (descending order)
        # TODO: generalize to bsz>1
        start_iids_idx = start_idx
        for j in range(end_idx-start_idx):
            k = topks[j]
            spec_toks = dlm_logits[:, j,:k]
            end_iids_idx = start_iids_idx+k
            dlm_decode_inputs["input_ids"][:, start_iids_idx:end_iids_idx] = spec_toks
            start_iids_idx += k
        #dlm_decode_inputs["past_key_values"] = dlm_outputs["past_key_values"]
        start_idx += end_idx
    cast_np(dlm_outputs)
    #del dlm_decode_inputs["num_logits_to_keep"]
    return dlm_outputs

def select_best_candidates(
        spec_candidates: np.ndarray,
        tlm_candidates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """select best candidate per batch

    Args:
        spec_candidates (np.ndarray): speculation candidate paths, shape: [decode_batch_size, leaf_nodes, num_speculative_tokens+1] 
        tlm_candidates (np.ndarray): tlm candidate paths, shape: [decode_batch_size, leaf_nodes, num_speculative_tokens+1] 
    Returns:
        tuple: _description_
    """
    bsz = spec_candidates.shape[0]
    batch_indices = np.arange(bsz)
    posterior_mask = spec_candidates[:, :, 1:] == tlm_candidates[:, :, :-1] # shape: [decode_batch_size, leaf_nodes, num_speculative_tokens]
    candidates_accept_length = (np.cumprod(posterior_mask, axis=2)).sum(axis=2) # shape: [decode_batch_size, leaf_nodes]
    max_accept_length = candidates_accept_length.max(axis=1) + 1 # shape: [deode_batch_size]
    best_path_idx = np.argmax(candidates_accept_length, axis=1).astype(np.int64) # shape: [decode_batch_size]
    best_tlm_paths = tlm_candidates[batch_indices, best_path_idx, :] # shape: [decode_batch_size, num_speculative_tokens+1]
    best_spec_paths = spec_candidates[batch_indices, best_path_idx, :] # shape: [decode_batch_size, num_speculative_tokens+1]
    assert best_tlm_paths.shape[1] == tlm_candidates.shape[2]
    return best_tlm_paths, best_spec_paths, max_accept_length, best_path_idx 



def cast_np(inputs: dict):
    for key,val in inputs.items():
        if isinstance(val, torch.Tensor):
            inputs[key] = val.detach().numpy()
        elif isinstance(val, tuple):
            # tuple kv$
            past_key_values = []
            for pkv in val:
                past_key, past_value = pkv
                pk = past_key.detach().numpy()
                pv = past_value.detach().numpy()
                past_key_values.append((pk, pv))
            inputs[key] = tuple(past_key_values)
        else:
            raise ValueError(f"key {key} has value of type {type(val)}")

def cast_pt(inputs: dict):
    for key,val in inputs.items():
        if isinstance(val, np.ndarray):
            inputs[key] = torch.from_numpy(val)
        elif isinstance(val, tuple):
            # tuple kv$
            past_key_values = []
            for pkv in val:
                past_key, past_value = pkv
                past_key, past_value = torch.from_numpy(past_key), torch.from_numpy(past_value)
                past_key_values.append((past_key, past_value))
            inputs[key] = tuple(past_key_values)
        else:
            raise ValueError(f"key {key} has value of type {type(val)}")

def create_pkvs(config, batch_size=1, seq_len=32):
    padding_shape = get_padding_shape_from_config(config=config, batch_size=batch_size, seq_len=seq_len)
    num_hidden_layers = config.num_hidden_layers
    past_key_values = []
    for _ in range(num_hidden_layers):
        past_key = np.zeros((padding_shape), dtype=np.float32)
        past_value = np.zeros((padding_shape), dtype=np.float32)
        pkv = (past_key, past_value)
        past_key_values.append(pkv)
    return tuple(past_key_values)

def tree_attn_inference(
    prompts: List[str],
    tree_attn_choices: List[list],
    ctx_len: int,
    draft_model_name: str,
    target_model_name: str,
    ignore_eos_token: bool = True,
    debug: bool = False
):
    num_tree_nodes = len(tree_attn_choices)+1 # +1 to account for root node
    # assumes dlm and tlm are compiled to the same prompt-chunk-size, context length and full_batch_size/batch-size
    # get vocab size
    tokenizer = AutoTokenizer.from_pretrained(target_model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)
    # generate tree attention buffers
    tree_buffers = generate_medusa_buffers(tree_attn_choices)
    tree_mask = tree_buffers["medusa_attn_mask"].numpy().astype(bool) # shape: [1, 1, num_tree_nodes, num_tree_nodes]
    tree_mask = ~tree_mask # invert to match QEff mask where 1 values will become large negative values
    tree_indices = tree_buffers["tree_indices"].numpy().astype(np.int64) # shape: [1, num_tree_nodes]
    tree_position_ids = tree_buffers["medusa_position_ids"].numpy().reshape(1, -1).astype(np.int64) # shape: [1, num_tree_nodes]
    retrieve_indices = tree_buffers["retrieve_indices"].numpy().astype(np.int64) # shape: [leaf_nodes, num_speculative_tokens+1]
    num_speculative_tokens = retrieve_indices.shape[1] - 1
    # get QEff model
    continuous_batching = False
    is_tlm=True
    include_4d_causal_mask=True
    topk_logits=None
    tlm = AutoModelForCausalLM.from_pretrained(
        target_model_name, continuous_batching=continuous_batching, is_tlm=is_tlm, include_4d_causal_mask=include_4d_causal_mask, topk_logits=topk_logits
    ).model
    topk_logits=TOPK
    dlm = AutoModelForCausalLM.from_pretrained(
        draft_model_name, continuous_batching=continuous_batching, is_tlm=is_tlm, include_4d_causal_mask=include_4d_causal_mask, topk_logits=topk_logits
    ).model
    # create pkvs
    target_pkvs = create_pkvs(tlm.config, batch_size=1, seq_len=32)
    draft_pkvs = create_pkvs(dlm.config, batch_size=1, seq_len=32)
    # tokenize prompts
    decode_batch_size = 1
    prompts_tokenized: List[dict] = []
    for p in prompts:
        p_tok: dict = tokenizer(p, return_tensors="np")
        input_len: int = p_tok.input_ids.shape[1]
        del p_tok["attention_mask"]
        position_ids = np.arange(input_len).reshape(decode_batch_size, -1)
        p_tok["position_ids"] = position_ids
        p_tok["attention_mask"] = create_4d_causal_mask(position_ids, ctx_len) # shape: [1, 1, input_len, ctx_len]
        p_tok["target_past_key_values"] = target_pkvs
        p_tok["draft_past_key_values"] = draft_pkvs
        prompts_tokenized.append(p_tok)
    # create caches to hold generated ids and input prompt lengths
    generated_ids = [[] for i in range(decode_batch_size)]
    input_lengths = [0] * decode_batch_size
    max_gen_len = [ctx_len] * decode_batch_size
    # create dlm decode buffers
    dlm_decode_inputs = dict(
        input_ids = np.full((decode_batch_size, 1), tokenizer.pad_token_id),
        position_ids = np.zeros((decode_batch_size, 1), np.int64),
    )
    # create tlm decode buffers
    tlm_precode_inputs = dict(
        input_ids=np.zeros((decode_batch_size, num_tree_nodes), dtype=np.int64),
        position_ids=np.zeros((decode_batch_size, num_tree_nodes), dtype=np.int64),
        batch_index=np.arange(decode_batch_size, dtype=np.int64).reshape(-1, 1),
        num_logits_to_keep=np.zeros((num_tree_nodes,1))
    )
    
    # calculate greedy candidate idx
    greedy_candidate_idx = np.where(retrieve_indices == -1, np.inf, retrieve_indices).sum(axis=1).argmin().item()
    #greedy_candidate_idx = retrieve_indices.sum(axis=1).argmin().item() # shape: [1]
    if debug:
        print(f"{num_tree_nodes=}")
        print(f"{tree_mask.astype(int)=}")
        print(f"{tree_mask.shape=}")
        print(f"{tree_indices=}")
        print(f"{tree_indices.shape=}")
        print(f"{tree_position_ids=}")
        print(f"{tree_position_ids.shape=}")
        print(f"{retrieve_indices=}")
        print(f"{retrieve_indices.shape=}")
        print(f"{greedy_candidate_idx=}")
    # run prefill
    ttfts = []
    e2e_start = perf_counter()
    for bi in range(decode_batch_size):
        start = perf_counter()
        tlm_outputs, dlm_outputs = run_prefill_on_draft_and_target( # shape: [1, 1, 1]
            tlm=tlm,
            dlm=dlm,
            inputs=prompts_tokenized[bi],
        ) 
        tlm_logits = tlm_outputs["logits"]
        ttft = perf_counter() - start
        ttfts.append(ttft)
        input_ids = tlm_logits.argmax(axis=2)[:, :, np.newaxis][:, :, 0].astype(np.int64) # shape: [1, 1]
        common_input_ids = input_ids.item() # common input ids of dlm/tlm
        generated_ids[bi].append(common_input_ids)
        dlm_decode_inputs["input_ids"][bi, 0] = common_input_ids
        tlm_precode_inputs["input_ids"][bi, 0] = common_input_ids
        input_len = prompts_tokenized[bi]["position_ids"].max().item() + 1
        dlm_decode_inputs["position_ids"][bi, 0] = input_len
        tlm_precode_inputs["position_ids"][bi] = tree_position_ids[0] + input_len
        # assumes that prefill queue will always be popped from the front
        if debug:
            print(f"{input_len=}")
        input_lengths[bi] = input_len
        max_gen_len[bi] -= input_lengths[bi]
    batch_ttft = perf_counter() - e2e_start
    dlm_decode_inputs["past_key_values"] = dlm_outputs["past_key_values"]
    tlm_precode_inputs["past_key_values"] = tlm_outputs["past_key_values"]

    # set decode logits buffers
    spec_decode_nst_topk = np.zeros((decode_batch_size, num_speculative_tokens, TOPK), dtype=np.int64) # buffer will hold logits of each speculation step
    # create frequency matrix to track number of accepted tokens with each respective candidate
    freq = np.zeros_like(retrieve_indices, dtype=np.int64) 
    # start decode phase
    valid_batch_indices = np.full(decode_batch_size, True, dtype=bool)
    if not ignore_eos_token:
        valid_batch_indices[dlm_decode_inputs["input_ids"][:,0] == tokenizer.eos_token_id] = False
    it = 0
    mean_num_accepted_tokens = 0
    target_seq_pids = np.arange(num_speculative_tokens+1).reshape(1,-1).repeat(decode_batch_size, axis=0)
    decode_start = perf_counter()
    while True:
        it += 1
        if debug:
            print('-'*60)
            print(f"{it=}")
            print(f"{dlm_decode_inputs['input_ids']=}")
            print(f"{dlm_decode_inputs['input_ids'].shape=}")
            print(f"{dlm_decode_inputs['position_ids']=}")
            print(f"{dlm_decode_inputs['position_ids'].shape=}")
            print(f"{dlm_decode_inputs['past_key_values'][0][0].shape=}")
        # generate proposals from draft model
        common_token_id: np.ndarray = dlm_decode_inputs["input_ids"][:, -1:].copy() # shape: [decode_batch_size, 1]
        common_position_id: np.ndarray = dlm_decode_inputs["position_ids"][:, -1:].copy() # shape: [decode_batch_size, 1]
        #print(f"{common_position_id=}")
        #print("dlm_decode_inputs=")
        #pprint(dlm_decode_inputs)
        if it==4:
            #breakpoint()
            pass
        dlm_output = speculate_tokens( 
            dlm, 
            dlm_decode_inputs, 
            valid_batch_indices, 
            num_speculative_tokens, 
            spec_decode_nst_topk)
        # generate speculative tree proposals
        (spec_candidates, # shape: [decode_batch_size, leaf_nodes, num_speculative_tokens+1]
        spec_tree_candidates # shape: [decode_batch_size, num_tree_nodes]
        ) = generate_candidates(common_token_id, spec_decode_nst_topk, tree_indices, retrieve_indices)
        assert spec_candidates.shape[1:] == retrieve_indices.shape
        assert spec_tree_candidates.shape[1] == tree_position_ids.shape[1]
        #print(f"{spec_candidates=}")
        #print(f"{spec_tree_candidates=}")
        # prepare TLM inputs
        past_seen_values: int = common_position_id.max()
        #print(f"{past_seen_values=}")
        # continue
        #print(f"{past_seen_values=}") # 24
        tlm_precode_inputs["input_ids"][:] = spec_tree_candidates
        tlm_pids = tlm_precode_inputs["position_ids"]
        causal_mask = create_4d_causal_mask(tlm_pids, ctx_len, past_seen_values, tree_mask) # shape: [decode_batch_size, 1, num_tree_nodes, ctx_len] TODO: generalize to bsz > 1
        tlm_precode_inputs["attention_mask"] = causal_mask
        # run TLM inferences
        if debug:
            print(f"{tlm_precode_inputs['input_ids']=}")
            print(f"{tlm_precode_inputs['position_ids']=}")
        cast_pt(tlm_precode_inputs)
        tlm_precode_inputs["position_ids"].is_tree = True
        tlm_outputs: dict = tlm(**tlm_precode_inputs) 
        cast_np(tlm_outputs)
        cast_np(tlm_precode_inputs)
        #target_logits: np.ndarray = tlm_outputs["logits"] # shape: [decode_batch_size, num_tree_nodes, 1]
        target_logits: np.ndarray = tlm_outputs["logits"].argmax(axis=2)[:, :, np.newaxis] # shape: [decode_batch_size, num_tree_nodes, 1]
        if debug:
            print(f"{target_logits.shape=}")
        tlm_candidates = target_logits[:, retrieve_indices, 0] # shape: [decode_batch_size, leaf_nodes, num_speculative_tokens+1]
        assert tlm_candidates.shape[1:] == retrieve_indices.shape
        if debug:
            print(f"{tlm_candidates=}")
            print(f"{spec_candidates=}")
        # select best tlm/spec pair of candidates
        (target_tokens, # shape: [decode_batch_size, num_speculative_tokens+1]
         spec_tokens, # shape: [decode_batch_size, num_speculative_tokens+1]
         num_tokens_selected, # shape: [decode_batch_size]
         best_path_indices, # shape: [decode_batch_size]
          ) = select_best_candidates(spec_candidates, tlm_candidates)
        if debug:
            print(f"{num_tokens_selected=}")
            print(f"{best_path_indices=}")
        # record mean number of accepted tokens
        mean_num_accepted_tokens += num_tokens_selected[valid_batch_indices].mean().item()
        # append selected tokens to the generated_ids
        for bi, valid in enumerate(valid_batch_indices):
            if not valid:
                continue
            # record chosen path and its accepted tokens
            num_accepted_tokens = num_tokens_selected[bi]
            best_path_idx = best_path_indices[bi]
            freq[best_path_idx, num_accepted_tokens-1] += 1
            # record accepted tokens
            num_gen_tokens_left = max_gen_len[bi] - len(generated_ids[bi])
            num_tokens_to_append = min(num_accepted_tokens, num_gen_tokens_left)
            accepted_tokens_arr = target_tokens[bi, :num_tokens_to_append]
            accepted_tokens = accepted_tokens_arr.tolist()
            generated_ids[bi].extend(accepted_tokens)
            if debug:
                print(f"{accepted_tokens=}")
                print(f"{generated_ids=}")
            if len(generated_ids[bi])+num_tree_nodes >= max_gen_len[bi] or ((not ignore_eos_token) and (accepted_tokens_arr == tokenizer.eos_token_id).any()):
                valid_batch_indices[bi] = False
        # check if all generations are done
        if not valid_batch_indices.any():
            break
        # align tlm KV$ 
        retrieve_best_indices = retrieve_indices[best_path_indices] # shape: [decode_batch_size, num_speculative_tokens+1] 
        align_tlm_kvs(tlm_outputs, tlm_precode_inputs["position_ids"], retrieve_best_indices)
        tlm_precode_inputs["past_key_values"] = tlm_outputs["past_key_values"]
        # prepare spec decode inputs for next decode iteration
        num_valid_batch_indices = valid_batch_indices.sum().item()
        valid_num_tokens_selected = num_tokens_selected[valid_batch_indices] # shape: [num_valid_batch_indices]
        # create target sequential pids (ignore non-valid batch indices)
        target_pids = target_seq_pids + (tlm_precode_inputs["position_ids"][:,0]+1) # shape: [decode_batch_size, num_speculative_tokens+1]
        valid_mask = np.zeros((decode_batch_size, num_speculative_tokens+1), dtype=np.int64)
        valid_mask[valid_batch_indices] = 1
        target_pids = np.where(valid_mask, target_pids, -1)
        # prepare decode dlm inputs for next iteration
        valid_best_path_indices = best_path_indices[valid_batch_indices]
        valid_greedy_spec_tokens = spec_decode_nst_topk[valid_batch_indices, :, 0] # shape: [num_valid_batch_indices, num_speculative_tokens]
        assert target_tokens.shape == target_pids.shape
        assert valid_greedy_spec_tokens.shape[1] == target_pids.shape[1]-1
        dlm_iids, dlm_pids = prepare_decode_dlm_inputs(
            target_tokens,
            target_pids,
            valid_greedy_spec_tokens,
            valid_best_path_indices,
            greedy_candidate_idx,
            valid_num_tokens_selected,
            valid_batch_indices,
            debug
        )
        dlm_decode_inputs["input_ids"] = dlm_iids
        dlm_decode_inputs["position_ids"] = dlm_pids
        dlm_decode_inputs["past_key_values"] = dlm_output["past_key_values"]
        # prepare tlm decode inputs for next iteration
        tlm_precode_inputs["position_ids"][valid_batch_indices == False] -1
        tlm_precode_inputs["position_ids"][valid_batch_indices] += valid_num_tokens_selected
        #if debug:
            #if it >= 315:
            #    breakpoint()
            #if 1135 in tlm_precode_inputs["position_ids"]:
            #    breakpoint()
    end = perf_counter()
    decode_end = end - decode_start
    e2e_end = end - e2e_start
    mean_ttft = sum(ttfts) / len(ttfts)
    generated_tokens_per_prompt = [len(gid) + 1 for gid in generated_ids]
    decode_throughput = sum(generated_tokens_per_prompt) / decode_end
    e2e_throughput = (sum(generated_tokens_per_prompt) + decode_batch_size) / e2e_end
    batch_decode = tokenizer.batch_decode(generated_ids)
    mean_num_accepted_tokens /= it
    perf_metrics = PerfMetrics(
        mean_ttft,
        batch_ttft,
        decode_throughput,
        e2e_throughput,
        mean_num_accepted_tokens,
        max_gen_len,
        generated_tokens_per_prompt,
        freq.tolist()
    )
    exec_info = CloudAI100ExecInfo(
        prompts,
        decode_batch_size,
        batch_decode,
        generated_ids,
        perf_metrics,
        num_speculative_tokens,
        ctx_len,
        draft_model_name,
        target_model_name,
        it+1
    )

    return exec_info



def main():
    exec_info = tree_attn_inference(
        prompts = ["My name is"],
        tree_attn_choices=[ [0], [0,0], [0,1], [0,2], [1], [1,0], [1,1], [1,2] ],
        ctx_len=32,
        draft_model_name="JackFram/llama-68m",
        target_model_name="JackFram/llama-160m",
        debug=True
    )
    print(exec_info)
    prompts = exec_info.prompts
    generated_texts = exec_info.generated_texts
    for prompt, generation in zip(prompts, generated_texts):
        print(f"{prompt=} {generation=}")



if __name__ == "__main__":
    main()