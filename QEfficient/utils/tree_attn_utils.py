# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

# https://github.com/FasterDecoding/Medusa/blob/main/medusa/model/utils.py

from typing import Optional, List

import numpy as np
import torch

TOPK = 10


def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_medusa_buffers(medusa_choices, device="cpu"):
    """
    Generate buffers for the Medusa structure based on the provided choices.

    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".

    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth

    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[: c + 1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1 : start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i - 1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[: c + 1]))
                retrieve_paths.append(cur_medusa_choice[: c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat(
        [torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1
    )

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers


def generate_candidates(candidates_logit, topk_spec_logits, tree_indices, retrieve_indices):
    """
    Generate candidates based on provided logits and indices.

    Parameters:
    - candidates_logit (np.ndarray): base logits, shape: [decode_batch_size, 1]
    - topk_spec_logits (torch.Tensor): Standard logits from a language model, shape: [decode_batch_size, num_speculative_tokens, TOPK]
    - tree_indices (list or torch.Tensor): Indices representing a tree structure, used for mapping candidates.
    - retrieve_indices (list or torch.Tensor): Indices for extracting specific candidate tokens.

    Returns:
    - tuple (torch.Tensor, torch.Tensor): A tuple containing two sets of candidates:
        1. Cartesian candidates derived from the combined original and Medusa logits.
        2. Tree candidates mapped from the Cartesian candidates using tree indices.
    """
    # Greedy decoding: Select the most probable candidate from the original logits.
    # candidates_logit = torch.argmax(tree_logits[:, 0]).unsqueeze(0) # shape: [1, decode_batch_size]
    # Extract the TOPK candidates from the medusa logits.
    # candidates_medusa_logits = torch.topk(spec_logits, TOPK, dim = -1).indices # shape: [decode_batch_size, num_speculative_tokens, TOPK]

    # Combine the selected candidate from the original logits with the topk medusa logits.
    bsz = candidates_logit.shape[0]
    candidates = np.concatenate(
        [candidates_logit, topk_spec_logits.reshape(bsz, -1)], axis=-1
    )  # shape: [bsz, num_speculative_tokens*TOPK+1]
    # candidates = torch.cat([candidates_logit, topk_spec_logits.view(-1)], dim=-1) # shape: [1, decode_batch_size*num_speculative_tokens*TOPK+1]

    # Map the combined candidates to the tree indices to get tree candidates.
    tree_candidates = candidates[:, tree_indices]  # shape: [bsz, num_nodes]

    # Extend the tree candidates by appending a zero.
    tree_candidates_ext = np.concatenate(
        #[tree_candidates, np.zeros((bsz, 1), dtype=np.int64)], axis=-1 # extend to zero so that if `retrieve_inices` has -1, it will just pick it up (acutaly retrieve_indices will never go to that lenght unless i'ts value is -1)
        [tree_candidates, np.full((bsz, 1), -1, dtype=np.int64)], axis=-1 # extend to zero so that if `retrieve_inices` has -1, it will just pick it up (acutaly retrieve_indices will never go to that lenght unless i'ts value is -1)
    )  # shape: [bsz, n_nodes+1]
    # tree_candidates_ext = torch.cat([tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device)], dim=0)

    # Retrieve the cartesian candidates using the retrieve indices.
    cart_candidates = tree_candidates_ext[:, retrieve_indices]  # shape: [num_leafs, num_speculative_tokens+1]

    # Unsqueeze the tree candidates for dimension consistency.
    # tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, tree_candidates


def create_4d_causal_mask(
    position_ids: np.ndarray, ctx_len: int, past_seen_tokens: Optional[int] = None, tree_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Creates a QEff 4D mask that also integrates tree attention mask

    Parameters:
    ----------
    position_ids (np.ndarray): position ids, shape: [decode_batch_size, seq_len]
    ctx_len (int): model context length
    past_seen_tokens (int): number of previous tokens seen
    tree_mask (Optional[np.ndarray]): tree attention mask, shape: [1, 1, tree_len, tree_len]

    Returns:
    -------
    causal_mask: array_like
        A 4D mask with shape `(batch_size, 1, sequence_length, target_length)`.
    """

    # compute target length
    query_indices = position_ids[:, :, np.newaxis]
    kv_indices = np.arange(ctx_len).reshape(1, 1, -1)
    causal_mask = kv_indices > query_indices
    causal_mask = causal_mask[:, np.newaxis]  # shape: [1, 1, position_ids.shape[1], ctx_len]

    if tree_mask is not None:
        assert past_seen_tokens is not None
        tree_len = tree_mask.shape[-1]
        causal_mask[:, :, :, past_seen_tokens:past_seen_tokens+tree_len] = tree_mask
        #causal_mask[:, :, :, -tree_len:] = tree_mask

    return causal_mask


def bfs_sort(lst):
    return sorted(lst, key=lambda x: (len(x), [i for i in x]))

def group_by_depth(tree_attn_choices):
    sorted_choices = bfs_sort(tree_attn_choices)
    grouped_list = []
    current_length = len(sorted_choices[0])
    current_group = []

    for elem in sorted_choices:
        if len(elem) == current_length:
            current_group.append(elem)
        else:
            grouped_list.append(current_group)
            current_length = len(elem)
            current_group = [elem]
    grouped_list.append(current_group)
    return grouped_list


def parent_child_counts(tree_attn_choices) -> List[List[List[int]]]:
    n_elms_per_path = [len(path) for path in tree_attn_choices]
    max_tree_depth = max(n_elms_per_path) # tree depth starts from 0
    out = [[] for i in range(max_tree_depth)]
    out[0].append([0,0]) # root node
    tree_attn_choices = bfs_sort(tree_attn_choices)
    nodes_per_depth = group_by_depth(tree_attn_choices)
    prev_parent_idx = [0]
    counter = 0
    prev_depth = 1
    for path in tree_attn_choices:
        n_nodes = len(path)
        if n_nodes == 1:
                out[0][0][1] += 1
                continue
        #parent_idx = path[-2]
        parent_idx = path[:-1]
        curr_depth = n_nodes-1
        if curr_depth-1 == prev_depth:
            depth_bfs_nodes = nodes_per_depth[curr_depth-2]
            relative_idx = depth_bfs_nodes.index(prev_parent_idx)
            out[prev_depth].append([relative_idx, counter])
            prev_depth += 1
            prev_parent_idx = parent_idx
            counter = 1
            continue
        if parent_idx == prev_parent_idx:
             counter += 1
        else:
             
             depth_bfs_nodes = nodes_per_depth[curr_depth-1]
             relative_idx = depth_bfs_nodes.index(prev_parent_idx)
             out[curr_depth].append([relative_idx, counter])
             counter = 1
             prev_parent_idx = parent_idx
    out[curr_depth].append([parent_idx[-1], counter])
    return out