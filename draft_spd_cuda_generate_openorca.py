from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
import pickle
import torch
from argparse import ArgumentParser
from datetime import datetime
from transformers import AutoTokenizer
from QEfficient.generation.cloud_infer import QAICInferenceSession
from typing import List
import QEfficient
from tqdm import tqdm
import sys
import os
TOPK=10
# Get the current working directory
qeff_init = QEfficient.__file__
qeff_root_dir = r"/prj/crd/austin/validation/scratch/users/eplatero/qeff_cuda/efficient-transformers"
sys.path.insert(0, qeff_root_dir)

prefill_seq_len = 128
ctx_len = 2048
num_speculative_tokens = 4
draft_model_name = r"/local/mnt/qt_drive/users/agokhale/cerebras_networks/Llama3.2-512M-draft/Llama3.2_512M_draftModel/"

def prepare_inputs(num_seqs):
    open_orca = r"/local/mnt/qt_drive/users/eplatero/datasets/open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
    with open(open_orca, 'rb') as f:
        # Use pickle.load() to de-serialize the object from the file
        prompts = pickle.load(f)["input"].tolist()[:num_seqs]
        #prompts = pickle.load(f)["input"].tolist()[256:512]
    #prompts = load_pickle(open_orca)["input"].tolist()[:1]
    return prompts

def get_models(draft_model_name, target_model_name, device, diverse_roots = False):
    continuous_batching = False
    is_tlm=True
    include_4d_causal_mask=True
    topk_logits=None
    tlm = AutoModelForCausalLM.from_pretrained(
        target_model_name, continuous_batching=continuous_batching, is_tlm=is_tlm, include_4d_causal_mask=include_4d_causal_mask, topk_logits=topk_logits
    ).model.to(device)
    is_tlm=False
    include_4d_causal_mask=False
    if diverse_roots:
        is_tlm = True
        include_4d_causal_mask = True
    topk_logits=TOPK
    dlm = AutoModelForCausalLM.from_pretrained(
        draft_model_name, continuous_batching=continuous_batching, is_tlm=is_tlm, include_4d_causal_mask=include_4d_causal_mask, topk_logits=topk_logits
    ).model.to(device)
    return dlm, tlm

def run_vanilla_spd(num_seqs, diverse_roots = False, device = "auto", debug = False):
    if diverse_roots:
        from validate_pt_diverse_roots_tree import tree_attn_inference
    else:
        from validate_pt_graph_tree import tree_attn_inference
    target_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    draft_model_name = r"/local/mnt/qt_drive/users/agokhale/cerebras_networks/Llama3.2-512M-draft/Llama3.2_512M_draftModel/"
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    dlm, tlm = get_models(draft_model_name=draft_model_name, target_model_name=target_model_name, device=device, diverse_roots=diverse_roots)
    models = (dlm, tlm)
    prompts: List[str] = prepare_inputs(num_seqs)
    tree_attn_choices = [[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]
    #tree_attn_choices = [ [0], [0,0], [0,1], [0,2], [1], [1,0], [1,1], [1,2] ]
    n_prompts = len(prompts)
    exec_infos = []
    for i in tqdm(range(n_prompts)):
        prompt = prompts[i]
        exec_info = tree_attn_inference(
            prompts=[prompt],
            tree_attn_choices=tree_attn_choices,
            ctx_len=ctx_len,
            draft_model_name=draft_model_name,
            target_model_name=target_model_name,
            ignore_eos_token=False,
            debug=debug,
            models=models,
            device=device,
        )
        exec_infos.append(exec_info)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_cuda_tree_dev_attention_inference.pkl"
    # Open a file in binary write mode
    with open(filename, 'wb') as f:
        # Use pickle.dump() to serialize the object and write it to the file
        pickle.dump(exec_infos, f)
    print(f"pickle file: {filename}")
    n_samples = min(3, len(exec_infos))
    for i in range(n_samples):
        gen_text = exec_infos[i].generated_texts
        print(f"{gen_text=}")

def comma_separated_ints(x: str):
    return [int(qid) for qid in x.split(",")]

def arg_parse():
    parser = ArgumentParser(description="Draft-based SpD Inference")
    parser.add_argument("--num-seqs", type=int, default=1024, help="number of open orca sequences to extract")
    parser.add_argument("--device", type=str, default="auto", help="device")
    parser.add_argument('--diverse-roots', action='store_true')
    parser.add_argument('--debug', action='store_true')
    #parser.add_argument("--target-model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Number of target cores")
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    run_vanilla_spd(**vars(args))

if __name__ == '__main__':
    main()
