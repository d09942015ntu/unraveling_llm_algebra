"""Shared code for vis_com_std.py and vis_ide_std.py.

Both scripts compare the hidden states at the ``[=]`` token for a group of related
inputs (e.g. reorderings of the same operands) between operator ``+`` and an
operator chosen by ``rtype``.
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, GPT2Tokenizer

from dataset_generator import addition_str_64, pos_lh_64, pos_rh_64, pos_z0_64
from mydataset import EvalDataset, dataset_tokenizer
from utils import latest_checkpoint

# rtype -> function that turns operands into an input string
OPERATORS = {0: addition_str_64, 1: pos_z0_64, 2: pos_lh_64, 3: pos_rh_64}
MAX_LENGTH = 15


def parse_operands(tokenizer, input_ids_row):
    """Recover the sorted operand values from one encoded ``[z.][+][z.]...[=]`` input."""
    return tuple(sorted(int(x) for x in re.findall(r'\[z(\d+)\]', tokenizer.decode(input_ids_row))))


def encode_operand_lists(operand_lists, n, rtype, tokenizer, position_ids_i, attention_mask_i, device):
    """Encode each operand list with operator ``rtype``; position ids and mask are copied from the original sample."""
    input_ids = []
    for operands in operand_lists:
        s1, _ = OPERATORS[rtype](operands, n)
        encoding = tokenizer(s1, truncation=True, max_length=MAX_LENGTH, padding='max_length')
        input_ids.append(torch.tensor(encoding['input_ids']).unsqueeze(0))
    count = len(operand_lists)
    return (torch.cat(input_ids, dim=0).to(device),
            torch.cat([position_ids_i] * count, dim=0).to(device),
            torch.cat([attention_mask_i] * count, dim=0).to(device))


def hidden_states_at_eq(model, input_ids, position_ids, attention_mask, eq_token):
    """For every layer, a [batch, hidden] tensor of the hidden state at each row's ``[=]`` token."""
    output = model(input_ids, position_ids=position_ids, attention_mask=attention_mask, output_hidden_states=True)
    rows = torch.arange(input_ids.shape[0])
    eq_positions = torch.tensor([row.tolist().index(eq_token) for row in input_ids.cpu()])
    return [layer[rows, eq_positions, :] for layer in output["hidden_states"]]


def visualize(model, dataloader, make_groups, compare, n, rtype, limit=np.inf):
    """Average, over the samples in ``dataloader``, of ``compare(hidden_+, hidden_rtype)`` per layer.

    ``make_groups(operands, used, rtype)`` returns the operand lists to encode for one
    sample, or None to skip the sample.
    """
    model.eval()
    tokenizer = dataset_tokenizer(dataloader)
    eq_token = tokenizer.convert_tokens_to_ids("[=]")
    used = set()
    results = []

    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            for k in range(len(batch['labels'])):
                input_ids_i = batch['input_ids'][k].unsqueeze(0)
                position_ids_i = batch['position_ids'][k].unsqueeze(0)
                attention_mask_i = batch['attention_mask'][k].unsqueeze(0)
                operands = parse_operands(tokenizer, input_ids_i[0])

                groups = make_groups(operands, used, rtype=0)
                if groups is None:
                    continue
                groups_r = make_groups(operands, used, rtype=rtype)

                hidden = {}
                for name, group, group_rtype in [("p", groups, 0), ("r", groups_r, rtype)]:
                    encoded = encode_operand_lists(group, n, group_rtype, tokenizer, position_ids_i,
                                                   attention_mask_i, model.device)
                    hidden[name] = hidden_states_at_eq(model, *encoded, eq_token)
                results.append(compare(hidden["p"], hidden["r"]))
            if j > limit:
                break
    return np.average(np.array(results), axis=0).tolist()


def vis_array(data, fname):
    """Save ``data`` as a heat map (``fname``) and as JSON (same name, .json)."""
    plt.clf()
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('2D Array Visualization')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(fname)
    with open(fname.replace(".png", ".json"), "w") as f:
        json.dump(data.tolist(), f, indent=2)


def run(data_prefix, kind, test_ftype, make_groups, compare):
    """Write results/7_<rtype>_<kind>_diff_test.{png,json} for rtype 1 (z0), 2 (lh) and 3 (rh)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    p = 7
    for rtype in [1, 2, 3]:
        diff_test = []
        for train_size in [100, 300, 1000, 3000, 10000]:
            dataset_name = f"{data_prefix}_{p}_{train_size}"
            print(f"dataset_name={dataset_name},rtype={rtype}")
            ckpt_dirs = sorted(glob.glob(f"results/{dataset_name}_seqnew-*/checkpoints"))
            if not ckpt_dirs:
                continue
            dataset_path = f'./data/{dataset_name}'
            n = int(os.path.basename(dataset_path).split("_")[-2])
            model = AutoModelForCausalLM.from_pretrained(latest_checkpoint(ckpt_dirs[-1]), trust_remote_code=True)
            model.to(device)
            dataloader = DataLoader(EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'), ftype=test_ftype),
                                    batch_size=32, shuffle=False)
            diff_test.append(visualize(model, dataloader, make_groups, compare, n=n, rtype=rtype))
        vis_array(np.array(diff_test), f"results/{p}_{rtype}_{kind}_diff_test.png")
