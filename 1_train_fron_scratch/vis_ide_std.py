import argparse
import glob
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import torch

from dataset_generator import addition_str_64, pos_z0_64, pos_lh_64, pos_rh_64
from mydataset import EvalDataset


def diff_between(a, b):
    return np.sum(np.abs(a - b))


def generate_all_permutations(used, n, tokenizer, input_ids_i, position_ids_i, attention_mask_i, rtype=0):
    max_length = 15
    decoded_x = [x.replace('[PAD]', '') for x in tokenizer.decode(input_ids_i[0]).split(" ") if
                 x not in ['[+]', '[=]', '[PAD]', '[EOS]']]
    decoded_x = [x for x in decoded_x if len(x) > 0]
    # print(f"decoded_x={decoded_x}")
    filtered_input = tuple(sorted([int(x.replace("[z", "").replace("]", "")) for x in decoded_x]))
    filtered_input_2 = [x for x in filtered_input if x != 0]
    if not rtype and filtered_input in used:
        return False, None, None, None
    elif len(filtered_input_2) < 2:
        return False, None, None, None
    else:
        used.add(filtered_input)
    # permutations= sorted(list(set(list(itertools.permutations(filtered_input, len(filtered_input))))))
    permutations = [filtered_input, filtered_input_2]
    input_ids = []
    for p in permutations:
        if rtype == 1:
            s1, _ = pos_z0_64(p, n)
        elif rtype == 2:
            s1, _ = pos_lh_64(p, n)
        elif rtype == 3:
            s1, _ = pos_rh_64(p, n)
        else:
            s1, _ = addition_str_64(p, n)
        input_text = s1
        encoding = tokenizer(input_text, truncation=True, max_length=max_length, padding='max_length')
        input_id = torch.tensor(encoding['input_ids']).unsqueeze(0)
        input_ids.append(input_id)
    input_ids = torch.cat(input_ids, dim=0).to(input_ids_i.device)
    position_ids = torch.cat([position_ids_i] * len(permutations), dim=0).to(input_ids_i.device)
    attention_masks = torch.cat([attention_mask_i] * len(permutations), dim=0).to(input_ids_i.device)
    return True, input_ids, position_ids, attention_masks


def analyze_hidden_states(hidden_states_p, hidden_states_r):
    std_p = [diff_between(p[0, :], p[1, :]) for p in hidden_states_p]
    std_r = [diff_between(p[0, :], p[1, :]) for p in hidden_states_r]
    std_diff = [p - r for p, r in zip(std_p, std_r)]
    return std_diff


def visualize(model, dataloader, n=5, limit=np.inf, rtype=1, verbose=False):
    model.eval()
    results = []
    used = set()

    if 'tokenizer' in dir(dataloader.dataset):
        tokenizer = dataloader.dataset.tokenizer  # 50258
    else:
        tokenizer = dataloader.dataset.datasets[0].tokenizer  # 50258

    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            if verbose:
                print(f'eval_step:{j}')
            device = model.device
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            input_ids = input_ids.to(device)
            position_ids = position_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                for k in range(len(labels)):
                    input_ids_i = input_ids[k].unsqueeze(0)
                    position_ids_i = position_ids[k].unsqueeze(0)
                    attention_mask_i = attention_mask[k].unsqueeze(0)
                    # filtered_input = [x for x in tokenizer.decode(input_ids_i[0]).split(" ") if x not in ['[+]', '[=]', '[PAD]', '[EOS]']]
                    eq_token = tokenizer.encode(["[=]"])[0]
                    not_repeated, input_ids_p, position_ids_p, attention_masks_p = generate_all_permutations(used, n,
                                                                                                             tokenizer,
                                                                                                             input_ids_i,
                                                                                                             position_ids_i,
                                                                                                             attention_mask_i)
                    if not not_repeated:
                        continue
                    _, input_ids_r, position_ids_r, attention_masks_r = generate_all_permutations(used, n, tokenizer,
                                                                                                  input_ids_i,
                                                                                                  position_ids_i,
                                                                                                  attention_mask_i,
                                                                                                  rtype=rtype)
                    output_p = model.forward(
                        input_ids_p,
                        position_ids=position_ids_p,
                        attention_mask=attention_masks_p,
                        output_hidden_states=True,
                    )

                    hidden_states_p_all = []
                    for idx in range(input_ids_p.shape[0]):
                        eq_position = input_ids_p[idx].cpu().numpy().tolist().index(eq_token)
                        hidden_states_p = [p[idx, eq_position, :][np.newaxis, :].cpu().numpy() for p in
                                           output_p["hidden_states"]]
                        hidden_states_p_all.append(hidden_states_p)
                    hidden_states_p_zip = [np.concatenate(x, ) for x in list(zip(*hidden_states_p_all))]

                    output_r = model.forward(
                        input_ids_r,
                        position_ids=position_ids_r,
                        attention_mask=attention_masks_r,
                        output_hidden_states=True,
                    )

                    hidden_states_r_all = []
                    for idx in range(input_ids_r.shape[0]):
                        eq_position = input_ids_r[idx].cpu().numpy().tolist().index(eq_token)
                        hidden_states_r = [p[idx, eq_position, :][np.newaxis, :].cpu().numpy() for p in
                                           output_r["hidden_states"]]
                        hidden_states_r_all.append(hidden_states_r)
                    hidden_states_r_zip = [np.concatenate(x, ) for x in list(zip(*hidden_states_r_all))]

                    std_diff = analyze_hidden_states(hidden_states_p_zip, hidden_states_r_zip)
                    results.append(std_diff)
            if j > limit:
                break
    results = np.average(np.array(results), axis=0).tolist()
    return results


def vis_array(data, fname):
    plt.clf()
    plt.imshow(data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('2D Array Visualization')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.savefig(fname)
    json.dump(data.tolist(), open(fname.replace(".png", ".json"), "w"), indent=2)


def visualize_wrap(ckpt_path, dataset_path, rtype=1, limit=20):
    # Access the parsed arguments
    checkpoint_path = sorted(glob.glob(os.path.join(ckpt_path, "*")))[-1]
    dataset_path = dataset_path

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n = int(os.path.basename(dataset_path).split("_")[-2])

    # Assuming EvalDataset is defined elsewhere in your code

    model = GPT2LMHeadModel.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.to(device)
    dataloader_eval_com = DataLoader(EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'), ftype='test_ide'),
                                     batch_size=32, shuffle=False)  # Multiple Batch Size requires paddings
    diff_test = visualize(model, dataloader_eval_com, n=n, rtype=rtype, limit=limit)
    return {"diff_test": diff_test}


def run(data_prefix):
    for p in [7]:
        for rtype in [1, 2, 3]:
            diff_test = []
            for i in [100, 300, 1000, 3000, 10000]:
                dataset_name = f"{data_prefix}_{p}_{i}"
                print(f"dataset_name={dataset_name},rtype={rtype}")
                ckpt_paths = glob.glob(f"results/{dataset_name}_seqnew-*/checkpoints")
                if len(ckpt_paths) > 0:
                    ckpt_path = ckpt_paths[0]
                    dataset_path = f'./data/{dataset_name}'
                    result = visualize_wrap(ckpt_path, dataset_path, rtype=rtype, limit=np.inf)
                    # diff_train.append(result["diff_train"])
                    diff_test.append(result["diff_test"])
            # diff_train = np.array(diff_train)
            diff_test = np.array(diff_test)
            # vis_array(diff_train,f"outputs/{p}_{rtype}_diff_train.png")
            vis_array(diff_test, f"results/{p}_{rtype}_ide_diff_test.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument('--data_prefix', type=str, default='all_64', help='dataset prefix')
    args = parser.parse_args()
    run(args.data_prefix)
