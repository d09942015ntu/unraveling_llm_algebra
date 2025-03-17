import argparse
import glob
import os

import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
import torch

from mydataset import EvalDataset


def evaluate(model, dataloader, limit=np.inf, verbose=False):
    model.eval()
    correct = 0
    incorrect = 0

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
                output2 = model.forward(
                    input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
                for k in range(len(labels)):
                    labels_m1 = labels[k].cpu().numpy()  # np.array([label.cpu().numpy()[0] for label in labels])
                    label_pos = batch['label_position'][k]
                    outputs_m1 = np.argmax(output2.logits[k][label_pos - 1].cpu().numpy())
                    if verbose:
                        input_token = tokenizer.decode(input_ids[k])
                        label_token = tokenizer.decode(labels_m1)
                        outputs_token = tokenizer.decode(outputs_m1)
                        print(f"input_token={input_token}, label_token={label_token}, output_token={outputs_token}")
                    correct += int(labels_m1 == outputs_m1)
                    incorrect += int(labels_m1 != outputs_m1)
                if j > limit:
                    break
    return correct / (correct + incorrect)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with checkpoint and dataset paths.')
    parser.add_argument('--ckpt_path', type=str, default="./results/checkpoints",  # "./results/checkpoint-245"
                        help='Path to the checkpoint file.')
    parser.add_argument('--dataset_path', type=str, default='./data/all_64_7_100',
                        help='Path to the dataset file.')

    args = parser.parse_args()

    # Access the parsed arguments
    checkpoint_path = sorted(glob.glob(os.path.join(args.ckpt_path, "*")))[-1]
    dataset_path = args.dataset_path

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Assuming EvalDataset is defined elsewhere in your code

    model = GPT2LMHeadModel.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.to(device)
    dataset_train_com = EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'),
                                    ftype='train_com')  # Multiple Batch Size requires paddings
    dataset_train_ide = EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'),
                                    ftype='train_ide')  # Multiple Batch Size requires paddings
    dataset_train = torch.utils.data.ConcatDataset([dataset_train_com, dataset_train_ide])
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=False)  # Multiple Batch Size requires paddings
    dataloader_eval_com = DataLoader(EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'), ftype='test_com'),
                                     batch_size=32, shuffle=False)  # Multiple Batch Size requires paddings
    dataloader_eval_ide = DataLoader(EvalDataset(dataset_path, GPT2Tokenizer.from_pretrained('gpt2'), ftype='test_ide'),
                                     batch_size=32, shuffle=False)  # Multiple Batch Size requires paddings
    print(f"{args.dataset_path},Train accuracy: {evaluate(model, dataloader_train)}")
    print(f"{args.dataset_path}, Test_com accuracy: {evaluate(model, dataloader_eval_com)}")
    print(f"{args.dataset_path}, Test_ide accuracy: {evaluate(model, dataloader_eval_ide)}")


if __name__ == '__main__':
    main()
