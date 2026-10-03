import argparse

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoModelForCausalLM, GPT2Tokenizer

from mydataset import EvalDataset, dataset_tokenizer
from utils import latest_checkpoint


def evaluate(model, dataloader, limit=np.inf, verbose=False):
    """Fraction of samples whose answer token is the model's top prediction."""
    model.eval()
    tokenizer = dataset_tokenizer(dataloader)
    correct = 0
    total = 0

    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            if verbose:
                print(f'eval_step:{j}')
            device = model.device
            input_ids = batch['input_ids'].to(device)
            output = model(
                input_ids,
                position_ids=batch['position_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
            )
            for k, label in enumerate(batch['labels'].numpy()):
                # The answer is predicted from the logits one position before it.
                pred = np.argmax(output.logits[k][batch['label_position'][k] - 1].cpu().numpy())
                if verbose:
                    print(f"input_token={tokenizer.decode(input_ids[k])}, label_token={tokenizer.decode(label)}, "
                          f"output_token={tokenizer.decode(pred)}")
                correct += int(label == pred)
                total += 1
            if j > limit:
                break
    return correct / total


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with checkpoint and dataset paths.')
    parser.add_argument('--ckpt_path', type=str, default="./results/checkpoints",
                        help='Directory of checkpoints; the one with the highest step is used.')
    parser.add_argument('--dataset_path', type=str, default='./data/all_64_7_100', help='Path to the dataset.')
    parser.add_argument('--dataset_type', type=str, default='com+ide', help='Dataset types, separated by `+`.')
    parser.add_argument('--max_length', type=int, default=15, help='Number of tokens every sample is padded to')
    args = parser.parse_args()

    checkpoint_path = latest_checkpoint(args.ckpt_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    model.to(device)

    def loader(ftype):
        return EvalDataset(args.dataset_path, GPT2Tokenizer.from_pretrained('gpt2'), ftype=ftype,
                           max_length=args.max_length)

    dataset_types = args.dataset_type.split("+")
    train_loader = DataLoader(ConcatDataset([loader(f'train_{t}') for t in dataset_types]), batch_size=32)
    print(f"{args.dataset_path}, Train accuracy: {evaluate(model, train_loader)}")
    for t in dataset_types:
        test_loader = DataLoader(loader(f'test_{t}'), batch_size=32)
        print(f"{args.dataset_path}, Test_{t} accuracy: {evaluate(model, test_loader)}")


if __name__ == '__main__':
    main()
