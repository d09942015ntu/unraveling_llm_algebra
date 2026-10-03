import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from mydataset import EvalDataset
from utils import latest_checkpoint


def evaluate(model, dataloader, limit=np.inf, verbose=False):
    """Generate a continuation for each input and check the token right after ``[=]`` against the label."""
    model.eval()
    tokenizer = dataloader.dataset.tokenizer
    equal_sign_id = tokenizer.encode(['[=]'])
    correct = 0
    incorrect = 0
    with torch.no_grad():
        for j, batch in enumerate(dataloader):
            if verbose:
                print(f'eval_step:{j}')
            input_ids = batch['input_ids'].to(model.device)
            outputs = model.generate(
                input_ids,
                num_beams=10,
                max_length=dataloader.dataset.max_length + 1,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_p=0.95,
                temperature=0.7,
                do_sample=True,
            )
            labels = np.array([label.numpy()[0] for label in batch['labels']])
            preds = np.zeros(labels.shape, dtype=np.int64)
            for k, output_seq in enumerate(outputs):
                for i in range(len(output_seq) - 1):
                    if output_seq[i].cpu().numpy() == equal_sign_id:
                        preds[k] = output_seq[i + 1].cpu().numpy()
                        break
            correct += np.sum(labels == preds)
            incorrect += np.sum(labels != preds)
            if j > limit:
                break
    return correct / (correct + incorrect)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with checkpoint and dataset paths.')
    parser.add_argument('--ckpt_path', type=str, default="./results/checkpoints",
                        help='Directory of checkpoints; the one with the highest step is used.')
    parser.add_argument('--dataset_path', type=str, default='./data/ide_41_11_9', help='Path to the dataset.')
    parser.add_argument('--tokenizer', type=str, default='./models/gpt2', help='Tokenizer name or path.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = GPT2LMHeadModel.from_pretrained(latest_checkpoint(args.ckpt_path), trust_remote_code=True)
    model.to(device)

    for ftype in ['train', 'test_com', 'test_ide']:
        # Batch size 1: inputs have different lengths and are not padded.
        dataloader = DataLoader(EvalDataset(args.dataset_path, GPT2Tokenizer.from_pretrained(args.tokenizer),
                                            ftype=ftype), batch_size=1, shuffle=False)
        print(f"{args.dataset_path}, {ftype} accuracy: {evaluate(model, dataloader, verbose=True)}")


if __name__ == '__main__':
    main()
