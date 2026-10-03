import json
import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class TrainDataset(Dataset):
    """Rows of ``<ftype>.csv``: ``s1`` is the expression (ending in ``[=]``), ``s2`` is the answer token."""

    def __init__(self, data_dir, tokenizer, ftype='train', rm_position=0, max_length=15):
        self.data = pd.read_csv(os.path.join(data_dir, f"{ftype}.csv"), sep=r'\s+')
        with open(os.path.join(data_dir, "tokens.json")) as f:
            tokens = json.load(f)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens({'additional_special_tokens': tokens})
        self.tokenizer = tokenizer
        self.rm_position = rm_position
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _encode(self, idx):
        row = self.data.iloc[idx]
        s1 = row['s1']
        s2 = row['s2']
        encoding = self.tokenizer(s1, truncation=True, max_length=self.max_length, padding='max_length')
        encoding_full = self.tokenizer(s1 + s2, truncation=True, max_length=self.max_length, padding='max_length')
        s1_len = len(self.tokenizer.encode(s1))
        if s1_len >= self.max_length:
            raise ValueError(f"'{s1}' has {s1_len} tokens; increase max_length (now {self.max_length}).")

        position_ids = list(range(len(encoding['input_ids'])))
        if self.rm_position:
            # All operands share position 0, so the model cannot use their order.
            position_ids[:s1_len - 1] = [0] * (s1_len - 1)
            position_ids[s1_len - 1:] = list(range(1, len(position_ids) - s1_len + 2))

        item = {
            'input_ids': torch.tensor(encoding['input_ids']),
            'position_ids': torch.tensor(position_ids),
            'attention_mask': torch.tensor(encoding['attention_mask']),
        }
        return item, encoding_full['input_ids'], s1_len

    def __getitem__(self, idx):
        item, full_ids, s1_len = self._encode(idx)
        # Only the answer token (right after s1) is trained on.
        labels = [-100] * len(full_ids)
        labels[s1_len] = full_ids[s1_len]
        item['labels'] = torch.tensor(labels)
        return item


class EvalDataset(TrainDataset):
    """Like TrainDataset, but ``labels`` is just the answer token id and ``label_position`` is its index."""

    def __getitem__(self, idx):
        item, full_ids, s1_len = self._encode(idx)
        item['label_position'] = s1_len
        item['labels'] = torch.tensor(full_ids[s1_len])
        return item


def dataset_tokenizer(dataloader):
    """The tokenizer of a DataLoader built on a dataset above or on a ConcatDataset of them."""
    dataset = dataloader.dataset
    if hasattr(dataset, 'tokenizer'):
        return dataset.tokenizer
    return dataset.datasets[0].tokenizer


if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset = TrainDataset('data/all_64_7_100', tokenizer, ftype="train_com", rm_position=1)
    for item in dataset:
        for key, value in item.items():
            if key in ('input_ids', 'labels'):
                raw = dataset.tokenizer.decode(value[value > 0])
            else:
                raw = "0"
            print(f"{key}: {raw} : {value}")
