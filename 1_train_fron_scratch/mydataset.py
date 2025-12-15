import os.path
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import torch


class TrainDataset(Dataset):
    def __init__(self, data_dir, tokenizer, ftype='train', rm_position=0):
        self.data = pd.read_csv(os.path.join(data_dir, f"{ftype}.csv"), delim_whitespace=True)
        tokens = json.load(open(os.path.join(data_dir, "tokens.json"), "r"))
        special_tokens_dict = {'additional_special_tokens': tokens}
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer = tokenizer
        self.max_length = 15
        self.rng = np.random.RandomState(0)
        self.rm_position = rm_position

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Randomly choose a starting point
        # Match s_i as input and s_{i+1} as the target
        s1 = row['s1']
        s2 = row['s2']

        input_text = s1
        full_text = s1 + s2

        encoding = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding='max_length')
        encoding_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding='max_length')
        position_ids = list(range(len(encoding['input_ids'])))
        s1_encoded = self.tokenizer.encode(s1)
        if self.rm_position:
            position_ids[:len(s1_encoded) - 1] = [0] * (len(s1_encoded) - 1)
            position_ids[len(s1_encoded) - 1:] = list(range(1, len(position_ids[len(s1_encoded) - 1:]) + 1))

        labels = encoding_full["input_ids"].copy()
        labels[:len(s1_encoded)] = [-100] * len(s1_encoded)  # Mask `s1` tokens
        labels[len(s1_encoded) + 1:] = [-100] * len(labels[len(s1_encoded) + 1:])  # Mask `s1` tokens

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'position_ids': torch.tensor(position_ids),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'labels': torch.tensor(labels)
        }


class EvalDataset(TrainDataset):
    def __init__(self, file_path, tokenizer, ftype='test', rm_position=0):
        super().__init__(file_path, tokenizer, ftype=ftype, rm_position=rm_position)
        self.max_length = 15

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Randomly choose a starting point
        # Match s_i as input and s_{i+1} as the target
        s1 = row['s1']
        s2 = row['s2']

        input_text = s1
        full_text = s1 + s2

        encoding = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding='max_length')
        encoding_full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, padding='max_length')
        position_ids = list(range(len(encoding['input_ids'])))
        s1_encoded = self.tokenizer.encode(s1)
        if self.rm_position:
            position_ids[:len(s1_encoded) - 1] = [0] * (len(s1_encoded) - 1)
            position_ids[len(s1_encoded) - 1:] = list(range(1, len(position_ids[len(s1_encoded) - 1:]) + 1))

        labels = encoding_full["input_ids"][len(s1_encoded)]

        return {
            'input_ids': torch.tensor(encoding['input_ids']),
            'position_ids': torch.tensor(position_ids),
            'attention_mask': torch.tensor(encoding['attention_mask']),
            'label_position': len(s1_encoded),
            'labels': torch.tensor(labels)
        }


if __name__ == '__main__':
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    data_name = 'data/all_64_7_100'
    dataset = TrainDataset(data_name, tokenizer, ftype="train_com", rm_position=1)
    for item in dataset:
        for key, value in item.items():
            if key == 'input_ids' or key == 'labels':
                raw = dataset.tokenizer.decode(value[value > 0])
            else:
                raw = "0"
            print(f"{key}: {raw} : {value}")  # Adjust processing logic as necessary
