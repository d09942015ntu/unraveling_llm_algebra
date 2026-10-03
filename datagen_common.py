"""Helpers shared by the dataset generator scripts."""
import csv
import json
import os
import re

import numpy as np


class RandomLabeler:
    """Gives every key a fixed random label in ``range(n)``.

    Used for the "random" operators (e.g. ``[x]``): the label depends only on the
    key, never on the order the operands appear in. The table is kept for the
    whole run, so the same key keeps its label across all datasets generated in
    one process.
    """

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.table = {}

    def __call__(self, key, n):
        if key not in self.table:
            self.table[key] = self.rng.randint(n)
        return self.table[key]


def count_descents(seq):
    """Number of adjacent pairs (a, b) in ``seq`` with a >= b."""
    return sum(a >= b for a, b in zip(seq[:-1], seq[1:]))


def write_to_csv(rows, csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(['s1', 's2'])
        writer.writerows(rows)


def write_tokens(rows, save_path):
    """Write every ``[...]`` token that appears in ``rows`` to ``tokens.json``."""
    tokens = set()
    for row in rows:
        for text in row:
            tokens.update(re.findall(r'\[.*?\]', text))
    with open(os.path.join(save_path, 'tokens.json'), 'w') as f:
        json.dump(sorted(tokens), f, indent=2)


def save_splits(save_path, splits):
    """Save ``{name: (train_rows, test_rows)}`` as train_<name>.csv / test_<name>.csv plus tokens.json."""
    os.makedirs(save_path, exist_ok=True)
    all_rows = []
    for name, (train_rows, test_rows) in splits.items():
        write_to_csv(train_rows, os.path.join(save_path, f'train_{name}.csv'))
        write_to_csv(test_rows, os.path.join(save_path, f'test_{name}.csv'))
        all_rows += train_rows + test_rows
    write_tokens(all_rows, save_path)
