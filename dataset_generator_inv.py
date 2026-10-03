"""Generate the inverse-element datasets (inv, invx, lh, rh)."""
import os

import numpy as np

from datagen_common import RandomLabeler, save_splits

random_label = RandomLabeler(seed=0)


def operand_tokens(S):
    return [f"[{s}]" for s in S]


def addition_str_64i(S, n):
    """Operator +: sum modulo n (operands may be negative)."""
    input_str = "[+]".join(operand_tokens(S)) + "[=]"
    label_str = f"[{sum(S) % n}]"
    return input_str, label_str


def additionx_str_64i(S, n):
    """Random operator: the label depends only on the positive operands."""
    s_key = tuple(sorted(x for x in S if x > 0))
    input_str = "[x]".join(operand_tokens(S)) + "[=]"
    label_str = f"[r{random_label(s_key, n)}]"
    return input_str, label_str


def pos_lh_64i(S, n):
    tokens = operand_tokens(S)
    return "[<=]".join(tokens) + "[=]", tokens[0]


def pos_rh_64i(S, n):
    tokens = operand_tokens(S)
    return "[=>]".join(tokens) + "[=]", tokens[-1]


def dataset_inv_64i(m=100, train_size=1000, test_size=1000, func=addition_str_64i):
    """Inverse: test on 4 operands with a pair (-s, s) or (s, -s) inserted at every position."""
    rng = np.random.RandomState(0)
    train_set = []
    test_set = []
    used = set()

    values = list(range(1, m)) + list(range(-m + 1, 0))
    while True:
        S = tuple(rng.choice(values, 4))
        s2 = rng.choice(range(1, m)) * rng.choice([-1, 1])
        key = S + (s2,)
        if key in used:
            continue
        used.add(key)

        with_pair = [list(S[:i]) + [-s2, s2] + list(S[i:]) for i in range(len(S) + 1)]
        with_pair += [list(S[:i]) + [s2, -s2] + list(S[i:]) for i in range(len(S) + 1)]
        if len(test_set) < test_size:
            test_set.extend(with_pair[:test_size - len(test_set)])
            train_set.append(list(S))
        elif len(train_set) < train_size:
            train_set.extend(with_pair[:train_size - len(train_set)])
            train_set.append(list(S))
        else:
            break

    return [func(s, m) for s in train_set], [func(s, m) for s in test_set]


def save_dataset(m=100, fname='inv_64', train_size=1000, test_size=1000):
    sizes = dict(m=m, train_size=train_size, test_size=test_size)
    splits = {
        'inv': dataset_inv_64i(func=addition_str_64i, **sizes),
        'invx': dataset_inv_64i(func=additionx_str_64i, **sizes),
        'lh': dataset_inv_64i(func=pos_lh_64i, **sizes),
        'rh': dataset_inv_64i(func=pos_rh_64i, **sizes),
    }
    save_splits(os.path.join('./data', f'{fname}_{m}_{train_size}'), splits)


def run():
    for m in [7, 11, 13]:
        for pn in [200, 600, 2000, 6000]:
            print(f"generate,m={m},pn={pn}")
            save_dataset(m, fname='inv_64', train_size=pn, test_size=1000)


if __name__ == '__main__':
    run()
