"""Generate the commutativity / identity datasets (com, ide, comx, idex, z0, lh, rh)."""
import itertools
import os

import numpy as np

from datagen_common import RandomLabeler, count_descents, save_splits

random_label = RandomLabeler(seed=0)


def operand_tokens(S):
    return [f"[z{abs(s)}]" for s in S]


def addition_str_64(S, n):
    """Operator +: sum modulo n."""
    input_str = "[+]".join(operand_tokens(S)) + "[=]"
    label_str = f"[z{sum(S) % n}]"
    return input_str, label_str


def additionx_str_64(S, n):
    """Operator (+) : a random but commutative operator, with 0 as identity."""
    s_key = tuple(sorted(x for x in S if x > 0))
    input_str = "[x]".join(operand_tokens(S)) + "[=]"
    label_str = f"[r{random_label(s_key, n)}]"
    return input_str, label_str


def pos_z0_64(S, n):
    """Operator (-): counts the descents, so it depends on the order."""
    input_str = "[0->]".join(operand_tokens(S)) + "[=]"
    label_str = f"[N{count_descents(S)}]"
    return input_str, label_str


def pos_lh_64(S, n):
    """Operator <|: returns the leftmost operand."""
    tokens = operand_tokens(S)
    return "[<=]".join(tokens) + "[=]", tokens[0]


def pos_rh_64(S, n):
    """Operator |>: returns the rightmost operand."""
    tokens = operand_tokens(S)
    return "[=>]".join(tokens) + "[=]", tokens[-1]


def dataset_com_64(m=100, train_size=1000, test_size=1000, func=addition_str_64):
    """Commutativity: train on one ordering of 6 operands, test on other orderings."""
    rng = np.random.RandomState(0)
    train_set = []
    test_set = []
    used = set()

    while True:
        S = tuple(sorted(rng.choice(range(1, m), 6)))
        if S in used:
            continue
        used.add(S)
        permutations = sorted(set(itertools.permutations(S, len(S))))
        rng.shuffle(permutations)
        permutations = permutations[:30]
        if len(permutations) == 1:
            continue
        if len(test_set) < test_size:
            test_set.extend(permutations[1:][:test_size - len(test_set)])
            train_set.append(permutations[0])
        elif len(train_set) < train_size:
            train_set.extend(permutations[:train_size - len(train_set)])
        else:
            break

    return [func(s, m) for s in train_set], [func(s, m) for s in test_set]


def dataset_ide_64(m=100, train_size=1000, test_size=1000, func=addition_str_64):
    """Identity: test on 5 operands with a 0 inserted at every position."""
    rng = np.random.RandomState(0)
    train_set = []
    test_set = []
    used = set()

    while True:
        S = tuple(rng.choice(range(1, m), 5))
        if S in used:
            continue
        used.add(S)

        with_zero = [list(S[:i]) + [0] + list(S[i:]) for i in range(len(S) + 1)]
        if len(test_set) < test_size:
            test_set.extend(with_zero[:test_size - len(test_set)])
            train_set.append(list(S))
        elif len(train_set) < train_size:
            train_set.extend(with_zero[:train_size - len(train_set)])
            train_set.append(list(S))
        else:
            break

    return [func(s, m) for s in train_set], [func(s, m) for s in test_set]


def save_dataset(m=100, fname='all_64', train_size=1000, test_size=1000):
    sizes = dict(m=m, train_size=train_size, test_size=test_size)
    splits = {
        'com': dataset_com_64(func=addition_str_64, **sizes),
        'ide': dataset_ide_64(func=addition_str_64, **sizes),
        'comx': dataset_com_64(func=additionx_str_64, **sizes),
        'idex': dataset_ide_64(func=additionx_str_64, **sizes),
    }
    # Operators without commutativity or identity: com and ide samples are stored together.
    for name, func in [("z0", pos_z0_64), ("lh", pos_lh_64), ("rh", pos_rh_64)]:
        train_com, test_com = dataset_com_64(func=func, **sizes)
        train_ide, test_ide = dataset_ide_64(func=func, **sizes)
        splits[name] = (train_com + train_ide, test_com + test_ide)

    save_splits(os.path.join('./data', f'{fname}_{m}_{train_size}'), splits)


def run():
    for m in [7, 11, 13]:
        for pn in [100, 300, 1000, 3000, 10000, 30000]:
            if m == 7 and pn > 10000:
                continue
            print(f"generate,m={m},pn={pn}")
            save_dataset(m, fname='all_64', train_size=pn, test_size=1000)


def quick_run():
    for m in [7, 11, 13]:
        for pn in [100, 300]:
            print(f"generate,m={m},pn={pn}")
            save_dataset(m, fname='quick_64', train_size=pn, test_size=200)


if __name__ == '__main__':
    quick_run()
    run()
