"""Generate the distributivity datasets (dist, distx, z0)."""
import os

import numpy as np

from datagen_common import RandomLabeler, count_descents, save_splits

random_label = RandomLabeler(seed=0)


def flatten(seq):
    for elem in seq:
        if isinstance(elem, (list, tuple)):
            yield from flatten(elem)
        else:
            yield elem


def partitions(lst):
    """All ways to split ``lst`` into consecutive non-empty groups, e.g. [1,2] -> [[(1,2)], [(1,),(2,)]]."""
    def _partitions(i):
        if i == len(lst):
            yield []
            return
        for j in range(i + 1, len(lst) + 1):
            for tail in _partitions(j):
                yield [tuple(lst[i:j])] + tail
    return list(_partitions(0))


def to_token(s):
    return f"[{s}]"


def dist_addmult_str_64(S, n):
    """S is a list of (a, (b1, b2, ...)) terms, read as a x (b1 + b2 + ...), all added together."""
    ans = 0
    terms = []
    for s_prod, s_sums in S:
        if len(s_sums) == 1:
            sub_str = to_token(s_sums[0])
        else:
            sub_str = f"[(]{'[+]'.join(to_token(s) for s in s_sums)}[)]"
        terms.append(f"{to_token(s_prod)}[x]{sub_str}")
        ans += s_prod * sum(s_sums)
    input_str = "[+]".join(terms) + "[=]"
    label_str = f"[{ans % n}]"
    return input_str, label_str


def dist_addmultx_str_64(S, n):
    """Random operators [o+] and [ox] that still satisfy distributivity."""
    input_str, _ = dist_addmult_str_64(S, n)
    input_str = input_str.replace("[+]", "[o+]").replace("[x]", "[ox]")
    s_key = tuple(sorted((s_prod, s) for s_prod, s_sums in S for s in s_sums))
    label_str = f"[r{random_label(s_key, n)}]"
    return input_str, label_str


def dist_z0_str_64(S, n):
    """Order-dependent operator [0->]: counts descents of the flattened operands."""
    input_str, _ = dist_addmult_str_64(S, n)
    input_str = input_str.replace("[+]", "[0->]").replace("[x]", "[0->]")
    label_str = f"[N{count_descents(list(flatten(S)))}]"
    return input_str, label_str


def dataset_dist_64(m=7, train_size=1000, test_size=1000, func=dist_addmultx_str_64):
    """Train on a*(b1+..)+c*(d1+..) in one grouping; test on the other groupings."""
    rng = np.random.RandomState(0)
    train_set = []
    test_set = []
    used = set()

    while True:
        n1 = rng.choice([2, 3, 4])
        n2 = 6 - n1
        S0 = tuple(rng.choice(range(1, m), 2))
        S1 = tuple(rng.choice(range(1, m), n1))
        S2 = tuple(rng.choice(range(1, m), n2))
        key = S0 + S1 + S2
        if key in used:
            continue
        used.add(key)

        samples = []
        for p1 in partitions(S1):
            for p2 in partitions(S2):
                samples.append([(S0[0], group) for group in p1] + [(S0[1], group) for group in p2])
        base_sample, other_samples = samples[0], samples[1:]

        if len(test_set) < test_size:
            test_set.extend(other_samples[:test_size - len(test_set)])
            train_set.append(base_sample)
        elif len(train_set) < train_size:
            train_set.extend(other_samples[:train_size - len(train_set)])
            train_set.append(base_sample)
        else:
            break

    return [func(s, m) for s in train_set], [func(s, m) for s in test_set]


def save_dataset(m=100, fname='dist_64', train_size=1000, test_size=1000):
    sizes = dict(m=m, train_size=train_size, test_size=test_size)
    splits = {
        'dist': dataset_dist_64(func=dist_addmult_str_64, **sizes),
        'distx': dataset_dist_64(func=dist_addmultx_str_64, **sizes),
        'z0': dataset_dist_64(func=dist_z0_str_64, **sizes),
    }
    save_splits(os.path.join('./data', f'{fname}_{m}_{train_size}'), splits)


def run():
    for m in [7, 11, 13]:
        for pn in [100, 300, 1000, 3000, 10000, 30000]:
            if m == 7 and pn > 10000:
                continue
            print(f"generate,m={m},pn={pn}")
            save_dataset(m, fname='dist_64', train_size=pn, test_size=1000)


if __name__ == '__main__':
    run()
