import csv
from functools import reduce
import itertools
import json
import numpy as np
import os.path
import re

x_map = {}
x_rng = np.random.RandomState(0)


def addition_str_64(S, n):
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str = "[+]".join(input_str_list) + "[=]"
    label_str = f"[z{str((sum(S)) % n)}]"
    return input_str, label_str


def additionx_str_64(S, n):
    global x_map, x_rng
    s_key = tuple(sorted([x for x in S if x > 0]))
    s_val = x_map.get(s_key, -1)
    if s_val < 0:
        s_val = x_rng.randint(n)
        x_map[s_key] = s_val
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str = "[x]".join(input_str_list) + "[=]"
    label_str = f"[r{s_val}]"
    return input_str, label_str


def pos_z0_64(S, n):
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str = "[0->]".join(input_str_list) + "[=]"
    label_str = f"[N{sum([a >= b for a, b in zip(S[:-1], S[1:])])}]"
    return input_str, label_str


def pos_lh_64(S, n):
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str = "[<=]".join(input_str_list) + "[=]"
    label_str = input_str_list[0]
    return input_str, label_str


def pos_rh_64(S, n):
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str = "[=>]".join(input_str_list) + "[=]"
    label_str = input_str_list[-1]
    return input_str, label_str


def add_traintest_com_64(S, train_set, test_set, rng, p):
    permutations = sorted(list(set(list(itertools.permutations(S, len(S))))))
    train_i = permutations[0]
    test_i = permutations[1:]
    train_set.append(train_i)
    if len(test_i) > 0:
        test_set.append(test_i)


def add_traintest_ide_64(S, train_set, test_set, rng, p):
    permutations = list(set(list(itertools.permutations(S, len(S)))))
    test_set.append(permutations)


def dataset_com_64(m=100, train_size=1000, test_size=1000, func=addition_str_64, shuffle=True):
    seed = 0
    rng = np.random.RandomState(seed)
    train_set = []
    test_set = []
    used = set()

    while True:
        S = tuple(sorted(rng.choice(range(1, m), 6)))
        if S in used:
            continue
        else:
            used.add(S)
        permutations = sorted(list(set(list(itertools.permutations(S, len(S))))))
        rng.shuffle(permutations)
        permutations = permutations[:30]
        if len(permutations) == 1:
            continue
        if len(test_set) < test_size:
            # t_idx = rng.choice(range(len(permutations)))
            permutation_add = permutations[1:]
            test_set.extend(permutation_add[:test_size - len(test_set)])
            train_set.append(permutations[0])
        elif len(train_set) < train_size:
            train_set.extend(permutations[:train_size - len(train_set)])
        else:
            break

    # train_set.extend(train_set_raw)
    train_set = [func(s, m) for s in train_set]
    test_set = [func(s, m) for s in test_set]

    return train_set, test_set


def dataset_ide_64(m=100, train_size=1000, test_size=1000, func=addition_str_64, shuffle=True):
    seed = 0
    rng = np.random.RandomState(seed)
    train_set = []
    test_set = []
    used = set()

    while True:
        S = tuple(rng.choice(range(1, m), 5))
        if S in used:
            continue
        else:
            used.add(S)

        permutations = [list(S[:i]) + [0] + list(S[i:]) for i in range(len(S) + 1)]
        if len(permutations) == 1:
            continue
        if len(test_set) < test_size:
            test_set.extend(permutations[:test_size - len(test_set)])
            train_set.append(list(S))
        elif len(train_set) < train_size:
            train_set.extend(permutations[:train_size - len(train_set)])
            train_set.append(list(S))
        else:
            break

    train_set = [func(s, m) for s in train_set]
    test_set = [func(s, m) for s in test_set]
    return train_set, test_set


def write_to_csv(save_set, csv_name):
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(['s1', 's2'])
        for t in save_set:
            writer.writerow(t)


def save_dataset(m=100, fname='all_64', train_size=1000, test_size=1000):
    save_path = os.path.join('./data', f'{fname}_{m}_{train_size}')
    os.makedirs(save_path, exist_ok=True)

    train_set_com, test_set_com = dataset_com_64(m=m, train_size=train_size, test_size=test_size, func=addition_str_64,
                                                 shuffle=True)
    train_set_ide, test_set_ide = dataset_ide_64(m=m, train_size=train_size, test_size=test_size, func=addition_str_64,
                                                 shuffle=True)

    train_set_comx, test_set_comx = dataset_com_64(m=m, train_size=train_size, test_size=test_size,
                                                   func=additionx_str_64, shuffle=True)
    train_set_idex, test_set_idex = dataset_ide_64(m=m, train_size=train_size, test_size=test_size,
                                                   func=additionx_str_64, shuffle=True)

    all_sets = reduce(lambda a, b: a + b, [
        train_set_com,
        test_set_com,
        train_set_ide,
        test_set_ide,

        train_set_comx,
        test_set_comx,
        train_set_idex,
        test_set_idex,
    ])

    write_to_csv(train_set_com, os.path.join(save_path, 'train_com.csv'))
    write_to_csv(test_set_com, os.path.join(save_path, 'test_com.csv'))
    write_to_csv(train_set_ide, os.path.join(save_path, 'train_ide.csv'))
    write_to_csv(test_set_ide, os.path.join(save_path, 'test_ide.csv'))

    write_to_csv(train_set_comx, os.path.join(save_path, 'train_comx.csv'))
    write_to_csv(test_set_comx, os.path.join(save_path, 'test_comx.csv'))
    write_to_csv(train_set_idex, os.path.join(save_path, 'train_idex.csv'))
    write_to_csv(test_set_idex, os.path.join(save_path, 'test_idex.csv'))

    pos_funcs = {
        "z0": pos_z0_64,
        "lh": pos_lh_64,
        "rh": pos_rh_64
    }
    for ikey, pos_func in pos_funcs.items():
        train_set_com, test_set_com = dataset_com_64(m=m, train_size=train_size, test_size=test_size, func=pos_func,
                                                     shuffle=True)
        train_set_ide, test_set_ide = dataset_ide_64(m=m, train_size=train_size, test_size=test_size, func=pos_func,
                                                     shuffle=True)

        write_to_csv(train_set_com + train_set_ide, os.path.join(save_path, f'train_{ikey}.csv'))
        write_to_csv(test_set_com + test_set_ide, os.path.join(save_path, f'test_{ikey}.csv'))

        all_sets_temp = reduce(lambda a, b: a + b, [
            train_set_com,
            test_set_com,
            train_set_ide,
            test_set_ide,
        ])

        all_sets.extend(all_sets_temp)

    all_tokens = set()
    for row in all_sets:
        for s in row:
            tokens = re.findall(r'\[.*?\]', s)
            for t in tokens:
                all_tokens.add(t)
    all_tokens = sorted(list(all_tokens))

    token_list_filename = os.path.join(save_path, 'tokens.json')
    json.dump(all_tokens, open(token_list_filename, 'w'), indent=2)


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
