import csv
import itertools
import json
import math
import os.path
import random
import re


from collections import Counter
from functools import reduce
import numpy as np

x_map = {}

x_rng = np.random.RandomState(0)


def addition_str_64i(S, n):
    input_str_list = [f"[{s}]" for s in S]
    input_str="[+]".join(input_str_list)+"[=]"
    label_str=f"[{str((sum(S))%n)}]"
    return input_str,label_str


def additionx_str_64i(S, n):
    global x_map, x_rng
    s_key = tuple(sorted([x for x in S if x > 0]))
    s_val = x_map.get(s_key,-1)
    if s_val < 0:
        s_val = x_rng.randint(n)
        x_map[s_key] = s_val
    input_str_list = [f"[{s}]" for s in S]
    input_str="[x]".join(input_str_list)+"[=]"
    label_str=f"[r{s_val}]"
    return input_str,label_str


def pos_z0_64i(S, n):
    input_str_list = [f"[{s}]" for s in S]
    input_str="[0->]".join(input_str_list)+"[=]"
    label_str = f"[N{sum([a>=b for a,b in zip(S[:-1],S[1:])])}]"
    return input_str,label_str


def pos_lh_64i(S, n):
    input_str_list = [f"[{s}]" for s in S]
    input_str="[<=]".join(input_str_list)+"[=]"
    label_str = input_str_list[0]
    return input_str,label_str

def pos_rh_64i(S, n):
    input_str_list = [f"[{s}]" for s in S]
    input_str="[=>]".join(input_str_list)+"[=]"
    label_str = input_str_list[-1]
    return input_str,label_str

def add_traintest_com_64i(S, train_set, test_set, rng, p):
    permutations = sorted(list(set(list(itertools.permutations(S, len(S))))))
    train_i = permutations[0]
    test_i = permutations[1:]
    train_set.append(train_i)
    if len(test_i) > 0:
        test_set.append(test_i)

def add_traintest_ide_64i(S, train_set, test_set, rng, p):
    permutations = list(set(list(itertools.permutations(S, len(S)))))
    test_set.append(permutations)



def dataset_ide_64i(m=100, train_size=1000, test_size=1000, func=addition_str_64i, shuffle=True):
    seed=0
    rng=np.random.RandomState(seed)
    train_set = []
    test_set = []
    used = set()

    values = list(range(1,m)) + list(range(-m+1,0))
    while True:
        S = tuple(rng.choice(values,4))
        s2 = rng.choice(range(1,m))*rng.choice([-1,1])
        skey = S+(s2,)
        if skey in used:
            continue
        else:
            used.add(skey)

        permutations = [list(S[:i]) +[-s2, s2] + list(S[i:]) for i in range(len(S)+1)]
        permutations += [list(S[:i]) +[s2, -s2] + list(S[i:]) for i in range(len(S)+1)]
        if len(permutations) == 1:
            continue
        if len(test_set) < test_size:
            test_set.extend(permutations[:test_size-len(test_set)])
            train_set.append(list(S))
        elif len(train_set) < train_size:
            train_set.extend(permutations[:train_size-len(train_set)])
            train_set.append(list(S))
        else:
            break


    train_set = [func(s,m) for s in train_set]
    test_set = [func(s,m) for s in test_set]
    return train_set, test_set





def write_to_csv(save_set,csv_name):
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(['s1','s2'])
        #save_set = sorted(save_set, key=lambda x: (len(x[0]),x[0]))
        for t in save_set:
            writer.writerow(t)

def save_dataset(m=100, fname='all_64i', train_size=1000, test_size=1000):

    save_path = os.path.join('./data', f'{fname}_{m}_{train_size}')
    os.makedirs(save_path,exist_ok=True)

    train_set_ide, test_set_ide = dataset_ide_64i(m=m, train_size=train_size, test_size=test_size, func=addition_str_64i, shuffle=True)

    train_set_idex, test_set_idex = dataset_ide_64i(m=m, train_size=train_size, test_size=test_size, func=additionx_str_64i, shuffle=True)

    all_sets = reduce(lambda a,b:a+b,[
        train_set_ide,
        test_set_ide,

        train_set_idex,
        test_set_idex,
    ])

    write_to_csv(train_set_ide, os.path.join(save_path, 'train_inv.csv'))
    write_to_csv(test_set_ide, os.path.join(save_path, 'test_inv.csv'))

    write_to_csv(train_set_idex, os.path.join(save_path, 'train_invx.csv'))
    write_to_csv(test_set_idex, os.path.join(save_path, 'test_invx.csv'))

    pos_funcs = {
        #"z0": pos_z0_64i,
        "lh": pos_lh_64i,
        "rh": pos_rh_64i
    }
    for ikey, pos_func in pos_funcs.items():
        train_set_ide, test_set_ide = dataset_ide_64i(m=m, train_size=train_size, test_size=test_size, func=pos_func, shuffle=True)

        write_to_csv(train_set_ide, os.path.join(save_path, f'train_{ikey}.csv'))
        write_to_csv(test_set_ide, os.path.join(save_path, f'test_{ikey}.csv'))

        all_sets_temp = reduce(lambda a, b: a + b, [
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

    token_list_filename = os.path.join(save_path,'tokens.json')
    json.dump(all_tokens, open(token_list_filename,'w'), indent=2)



def run():
    for m in [7, 11, 13]:
        for pn in [200,600,2000,6000]:
            if m==7 and pn > 10000:
                continue
            print(f"generate,m={m},pn={pn}")
            save_dataset(m, fname='inv_64', train_size=pn, test_size=1000)

if __name__ == '__main__':
    run()
