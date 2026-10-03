"""Generate the ide_41_<m>_<pn> datasets (commutativity, identity and inverse for +/-)."""
import csv
import itertools
import json
import math
import os.path
from collections import Counter

import numpy as np


def addition_str_41(S, n):
    input_str = ""
    for i,s in enumerate(S):
        if i == 0:
            if s >=0:
                input_str += f"[{abs(s)}]"
            else:
                input_str += f"[-][{abs(s)}]"
        else:
            if s >=0:
                input_str += f"[+][{abs(s)}]"
            else:
                input_str += f"[-][{abs(s)}]"
    input_str+="[=]"
    label_str=f"[{str((sum(S))%n)}]"
    return input_str,label_str

def add_traintest_com_41(S, train_set, test_set, rng, p):
    permutations = list(set(list(itertools.permutations(S, len(S)))))
    rng.shuffle(permutations)
    if len(permutations) == 1:
        split = 1
    elif len(permutations) == 2:
        split = int(rng.rand() > p)
    else:
        split = int(math.ceil(len(permutations) * (1-p)))
    train_i = permutations[:split]
    test_i = permutations[split:]
    train_set.extend(train_i)
    test_set.extend(test_i)

def add_traintest_ide_41(S, train_set, test_set, rng, p):
    permutations = list(set(list(itertools.permutations(S, len(S)))))
    if rng.rand() > p:
        train_set.extend(permutations)
    else:
        test_set.extend(permutations)


def add_traintest_inv_41(S, train_set, test_set, rng, p):
    is_test = int(rng.rand() < p)
    permutations = list(set(list(itertools.permutations(S, len(S)))))
    signs = list(itertools.product([-1, 1], repeat=len(S)))
    results = []
    for p in permutations:
        for s in signs:
            results.append(tuple(int(x*y) for x,y in zip(p,s)))
    results = list(set(results))

    counts = Counter([s for s in S if s > 0])
    if (len(counts) == 1 and list(counts.items())[0][-1] == 2) or \
            (len(counts) == 1 and list(counts.items())[0][-1] == 4) or \
            (len(counts) == 2 and list(counts.items())[0][-1] == 2 and list(counts.items())[1][-1] == 2):
        for result in results:
            if sum(result) == 0:
                if is_test:
                    if len([s for s in result if s < 0]) > 0:
                        test_set.append(result)
                else:
                    if len([s for s in result if s < 0]) > 0:
                        train_set.append(result)
            else:
                if len([s for s in result if s < 0]) > 0:
                    train_set.append(result)
    else:
        for result in results:
            if len([s for s in result if s < 0]) > 0:
                train_set.append(result)

def dataset_inv_41(rng, m=100, p=0.2):
    train_set_1=[]
    train_set_2=[]
    train_set_3=[]
    test_set_1=[]
    test_set_2=[]
    test_set_3=[]
    for a in range(0,m):
        for b in range(a,m):
            if a*b > 0:
                add_traintest_com_41([a, b], train_set_1, test_set_1, rng, p)
            if a == 0:
                add_traintest_ide_41([a, b], train_set_2, test_set_2, rng, p)
            add_traintest_inv_41([a, b], train_set_3, test_set_3, rng, p)
            for c in range(b, m):
                if a * b*c > 0:
                    add_traintest_com_41([a, b, c], train_set_1, test_set_1, rng, p)
                if a == 0:
                    add_traintest_ide_41([a, b, c], train_set_2, test_set_2, rng, p)
                add_traintest_inv_41([a, b, c], train_set_3, test_set_3, rng, p)
                for d in range(c, m):
                    if a * b * c * d > 0:
                        add_traintest_com_41([a, b, c, d], train_set_1, test_set_1, rng, p)
                    if a == 0:
                        add_traintest_ide_41([a, b, c, d], train_set_2, test_set_2, rng, p)
                    add_traintest_inv_41([a,b,c,d], train_set_3, test_set_3, rng, p)
    train_set_1 = [addition_str_41(s,m) for s in train_set_1]
    train_set_2 = [addition_str_41(s,m) for s in train_set_2]
    train_set_3 = [addition_str_41(s,m) for s in train_set_3]
    test_set_1 = [addition_str_41(s,m) for s in test_set_1]
    test_set_2 = [addition_str_41(s,m) for s in test_set_2]
    test_set_3 = [addition_str_41(s,m) for s in test_set_3]
    return train_set_1, train_set_2, train_set_3, test_set_1, test_set_2, test_set_3


def write_to_csv(save_set,csv_name,  rng):
    rng.shuffle(save_set)
    with open(csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['s1','s2'])
        for t in save_set:
            writer.writerow(t)

def save_dataset(m=100, pn=2):
    """Save to ./data/ide_41_<m>_<pn>; pn/10 is the fraction of samples held out for testing."""
    seed=0
    rng=np.random.RandomState(seed)
    train_set_1, train_set_2, train_set_3, test_set_1, test_set_2, test_set_3 = dataset_inv_41(rng, m, pn/10)
    save_path = os.path.join('./data', f'ide_41_{m}_{pn}')
    os.makedirs(save_path,exist_ok=True)

    write_to_csv(train_set_1+train_set_2+train_set_3, os.path.join(save_path,'train_all.csv'), rng)

    write_to_csv(train_set_1, os.path.join(save_path,'train_com.csv'), rng)
    write_to_csv(train_set_2, os.path.join(save_path,'train_ide.csv'), rng)
    write_to_csv(train_set_3, os.path.join(save_path,'train_inv.csv'), rng)

    write_to_csv(test_set_1+test_set_2+test_set_3, os.path.join(save_path,'test.csv'), rng)

    write_to_csv(test_set_1, os.path.join(save_path,'test_com.csv'), rng)
    write_to_csv(test_set_2, os.path.join(save_path,'test_ide.csv'), rng)
    write_to_csv(test_set_3, os.path.join(save_path,'test_inv.csv'), rng)


    write_to_csv(train_set_1+train_set_2, os.path.join(save_path,'train.csv'), rng)


    token_list_filename = os.path.join(save_path,'tokens.json')
    token_list = []
    token_list.append('[=]')
    token_list.append('[+]')
    token_list.append('[-]')
    token_list.append('[x]')
    token_list.append('[/]')
    for i in range(m):
        token_list.append(f'[{i}]')
    json.dump(token_list, open(token_list_filename,'w'), indent=2)


if __name__ == '__main__':
    for n in [5,7, 11]:
        for pn in [1,2,3,4,5,6,7,8,9]:
            save_dataset(n, pn=pn)
