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


def flatten(seq):
    for elem in seq:
        if isinstance(elem, (list, tuple)):
            yield from flatten(elem)
        else:
            yield elem

def flatten_to_list(seq):
    return list(flatten(seq))


def selection_map_old(rng, S, m, n=7, seed=0):
    k = 4

    m1 = m - 1

    out_strs = []
    list_nm1 = list(range(m1))
    SG = [tuple(list_nm1)]
    sg_cache = set(SG)

    while len(sg_cache) < k:
        selected_group = sorted(list_nm1 + rng.choice(list_nm1, m1-1).tolist())
        sg = tuple(selected_group[:m1])
        if sg in sg_cache:
            continue
        sg_cache.add(sg)
        SG.append(sg)

    for sg in SG:
        A = []
        B = []
        out_str = []

        for si, ss in zip(list(sg)+[999],list(S[1:])+[999]):
            if len(A) == 0:
                pass
            elif si in A:
                pass
            else:
                if len(A) > 1:
                    temp_str = []
                    while len(A) > 0:
                        ssb = B.pop()
                        temp_str.append(ssb)
                        A.pop()
                    temp_str = (S[0], tuple(reversed(temp_str)))
                    out_str.append(temp_str)
                else:
                    ssb = B.pop()
                    A.pop()
                    temp_str = (S[0], (ssb,))
                    out_str.append(temp_str)
            A.append(si)
            B.append(ss)
        out_strs.append(out_str)
    #print(1)



def addition_str_64(S, n):
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str="[+]".join(input_str_list)+"[=]"
    label_str=f"[z{str((sum(S))%n)}]"
    return input_str, label_str


def additionx_str_64(S, n):
    global x_map, x_rng
    s_key = tuple(sorted([x for x in S if x > 0]))
    s_val = x_map.get(s_key,-1)
    if s_val < 0:
        s_val = x_rng.randint(n)
        x_map[s_key] = s_val
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str="[x]".join(input_str_list)+"[=]"
    label_str=f"[r{s_val}]"
    return input_str, label_str


def pos_z0_64(S, n):
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str="[0->]".join(input_str_list)+"[=]"
    label_str = f"[N{sum([a>=b for a,b in zip(S[:-1],S[1:])])}]"
    return input_str,label_str


def pos_lh_64(S, n):
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str="[<=]".join(input_str_list)+"[=]"
    label_str = input_str_list[0]
    return input_str, label_str

def pos_rh_64(S, n):
    input_str_list = [f"[z{abs(s)}]" for s in S]
    input_str="[=>]".join(input_str_list)+"[=]"
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
    seed=0
    rng=np.random.RandomState(seed)
    train_set = []
    test_set = []
    used = set()

    while True:
        S = tuple(sorted(rng.choice(range(1,m),6)))
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
            #t_idx = rng.choice(range(len(permutations)))
            permutation_add = permutations[1:]
            test_set.extend(permutation_add[:test_size-len(test_set)])
            train_set.append(permutations[0])
        elif len(train_set) < train_size:
            train_set.extend(permutations[:train_size-len(train_set)])
        else:
            break

    #train_set.extend(train_set_raw)
    train_set = [func(s,m) for s in train_set]
    test_set = [func(s,m) for s in test_set]

    return train_set, test_set


def dataset_ide_64(m=100, train_size=1000, test_size=1000, func=addition_str_64, shuffle=True):
    seed=0
    rng=np.random.RandomState(seed)
    train_set = []
    test_set = []
    used = set()

    while True:
        S = tuple(rng.choice(range(1,m),5))
        if S in used:
            continue
        else:
            used.add(S)

        permutations = [list(S[:i]) +[0] + list(S[i:]) for i in range(len(S)+1)]
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

def save_dataset(m=100, fname='dist_64', train_size=1000, test_size=1000):

    save_path = os.path.join('./data', f'{fname}_{m}_{train_size}')
    os.makedirs(save_path,exist_ok=True)

    train_set_dist, test_set_dist = dataset_dist_64(m=m, train_size=train_size, test_size=test_size, func=dist_addmult_str_64, shuffle=True)
    train_set_distx, test_set_distx = dataset_dist_64(m=m, train_size=train_size, test_size=test_size, func=dist_addmultx_str_64, shuffle=True)
    train_set_z0, test_set_z0 = dataset_dist_64(m=m, train_size=train_size, test_size=test_size, func=dist_z0_str_64, shuffle=True)

    all_sets = reduce(lambda a,b:a+b,[
        train_set_dist,
        test_set_dist,
        train_set_distx,
        test_set_distx,
        train_set_z0,
        test_set_z0,
    ])

    write_to_csv(train_set_dist, os.path.join(save_path, 'train_dist.csv'))
    write_to_csv(test_set_dist, os.path.join(save_path, 'test_dist.csv'))
    write_to_csv(train_set_distx, os.path.join(save_path, 'train_distx.csv'))
    write_to_csv(test_set_distx, os.path.join(save_path, 'test_distx.csv'))

    write_to_csv(train_set_z0, os.path.join(save_path, 'train_z0.csv'))
    write_to_csv(test_set_z0, os.path.join(save_path, 'test_z0.csv'))

    all_tokens = set()
    for row in all_sets:
        for s in row:
            tokens = re.findall(r'\[.*?\]', s)
            for t in tokens:
                all_tokens.add(t)
    all_tokens = sorted(list(all_tokens))

    token_list_filename = os.path.join(save_path,'tokens.json')
    json.dump(all_tokens, open(token_list_filename,'w'), indent=2)



def partitions(lst):
    def _partitions(i):
        if i == len(lst):
            yield []
            return
        for j in range(i+1, len(lst)+1):
            for tail in _partitions(j):
                yield [tuple(lst[i:j])] + tail
    return list(_partitions(0))

# Example usage:


def to_token(s):
    return f"[{s}]"


def dist_addmult_str_64(S, n):
    #print(S)
    ans = 0
    input_str_list = []
    for s_prod, s_sums in S:
        if len(s_sums) == 1:
            sub_str = to_token(s_sums[0])
            pass
        else:
            sub_str=f"[(]{'[+]'.join([to_token(s) for s in s_sums])}[)]"
        sub_str = f"{to_token(s_prod)}[x]{sub_str}"
        ans += s_prod*sum(s_sums)
        input_str_list.append(sub_str)
    input_str="[+]".join(input_str_list)+"[=]"
    label_str=f"[{str((ans)%n)}]"
    return input_str, label_str


def dist_addmultx_str_64(S, n):
    global x_map, x_rng
    input_str, _ = dist_addmult_str_64(S, n)
    input_str = input_str.replace("[+]","[o+]").replace("[x]","[ox]")
    s_new = []
    for si in S:
        for sii in si[1]:
            s_new.append((si[0],sii))
    s_key = tuple(sorted(s_new))
    s_val = x_map.get(s_key,-1)
    if s_val < 0:
        s_val = x_rng.randint(n)
        x_map[s_key] = s_val
    label_str=f"[r{s_val}]"
    return input_str, label_str


def dist_z0_str_64(S, n):
    global x_map, x_rng
    input_str, _ = dist_addmult_str_64(S, n)
    input_str = input_str.replace("[+]","[0->]").replace("[x]","[0->]")
    sf = flatten_to_list(S)
    label_str = f"[N{sum([a>=b for a,b in zip(sf[:-1],sf[1:])])}]"
    return input_str, label_str



def dataset_dist_64(m=7, train_size=1000, test_size=1000, func=dist_addmultx_str_64, shuffle=True):
    seed=0
    rng=np.random.RandomState(seed)
    train_set = []
    test_set = []
    used = set()

    while True:
        n1 = rng.choice([2,3,4])
        n2 = 6-n1
        S0 = tuple(rng.choice(range(1, m), 2))
        S1 = tuple(rng.choice(range(1, m), n1))
        S2 = tuple(rng.choice(range(1, m), n2))
        ikey = S0+S1+S2
        if ikey in used:
            continue
        used.add(ikey)

        p1 = partitions(S1)
        p2 = partitions(S2)
        samples = []
        for p1i in p1:
            for p2i in p2:
                sample=[]
                for p1ii in p1i:
                    sample.append((S0[0],p1ii))
                for p2ii in p2i:
                    sample.append((S0[1], p2ii))
                samples.append(sample)
        samples_b = samples[0]
        samples = samples[1:]

        if len(test_set) < test_size:
            test_set.extend(samples[:test_size-len(test_set)])
            train_set.append(samples_b)
        elif len(train_set) < train_size:
            train_set.extend(samples[:train_size-len(train_set)])
            train_set.append(samples_b)
        else:
            break

    train_set = [func(s,m) for s in train_set]
    test_set = [func(s,m) for s in test_set]
    return train_set, test_set





def run():
    for m in [7, 11, 13]:
        for pn in [100,300,1000,3000,10000,30000]:
            if m==7 and pn > 10000:
                continue
            print(f"generate,m={m},pn={pn}")
            save_dataset(m, fname='dist_64', train_size=pn, test_size=1000)


if __name__ == '__main__':
    run()
