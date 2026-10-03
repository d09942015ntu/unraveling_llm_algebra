"""Hidden-state spread across reorderings of the same operands (commutativity).

For each test sample, up to 25 reorderings of its operands are fed to the model with
operator + and with the operator chosen by rtype (z0, lh, rh). For every layer we take
the std of the [=] hidden state across reorderings, summed over hidden units, and
report (std for +) - (std for rtype).
"""
import argparse
import itertools

import numpy as np

from vis_hidden_common import run


def reorderings(operands, used, rtype):
    """Up to 25 random reorderings of ``operands``; None if these operands were already used (for +)."""
    if not rtype and operands in used:
        return None
    used.add(operands)
    permutations = sorted(set(itertools.permutations(operands, len(operands))))
    np.random.shuffle(permutations)
    return permutations[:25]


def std_difference(hidden_p, hidden_r):
    std_p = [float(sum(h.std(dim=0))) for h in hidden_p]
    std_r = [float(sum(h.std(dim=0))) for h in hidden_r]
    return [p - r for p, r in zip(std_p, std_r)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument('--data_prefix', type=str, default='all_64', help='dataset prefix')
    args = parser.parse_args()
    run(args.data_prefix, "com", "test_com", reorderings, std_difference)
