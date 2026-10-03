"""Hidden-state distance between an input with and without its 0 operand (identity).

For each test sample (operands containing a 0), the sorted operands are fed to the
model with and without the 0, once with operator + and once with the operator chosen
by rtype (z0, lh, rh). For every layer we take the L1 distance between the two [=]
hidden states and report (distance for +) - (distance for rtype).
"""
import argparse

import numpy as np

from vis_hidden_common import run


def with_and_without_zero(operands, used, rtype):
    """[operands, operands without 0]; None if already used (for +) or fewer than 2 non-zero operands."""
    nonzero = [x for x in operands if x != 0]
    if not rtype and operands in used:
        return None
    if len(nonzero) < 2:
        return None
    used.add(operands)
    return [operands, nonzero]


def distance_difference(hidden_p, hidden_r):
    def l1(h):
        h = h.cpu().numpy()
        return np.sum(np.abs(h[0, :] - h[1, :]))
    return [l1(p) - l1(r) for p, r in zip(hidden_p, hidden_r)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument('--data_prefix', type=str, default='all_64', help='dataset prefix')
    args = parser.parse_args()
    run(args.data_prefix, "ide", "test_ide", with_and_without_zero, distance_difference)
