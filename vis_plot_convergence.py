import argparse
import json
import glob

import numpy as np


def run(data_prefix, n=7, p=3000):
    dataset_dict = {
        "Training: $+$'s commutativity and identity": ('train_com', 'train_ide'),
        "Testing:  $+$'s commutativity": ('eval_com',),
        "Testing:  $+$'s identity": ('eval_ide',),
        "Training: $\\oplus$'s commutativity and identity": ('train_comx', 'train_idex'),
        "Testing:  $\\oplus$'s commutativity": ('eval_comx',),
        "Testing:  $\\oplus$'s identity": ('eval_idex',),
        "Training: $\\ominus$, $\\triangleleft$ and $\\triangleright$, no commutativity and identity": (
        'train_z0', 'train_lh', 'train_rh'),
        "Testing:  $\\ominus$, no commutativity and identity": ('eval_z0',),
        "Testing:  $\\triangleleft$ and $\\triangleright$, no commutativity and identity ": ('eval_lh', 'eval_rh')
    }

    color_dict = {
        "Training: $+$'s commutativity and identity": "pc11, thick, dashed",
        "Testing:  $+$'s commutativity": "pc12, thick, dashed",
        "Testing:  $+$'s identity": "pc13, thick, dashed",
        "Training: $\\oplus$'s commutativity and identity": "pc21, thick, densely dotted",
        "Testing:  $\\oplus$'s commutativity": "pc22, thick, densely dotted",
        "Testing:  $\\oplus$'s identity": "pc23, thick, densely dotted",
        "Training: $\\ominus$, $\\triangleleft$ and $\\triangleright$, no commutativity and identity": "pc31, very thick, loosely dotted",
        "Testing:  $\\ominus$, no commutativity and identity": "pc32, very thick, loosely dotted",
        "Testing:  $\\triangleleft$ and $\\triangleright$, no commutativity and identity ": "pc33,  very thick, loosely dotted",
    }

    f = open("results/convergence_%s.tex" % p, "w")
    f.write("""
\\begin{tikzpicture}
\\begin{axis}[
    xmode=log,
    width=10cm,
    height=4cm,
    xlabel={steps},
    ylabel={accuracy},
    legend pos=outer north east,
    legend cell align={left},
    grid=major,
    grid style={dashed,gray!30},
    xmin=10, xmax=60000,
    ymin=-0.1, ymax=1.1,
    title={Training Dynamics},
    title style={font=\\scriptsize},
    label style={font=\\scriptsize},
    tick label style={font=\\tiny},
    legend style={font=\\tiny},
]
""")
    for i, (dname, dname_list) in enumerate(dataset_dict.items()):
        f.write("\\addplot[%s] table[row sep=\\\\] {\n" % (color_dict[dname]))
        f.write("  x y \\\\ \n")
        f.write("  1 0 \\\\ \n")
        log_path = f"results/{data_prefix}_{n}_{p}_*/*.log"
        print(log_path)
        result_file = sorted(glob.glob(log_path))[-1]
        jlines = [line for line in open(result_file, "r").readlines() if "step" in line]
        for jline in jlines:
            json_item = json.loads(jline)
            acc_dict = {}
            acc_dict.update(json_item['train_acc'])
            acc_dict.update(json_item['eval_acc'])
            acc_result = np.average([acc_dict[dn] for dn in dname_list])
            f.write(f"  {json_item['step']} {acc_result} \\\\  \n")
        f.write("}; \n")
        f.write("\\addlegendentry{%s}" % dname)
    f.write("""\\end{axis} \n
\\end{tikzpicture} \n
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument('--data_prefix', type=str, default='all_64', help='dataset prefix')
    parser.add_argument('--n', type=str, default=7, help='n')
    parser.add_argument('--scale', type=str, default=3000, help='scale')
    args = parser.parse_args()
    run(args.data_prefix, n=args.n, p=args.scale)
