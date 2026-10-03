"""Write results/convergence_<scale>.tex: accuracy vs training step for one run."""
import argparse

import numpy as np

from results_log import latest_log, read_accuracy_log

# legend -> (log keys averaged together, pgfplots style)
CURVES = {
    "Training: $+$'s commutativity and identity": (('train_com', 'train_ide'), "pc11, thick, dashed"),
    "Testing:  $+$'s commutativity": (('eval_com',), "pc12, thick, dashed"),
    "Testing:  $+$'s identity": (('eval_ide',), "pc13, thick, dashed"),
    "Training: $\\oplus$'s commutativity and identity": (('train_comx', 'train_idex'), "pc21, thick, densely dotted"),
    "Testing:  $\\oplus$'s commutativity": (('eval_comx',), "pc22, thick, densely dotted"),
    "Testing:  $\\oplus$'s identity": (('eval_idex',), "pc23, thick, densely dotted"),
    "Training: $\\ominus$, $\\triangleleft$ and $\\triangleright$, no commutativity and identity": (
        ('train_z0', 'train_lh', 'train_rh'), "pc31, very thick, loosely dotted"),
    "Testing:  $\\ominus$, no commutativity and identity": (('eval_z0',), "pc32, very thick, loosely dotted"),
    "Testing:  $\\triangleleft$ and $\\triangleright$, no commutativity and identity ": (
        ('eval_lh', 'eval_rh'), "pc33,  very thick, loosely dotted"),
}

AXIS_HEADER = """
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
"""

AXIS_FOOTER = """\\end{axis} \n
\\end{tikzpicture} \n
"""


def run(data_prefix, n=7, p=3000):
    log_path = f"results/{data_prefix}_{n}_{p}_*/*.log"
    print(log_path)
    log_file = latest_log(log_path)
    if log_file is None:
        raise FileNotFoundError(f"No log file matches {log_path}")
    entries = read_accuracy_log(log_file)

    with open(f"results/convergence_{p}.tex", "w") as f:
        f.write(AXIS_HEADER)
        for legend, (keys, style) in CURVES.items():
            f.write("\\addplot[%s] table[row sep=\\\\] {\n" % style)
            f.write("  x y \\\\ \n")
            f.write("  1 0 \\\\ \n")
            for step, acc in entries:
                f.write(f"  {step} {np.average([acc[k] for k in keys])} \\\\  \n")
            f.write("}; \n")
            f.write("\\addlegendentry{%s}" % legend)
        f.write(AXIS_FOOTER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument('--data_prefix', type=str, default='all_64', help='dataset prefix')
    parser.add_argument('--n', type=int, default=7, help='modulus n of the dataset')
    parser.add_argument('--scale', type=int, default=3000, help='training-set size of the dataset')
    args = parser.parse_args()
    run(args.data_prefix, n=args.n, p=args.scale)
