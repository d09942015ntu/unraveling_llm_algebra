"""Write results/test_n_<n>.tex: best accuracy vs training-set size K, for each operator group."""
import argparse

import numpy as np

from results_log import latest_log, read_accuracy_log

# legend -> (log keys averaged together, pgfplots style)
CURVES = {
    "Training: Operator $+$'s Commutativity and Identity": (('train_com', 'train_ide'), "pc11, thick, dashed"),
    "Testing: Operator $+$'s Commutativity": (('eval_com',), "pc12, thick, dashed"),
    "Testing: Operator $+$'s Identity": (('eval_ide',), "pc13, thick, dashed"),
    "Training: Operator $\\oplus$'s Commutativity and Identity": (('train_comx', 'train_idex'),
                                                                   "pc21, thick, densely dotted"),
    "Testing: Operator $\\oplus$'s Commutativity": (('eval_comx',), "pc22, thick, densely dotted"),
    "Testing: Operator $\\oplus$'s Identity": (('eval_idex',), "pc23, thick, densely dotted"),
    "Training: Operator $\\ominus$, $\\triangleleft$ and $\\triangleright$": (('train_z0', 'train_lh', 'train_rh'),
                                                                              "pc31, very thick, loosely dotted"),
    "Testing: Operator $\\ominus$": (('eval_z0',), "pc32, very thick, loosely dotted"),
    "Testing: Operator $\\triangleleft$ and $\\triangleright$ ": (('eval_lh', 'eval_rh'),
                                                                  "pc33,  very thick, loosely dotted"),
}

AXIS_HEADER = """
    \\begin{tikzpicture}
    \\begin{axis}[
        width=5.2cm,
        height=3.5cm,
        xmode=log,
        xlabel={$K$}, 
   ylabel={accuracy},
        legend pos=north west,
        grid=major,
        grid style={dashed,gray!30},
        xmin=100, xmax=%s,
        ymin=-0.1, ymax=1.05,
        title={Varying K, for
        $\\mathbb{Z}_{%s}$},
        title style={font=\\scriptsize},
        label style={font=\\scriptsize},
        tick label style={font=\\tiny},
        legend style={font=\\tiny},
            xlabel style={
            at={(current axis.south east)}, 
            anchor=north east,              
            yshift=-5pt,                   
            xshift=20pt                      
        },
    ]
    """

AXIS_FOOTER = """\\end{axis} \n
    \\end{tikzpicture} \n
    """


def run(data_prefix):
    x_max = {7: 10000, 11: 30000, 13: 30000}
    for n in [7, 11, 13]:
        with open(f"results/test_n_{n}.tex", "w") as f:
            f.write(AXIS_HEADER % (x_max[n], n))
            for keys, style in CURVES.values():
                f.write("\\addplot[%s] table[row sep=\\\\] {\n" % style)
                f.write("  x y \\\\ \n")
                for train_size in [100, 300, 1000, 3000, 10000, 30000]:
                    log_file = latest_log(f"results/{data_prefix}_{n}_{train_size}_*/*.log")
                    if log_file is None:
                        continue
                    print(f"n={n},p={train_size},result_file={log_file}")
                    accs = [np.average([acc[k] for k in keys]) for _, acc in read_accuracy_log(log_file)]
                    if accs:
                        # Mean of the two best logged accuracies.
                        f.write(f"  {train_size} {np.average(sorted(accs, reverse=True)[:2])} \\\\  \n")
                f.write("}; \n")
            f.write(AXIS_FOOTER)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument('--data_prefix', type=str, default='all_64', help='dataset prefix')
    args = parser.parse_args()
    run(args.data_prefix)
