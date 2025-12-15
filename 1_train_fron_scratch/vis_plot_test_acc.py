import argparse
import json
import glob
import numpy as np


def run(data_prefix):
    dataset_dict = {
        "Training: Operator $+$'s Commutativity and Identity": ('train_com', 'train_ide'),
        "Testing: Operator $+$'s Commutativity": ('eval_com',),
        "Testing: Operator $+$'s Identity": ('eval_ide',),
        "Training: Operator $\\oplus$'s Commutativity and Identity": ('train_comx', 'train_idex'),
        "Testing: Operator $\\oplus$'s Commutativity": ('eval_comx',),
        "Testing: Operator $\\oplus$'s Identity": ('eval_idex',),
        "Training: Operator $\\ominus$, $\\triangleleft$ and $\\triangleright$": ('train_z0', 'train_lh', 'train_rh'),
        "Testing: Operator $\\ominus$": ('eval_z0',),
        "Testing: Operator $\\triangleleft$ and $\\triangleright$ ": ('eval_lh', 'eval_rh')
    }

    color_dict = {
        "Training: Operator $+$'s Commutativity and Identity": "pc11, thick, dashed",
        "Testing: Operator $+$'s Commutativity": "pc12, thick, dashed",
        "Testing: Operator $+$'s Identity": "pc13, thick, dashed",
        "Training: Operator $\\oplus$'s Commutativity and Identity": "pc21, thick, densely dotted",
        "Testing: Operator $\\oplus$'s Commutativity": "pc22, thick, densely dotted",
        "Testing: Operator $\\oplus$'s Identity": "pc23, thick, densely dotted",
        "Training: Operator $\\ominus$, $\\triangleleft$ and $\\triangleright$": "pc31, very thick, loosely dotted",
        "Testing: Operator $\\ominus$": "pc32, very thick, loosely dotted",
        "Testing: Operator $\\triangleleft$ and $\\triangleright$ ": "pc33,  very thick, loosely dotted",
    }

    label_n = {
        7: "\n   ylabel={accuracy},",
        11: "\n   ylabel={accuracy},",
        13: "\n   ylabel={accuracy},",
    }

    end_n = {
        7: 10000,
        11: 30000,
        13: 30000,
    }
    for n in [7, 11, 13]:
        f = open("results/test_n_%s.tex" % n, "w")
        f.write("""
    \\begin{tikzpicture}
    \\begin{axis}[
        width=5.2cm,
        height=3.5cm,
        xmode=log,
        xlabel={$K$}, %s
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
    """ % (label_n[n], end_n[n], n))
        for i, (dname, dname_list) in enumerate(dataset_dict.items()):
            f.write("\\addplot[%s] table[row sep=\\\\] {\n" % (color_dict[dname]))
            f.write("  x y \\\\ \n")
            for p in [100, 300, 1000, 3000, 10000, 30000]:
                result_files = sorted(glob.glob(f"results/{data_prefix}_{n}_{p}_*/*.log"))
                if len(result_files) > 0:
                    print(f"n={n},p={p},result_files={result_files}")
                    result_file = result_files[-1]
                    jlines = [line for line in open(result_file, "r").readlines() if "step" in line]
                    acc_results = []
                    for jline in jlines:
                        json_item = json.loads(jline)
                        acc_dict = {}
                        acc_dict.update(json_item['train_acc'])
                        acc_dict.update(json_item['eval_acc'])
                        acc_result = np.average([acc_dict[dn] for dn in dname_list])
                        acc_results.append(acc_result)
                    if len(acc_results) > 0:
                        acc_result = np.average(sorted(acc_results, reverse=True)[:2])
                        f.write(f"  {p} {acc_result} \\\\  \n")
            f.write("}; \n")
        f.write("""\\end{axis} \n
    \\end{tikzpicture} \n
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize')
    parser.add_argument('--data_prefix', type=str, default='all_64', help='dataset prefix')
    args = parser.parse_args()
    run(args.data_prefix)
