"""Turn results/7_<t>_{com,ide}_diff_test.json (from vis_com_std.py / vis_ide_std.py) into TikZ heat maps."""
import json

TIKZ_BODY = """ {
          \\foreach \\x [count=\\m] in \\y {
               \\ifnum \\x < 0
                    \\node[fill=yellow!\\x!purple, minimum width=1.5mm, text=white] at (\\m*0.8, -\\n*0.8) {};
                \\else
                     \\node[fill=lime!\\x!green, minimum width=1.5mm, text=white] at (\\m*0.8, -\\n*0.8) {};
                \\fi
                  \\ifnum \\n < 2
                    \\node[minimum size=4mm] at (\\m*0.8, 0) {\\tiny \\m};
                \\fi
      }
    }
  % row labels
  \\foreach \\a [count=\\i] in {100,300,1000,3000,10000} {
    \\node[minimum size=4mm] at (-0.5, -\\i*0.8) {\\tiny \\a};

  }
\\end{tikzpicture}
"""


def to_latex(kind, t):
    with open(f"results/7_{t}_{kind}_diff_test.json") as f:
        rows = json.load(f)
    row_strs = ["{" + ",".join(str(int(value / 10)).zfill(4) for value in row) + "}" for row in rows]
    out_str = "{" + ",\n".join(row_strs) + "}"
    with open(f"results/hidden_{kind}_p{t}.tex", "w") as f:
        f.write("\\begin{tikzpicture}[scale=0.3] \\foreach \\y [count=\\n] in \n%s" % out_str)
        f.write(TIKZ_BODY)


if __name__ == '__main__':
    for kind in ["com", "ide"]:
        for t in [1, 2, 3]:
            to_latex(kind, t)
