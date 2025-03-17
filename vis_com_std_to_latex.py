import json

for t in [1,2,3]:
    f=open(f"results/hidden_com_p{t}.tex","w")
    jfile=f"results/7_{t}_com_diff_test.json"
    json_item = json.load(open(jfile,"r"))
    temp2=[]
    for i,r1 in enumerate(json_item):
        temp1=[]
        for j,r2 in enumerate(r1):
            if r2 >= 0:
                result=f"{str(int(r2/10)).zfill(4)}"
            else:
                result=f"{str(int(r2/10)).zfill(4)}"
            temp1.append(result)
        temp2.append("{"+f"{','.join(temp1)}"+"}")

    out_str="{"+f"{',\n'.join(temp2)}"+"}"
    f.write("\\begin{tikzpicture}[scale=0.3] \\foreach \\y [count=\\n] in \n%s"%(out_str))
    f.write(""" {
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
""")