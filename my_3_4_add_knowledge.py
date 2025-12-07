import csv
import json
import os.path
import re

def process_file(input_file):
    input_dir = os.path.dirname(input_file)
    knowledge_file_fw = open(os.path.join(input_dir, "knowledge_1_fw.jsonl"), "w")
    knowledge_file_bw = open(os.path.join(input_dir, "knowledge_2_bw.jsonl"), "w")

    data_file = open(os.path.join(input_dir, "data.jsonl"), "w")
    for line in open(input_file,"r").readlines():
        #print(f"processing:{line}")
        item = json.loads(line)
        eq = item['answer'].split("=")[0]
        eq = eq.replace(")","").replace("(","")
        cond_or1 = ("+" in eq and "*" not in eq)
        cond_or2 = ("+" not in eq and "*" in eq)
        #cond_and1 = ("(" not in eq)
        #cond_and2 = (")" not in eq)
        cond_and3 = ("/" not in eq)
        cond_and4 = ("-" not in eq)
        if not((cond_or1 or cond_or2) and cond_and3 and cond_and4):
            print(f"not exist:{item['answer']}")
            continue
        data_file.write(line)
        op = ""
        if "+" in eq:
            eq = eq.split("+")
            op = "+"
        elif "*" in eq:
            eq = eq.split("*")
            op = "*"

        eq = [x.strip() for x in eq]
        eq0 = eq[0]
        knowledge = []
        for eq1 in eq[1:]:
            eq_str = f"{eq0}{op}{eq1}"
            result = eval(eq_str)
            knowledge.append(f"{eq_str}={result}")
            eq0 = result
        knowledge_str = "\n".join(knowledge)
        json.dump({"knowledge":knowledge_str,"index":item["index"]},knowledge_file_fw)
        knowledge_file_fw.write("\n")



        eq = list(reversed(eq))
        eq0 = eq[0]
        knowledge = []
        for eq1 in eq[1:]:
            eq_str = f"{eq0}{op}{eq1}"
            result = eval(eq_str)
            knowledge.append(f"{eq_str}={result}")
            eq0 = result
        knowledge_str = "\n".join(knowledge)
        json.dump({"knowledge":knowledge_str,"index":item["index"]},knowledge_file_bw)
        knowledge_file_bw.write("\n")

if __name__ == "__main__":
    process_file("./data/MAWPS/data_raw_f2.jsonl")
