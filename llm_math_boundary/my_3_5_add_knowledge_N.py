import csv
import json
import os.path
import copy
import re
import numpy as np

RNG = np.random.RandomState(0)


def eval_result(eq_str):
    result = eval(eq_str)
    result = round(result, 2)
    if result == int(result):
        result = int(result)
    return result


def gen_knowledge_N(input_file="./data/MAWPS/data_raw_f2.jsonl"):
    input_dir = os.path.dirname(input_file)
    #knowledge_file_fw = open(os.path.join(input_dir, "knowledge_1_fw.jsonl"), "w")
    #knowledge_file_bw = open(os.path.join(input_dir, "knowledge_2_bw.jsonl"), "w")
    #knowledge_file_full = open(os.path.join(input_dir, "knowledge_3_full.jsonl"), "w")

    knowledge_file_fwN = open(os.path.join(input_dir, "knowledge_11_fwN.jsonl"), "w")
    knowledge_file_bwN = open(os.path.join(input_dir, "knowledge_21_bwN.jsonl"), "w")
    knowledge_file_fullN = open(os.path.join(input_dir, "knowledge_31_fullN.jsonl"), "w")

    data_file = open(os.path.join(input_dir, "data.jsonl"), "w")
    for line in open(input_file,"r").readlines():
        print(f"processing:{line}")
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
        eq_original = copy.deepcopy(eq)


        #------------- 1_fw -------------
        eq = copy.deepcopy(eq_original)
        eq0 = eq[0]
        knowledge = []
        for eq1 in eq[1:]:
            eq_str = f"{eq0}{op}{eq1}"
            result = eval_result(eq_str)
            knowledge.append(f"{eq_str}={result}")
            eq0 = result
        knowledge_fw = copy.deepcopy(knowledge)
        #json.dump({"knowledge":knowledge_str,"index":item["index"]},knowledge_file_fw)
        #knowledge_file_fw.write("\n")

        #------------- 2_bw -------------
        eq = list(reversed(eq_original))
        eq0 = eq[0]
        knowledge = []
        for eq1 in eq[1:]:
            eq_str = f"{eq1}{op}{eq0}"
            result = eval_result(eq_str)
            knowledge.append(f"{eq_str}={result}")
            eq0 = result
        knowledge_bw = copy.deepcopy(knowledge)
        #json.dump({"knowledge":knowledge_str,"index":item["index"]},knowledge_file_bw)
        #knowledge_file_bw.write("\n")

        #------------- 3_full -------------
        knowledge = []
        eq = copy.deepcopy(eq_original)
        eq_str = f"{op}".join(eq)
        result = eval_result(eq_str)
        knowledge.append(f"{eq_str}={result}")
        knowledge_full = copy.deepcopy(knowledge)
        #json.dump({"knowledge":knowledge_str,"index":item["index"]},knowledge_file_full)
        #knowledge_file_full.write("\n")

        #------------- 11_fwN -------------
        eq = copy.deepcopy(eq_original)
        eq0_1 = eq[0]
        eq0_2 = eq[0]
        knowledge = []
        for eq1 in eq[1:]:
            eq_str_1 = f"{eq0_1}+{eq1}"
            eq_str_2 = f"{eq0_2}*{eq1}"
            result_1 = eval_result(eq_str_1)
            result_2 = eval_result(eq_str_2)
            knowledge.append(f"{eq_str}={result_1}")
            knowledge.append(f"{eq_str}={result_2}")
            eq0_old_1 = eq0_1
            eq0_old_2 = eq0_2
            eq0_1 = result_1
            eq0_2 = result_2

            for op_temp in ['+', '*']:
                for eq0_temp in [eq0_1, eq0_2]:
                    for eq_temp in RNG.permutation([eq1, eq0_old_1, eq1, eq0_old_2]): #, eq1, eq0_old, eq1, eq0_old, eq1, eq0_old]):
                        eq_str_temp = f"{eq0_temp}{op_temp}{eq_temp}"
                        result_temp = eval_result(eq_str_temp)
                        knowledge.append(f"{eq_str_temp}={result_temp}")
                        if result_temp > 10000000:
                            result_temp = int(result_temp/1000000)
                        eq0_temp = result_temp

        knowledge = [str(x) for x in RNG.permutation(knowledge)]
        knowledge_fwN = copy.deepcopy(knowledge)
        knowledge_str = "\n".join(knowledge)
        json.dump({"knowledge":knowledge_str,"index":item["index"]},knowledge_file_fwN)
        knowledge_file_fwN.write("\n")

        #------------- 21_bwN -------------
        eq = list(reversed(eq_original))
        eq = [x.strip() for x in eq]
        eq0_1 = eq[0]
        eq0_2 = eq[0]
        knowledge = []
        for eq1 in eq[1:]:
            eq_str_1 = f"{eq1}+{eq0_1}"
            eq_str_2 = f"{eq1}*{eq0_2}"
            result_1 = eval_result(eq_str_1)
            result_2 = eval_result(eq_str_2)
            knowledge.append(f"{eq_str}={result_1}")
            knowledge.append(f"{eq_str}={result_2}")
            eq0_old_1 = eq0_1
            eq0_old_2 = eq0_2
            eq0_1 = result_1
            eq0_2 = result_2

            for op_temp in ['+', '*']:
                for eq0_temp in [eq0_1, eq0_2]:
                    for eq_temp in RNG.permutation([eq1, eq0_old_1, eq1, eq0_old_2]): #, eq1, eq0_old, eq1, eq0_old, eq1, eq0_old]):
                        eq_str_temp = f"{eq_temp}{op_temp}{eq0_temp}"
                        result_temp = eval_result(eq_str_temp)
                        knowledge.append(f"{eq_str_temp}={result_temp}")
                        if result_temp > 10000000:
                            result_temp = int(result_temp/1000000)
                        eq0_temp = result_temp

        knowledge = [str(x) for x in RNG.permutation(knowledge)]
        knowledge_bwN = copy.deepcopy(knowledge)
        knowledge_str = "\n".join(knowledge)
        json.dump({"knowledge":knowledge_str,"index":item["index"]},knowledge_file_bwN)
        knowledge_file_bwN.write("\n")


        #------------- 31_fullN -------------
        knowledge_fwN= knowledge_fwN[:int(len(knowledge_fwN)/2)]
        knowledge_bwN= knowledge_bwN[:int(len(knowledge_bwN)/2)]
        knowledge_fullN = knowledge_fwN + knowledge_bwN + knowledge_full

        for k in knowledge_fw:
            if k in knowledge_fullN:
                knowledge_fullN.remove(k)
        for k in knowledge_bw:
            if k in knowledge_fullN:
                knowledge_fullN.remove(k)

        knowledge = [str(x) for x in RNG.permutation(knowledge_fullN)]
        knowledge_str = "\n".join(knowledge)
        json.dump({"knowledge":knowledge_str,"index":item["index"]},knowledge_file_fullN)
        knowledge_file_fullN.write("\n")

if __name__ == "__main__":
    gen_knowledge_N("./data/MAWPS/data_raw_f2.jsonl")
