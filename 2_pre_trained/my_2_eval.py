
import numpy as np

import argparse
import json
import random
import re


random.seed(42)

def judge_error(pred):
    try:
        float(pred)
    except:
        return False
    return True


def get_origin_input(data):
    return data["origin"]

def get_pred_text(data):
    return data["pred"]

def get_math_answer(data):
    pred_str = [s for s in [""] + re.findall(r'\$(.*?)\$', get_pred_text(data).replace("\$", "").replace("$$", "$"))][-1].replace("$", "")
    if pred_str == "":
        pred_str = [s for s in [""] + re.findall(r'\\\((.*?)\\\)', get_pred_text(data))][-1].replace("\(", "").replace("\)", "")
    return pred_str


def get_parsed_pred_answer(data):
    pred_str = get_pred_text(data)
    if "var1" not in pred_str or "<<" not in pred_str:
        pred_list = [s for s in re.findall(r'-?\d+\.?\,?\d*', pred_str.replace(",", "").strip(".").split("=")[-1])]
        if len(pred_list) == 0:
            pred1 = -1
        else:
            pred1 = pred_list[-1]
        return pred1
    else:
        try:
            eqs = [s for s in re.findall(r'<<(.*?)>>', pred_str) if "=" in s]
            eqs = sorted(eqs, key=lambda x: int(x.split("=")[0].strip("var")))
            var_list = {eq.split("=")[0]: None for eq in eqs}
            for eq in eqs:
                if "=" in eq:
                    func_str = eq.split("=")[1]
                    for var in var_list:
                        if var_list[var] is not None:
                            func_str = func_str.replace(var, str(var_list[var]))
                    if var_list[eq.split("=")[0]] is None:
                        try:
                            var_list[eq.split("=")[0]] = eval(func_str)
                        except:
                            return -1
                elif "var" in eq:
                    pred_str += "#### " + eq
            if "####" in pred_str:
                var_key = pred_str.split("####")[-1].strip().strip(".").replace("<", "").replace(">", "")
                if var_key in var_list:
                    return var_list[var_key]
            last_var = var_list[list(var_list.keys())[-1]]
            if last_var is None:
                last_var = -1
        except:
            return -1
        return last_var

def judge_correct(data, idx):
    golden_answer_str = get_origin_input(data)["answer"].replace(",", "").strip(".").split("#### ")[-1]
    golden_answer = round(float(golden_answer_str), 2)
    pred = get_parsed_pred_answer(data)


    correct = judge_error(pred) and abs(abs(round(float(pred), 2)) - abs(round(golden_answer, 2))) < 0.01

    origin_answer_json = get_origin_input(data)
    pred_str = get_pred_text(data)
    print("--------------------------")
    print(f"# question_idx: {idx}\n")
    print(f"# origin_text:\n{origin_answer_json['text']}\n")
    print("-------------")
    print(f"# origin_question:\n{origin_answer_json['question']}\n")
    print("-------------")
    print(f"# pred_ans:\n{pred_str}\n")
    print("-------------")
    print(f"# origin_ans:\n{origin_answer_json['answer']}\n")
    print("-------------")
    print(f"# correct:\n{correct}\n")
    print("--------------------------")

    return correct

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input json file", default="test/qwen-2.5-7b-instruct.jsonl")
    args = parser.parse_args()

    total_correct = []
    with open(args.input_file, "r") as f:
        for line in f.readlines():
            json_data = json.loads(line)
            idx = json_data["index"]
            correct = judge_correct(json_data, idx)
            total_correct.append(correct)
    print(f"averaged_correct:{np.average(total_correct):.5f}")




if __name__ == "__main__":
    run()
