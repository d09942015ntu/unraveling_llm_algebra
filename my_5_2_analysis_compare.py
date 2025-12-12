'''
Author: Qiguang Chen
LastEditors: Qiguang Chen
Date: 2023-12-18 14:54:57
LastEditTime: 2024-05-18 16:38:28
Description: 

'''

import asyncio
from functools import partial
import random
import numpy as np
from openai import OpenAI
import queue as queue_package
import argparse
from collections import defaultdict



from copy import deepcopy
import json
import os
import re
from typing import List
from tqdm import tqdm

from utils.tools import read_jsonl, write_jsonl

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

def judge_correct(data, model, ttype, knowledge, full_knowledge):
    golden_answer_str = get_origin_input(data)["answer"].replace(",", "").strip(".").split("#### ")[-1]
    golden_answer = round(float(golden_answer_str), 2)
    pred = get_parsed_pred_answer(data)


    correct = judge_error(pred) and abs(abs(round(float(pred), 2)) - abs(round(golden_answer, 2))) < 0.01

    correct = int(correct)
    origin_answer_json = get_origin_input(data)
    pred_str = get_pred_text(data)
    details= {
        "origin_ans_len":len(origin_answer_json['answer']),
        "origin_question":origin_answer_json['question'],
        "origin_ans":origin_answer_json['answer'],
        "full_knowledge":full_knowledge,
        "knowledge":knowledge,
        "pred_ans":pred_str,
        "model":model,
        "ttype":ttype,
        "correct":int(correct),
        }

    a_len = len(origin_answer_json['answer'])
    a_key = f"{model}-{ttype}-{correct}"
    return a_key, a_len, details



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input json file", default="outputs/qwen72b_knowledge_1_full_0.jsonl")
    #parser.add_argument("--knowledge_file", type=str, help="knowledge json file", default="data/biggsm/knowledge_1_full.jsonl")
    #parser.add_argument("--full_knowledge_file", type=str, help="knowledge json file", default="data/biggsm/knowledge_1_full.jsonl")
    #args = parser.parse_args()

    ttypes = ["0_none", "1_full", "2_com", "3_ide", "11_fullN", "21_comN", "31_ideN"]
    models = ["qwen7b", "qwen72b", "llama8b", "llama70b"]

    # qwen/qwen-2.5-72b-instruct
    # qwen/qwen-2.5-7b-instruct
    # meta-llama/llama-3.1-8b-instruct
    # meta-llama/llama-3.1-70b-instruct

    seed = 0

    full_knowledge_file = "data/biggsm/knowledge_1_full.jsonl"
    full_knowledge_dict = {}
    if os.path.exists(full_knowledge_file):
        print(f"loading knowledge:{full_knowledge_file}")
        for i, data in enumerate(read_jsonl(full_knowledge_file)):
            full_knowledge_dict[data["index"]] = data["knowledge"]


    result_all_details = defaultdict(list)
    result_all_quick = defaultdict(str)
    for ttype in ttypes:
        knowledge_file = f"data/biggsm/knowledge_{ttype}.jsonl"
        knowledge_dict = {}
        if os.path.exists(knowledge_file):
            print(f"loading knowledge:{knowledge_file}")
            for i, data in enumerate(read_jsonl(knowledge_file)):
                knowledge_dict[data["index"]] = data["knowledge"]

        for model in models:
            print(f"{ttype}:{model}")

            input_file=f"outputs/{model}_knowledge_{ttype}_{seed}.jsonl"

            with open(input_file, "r") as f:
                for line in f.readlines():
                    json_data = json.loads(line)
                    idx = json_data['index']
                    a_key, a_len, details  = judge_correct(json_data, model, ttype, knowledge_dict.get(idx,""), full_knowledge_dict.get(idx, ""))
                    details['ttype'] = ttype
                    result_all_details[idx].append(details)
                    result_all_quick[f"({idx},{a_len})"] += a_key +","
    json.dump(result_all_details, open("outputs_analysis/result_all_details.json", "w"))
    json.dump(result_all_quick, open("outputs_analysis/result_all_quick.json", "w"))




if __name__ == "__main__":
    run()
