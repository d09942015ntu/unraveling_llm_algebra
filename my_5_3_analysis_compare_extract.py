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
from functools import reduce
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

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input json file", default="outputs/qwen72b_knowledge_1_full_0.jsonl")
    #parser.add_argument("--knowledge_file", type=str, help="knowledge json file", default="data/biggsm/knowledge_1_full.jsonl")
    #parser.add_argument("--full_knowledge_file", type=str, help="knowledge json file", default="data/biggsm/knowledge_1_full.jsonl")
    #args = parser.parse_args()

    ttypes = ["0_none", "1_full", "2_com", "3_ide", "11_fullN", "21_comN", "31_ideN"]
    models = ["qwen7b", "qwen72b", "llama8b", "llama70b"]

    result_all_details = json.load(open("outputs_analysis/result_all_details.json","r"))
    result_all_quick_temp = json.load(open("outputs_analysis/result_all_quick.json","r"))
    result_all_quick = []

    for k in result_all_quick_temp.keys():
        result_all_quick.append(
           {
            "len":eval(k)[1],
            "id":str(eval(k)[0]),
            "info":result_all_quick_temp[k],
            }
        )
    result_all_quick = sorted(result_all_quick, key=lambda x: x["len"])

    constraints = ['qwen7b-0_none-0','qwen7b-1_full-0''llama8b-1_full-0']
    #constraints = ['qwen7b-0_none-0','qwen7b-11_fullN-0','qwen7b-21_comN-0', 'llama8b-0_none-0','llama8b-11_fullN-0','llama8b-21_comN-0']
    #constraints = ['llama8b-21_comN-0']
    matches = []
    for i,item in enumerate(result_all_quick):
        match = reduce(lambda a,b:a*b, [int(constraint in item["info"]) for constraint in constraints])
        if match:
            matches.append(item)

    for i, item in enumerate(matches[:10]):
        results = result_all_details[item["id"]]
        result_print={}

        for constraint in constraints:
            for result in results:
                rkey=f"{result['model']}-{result['ttype']}-{result['correct']}"
                if 'origin_question' not in result_print:
                    result_print['origin_question'] = result['origin_question']
                if 'origin_ans' not in result_print:
                    result_print['origin_ans'] = result['origin_ans']
                if 'full_knowledge' not in result_print:
                    result_print['full_knowledge'] = result['full_knowledge']
                if rkey==constraint:
                    result_print[rkey] = result['pred_ans']
        f = open(f"outputs_view/result_{i}.txt","w")
        f.write(f"id:{item['id']}\n")
        f.write("--------------\n")
        for ikey in result_print.keys():
            f.write(f"{ikey}:\n")
            f.write(f"{result_print[ikey]}:\n")
            f.write("--------------\n")
        f.close()




if __name__ == "__main__":
    run()
