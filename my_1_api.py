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
from openai import OpenAI
import queue as queue_package
import argparse

import datetime



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



class MMRequestor:
    def __init__(self,
                 model_name="MODEL_NAME",
                 api_key="YOUR_API_KEY",
                 enable_multi_turn=False,
                 seed=0,
                 request_proxy=None) -> None:

        self.model_name = model_name
        self.enable_multi_turn = enable_multi_turn
        self.chat = []
        self.seed=seed
        #print(f"api key = {api_key}")
        client = OpenAI(api_key=api_key, base_url=request_proxy)
        self.requestor = client

    def request(self, prompts):
        #response = self.requestor.responses.create(
        #    model=self.model_name,
        #    input=prompts,
        #    max_output_tokens=512,
        #    #seed=12345,
        #)
        try:
            response = self.requestor.chat.completions.create(
                model=self.model_name,  # Or any other model supported by OpenRouter
                messages=[
                    {"role": "user", "content": prompts},
                ],
                seed=self.seed,  # The seed parameter for reproducibility
                temperature=0,  # Other optional parameters
                max_tokens=512,
            )

            #res_str = response.output[0].content[0].text
            res_str = response.choices[0].message.content
        except Exception as e:
            res_str = f"Error: {e}"
        return res_str


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data, ensure_ascii=False)
    with open(filename, "a", encoding="utf8") as f:
        f.write(json_string + "\n")


def producer(queue, dataset, save_path, bar, create_prompt):
    if os.path.exists(save_path):
        last_request = [x["index"] for x in read_jsonl(save_path)]
    else:
        last_request = []
    for i, data in enumerate(dataset.data):
        #if data["index"] in last_request:
        #    bar.update(1)
        #    print(f"Skip {data['index']}")
        #    continue
        prompt = create_prompt(data)
        #print(f"Loaded\t\t {i}, # {data['index']} ")
        data.update({"index": data["index"], "text": prompt})
        queue.put(data)
    print("Dataset Loaded.")


def consumer(queue, bar, model_name,
                   api_key, enable_multi_turn, request_proxy, seed=0):
    output_data = []

    requestor = MMRequestor(
                            model_name=model_name,
                            api_key=api_key,
                            enable_multi_turn=enable_multi_turn,
                            request_proxy=request_proxy,
                            seed=seed,
                            )
    while True:
        item = queue.get()
        if item is None:
            print("Consumer Break")
            break
        text = item["text"]

        try:
            print("Requesting\t\t#", item["index"])
            result = requestor.request(
                prompts=text,
            )
            output_data.append({"index": item["index"], "pred": result, "origin": item})
        except Exception as e:
            print(e)
        print("Saved\t\t#", item["index"])
        bar.update(1)
        print(f"Queue left: {queue.qsize()}, Finished: {item['index']}")
        if queue.qsize() == 0:
            break
    return output_data


def request_LLM(total, model_name, api_key, enable_multi_turn, split=0, dataset=None, save_path="",
                      create_prompt_fn=None, request_proxy=None, seed=0):

    queue = queue_package.Queue()

    if dataset is None:
        return

    step = int(len(dataset.data) / total)
    dataset.data = dataset.data[step * split:min(len(dataset.data), step * (split + 1))]
    bar = tqdm(total=len(dataset.data), desc=f"Total: {total} Split: {split}")
    producer(queue, dataset, save_path, bar, create_prompt_fn)
    output_data = consumer(queue, bar,  model_name, api_key, enable_multi_turn, request_proxy, seed)
    with open(save_path, "w") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")




def create_prompt(data, prompt_config):
    instruction = ""
    if "knowledge" in data.keys():
        instruction += f"""
You are supplied with the following knowledge to answer the questions.

Knowledge: {data["knowledge"]}

"""
    instruction += f"""
Please reason and answer the following questions, and finally use the format of "### xxx" to give "xxx" as the final answer.

Question: {data["question"]}

"""
    return instruction


class DataLoader:
    def __init__(self, data_path, knowledge_path=None) -> None:
        input_data = {}
        for i, data in enumerate(read_jsonl(data_path)):
            input_data[data["index"]] = data

        if os.path.exists(knowledge_path):
            print(f"loading knowledge:{knowledge_path}")
            for i, data in enumerate(read_jsonl(knowledge_path)):
                input_data[data["index"]]["knowledge"] = data["knowledge"]
        else:
            print(f"knowledge not found:{knowledge_path}")
        self.data = list(input_data.values())


# client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
# api_key = os.environ.get("OPENAI_API_KEY", "")
# MODEL_NAME = "qwen/qwen3-30b-a3b-instruct-2507"  # change to your preferred model
# MODEL_NAME = "meta-llama/llama-3.2-1b-instruct"
# MODEL_NAME = "qwen/qwen-2.5-7b-instruct"

#openai/gpt-4o-mini-2024-07-18
def run(total=1,
        split=0,
        request_proxy="https://openrouter.ai/api/v1",  # base_url; None means OpenAI base_url by default
        ):
    print(f"api_key = {os.environ.get("OPENAI_API_KEY", '')}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--data_file", type=str, default="data/biggsm/data.jsonl")
    parser.add_argument("--output_file", type=str, default="outputs/qwen_knowledge_4_noop.jsonl")
    parser.add_argument("--knowledge_file", type=str, default="data/biggsm/knowledge_4_noop.jsonl")
    parser.add_argument("--api_key", type=str, default=f"{os.environ.get("OPENAI_API_KEY", '')}")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    print(args)
    model_name = args.model_name
    data_file = args.data_file
    knowledge_file = args.knowledge_file
    output_file = args.output_file
    api_key = args.api_key
    seed = args.seed

    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)



    request_LLM(
        total=total,
        split=split,
        dataset=DataLoader(data_file, knowledge_file),
        save_path=output_file,
        create_prompt_fn=partial(create_prompt, prompt_config=None),
        model_name=model_name,
        api_key=api_key,
        enable_multi_turn = False,
        request_proxy=request_proxy,
        seed=seed,
        )

if __name__ == "__main__":
    run()
