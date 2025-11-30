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
                 model_type="gpt",
                 model_name="MODEL_NAME",
                 api_key="YOUR_API_KEY",
                 enable_multi_turn=False,
                 request_proxy=None) -> None:

        self.model_type = model_type
        self.model_name = model_name
        self.enable_multi_turn = enable_multi_turn
        self.chat = []
        #print(f"api key = {api_key}")
        client = OpenAI(api_key=api_key, base_url=request_proxy)
        self.requestor = client

    def request(self, prompts):
        response = self.requestor.responses.create(
            model=self.model_name,
            input=prompts,
            max_output_tokens=512,
        )
        res_str = response.output[0].content[0].text
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
        if data["index"] in last_request:
            bar.update(1)
            print(f"Skip {data['index']}")
            continue
        prompt = create_prompt(data)
        print("Loaded\t\t#", data['index'])
        data.update({"index": data["index"], "text": prompt})
        queue.put(data)
    print("Dataset Loaded.")


def consumer(queue, save_path, bar, model_type, model_name,
                   api_key, enable_multi_turn, request_proxy, return_origin=True):
    output_data = []

    requestor = MMRequestor(model_type=model_type,
                            model_name=model_name,
                            api_key=api_key,
                            enable_multi_turn=enable_multi_turn,
                            request_proxy=request_proxy)
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


def request_LLM(total, model_type, model_name, api_key, enable_multi_turn, split=0, dataset=None, save_path="",
                      create_prompt_fn=None, request_proxy=None, return_origin=True):

    queue = queue_package.Queue(maxsize=120)

    if dataset is None:
        return

    step = int(len(dataset.data) / total)
    dataset.data = dataset.data[step * split:min(len(dataset.data), step * (split + 1))]
    bar = tqdm(total=len(dataset.data), desc=f"Total: {total} Split: {split}")
    producer(queue, dataset, save_path, bar, create_prompt_fn)
    output_data = consumer(queue, save_path, bar, model_type, model_name, api_key, enable_multi_turn, request_proxy, return_origin)
    with open(save_path, "w") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")




def create_prompt(data, prompt_config):
    instruction = """Please reason and answer the following questions, and finally use the format of "### xxx" to give "xxx" as the final answer.

Question: """
    instruction += data["question"]
    return instruction


class DataLoader:
    def __init__(self, load_path: str) -> None:
        input_data = []
        for i, data in enumerate(read_jsonl(load_path)):
            if "index" not in data:
                data["index"] = str(i)
            input_data.append(data)
        input_data.reverse()
        self.data = input_data


# client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
# api_key = os.environ.get("OPENAI_API_KEY", "")
# MODEL_NAME = "qwen/qwen3-30b-a3b-instruct-2507"  # change to your preferred model
# MODEL_NAME = "meta-llama/llama-3.2-1b-instruct"
# MODEL_NAME = "qwen/qwen-2.5-7b-instruct"

#openai/gpt-4o-mini-2024-07-18
def run(total=1, 
        split=0, 
        model_type="qwen",
        model_name="qwen/qwen-2.5-7b-instruct",
        api_key= f"{os.environ.get("OPENAI_API_KEY", '')}",
        request_proxy="https://openrouter.ai/api/v1", # base_url; None means OpenAI base_url by default
        temperature=0.0):
    print(f"api_key = {os.environ.get("OPENAI_API_KEY", '')}")
    
    model_config = {
        "temperature": temperature,
    }
    

    data_dir="data/biggsm_samples/data.jsonl"
    save_dir= "outputs"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{os.path.basename(model_name)}.jsonl")

    request_LLM(
        total=total,
        split=split,
        dataset=DataLoader(data_dir),
        save_path=save_path,
        create_prompt_fn=partial(create_prompt, prompt_config=None),
        model_type=model_type,
        model_name=model_name,
        api_key=api_key,
        enable_multi_turn = False,
        request_proxy=request_proxy,
        return_origin=True,
        )

if __name__ == "__main__":
    run()
