"""Ask an LLM (through an OpenAI-compatible API, OpenRouter by default) every question in a dataset.

Each question can be given extra "knowledge" (equations) from a knowledge_*.jsonl file.
The API key is read from --api_key or the OPENAI_API_KEY environment variable.
"""
import argparse
import json
import os

from openai import OpenAI
from tqdm import tqdm

from utils.tools import read_jsonl


class Requestor:
    def __init__(self, model_name, api_key, seed=0, base_url=None):
        self.model_name = model_name
        self.seed = seed
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def request(self, prompt):
        """The model's reply, or "Error: ..." if the request failed."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                seed=self.seed,
                temperature=0,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"


def create_prompt(data):
    instruction = ""
    if "knowledge" in data:
        instruction += f"""
You are supplied with the following knowledge to answer the questions.

Knowledge: {data["knowledge"]}

"""
    instruction += f"""
Please reason and answer the following questions, and finally use the format of "### xxx" to give "xxx" as the final answer.

Question: {data["question"]}

"""
    return instruction


def load_questions(data_path, knowledge_path=None):
    """Questions from ``data_path``, each with a "knowledge" field if ``knowledge_path`` has one for it."""
    questions = {data["index"]: data for data in read_jsonl(data_path)}
    if knowledge_path and os.path.exists(knowledge_path):
        print(f"loading knowledge:{knowledge_path}")
        for data in read_jsonl(knowledge_path):
            questions[data["index"]]["knowledge"] = data["knowledge"]
    else:
        print(f"knowledge not found:{knowledge_path}")
    return list(questions.values())


def request_all(questions, requestor, save_path):
    output_data = []
    for item in tqdm(questions):
        item["text"] = create_prompt(item)
        print("Requesting\t\t#", item["index"])
        output_data.append({"index": item["index"], "pred": requestor.request(item["text"]), "origin": item})
    with open(save_path, "w") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--data_file", type=str, default="data/biggsm/data.jsonl")
    parser.add_argument("--output_file", type=str, default="outputs/qwen_knowledge_4_noop.jsonl")
    parser.add_argument("--knowledge_file", type=str, default="data/biggsm/knowledge_4_noop.jsonl")
    parser.add_argument("--api_key", type=str, default=os.environ.get('OPENAI_API_KEY', ''))
    parser.add_argument("--base_url", type=str, default="https://openrouter.ai/api/v1",
                        help="API base url (OpenRouter by default)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    requestor = Requestor(args.model_name, args.api_key, seed=args.seed, base_url=args.base_url)
    request_all(load_questions(args.data_file, args.knowledge_file), requestor, args.output_file)


if __name__ == "__main__":
    run()
