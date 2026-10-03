"""Print every answer from a my_1_api.py output file with whether it is correct, then the accuracy."""
import argparse
import json

import numpy as np

from my_common import is_correct


def print_result(data, correct):
    origin = data["origin"]
    print("--------------------------")
    print(f"# question_idx: {data['index']}\n")
    print(f"# origin_text:\n{origin['text']}\n")
    print("-------------")
    print(f"# origin_question:\n{origin['question']}\n")
    print("-------------")
    print(f"# pred_ans:\n{data['pred']}\n")
    print("-------------")
    print(f"# origin_ans:\n{origin['answer']}\n")
    print("-------------")
    print(f"# correct:\n{correct}\n")
    print("--------------------------")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input json file", default="test/qwen-2.5-7b-instruct.jsonl")
    args = parser.parse_args()

    total_correct = []
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            correct = is_correct(data)
            print_result(data, correct)
            total_correct.append(correct)
    print(f"averaged_correct:{np.average(total_correct):.5f}")


if __name__ == "__main__":
    run()
