"""Print every wrong answer in a my_1_api.py output file, together with the knowledge given to the model."""
import argparse
import json

from my_common import is_correct, load_knowledge


def print_failure(data, failure_id, knowledge, full_knowledge):
    origin = data["origin"]
    print("-------------------------------------")
    print(f"# question_idx: {data['index']}\n")
    print("-------------")
    print(f"# error_id: {failure_id}\n")
    print("-------------")
    print(f"# origin_question:\n{origin['question']}\n")
    print("-------------")
    print(f"# full_knowledge:\n{full_knowledge}\n")
    print("-------------")
    print(f"# knowledge:\n{knowledge}\n")
    print("-------------")
    print(f"# origin_ans:\n{origin['answer']}\n")
    print("-------------")
    print(f"# pred_ans:\n{data['pred']}\n")
    print("-------------")
    print("# correct:\nFalse\n")
    print("-------------------------------------")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="input json file",
                        default="outputs/qwen72b_knowledge_1_full_0.jsonl")
    parser.add_argument("--knowledge_file", type=str, help="knowledge json file",
                        default="data/biggsm/knowledge_1_full.jsonl")
    parser.add_argument("--full_knowledge_file", type=str, help="knowledge json file",
                        default="data/biggsm/knowledge_1_full.jsonl")
    args = parser.parse_args()

    knowledge = load_knowledge(args.knowledge_file)
    full_knowledge = load_knowledge(args.full_knowledge_file)

    total_failure = 0
    with open(args.input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            idx = data['index']
            if not is_correct(data):
                print_failure(data, total_failure, knowledge.get(idx, ""), full_knowledge.get(idx, ""))
                total_failure += 1
    print(f"failure counts:{total_failure}")


if __name__ == "__main__":
    run()
