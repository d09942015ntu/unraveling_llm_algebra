"""Collect the correctness of every (model, knowledge type) per question into outputs_analysis/*.json."""
from collections import defaultdict
import json
import os

from my_common import is_correct, load_knowledge

TTYPES = ["0_none", "1_full", "2_com", "3_ide", "11_fullN", "21_comN", "31_ideN"]
MODELS = ["qwen7b", "qwen72b", "llama8b", "llama70b"]


def run(seed=0, output_dir="outputs_analysis"):
    full_knowledge = load_knowledge("data/biggsm/knowledge_1_full.jsonl")

    result_all_details = defaultdict(list)
    result_all_quick = defaultdict(str)
    for ttype in TTYPES:
        knowledge = load_knowledge(f"data/biggsm/knowledge_{ttype}.jsonl")
        for model in MODELS:
            print(f"{ttype}:{model}")
            with open(f"outputs/{model}_knowledge_{ttype}_{seed}.jsonl", "r") as f:
                for line in f:
                    data = json.loads(line)
                    idx = data['index']
                    origin = data["origin"]
                    correct = int(is_correct(data))
                    answer_len = len(origin['answer'])
                    result_all_details[idx].append({
                        "origin_ans_len": answer_len,
                        "origin_question": origin['question'],
                        "origin_ans": origin['answer'],
                        "full_knowledge": full_knowledge.get(idx, ""),
                        "knowledge": knowledge.get(idx, ""),
                        "pred_ans": data["pred"],
                        "model": model,
                        "ttype": ttype,
                        "correct": correct,
                    })
                    result_all_quick[f"({idx},{answer_len})"] += f"{model}-{ttype}-{correct},"

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result_all_details.json"), "w") as f:
        json.dump(result_all_details, f)
    with open(os.path.join(output_dir, "result_all_quick.json"), "w") as f:
        json.dump(result_all_quick, f)


if __name__ == "__main__":
    run()
