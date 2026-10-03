"""From my_5_2's output, write the first 10 questions matching all CONSTRAINTS to outputs_view/result_<i>.txt.

A constraint is "<model>-<knowledge type>-<correct>", e.g. 'qwen7b-0_none-0' means
qwen7b answered wrongly without knowledge.
"""
import json
import os

CONSTRAINTS = ['qwen7b-0_none-0', 'qwen7b-1_full-0', 'llama8b-1_full-0']


def run(constraints=CONSTRAINTS, input_dir="outputs_analysis", output_dir="outputs_view"):
    with open(os.path.join(input_dir, "result_all_details.json")) as f:
        result_all_details = json.load(f)
    with open(os.path.join(input_dir, "result_all_quick.json")) as f:
        result_all_quick = json.load(f)

    # Keys look like "(<index>,<answer length>)"; sort questions by answer length.
    questions = []
    for key, info in result_all_quick.items():
        idx, answer_len = key.strip("()").rsplit(",", 1)
        questions.append({"len": int(answer_len), "id": idx, "info": info})
    questions.sort(key=lambda x: x["len"])

    matches = [q for q in questions if all(c in q["info"] for c in constraints)]

    os.makedirs(output_dir, exist_ok=True)
    for i, question in enumerate(matches[:10]):
        results = result_all_details[question["id"]]
        first = results[0]
        result_print = {
            'origin_question': first['origin_question'],
            'origin_ans': first['origin_ans'],
            'full_knowledge': first['full_knowledge'],
        }
        for constraint in constraints:
            for result in results:
                if f"{result['model']}-{result['ttype']}-{result['correct']}" == constraint:
                    result_print[constraint] = result['pred_ans']
        with open(os.path.join(output_dir, f"result_{i}.txt"), "w") as f:
            f.write(f"id:{question['id']}\n")
            f.write("--------------\n")
            for key, value in result_print.items():
                f.write(f"{key}:\n")
                f.write(f"{value}:\n")
                f.write("--------------\n")


if __name__ == "__main__":
    run()
