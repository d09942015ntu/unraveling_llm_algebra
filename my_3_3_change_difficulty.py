import csv
import json
import re
import numpy as np
import math
from my_3_4_add_knowledge import gen_knowledge

# Regex to capture numbers: decimals first, then integers
NUM_RE = re.compile(r'\d+\.\d+|\d+')
RNG = np.random.RandomState(0)

def map_integer(s: str, scale=1000) -> str:
    """Map an integer string to a larger integer."""
    n = int(s)
    new_n = n * scale + RNG.randint(scale,scale*5)
    return str(new_n)

def map_decimal(s: str) -> str:
    """Map a decimal string to a 2-decimal-place number."""
    x = float(s)
    new_x = round(x*100) + RNG.randint(100,500)
    return f"{new_x}"

def build_number_map(question: str, lhs_expr: str):
    """
    Build a mapping from original numeric strings to new numeric strings.
    """
    mapping = {}
    all_nums = set(NUM_RE.findall(question) + NUM_RE.findall(lhs_expr))
    print(f"lhs_expr:{lhs_expr}")

    for num in all_nums:
        if '.' in num:
            mapping[num] = map_decimal(num)
        else:
            if '*' in lhs_expr:
                mapping[num] = map_integer(num, scale=2)
            else:
                mapping[num] = map_integer(num, scale=10000)

    return mapping

def replace_numbers(text: str, mapping: dict) -> str:
    """Replace numbers in text according to mapping."""
    def repl(match):
        s = match.group(0)
        return mapping.get(s, s)
    return NUM_RE.sub(repl, text)

def recompute_rhs(lhs_expr: str) -> str:
    """Recompute RHS after modifying LHS."""
    lhs_part = lhs_expr.split('=')[0].strip()
    try:
        value = eval(lhs_part, {"__builtins__": None}, {})
    except Exception as e:
        raise RuntimeError(f"Error evaluating: {lhs_part}") from e

    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        s = f"{value}"
        s = s.rstrip("0").rstrip(".")
        return s
    return str(value)

def inner_loop(input_csv, f_out, operator, index_offset=0):
    with open(input_csv, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader):
            q = row["question"]
            ans = row["answer"]
            if operator not in ans:
                continue
            expr_part = ans.split("####")[0].strip()
            lhs_raw = expr_part.split("=")[0].strip()

            mapping = build_number_map(q, lhs_raw)

            q_new = replace_numbers(q, mapping)
            lhs_new = replace_numbers(lhs_raw, mapping)

            rhs_new = recompute_rhs(lhs_new)

            ans_new = f"{lhs_new} = {rhs_new} #### {rhs_new}"

            record = {
                "question": q_new,
                "answer": ans_new,
                "index": str(i+index_offset),
            }
            index_offset += 1
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return index_offset


def process_file(input_csv: str, output_jsonl: str):
    operator_add = 0
    with open(input_csv, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader):
            ans = row["answer"]
            if '+' in ans:
                operator_add += 1

    operator_mult = 0
    with open(input_csv, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        for i, row in enumerate(reader):
            ans = row["answer"]
            if '*' in ans:
                operator_mult += 1

    with open(output_jsonl, "w", encoding="utf-8") as f_out:
        index_offset = inner_loop(input_csv, f_out, '+')
        for _ in range(int(math.ceil(operator_add/operator_mult))):
            index_offset = inner_loop(input_csv, f_out, '*', index_offset)

    print(f"Wrote transformed data to {output_jsonl}")


if __name__ == "__main__":
    process_file("./data/MAWPS/data_raw_filtered.csv", "./data/MAWPS/data_raw_f2.jsonl")
    gen_knowledge("./data/MAWPS/data_raw_f2.jsonl")
