import csv
import json
import re

# Regex to capture numbers: decimals first, then integers
NUM_RE = re.compile(r'\d+\.\d+|\d+')

def map_integer(s: str) -> str:
    """Map an integer string to a larger integer."""
    n = int(s)
    new_n = n * 1000 + 123
    return str(new_n)

def map_decimal(s: str) -> str:
    """Map a decimal string to a 4-decimal-place number."""
    x = float(s)
    new_x = x + 0.1234
    return f"{new_x:.4f}"

def build_number_map(question: str, lhs_expr: str):
    """
    Build a mapping from original numeric strings to new numeric strings.
    """
    mapping = {}
    all_nums = set(NUM_RE.findall(question) + NUM_RE.findall(lhs_expr))

    for num in all_nums:
        if '.' in num:
            mapping[num] = map_decimal(num)
        else:
            mapping[num] = map_integer(num)

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
        s = f"{value:.4f}"
        s = s.rstrip("0").rstrip(".")
        return s
    return str(value)

def process_file(input_csv: str, output_jsonl: str):
    with open(input_csv, newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)

        with open(output_jsonl, "w", encoding="utf-8") as f_out:
            for i, row in enumerate(reader):
                q = row["question"]
                ans = row["answer"]

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
                    "index": str(i),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote transformed data to {output_jsonl}")


if __name__ == "__main__":
    process_file("./data/MAWPS/data_raw_filtered.csv", "./data/MAWPS/data.jsonl")
