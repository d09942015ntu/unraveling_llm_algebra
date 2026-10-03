import csv
import re

OPS = ["+", "-", "*", "/"]
#OPS = ["+", "*"]

def extract_expr(answer):
    # Get text before "####"
    return answer.split("####")[0].strip()

def count_operators(expr):
    """
    Return a dict {op: count} for each operator in expr.
    """
    return {op: expr.count(op) for op in OPS}

def qualifies(expr):
    """
    True iff:
      1. Only 1 operator type is present.
      2. That operator appears at least twice.
    """
    counts = count_operators(expr)

    # operator types present (count ≥ 1)
    present_ops = [op for op, c in counts.items() if c > 0]

    # must be exactly one type
    if len(present_ops) != 1:
        return False

    # that operator must appear at least twice
    op = present_ops[0]
    return counts[op] >= 2


# --------------------------
# Main processing
# --------------------------

input_file = "data/MAWPS/data_raw_all.csv"
output_file = "data/MAWPS/data_raw_filtered.csv"

kept_rows = []

with open(input_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        expr = extract_expr(row["answer"])
        if qualifies(expr):
            kept_rows.append(row)

# Save filtered results
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "answer"])
    writer.writeheader()
    writer.writerows(kept_rows)

print(f"Saved {len(kept_rows)} rows to {output_file}")
