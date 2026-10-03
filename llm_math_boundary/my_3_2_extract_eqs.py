"""Keep the MAWPS rows whose answer uses a single operator type at least twice."""
import csv

OPS = ["+", "-", "*", "/"]

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


def main(input_file="data/MAWPS/data_raw_all.csv", output_file="data/MAWPS/data_raw_filtered.csv"):
    with open(input_file, newline="", encoding="utf-8") as f:
        kept_rows = [row for row in csv.DictReader(f) if qualifies(extract_expr(row["answer"]))]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Saved {len(kept_rows)} rows to {output_file}")


if __name__ == "__main__":
    main()
