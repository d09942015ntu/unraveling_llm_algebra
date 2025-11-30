import json
import os
import re

def normalize_equation(eq: str) -> str:
    """
    Normalize an equation string by:
      1. Rewriting x - y  as  x+(-y)
      2. Rewriting -x + y as (-x)+y
    """
    # 1) x - y  ->  x+(-y)
    #    we assume x,y are (possibly multi-digit) numbers
    eq = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1+(-\2)', eq)

    # 2) -x + y  ->  (-x)+y  (only when the '-' is at the beginning)
    eq = re.sub(r'^-\s*(\d+)\s*\+\s*(\d+)', r'(-\1)+\2', eq)

    return eq


def process_file(
    in_path: str = "data/biggsm/data.jsonl",
):

    """
    Read the original data.jsonl, extract equations from each 'answer',
    normalize them, and write:
      - a (possibly unchanged) copy of the data to out_data_path
      - a knowledge.jsonl file where each line contains:
            {"knowledge": "...", "index": "..."}
        with all normalized equations joined by newlines.
    """

    knowledge_types = ["1_full", "2_com", "3_xop", "4_noop"]

    knowledge_dir = os.path.dirname(in_path)
    fouts = dict([(k,open(knowledge_dir + "/" + f"knowledge_{k}.jsonl", "w", encoding="utf-8") ) for k in knowledge_types])


    with open(in_path, "r", encoding="utf-8") as fin:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            answer = obj.get("answer", "")
            index = obj.get("index", None)

            # 1. extract everything inside << ... >>
            raw_eqs = re.findall(r"<<([^<>]+)>>", answer)

            # 2. normalize each equation
            normalized_eqs = [normalize_equation(eq.strip()) for eq in raw_eqs]

            # 3a. write (possibly unchanged) data entry
            #json.dump(obj, fdata, ensure_ascii=False)
            #fdata.write("\n")

            # 3b. write knowledge entry (only if we have equations)


            for knowledge_type in knowledge_types:
                if knowledge_type == "1_full":
                    normalized_eqs_new = [x for x in normalized_eqs]
                    pass
                elif knowledge_type == "2_com" or knowledge_type == "3_xop" or knowledge_type == "4_noop":
                    normalized_eqs_new = []
                    for idx,normalized_eq in enumerate(normalized_eqs):
                        if "=" in normalized_eq and ("+" in normalized_eq or "*" in normalized_eq):
                            left_side, right_side = normalized_eq.split("=")
                            if "+" in left_side:
                                ops = left_side.split("+")
                                ops = list(reversed(ops))
                                if knowledge_type == "2_com":
                                    normalized_eqs_new.append(f"{'+'.join(ops)}={right_side}")
                                elif knowledge_type == "3_xop":
                                    normalized_eqs_new.append(f"{'*'.join(ops)}={right_side}")
                                elif knowledge_type == "4_noop":
                                    normalized_eqs_new.append(f"{','.join(ops)}={right_side}")
                                else:
                                    assert 0, f"unknown knowledge type {knowledge_type}"
                            elif "*" in normalized_eq:
                                ops = left_side.split("*")
                                ops = list(reversed(ops))
                                if knowledge_type == "2_com":
                                    normalized_eqs_new.append(f"{'*'.join(ops)}={right_side}")
                                elif knowledge_type == "3_xop":
                                    normalized_eqs_new.append(f"{'+'.join(ops)}={right_side}")
                                elif knowledge_type == "4_noop":
                                    normalized_eqs_new.append(f"{','.join(ops)}={right_side}")
                                else:
                                    assert 0, f"unknown knowledge type {knowledge_type}"
                            else:
                                assert 0, f"unknown operator"
                        else:
                            normalized_eqs_new.append(normalized_eq)
                else:
                    assert 0, f"unknown knowledge type {knowledge_type}"

                knowledge_str = "\n".join(normalized_eqs_new)
                know_obj = {
                    "knowledge": knowledge_str,
                    "index": index
                }
                json.dump(know_obj, fouts[knowledge_type], ensure_ascii=False)
                fouts[knowledge_type].write("\n")


if __name__ == "__main__":
    # By default, read from 'data.jsonl' and write:
    #   - processed data to 'data.jsonl' (can change to 'data_out.jsonl' if desired)
    #   - knowledge to 'knowledge.jsonl'
    process_file()
