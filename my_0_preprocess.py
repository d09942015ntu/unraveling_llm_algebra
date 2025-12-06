import json
import os
import re
import numpy as np

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

def gen_eq_common(operands, knowledge_type):
    normalize_equations = []
    results = []
    for ops in operands:
        ops = [op if eval(op) >= 0 else f"({eval(op)})" for op in ops]
        for operator in ['+', '*']:
            if "ide" in knowledge_type:
                if operator == '+':
                    eq = f"{'+0+'.join(ops)}"
                elif operator == '*':
                    eq = f"{'*1*'.join(ops)}"
            else:
                eq = f"{operator.join(ops)}"
            print(f"ops:{ops}, eq:{eq}")
            result = eval(eq)
            normalize_equations.append(f"{eq}={result}")
            results.append(str(result) if result >=0 else f"({str(result)})")
    return normalize_equations, results

def gen_noisy_eqs(operands, knowledge_type, rng):
    normalize_equations = []
    normalize_equations_new, _ = gen_eq_common(operands, knowledge_type)
    normalize_equations.extend(normalize_equations_new)
    ops_set = set()
    for _ in range(len(operands)**2):
        choice_indices = sorted(rng.choice(range(len(operands)),2))
        op0 =  rng.choice(operands[choice_indices[0]],1)[0]
        op1 = rng.choice(operands[choice_indices[1]],1)[0]
        if op0 != op1 and op0 != '0' and op1 !='0': #and op0 != '1' and op1 != '1':
            ops_set.add((op0,op1))
    new_operands = list(ops_set)
    normalize_equations_new, result_new = gen_eq_common(new_operands, knowledge_type)
    normalize_equations.extend(normalize_equations_new)
    operand_flatten = []
    for ops in operands:
        operand_flatten.extend(ops)

    ops_set = set()
    if len(result_new) > 0:
        print(f"result_new:{result_new}, operand_flatten:{operand_flatten}")
        for _ in range(len(operand_flatten)*2):
            if 'com' in knowledge_type:
                op0 = rng.choice(result_new,1)[0]
                op1 = rng.choice(operand_flatten,1)[0]
            else:
                op0 = rng.choice(operand_flatten, 1)[0]
                op1 = rng.choice(result_new, 1)[0]
            if op0 != op1 and op0 != '0' and op1 !='0': # and op0 != '1' and op1 != '1':
                ops_set.add((op0,op1))
        new_operands = list(ops_set)
        normalize_equations_new, result_new = gen_eq_common(new_operands, knowledge_type)
        normalize_equations.extend(normalize_equations_new)
    normalize_equations = [str(x) for x in list(rng.permutation(normalize_equations))]
    return normalize_equations

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

    #knowledge_types = ["1_full", "2_com", "5_xop", "4_noop", "41_noop"]
    knowledge_types = ["1_full", "2_com", "3_ide", "11_fullN", "21_comN", "31_ideN", "4_noop", "41_noop", "5_xop"]
    #knowledge_types = ["11_fullN","21_comN", "31_ideN"]

    knowledge_dir = os.path.dirname(in_path)
    fouts = dict([(k,open(knowledge_dir + "/" + f"knowledge_{k}.jsonl", "w", encoding="utf-8") ) for k in knowledge_types])
    rng = np.random.RandomState(0)


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
                else:
                    normalized_eqs_new = []
                    for idx,normalized_eq in enumerate(normalized_eqs):
                        if "=" in normalized_eq and ("+" in normalized_eq or "*" in normalized_eq):
                            left_side, right_side = normalized_eq.split("=")
                            if "+" in left_side:
                                ops = left_side.split("+")
                                ops = list(reversed(ops))
                                if knowledge_type == "2_com":
                                    normalized_eqs_new.append(f"{'+'.join(ops)}={right_side}")
                                elif knowledge_type == "5_xop":
                                    normalized_eqs_new.append(f"{'*'.join(ops)}={right_side}")
                                elif knowledge_type == "4_noop":
                                    normalized_eqs_new.append(f"{','.join(ops)}={right_side}")
                                elif knowledge_type == "41_noop":
                                    normalized_eqs_new.append(f"({','.join(ops)})->{right_side}")
                                elif knowledge_type == "3_ide":
                                    ops = list(reversed(ops))
                                    normalized_eqs_new.append(f"{'+0+'.join(ops)}={right_side}")
                                elif knowledge_type == "11_fullN" or knowledge_type == "21_comN" or knowledge_type == "31_ideN":
                                    if knowledge_type == "11_fullN" or knowledge_type == "31_ideN":
                                        ops = list(reversed(ops))
                                    normalized_eqs_new.append(ops)
                                else:
                                    assert 0, f"unknown knowledge type {knowledge_type}"
                            elif "*" in normalized_eq:
                                ops = left_side.split("*")
                                ops = list(reversed(ops))
                                if knowledge_type == "2_com":
                                    normalized_eqs_new.append(f"{'*'.join(ops)}={right_side}")
                                elif knowledge_type == "5_xop":
                                    normalized_eqs_new.append(f"{'+'.join(ops)}={right_side}")
                                elif knowledge_type == "4_noop":
                                    normalized_eqs_new.append(f"{','.join(ops)}={right_side}")
                                elif knowledge_type == "41_noop":
                                    normalized_eqs_new.append(f"({','.join(ops)})->{right_side}")
                                elif knowledge_type == "3_ide":
                                    ops = list(reversed(ops))
                                    normalized_eqs_new.append(f"{'*1*'.join(ops)}={right_side}")
                                elif knowledge_type == "11_fullN" or knowledge_type == "21_comN" or knowledge_type == "31_ideN":
                                    if knowledge_type == "11_fullN" or knowledge_type == "31_ideN":
                                        ops = list(reversed(ops))
                                    normalized_eqs_new.append(ops)
                                else:
                                    assert 0, f"unknown knowledge type {knowledge_type}"
                            else:
                                assert 0, f"unknown operator"
                        else:
                            if knowledge_type != "21_comN" and knowledge_type != "31_ideN":
                                normalized_eqs_new.append(normalized_eq)

                if knowledge_type == "11_fullN" or knowledge_type == "21_comN" or knowledge_type == "31_ideN":
                    normalized_eqs_new = gen_noisy_eqs(normalized_eqs_new, knowledge_type, rng)
                    if knowledge_type == "21_comN":
                        for eq in normalized_eqs:
                            if eq in normalized_eqs_new:
                                normalized_eqs_new.remove(eq)


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
