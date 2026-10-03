"""Answer checking and file helpers shared by the my_*.py scripts."""
import os
import re

from utils.tools import read_jsonl


def is_number(pred):
    try:
        float(pred)
    except Exception:
        return False
    return True


def get_parsed_pred_answer(data):
    """Final numeric answer in the model output ``data["pred"]`` (-1 if none is found).

    Normally the last number after the last "=". If the output defines variables as
    ``<<var1=...>>``, they are evaluated and the variable named after "####" (or the
    last one) is returned.
    """
    pred_str = data["pred"]
    if "var1" not in pred_str or "<<" not in pred_str:
        pred_list = re.findall(r'-?\d+\.?\,?\d*', pred_str.replace(",", "").strip(".").split("=")[-1])
        return pred_list[-1] if pred_list else -1

    try:
        eqs = [s for s in re.findall(r'<<(.*?)>>', pred_str) if "=" in s]
        eqs = sorted(eqs, key=lambda x: int(x.split("=")[0].strip("var")))
        var_list = {eq.split("=")[0]: None for eq in eqs}
        for eq in eqs:
            func_str = eq.split("=")[1]
            for var in var_list:
                if var_list[var] is not None:
                    func_str = func_str.replace(var, str(var_list[var]))
            if var_list[eq.split("=")[0]] is None:
                try:
                    var_list[eq.split("=")[0]] = eval(func_str)
                except Exception:
                    return -1
        if "####" in pred_str:
            var_key = pred_str.split("####")[-1].strip().strip(".").replace("<", "").replace(">", "")
            if var_key in var_list:
                return var_list[var_key]
        last_var = var_list[list(var_list.keys())[-1]]
        return -1 if last_var is None else last_var
    except Exception:
        return -1


def is_correct(data):
    """Whether the predicted answer matches the gold answer after "#### " (to 2 decimals, ignoring sign)."""
    golden_answer_str = data["origin"]["answer"].replace(",", "").strip(".").split("#### ")[-1]
    golden_answer = round(float(golden_answer_str), 2)
    pred = get_parsed_pred_answer(data)
    return is_number(pred) and abs(abs(round(float(pred), 2)) - abs(golden_answer)) < 0.01


def load_knowledge(knowledge_file):
    """{index: knowledge text} from a knowledge_*.jsonl file; empty if the file does not exist."""
    if not os.path.exists(knowledge_file):
        return {}
    print(f"loading knowledge:{knowledge_file}")
    return {data["index"]: data["knowledge"] for data in read_jsonl(knowledge_file)}
