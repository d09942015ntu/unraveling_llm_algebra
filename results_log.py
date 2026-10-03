"""Read the trainer.log files written by trainer.py."""
import glob
import json


def latest_log(pattern):
    """The last (by name, i.e. newest timestamp) log file matching ``pattern``, or None."""
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def read_accuracy_log(log_file):
    """List of (step, {"train_com": acc, "eval_com": acc, ...}) for every logged step."""
    entries = []
    with open(log_file) as f:
        for line in f:
            if "step" not in line:
                continue
            item = json.loads(line)
            entries.append((item['step'], {**item['train_acc'], **item['eval_acc']}))
    return entries
