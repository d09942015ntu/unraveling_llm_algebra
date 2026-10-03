import glob
import os
import re

import torch.nn as nn


def reinitialize_weights(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def latest_checkpoint(ckpt_dir):
    """The checkpoint in ``ckpt_dir`` with the highest step number (e.g. checkpoint-1000 beats checkpoint-500)."""
    def step(path):
        numbers = re.findall(r'\d+', os.path.basename(path))
        return int(numbers[-1]) if numbers else -1
    return max(glob.glob(os.path.join(ckpt_dir, "*")), key=step)
