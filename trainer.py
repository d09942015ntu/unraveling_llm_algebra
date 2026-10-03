import argparse
from datetime import datetime
import glob
import json
import logging
import os
import shutil

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from evaluator import evaluate
from mydataset import EvalDataset, TrainDataset
from utils import reinitialize_weights


def setup_logger(name, log_file, level=logging.DEBUG):
    """Logger that writes plain messages to both the terminal and ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(message)s')
    for handler in [logging.FileHandler(log_file), logging.StreamHandler()]:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class AccuracyLogger(TrainerCallback):
    """At every logging step, measure train/test accuracy of each dataset type and stop once the loss settles."""

    def __init__(self, model, tokenizer, dataset_dir, logger, dataset_types, rm_position=0, max_length=15,
                 batch_size=32):
        self.model = model
        self.logger = logger
        self.wait = 0

        def loader(ftype):
            dataset = EvalDataset(dataset_dir, tokenizer, ftype=ftype, rm_position=rm_position,
                                  max_length=max_length)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.train_dataloaders = {t: loader(f'train_{t}') for t in dataset_types}
        self.eval_dataloaders = {t: loader(f'test_{t}') for t in dataset_types
                                 if os.path.isfile(os.path.join(dataset_dir, f'test_{t}.csv'))}

    def on_log(self, args, state, control, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        train_accuracy = {f"train_{t}": evaluate(self.model, dl) for t, dl in self.train_dataloaders.items()}
        eval_accuracy = {f"eval_{t}": evaluate(self.model, dl) for t, dl in self.eval_dataloaders.items()}
        last = state.log_history[-1]
        loss = last.get('loss', np.inf)

        # Average loss over two recent windows of the log history, used to detect a plateau.
        older_window, recent_window = 50, 20
        older_avg = np.average([x.get('loss', 9999) for x in state.log_history[-older_window:-recent_window]])
        recent_avg = np.average([x.get('loss', 9999) for x in state.log_history[-recent_window:]])

        self.logger.info(json.dumps({
            'step': last['step'],
            'epoch': last['epoch'],
            'loss': loss,
            'train_acc': train_accuracy,
            'eval_acc': eval_accuracy,
            "history": {
                "history_len_2": older_window,
                "history_len_4": recent_window,
                "history_avg_1": older_avg,
                "history_avg_2": recent_avg,
            }}))

        loss_is_tiny = loss < 0.0001
        loss_plateaued = len(state.log_history) > 100 and abs(older_avg - recent_avg) < 0.0001
        self.wait += int(loss_is_tiny) + int(loss_plateaued)
        if self.wait > 2:
            control.should_training_stop = True


def main():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Pre-trained model name or path')
    parser.add_argument('--dataset_dir', type=str, default='./data/all_64_7_1000', help='Path to the training dataset')
    parser.add_argument('--dataset_type', type=str, default='com+ide',
                        help='Categories of tasks, separated by `+`. e.g. com+ide represents commutativity+identity')
    parser.add_argument('--output_name', type=str, default='', help='Name of the output directory under ./results')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--logging_step', type=int, default=1000, help='Logging step')
    parser.add_argument('--num_train_epochs', type=int, default=2000000, help='Total number of training epoch')
    parser.add_argument('--rm_position', type=int, default=0,
                        help='1: give all operands the same position id, so their order is hidden from the model')
    parser.add_argument('--max_length', type=int, default=15, help='Number of tokens every sample is padded to')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, ignore_mismatched_sizes=True)
    reinitialize_weights(model)
    dataset_types = args.dataset_type.split("+")
    datasets = [TrainDataset(args.dataset_dir, AutoTokenizer.from_pretrained("gpt2"), ftype=f'train_{t}',
                             rm_position=args.rm_position, max_length=args.max_length)
                for t in dataset_types]
    tokenizer = datasets[0].tokenizer

    if hasattr(model, 'resize_token_embeddings_by_tokenizer'):
        model.resize_token_embeddings_by_tokenizer(tokenizer, fix_transformer=0)
    else:
        model.resize_token_embeddings(len(tokenizer))

    output_dir = os.path.join("./results", f"{args.output_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ckpt_path = os.path.join(output_dir, "checkpoints")
    model_file_path = os.path.join(output_dir, "model")
    os.makedirs(model_file_path, exist_ok=True)
    # Keep a copy of custom model code (if any) next to the results.
    for model_file in glob.glob(os.path.join(args.model_name, "modeling_*")):
        shutil.copy(model_file, model_file_path)
    logger = setup_logger("my_logger", os.path.join(output_dir, "trainer.log"))

    training_args = TrainingArguments(
        output_dir=ckpt_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.logging_step,
        save_total_limit=1,
        logging_steps=args.logging_step,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ConcatDataset(datasets),
        processing_class=tokenizer,
        callbacks=[AccuracyLogger(model, tokenizer, args.dataset_dir, logger, dataset_types,
                                  rm_position=args.rm_position, max_length=args.max_length)],
    )
    trainer.train()


if __name__ == '__main__':
    main()
