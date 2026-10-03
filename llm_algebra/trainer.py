import argparse
from datetime import datetime
import json
import logging
import os
import sys

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainerCallback, TrainingArguments
from torch.utils.data import DataLoader

from evaluator import evaluate
from mydataset import EvalDataset, TrainDataset
from utils import latest_checkpoint, reinitialize_weights


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


class StringOutputEvaluator(TrainerCallback):
    """At every logging step, reload the latest checkpoint and log its train / test accuracy."""

    def __init__(self, tokenizer, ckpt_path, dataset_dir, logger):
        self.ckpt_path = ckpt_path
        self.logger = logger
        self.wait = 0

        def loader(ftype):
            # Batch size 1: inputs have different lengths and are not padded.
            return DataLoader(EvalDataset(dataset_dir, tokenizer, ftype=ftype), batch_size=1, shuffle=False)

        self.train_dataloader = loader('train')
        self.eval_com_dataloader = loader('test_com')
        self.eval_ide_dataloader = loader('test_ide')

    def on_log(self, args, state, control, **kwargs):
        if not os.path.isdir(self.ckpt_path) or not os.listdir(self.ckpt_path):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2LMHeadModel.from_pretrained(latest_checkpoint(self.ckpt_path))
        model.to(device)

        def accuracy(dataloader):
            return evaluate(model, dataloader, limit=100, verbose=True)

        last = state.log_history[-1]
        log_str = json.dumps({'step': last['step'],
                              'epoch': last['epoch'],
                              'loss': last['loss'],
                              'acc_train': accuracy(self.train_dataloader),
                              'acc_eval_com': accuracy(self.eval_com_dataloader),
                              'acc_eval_ide': accuracy(self.eval_ide_dataloader),
                              })
        self.logger.info(log_str)
        if last['loss'] < 0.001:
            self.wait += 1
            if self.wait > 2:
                sys.exit()


def main():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('--model_name', type=str, default='./models/gpt2', help='Pre-trained model name or path')
    parser.add_argument('--dataset_dir', type=str, default='./data/ide_41_11', help='Path to the training dataset')
    parser.add_argument('--output_dir', type=str, default='./results', help='Path to output directory')
    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained(args.model_name, trust_remote_code=True)
    reinitialize_weights(model)
    dataset_train = TrainDataset(args.dataset_dir, GPT2Tokenizer.from_pretrained(args.model_name))
    model.resize_token_embeddings(len(dataset_train.tokenizer))

    formatted_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{formatted_datetime}-{os.path.basename(args.dataset_dir)}")
    ckpt_path = os.path.join(output_dir, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("my_logger", os.path.join(output_dir, "trainer.log"))

    logging_step = 1000
    training_args = TrainingArguments(
        output_dir=ckpt_path,
        num_train_epochs=200000,
        per_device_train_batch_size=1024,
        save_steps=logging_step,
        save_total_limit=3,
        logging_steps=logging_step,
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        processing_class=dataset_train.tokenizer,
        callbacks=[StringOutputEvaluator(dataset_train.tokenizer, ckpt_path, args.dataset_dir, logger)],
    )
    trainer.train()


if __name__ == '__main__':
    main()
