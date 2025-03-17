import argparse
from datetime import datetime
import glob
import json
import logging
import os
import shutil
import sys

import numpy as np
from transformers import Trainer, TrainingArguments, TrainerCallback, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import DataLoader

from evaluator import evaluate
from mydataset import TrainDataset, EvalDataset
from utils import reinitialize_weights


def setup_logger(name, log_file, level=logging.DEBUG):
    """Sets up a logger to output to both terminal and file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class StringOutputEvaluator(TrainerCallback):
    def __init__(self, model, tokenizer, ckpt_path, dataset_dir, logger, dataset_type_list):
        self.model = model
        self.tokenizer = tokenizer
        self.ckpt_path = ckpt_path
        self.dataset_dir = dataset_dir
        self.logger = logger
        self.wait = 0
        self.batch_size = 32
        self.train_dataloaders = dict([(data_type, DataLoader(
            EvalDataset(self.dataset_dir, self.tokenizer, ftype=f'train_{data_type}', rm_position=0),
            batch_size=self.batch_size, shuffle=False)) for data_type in dataset_type_list])
        self.eval_dataloaders = dict([(data_type, DataLoader(
            EvalDataset(self.dataset_dir, self.tokenizer, ftype=f'test_{data_type}', rm_position=0),
            batch_size=self.batch_size, shuffle=False)) for data_type in dataset_type_list
                                      if os.path.isfile(os.path.join(self.dataset_dir, f'test_{data_type}.csv'))])

    def on_log(self, args, state, control, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(device)
        self.model.eval()

        train_accuracy = dict([(f"train_{ikey}", evaluate(self.model, self.train_dataloaders[ikey])) for ikey in
                               self.train_dataloaders.keys()])
        eval_accuracy = dict([(f"eval_{ikey}", evaluate(self.model, self.eval_dataloaders[ikey])) for ikey in
                              self.eval_dataloaders.keys()])
        epoch = state.log_history[-1]['epoch']
        loss = state.log_history[-1].get('loss', np.inf)
        step = state.log_history[-1]['step']

        history_len = len(state.log_history)
        history_len_2 = 50
        history_len_4 = 20
        history_avg_1 = np.average([x.get('loss', 9999) for x in state.log_history[-history_len_2:-history_len_4]])
        history_avg_2 = np.average([x.get('loss', 9999) for x in state.log_history[-history_len_4:]])

        log_str = json.dumps({'step': step,
                              'epoch': epoch,
                              'loss': loss,
                              'train_acc': train_accuracy,
                              'eval_acc': eval_accuracy,
                              "history": {
                                  "history_len_2": history_len_2,
                                  "history_len_4": history_len_4,
                                  "history_avg_1": history_avg_1,
                                  "history_avg_2": history_avg_2
                              }}
                             )

        self.logger.info(log_str)
        print(log_str)
        if loss < 0.0001:
            self.wait += 1
            if self.wait > 2:
                sys.exit()

        if history_len > 100 and abs(history_avg_1 - history_avg_2) < 0.0001:
            self.wait += 1
            if self.wait > 2:
                sys.exit()


def main():
    parser = argparse.ArgumentParser(description='Train a GPT-2 model.')
    parser.add_argument('--model_name', type=str, default='./mymodels/toytrans', help='Pre-trained model name or path')
    parser.add_argument('--dataset_dir', type=str, default='./data/all_62_7_1000', help='Path to the training dataset')
    parser.add_argument('--dataset_type', type=str, default='com+ide',
                        help='Categories of tasks, seperated by `+`. e.g. com+ide represents commutativity+identity')
    parser.add_argument('--output_name', type=str, default='', help='path to output directory')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--logging_step', type=int, default=1000, help='Logging step')
    parser.add_argument('--num_train_epochs', type=int, default=2000000, help='Total number of training epoch')

    args = parser.parse_args()

    tokenizer_path = "gpt2"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, ignore_mismatched_sizes=True)
    reinitialize_weights(model)
    dataset_list = []
    dataset_type_list = args.dataset_type.split("+")
    for dataset_type in dataset_type_list:
        dataset_list.append(TrainDataset(args.dataset_dir, AutoTokenizer.from_pretrained(tokenizer_path),
                                         ftype=f'train_{dataset_type}', rm_position=0))

    dataset_train = torch.utils.data.ConcatDataset(dataset_list)

    if 'resize_token_embeddings_by_tokenizer' in dir(model):
        model.resize_token_embeddings_by_tokenizer(dataset_list[0].tokenizer, fix_transformer=0)
    else:
        model.resize_token_embeddings(len(dataset_list[0].tokenizer))

    # Get the current date and time
    current_datetime = datetime.now()

    # Print the current date and time
    formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./results", f"{args.output_name}-{formatted_datetime}")
    ckpt_path = os.path.join(output_dir, "checkpoints")
    model_file_path = os.path.join(output_dir, "model")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
    model_files = glob.glob(os.path.join(args.model_name, "modeling_*"))
    for model_file in model_files:
        shutil.copy(os.path.join(model_file), model_file_path)
    logger = setup_logger("my_logger", os.path.join(output_dir, "trainer.log"))

    training_args = TrainingArguments(
        output_dir=ckpt_path,  # Directory to save the training results
        num_train_epochs=args.num_train_epochs,  # Total number of training epochs
        per_device_train_batch_size=args.batch_size,  # 1024, # Batch size per device during training
        save_steps=args.logging_step,  # Save the model every 50 steps
        save_total_limit=1,  # Keep a maximum of 3 checkpoints
        logging_steps=args.logging_step,  # Log(output) after every 10 steps
        learning_rate=5e-5,  # Initial learning rate
        weight_decay=0.01  # L2 weight decay (regularization)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_train,
        tokenizer=dataset_list[0].tokenizer,
        callbacks=[StringOutputEvaluator(model, dataset_list[0].tokenizer, ckpt_path, args.dataset_dir, logger,
                                         dataset_type_list)]
    )

    # Train the model
    trainer.train()


# Initialize Trainer
if __name__ == '__main__':
    main()
