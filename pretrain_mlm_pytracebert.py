# -*- coding : utf-8-*-
# import sys
#
# sys.path.append("../")
# sys.path.append("../../")
import torch
import pandas as pd
import argparse
import logging
import json
from datasets import Dataset
from transformers import (TrainingArguments, Trainer, BertTokenizer,
                          DataCollatorForLanguageModeling, BertForMaskedLM)
from util.utils import seed_everything
from sklearn.model_selection import train_test_split
from model.tokenizer import WordEncodeTokenizer
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(format='%(message)s', level=logging.INFO)

seed_everything(seed=1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device: {device} GPUs count: {torch.cuda.device_count()}\n")

torch.cuda.empty_cache()


def encode_data(data, tokenize_function, with_seg=True):
    tracebacks = data['tracebacks']
    if 'next_sentence_label' in data.keys():
        labels = data['next_sentence_label']
        dataset = Dataset.from_dict({'text': tracebacks, 'next_sentence_label': labels})
    else:
        dataset = Dataset.from_dict({'text': tracebacks})
    encoded_dataset = dataset.map(tokenize_function, batched=True, num_proc=args.num_proc, remove_columns=["text"])
    return encoded_dataset


def load_data(valid_ratio=0.1):
    data_file = args.pretrain_data
    df = pd.read_csv(data_file)

    if 'next_label' in df.columns:
        X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.next_label.values,
                                                          test_size=valid_ratio,
                                                          random_state=1234)
        train_next_labels = df.loc[X_train]['next_label']
        val_next_labels = df.loc[X_val]['next_label']
    else:
        X_train, X_val = train_test_split(df.index.values,
                                          test_size=valid_ratio,
                                          random_state=1234)

    train_dataset = df.loc[X_train]['Templates']
    val_dataset = df.loc[X_val]['Templates']

    if 'next_label' in df.columns:
        train_data = {'tracebacks': train_dataset, 'next_sentence_label': train_next_labels}
        val_data = {'tracebacks': val_dataset, 'next_sentence_label': val_next_labels}
    else:
        train_data = {'tracebacks': train_dataset}
        val_data = {'tracebacks': val_dataset}

    logging.info(f"df length: {len(df)}")
    logging.info(f"train_dataset length: {len(train_dataset)}\nval_dataset length: {len(val_dataset)}")
    logging.info(f"train_data keys: {train_data.keys()}")
    logging.info(f"val_data keys: {val_data.keys()}")
    return train_data, val_data


def get_exceed_frame_length_index(args, tracebacks_df):
    exceed_indexes = []
    for traceback in tqdm(tracebacks_df):
        frames = traceback.split('[SEP]')
        last_frame = frames[-2]
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
        last_frames_tokens = tokenizer.tokenize(last_frame)
        if len(last_frames_tokens) > args.max_frames_len:
            exceed_indexes.append(traceback.index)

    return exceed_indexes


def parse_args():
    args = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # args.add_argument('--local_rank', type=int, default=-1,
    #                   help='Local rank passed from distributed launcher')
    args.add_argument("-num_proc", "--num_proc", type=int, default=4, help="")
    args.add_argument("-pretrain_data", "--pretrain_data", type=str,
                      default="./dataset/pretrain/templates_train.csv",
                      help="pre-train data directory")
    args.add_argument("-outfolder", "--outfolder", type=str,
                      default="./output/pytracebert/pretrain/", help="Folder name to save the models.")
    args.add_argument("-base_model", "--base_model", type=str,
                      default="./pretrained/bert/", help="base_model")
    args.add_argument("-tokenizer_path", "--tokenizer_path", type=str,
                      default="./pretrained/bert/", help="tokenizer_path")

    args.add_argument("-max_len", "--max_len", type=int, default=512, help="Max length of sequence")
    args.add_argument("-mlm_probability", "--mlm_probability", type=float,
                      default=0.15, help="probability of masking text.")

    args.add_argument("-epochs", "--epochs", type=int, default=5, help="Number of epochs")
    args.add_argument("-batch_size", "--batch_size", type=int, default=16, help="Batch Size")
    args.add_argument("-learning_rate", "--learning_rate", type=float, default=2e-5, help="learning_rate")
    args.add_argument("-warmup_ratio", "--warmup_ratio", type=float, default=0.05, help="warmup_ratio")
    args.add_argument("-weight_decay", "--weight_decay", type=float, default=0.01, help=".")

    args.add_argument("-max_excep_len", "--max_excep_len", type=int, default=50, help="Max length of sequence")
    args.add_argument("-max_frames_len", "--max_frames_len", type=int, default=460, help="Max length of sequence")

    args = args.parse_args()
    return args


def pretraining():
    logging.info("Spliting Train and Valid Dataset...")
    train_dataset, val_dataset = load_data()

    tokenizer = WordEncodeTokenizer(args.tokenizer_path, max_length=args.max_len, max_excep_len=args.max_excep_len,
                                    max_frames_len=args.max_frames_len)

    def tokenize_function(tracebacks):
        tokenized_batch = tokenizer.encode_tracebacks_without_segment(tracebacks['text'])
        return tokenized_batch

    logging.info(f"Encoding data:\n")
    encoded_train_dataset = encode_data(train_dataset, tokenize_function)
    encoded_val_dataset = encode_data(val_dataset, tokenize_function)

    logging.debug(f"Loading data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer.get_tokenizer(),
        mlm=True,
        mlm_probability=args.mlm_probability
    )

    logging.debug(f"Initial model...")
    model = BertForMaskedLM.from_pretrained(args.base_model)

    training_args = TrainingArguments(
        output_dir=args.outfolder,
        overwrite_output_dir=True,
        logging_dir=args.outfolder + 'logs/',
        logging_strategy="steps",
        logging_steps=500,

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,  # batch_size of every GPU
        per_device_eval_batch_size=args.batch_size,  # batch_size of every GPU
        gradient_accumulation_steps=8,

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,

        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        dataloader_num_workers=4,
    )

    # logging.info(f"Initial Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train_dataset,
        eval_dataset=encoded_val_dataset,
        data_collator=data_collator,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=7)],
    )
    train_result = trainer.train()

    logging.info(f"Saving model...")
    trainer.save_model(args.outfolder + 'model/')  # save model and tokenizer

    logging.info(f"Training results: {train_result}")
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    # save env info
    save_infos()

    # plot
    # plot_evaluation_results(model='pytracebert', 
    #                         epoch=args.epochs, 
    #                         dropout=model.config.hidden_dropout_prob, 
    #                         lr=args.learning_rate, 
    #                         output_dir=args.outfolder+'imgs',
    #                         input_dir = args.outfolder,
    #                         fold = 'pretrain',
    #                         metric=trainer.args.metric_for_best_model
    #                         )


def save_infos():
    args_dict = vars(args)
    args_dict["Pytorch Version"] = torch.__version__
    args_dict["GPU"] = torch.cuda.get_device_name(0)

    with open(args.outfolder + "parameters_" + str(datetime.now()) + ".txt", 'a') as f:
        json.dump(args_dict, f, indent=4)


if __name__ == '__main__':
    args = parse_args()
    pretraining()
