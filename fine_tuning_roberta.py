# -*- coding : utf-8-*-
# import sys
# sys.path.append("../")
# sys.path.append("../../")

from datasets import Dataset
from transformers import (TrainingArguments, Trainer, DataCollatorWithPadding)
from model.models import RobertaForClassification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (f1_score, precision_score, recall_score)
from util.utils import seed_everything
from datetime import datetime
from model.tokenizer import WordEncodeTokenizer
from data.data_augmentation import random_inplace_number

import torch
import pandas as pd
import numpy as np
import argparse
import logging
import time
import gc
import json

logging.basicConfig(format='%(message)s', level=logging.INFO)

seed_everything(seed=1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"device: {device} GPUs count: {torch.cuda.device_count()}\n")
torch.cuda.empty_cache()


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = torch.nn.functional.softmax(torch.tensor(preds), dim=1).numpy()
    preds = np.argmax(preds, axis=-1).flatten()  # 根据正负例概率选择最大值作为预测类别
    labels = labels.flatten()

    return {
        'f1': f1_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average='weighted'),
        'precision_weighted': precision_score(labels, preds, average='weighted'),
        'recall_weighted': recall_score(labels, preds, average='weighted'),
    }


def encode_data(data, tokenize_function):
    tracebacks = data['tracebacks']
    labels = data['labels']

    dataset = Dataset.from_dict({'text': tracebacks, 'labels': labels})
    encoded_dataset = dataset.map(tokenize_function, batched=True, num_proc=args.num_proc, remove_columns=["text"])
    return encoded_dataset


def load_data(data_file):
    df = pd.read_csv(data_file)
    tracebacks = df['Templates'].tolist()
    labels = df['label'].tolist() if 'label' in df.columns else []
    return {'tracebacks': tracebacks, 'labels': labels}


def parse_args():
    args = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter_class意味着您希望在生成的帮助文本中包含每个参数的默认值

    args.add_argument("-num_proc", "--num_proc", type=int, default=1, help="")
    # args.add_argument('--local_rank', type=int, default=-1,
    #                     help='Local rank passed from distributed launcher')
    args.add_argument("-labeled_data", "--labeled_data", type=str,
                      default="./dataset/task/templates_fine_tuning.csv", help="fine_tuning data directory")

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default=f"./output/roberta/kfold_fine_tuning/", help="Folder name to save the models.")

    args.add_argument("-pretrained_model_path", "--pretrained_model_path", type=str,
                      default=f"./pretrained/roberta/", help="pretrained model")
    args.add_argument("-tokenizer_path", "--tokenizer_path", type=str,
                      default="./pretrained/bert/", help="tokenizer_path")
    args.add_argument("-max_len", "--max_len", type=int, default=512, help="Max length of sequence")

    args.add_argument("-dropout", "--dropout", type=int, default=0.15, help="")
    args.add_argument("-epochs", "--epochs", type=int, default=15, help="Number of epochs")
    args.add_argument("-batch_size", "--batch_size", type=int, default=16, help="Batch Size")
    args.add_argument("-learning_rate", "--learning_rate", type=float, default=1e-5, help=".")
    args.add_argument("-warmup_ratio", "--warmup_ratio", type=float, default=0.05, help=".")
    args.add_argument("-weight_decay", "--weight_decay", type=float, default=0.01, help=".")

    args.add_argument("-max_excep_len", "--max_excep_len", type=int, default=50, help="Max length of sequence")
    args.add_argument("-max_frames_len", "--max_frames_len", type=int, default=460, help="Max length of sequence")

    args = args.parse_args()

    return args


def train(MODEL='pytracebert', augmentation=True):
    weighted_f1 = []
    weighted_recall = []
    weighted_precision = []
    # split dataset into 5 fold train and test
    train_folds, test_folds, train_labels, test_labels = split_kfold(outfolder=None)
    for fold in range(len(train_folds)):
        logging.info(f"Fold: {fold}")
        logging.debug("Spliting Train and Valid Dataset...")

        X_train, X_val, y_train, y_val = train_test_split(train_folds[fold], train_labels[fold], test_size=0.1,
                                                          stratify=train_labels[fold], random_state=42)

        if augmentation:
            filename = args.labeled_data.split('/')[-1]
            X_train, y_train = random_inplace_number(X_train, y_train, save_file=None)

        train_dataset = {'tracebacks': X_train, 'labels': y_train}
        val_dataset = {'tracebacks': X_val, 'labels': y_val}

        tokenizer = WordEncodeTokenizer(args.tokenizer_path, max_length=args.max_len, max_excep_len=args.max_excep_len,
                                        max_frames_len=args.max_frames_len)

        def tokenize_function(tracebacks):
            # tokenized_batch = tokenizer.encode_tracebacks_with_segment(tracebacks['text'])
            tokenized_batch = tokenizer.encode_tracebacks_without_segment(tracebacks['text'])
            return tokenized_batch

        encoded_train_dataset = encode_data(train_dataset, tokenize_function)
        encoded_val_dataset = encode_data(val_dataset, tokenize_function)

        logging.info(f"Loading data collator...")
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer.get_tokenizer(),
            # max_length=args.max_len   
            # padding='max_length',
        )

        logging.info(f"Loading model...")
        model = RobertaForClassification.from_pretrained(args.pretrained_model_path, num_labels=2)
        model.config.hidden_dropout_prob = args.dropout

        # 4.pre-train
        logging.info(f"Initial training arguments...")
        training_args = TrainingArguments(
            output_dir=args.outfolder,
            overwrite_output_dir=True,
            logging_dir=args.outfolder + 'logs/',
            logging_strategy="epoch",

            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,  # batch_size of every GPU
            per_device_eval_batch_size=args.batch_size,  # batch_size of every GPU
            gradient_accumulation_steps=8,

            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,

            evaluation_strategy="epoch",
            save_strategy='epoch',
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            dataloader_num_workers=4,
        )

        logging.info(f"Initial Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_train_dataset,
            eval_dataset=encoded_val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
        )

        logging.info(f"Training...")
        trainer.train()

        logging.info(f"Saving model...")
        trainer.save_model(args.outfolder + 'model/fold_' + str(fold) + '/')

        # test phase
        logging.info(f"Test set phase...")
        test_dataset = {'tracebacks': test_folds[fold], 'labels': test_labels[fold]}
        encoded_test_dataset = encode_data(test_dataset, tokenize_function)
        eval_results = trainer.evaluate(encoded_test_dataset)
        logging.info(f"Test set results: {eval_results}")

        weighted_f1.append(eval_results['eval_f1_weighted'] * 100)
        weighted_precision.append(eval_results['eval_precision_weighted'] * 100)
        weighted_recall.append(eval_results['eval_recall_weighted'] * 100)

        # plot
        # plot_evaluation_results(model=MODEL,
        #                         epoch=args.epochs,
        #                         dropout=model.config.hidden_dropout_prob,
        #                         lr=args.learning_rate,
        #                         input_dir=args.outfolder,
        #                         output_dir=args.outfolder + 'imgs',
        #                         fold=fold,
        #                         metric=trainer.args.metric_for_best_model
        #                         )

        del encoded_train_dataset
        del encoded_val_dataset
        del encoded_test_dataset
        gc.collect()

    avg_weighted_f1_TEXT = f"Weighted Average F1: {np.mean(weighted_f1)} (+/- {np.std(weighted_f1)})"
    avg_weighted_recall_TEXT = f"Weighted Average Recall: {np.mean(weighted_recall)} (+/- {np.std(weighted_recall)})"
    avg_weighted_precision_TEXT = f"Weighted Average Precision: {np.mean(weighted_precision)} (+/- {np.std(weighted_precision)})"

    logging.info(avg_weighted_f1_TEXT)
    logging.info(avg_weighted_recall_TEXT)
    logging.info(avg_weighted_precision_TEXT)

    with open(args.outfolder + f'{MODEL}_avg_metrics.txt', mode='a') as f:
        f.write(f"TIME: {datetime.now()}\n")
        f.write(
            f"epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.learning_rate}\n")
        f.write(f"MODEL: {MODEL}, attention: True, data_augmentation: {augmentation}\n")
        f.write(avg_weighted_f1_TEXT + '\n')
        f.write(avg_weighted_recall_TEXT + '\n')
        f.write(avg_weighted_precision_TEXT + '\n')
        f.write('\n')

    save_infos()


# save args params
def save_infos():
    args_dict = vars(args)
    args_dict["Pytorch Version"] = torch.__version__
    args_dict["GPU"] = torch.cuda.get_device_name(0)

    with open(args.outfolder + "parameters_" + str(datetime.now()) + ".txt", 'a') as f:
        json.dump(args_dict, f, indent=4)


def split_kfold(outfolder=None):
    df = pd.read_csv(args.labeled_data)
    X = df['Templates'].values
    y = df['label'].values
    # 5-cross split
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_folds = []
    train_fold_labels = []
    test_folds = []
    test_fold_labels = []
    for i, (train_index, test_index) in enumerate(folds.split(X, y)):
        logging.info("=" * 40)
        logging.info(f"Fold: {i + 1}")
        logging.info("=" * 40)
        train_fold_data, test_fold_data = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]

        train_folds.append(train_fold_data.tolist())
        train_fold_labels.append(train_labels.tolist())
        test_folds.append(test_fold_data.tolist())
        test_fold_labels.append(test_labels.tolist())

        # save
        # train_fold_df = pd.DataFrame({"Templates":train_fold_data, "label":train_labels})
        # test_fold_df = pd.DataFrame({"Templates":test_fold_data, "label":test_labels})
        # logging.info(f"train_fold_df: {len(train_fold_df)}")
        # logging.info(f"test_fold_df: {len(test_fold_df)}")

        # train_fold_df.to_csv(f"{outfolder}/train_fold_{i}.csv", index=False)
        # test_fold_df.to_csv(f"{outfolder}/test_fold_{i}.csv", index=False)

    logging.info(f"train_folds: {len(train_folds[0])}")
    logging.info(f"test_folds: {len(test_folds[0])}")
    logging.info(f"train_fold_labels: {len(train_fold_labels[0])}")
    logging.info(f"test_fold_labels: {len(test_fold_labels[0])}")
    return train_folds, test_folds, train_fold_labels, test_fold_labels


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    train(args, augmentation=True)

    end_time = time.time()
    logging.info(f"total time: {(end_time - start_time) / 60}min")
