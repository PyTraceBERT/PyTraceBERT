# import sys
# sys.path.append("../")
# sys.path.append("../../")

import torch
import argparse
import pandas as pd
import numpy as np
import logging
# import glob
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from evaluation.WordEmbeddings import WordEmbeddings
from evaluation.BiLSTMAttention import BiLSTMAttention
from evaluation.CNN import CNN
from evaluation.tokenizer import SpacyTokenizer
from data.data_augmentation import random_inplace_number

from util.utils import seed_everything

# logging = logging.getlogging(__name__)
logging.basicConfig(format='%(message)s', level=logging.INFO)
seed_everything()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    args = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # formatter_class意味着您希望在生成的帮助文本中包含每个参数的默认值

    args.add_argument("-dataset_folder", "--dataset_folder", type=str,
                      default="./dataset/Kfold/unsep", help="fine_tuning data directory")

    args.add_argument("-outfolder", "--outfolder", type=str,
                      default=f"./output/bilstm/", help="Folder name to save the models.")

    args.add_argument("-dropout", "--dropout", type=int, default=0.15, help="")
    args.add_argument("-epochs", "--epochs", type=int, default=15, help="Number of epochs")
    args.add_argument("-batch_size", "--batch_size", type=int, default=16, help="Batch Size")
    args.add_argument("-learning_rate", "--learning_rate", type=float, default=1e-5, help="learning_rate")

    args = args.parse_args()

    return args


def create_datasets(texts, labels, word_embeddings):
    tokenized_data = word_embeddings.tokenize(texts)

    input_ids = tokenized_data['input_ids']
    attention_mask = tokenized_data['attention_mask']
    sentence_lengths = tokenized_data['sentence_lengths']

    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = TensorDataset(input_ids, attention_mask, sentence_lengths, labels)
    return dataset


def load_data(args):
    train_folds = []
    test_folds = []
    val_folds = []
    train_fold_labels = []
    test_fold_labels = []
    val_fold_labels = []

    folds = 5
    for fold in range(folds):
        train_df = pd.read_csv(f"{args.dataset_folder}/train_fold_unsep_{fold}.csv")
        test_df = pd.read_csv(f"{args.dataset_folder}/test_fold_unsep_{fold}.csv")

        train_fold_data, test_fold_data = train_df['Templates'].values.tolist(), test_df['Templates'].values.tolist()
        train_labels, test_labels = train_df['label'].values.tolist(), test_df['label'].values.tolist()
        train_fold_data, val_fold_data, train_labels, val_labels = train_test_split(train_fold_data, train_labels,
                                                                                    test_size=0.1,
                                                                                    stratify=train_labels,
                                                                                    random_state=42)
        # augmentation
        train_fold_data, train_labels = random_inplace_number(train_fold_data, train_labels, save_file=None)

        train_folds.append(train_fold_data)
        train_fold_labels.append(train_labels)
        test_folds.append(test_fold_data)
        test_fold_labels.append(test_labels)
        val_folds.append(val_fold_data)
        val_fold_labels.append(val_labels)

    logging.info(f"train_folds: {len(train_folds[0])}  test_folds: {len(test_folds[0])} val_folds: {len(val_folds[0])}")
    logging.info(
        f"train_fold_labels: {len(train_fold_labels[0])}  test_fold_labels: {len(test_fold_labels[0])} val_fold_labels: {len(val_fold_labels[0])}")
    return train_folds, test_folds, val_folds, train_fold_labels, test_fold_labels, val_fold_labels


def train(args, model, word_embeddings, train_loader, val_loader, criterion, optimizer):
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, attention_mask, sentence_lengths, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                embeddings_output = word_embeddings(
                    {'input_ids': inputs, 'attention_mask': None, 'sentence_lengths': sentence_lengths})

            logits = model(embeddings_output)
            if not logits.is_floating_point():
                logits = logits.float()

            labels = labels.long()

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        logging.info(f'Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}')
        result = evaluate(model, word_embeddings, val_loader, criterion)

    return train_loss, result


def evaluate(model, word_embeddings, test_loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    all_result = {}
    val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, attention_mask, sentence_lengths, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            embeddings_output = word_embeddings(
                {'input_ids': inputs, 'attention_mask': None, 'sentence_lengths': sentence_lengths})
            logits = model(embeddings_output)

            if not logits.is_floating_point():
                logits = logits.float()

            labels = labels.long()

            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(test_loader)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')

    logging.info(
        f"Eval Loss: {val_loss:.4f}"
        f"weighted_f1: {f1_weighted:.4f}, weighted_precision: {precision_weighted:.4f}, weighted_recall: {recall_weighted:.4f} "
    )

    all_result['val_loss'] = val_loss
    all_result['f1_weighted'] = f1_weighted
    all_result['precision_weighted'] = precision_weighted
    all_result['recall_weighted'] = recall_weighted
    return all_result


def main(args):
    embedding_file_path = './evaluation/word2vec/glove.6B.300d.txt'
    tokenizer = SpacyTokenizer()
    word_embeddings = WordEmbeddings.from_text_file(embedding_file_path, tokenizer=tokenizer)

    MODEL = 'bilstm'
    args.outfolder = f"./output/{MODEL}/"
    # BiLSTM
    model = BiLSTMAttention(word_embedding_dimension=word_embeddings.get_word_embedding_dimension(),bidirectional=True)
    # CNN
    # model = CNN(in_word_embedding_dimension=300, dropout=args.dropout)

    logging.info(f"Loading data...")
    train_folds, test_folds, val_folds, train_fold_labels, test_fold_labels, val_fold_labels = load_data(args)
    for fold in range(len(train_folds)):
        logging.info(f"fold {fold}")

        logging.info(f"Preparing Data...")
        train_texts, test_texts, val_texts, train_labels, test_labels, val_labels = train_folds[fold], test_folds[fold], \
                                                                                    val_folds[fold], \
                                                                                    train_fold_labels[fold], \
                                                                                    test_fold_labels[fold], \
                                                                                    val_fold_labels[fold]

        train_dataset = create_datasets(train_texts, train_labels, word_embeddings)
        val_dataset = create_datasets(val_texts, val_labels, word_embeddings)
        test_dataset = create_datasets(test_texts, test_labels, word_embeddings)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # train
        model.to(device)
        word_embeddings.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        logging.info(f"Training...")
        _, _ = train(args, model, word_embeddings, train_loader, val_loader, criterion, optimizer)

        logging.info(f"Test...")
        result = evaluate(model, word_embeddings, test_loader, criterion)

        eval_losses = []
        weighted_f1 = []
        weighted_recall = []
        weighted_precision = []
        eval_losses.append(result['val_loss'])
        weighted_f1.append(result['f1_weighted'] * 100)
        weighted_precision.append(result['precision_weighted'] * 100)
        weighted_recall.append(result['recall_weighted'] * 100)

    # 5-fold average
    avg_eval_loss_TEXT = f"Average Eval Loss: {np.mean(eval_losses)} (+/- {np.std(eval_losses)})"
    avg_weighted_f1_TEXT = f"Weighted Average F1: {np.mean(weighted_f1)} (+/- {np.std(weighted_f1)})"
    avg_weighted_recall_TEXT = f"Weighted Average Recall: {np.mean(weighted_recall)} (+/- {np.std(weighted_recall)})"
    avg_weighted_precision_TEXT = f"Weighted Average Precision: {np.mean(weighted_precision)} (+/- {np.std(weighted_precision)})"

    logging.info(avg_eval_loss_TEXT)
    logging.info(avg_weighted_f1_TEXT)
    logging.info(avg_weighted_recall_TEXT)
    logging.info(avg_weighted_precision_TEXT)

    # 保存模型未增强数据的最终平均评估分数
    with open(args.outfolder + f'{MODEL}_avg_metrics.txt', mode='a') as f:
        # f.write(f"TIME: {datetime.now()}\n")
        f.write(f"MODEL: {MODEL}\n")
        f.write(avg_eval_loss_TEXT + '\n')
        f.write(avg_weighted_f1_TEXT + '\n')
        f.write(avg_weighted_recall_TEXT + '\n')
        f.write(avg_weighted_precision_TEXT + '\n')
        f.write('\n')


if '__main__' == __name__:
    args = parse_args()
    main(args)
