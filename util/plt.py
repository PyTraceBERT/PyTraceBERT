import json
import matplotlib.pyplot as plt
import logging
import os
import pandas as pd
from tqdm import tqdm
from ordered_set import OrderedSet


logging.basicConfig(format='%(message)s', level=logging.INFO)


def find_ckpt(folder):
    check_folders = [name for name in os.listdir(folder)
                     if name.startswith('checkpoint')
                     and os.path.isdir(os.path.join(folder, name))]

    max_number = max(check_folders, key=lambda x: int(x.split('-')[1]))
    logging.info(f'max_number: {max_number}')
    target_folder = os.path.join(folder, max_number)
    target_file = os.path.join(target_folder, 'trainer_state.json')
    logging.info(f'target_file: {target_file}')
    return target_file


def plot_loss(log_history, epochs, loss_output_file):
    """绘制模型的loss随训练轮次的变化曲线。"""
    train_losses = []
    eval_losses = []
    for entry in log_history:
        if "eval_loss" in entry:
            eval_losses.append(entry["eval_loss"])
        elif "loss" in entry:
            train_losses.append(entry["loss"])

    logging.info(f'train_loss: {len(train_losses)}, eval_losses: {len(eval_losses)}')

    plt.figure()
    plt.plot(epochs, eval_losses, label='Eval_Loss', marker='o', linestyle='-', color='b')
    plt.plot(epochs, train_losses, label='Train_Loss', marker='o', linestyle='-')
    plt.title('Eval Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Eval Loss')
    plt.grid(True)
    plt.legend()

    plt.savefig(loss_output_file)


def plot_metrics(log_history, epochs, metrics_output_file):
    """绘制模型在验证集上的评估指标随训练轮次的变化曲线。"""
    precisions = [entry["eval_precision"] for entry in log_history if "eval_precision" in entry]
    recalls = [entry["eval_recall"] for entry in log_history if "eval_recall" in entry]
    f1_scores = [entry["eval_f1"] for entry in log_history if "eval_f1" in entry]
    recalls1 = [entry["eval_recall1"] for entry in log_history if "eval_recall1" in entry]

    logging.info(f'precision: {len(precisions)}, recalls: {len(recalls)}, f1_scores: {len(f1_scores)}')

    # 绘制图像
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precisions, label='Precision', marker='o', linestyle='-')
    plt.plot(epochs, recalls, label='Recall', marker='o', linestyle='-')
    plt.plot(epochs, f1_scores, label='F1 Score', marker='o', linestyle='-')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics Over Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig(metrics_output_file)
    # plt.show()


def plot_evaluation_results(model, epoch, lr, dropout, input_dir, output_dir, input_file=None, fold=None, metric='f1'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if input_file is None:
        file = find_ckpt(input_dir)
    else:
        file = input_file

    logging.info(f"Reading file: {file}")

    loss_output_file = f'{output_dir}/eval_loss_over_epochs_{dropout}_{epoch}_{lr}_{metric}_{fold}.png'
    metrics_output_file = f'{output_dir}/metrics_over_epochs_{dropout}_{epoch}_{lr}_{metric}_{fold}.png'

    with open(file, 'r') as file:
        log_history = json.load(file)['log_history']

    epoch_set = OrderedSet()
    for entry in log_history:
        epoch_set.add(entry["epoch"])
    epochs = list(epoch_set)

    plot_loss(log_history, epochs, loss_output_file)
    if fold != 'pretrain':
        plot_metrics(log_history, epochs, metrics_output_file)


def plot_cumulate_numbers(token_lengths, text_counts, save_file):
    """绘制累计数量曲线图"""
    cumulative_counts = []
    total_count = 0
    for count in tqdm(text_counts):
        total_count += count
        cumulative_counts.append(total_count)
    # 绘制累计曲线图
    plt.plot(token_lengths, cumulative_counts)
    plt.xlabel('Token Length')
    plt.ylabel('Cumulative Traceback Count')
    plt.title('Cumulative Traceback Count by Token Length')
    plt.grid(True)

    plt.savefig(save_file)
    # plt.show()
