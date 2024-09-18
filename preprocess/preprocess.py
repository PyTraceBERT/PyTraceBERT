# import sys
# sys.path.append("../")
# sys.path.append("../../")
# from PyTracebert.utils import seed_everything
from ..util.utils import seed_everything

from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import pandas as pd
# import pprint
# import plogging.debug as pp
import re
import logging
import random

logging.basicConfig(format='%(message)s', level=logging.INFO)
seed_everything()


def count_pos_neg(df):
    pos = df[df['label'] == 1]
    neg = df[df['label'] == 0]
    print(f"正样本数量: {len(pos)}")
    print(f"负样本数量: {len(neg)}")


def concatenate_traceback(fine_tuning_file, outfile):
    """直接将文本处理成一行, 并添加[SEP]分隔符"""
    df = pd.read_csv(fine_tuning_file)
    logging.info(df.head())
    logging.info(f"lrngth: {len(df)}")
    templates = df['Templates'].tolist()
    modified_texts = []
    for temp in tqdm(templates):
        # Split the temp into lines and rejoin them with [SEP]
        lines = temp.split('\n')
        modified_text = '[SEP]'.join(lines)
        modified_texts.append(modified_text)

    df['Templates'] = modified_texts
    df.to_csv(outfile, index=False)


def _generate_train_test(file, test_ratio=0.2, outfolder=None):
    """分割训练集、验证集和测试集"""
    df = pd.read_csv(file)
    logging.info(df.head())
    X_train, X_temp, y_train, y_temp = train_test_split(df.index.values,
                                                        df.label.values,
                                                        test_size=test_ratio,
                                                        random_state=1234,
                                                        stratify=df.label.values)

    X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                    y_temp,
                                                    test_size=0.5,
                                                    random_state=1234,
                                                    stratify=y_temp)

    train_dataset = df.loc[X_train]
    val_dataset = df.loc[X_val]
    test_dataset = df.loc[X_test]

    logging.info(f"df: {len(df)}")
    logging.info(f"train_dataset: {len(train_dataset)}")
    logging.info(f"val_dataset: {len(val_dataset)}")
    logging.info(f"test_dataset: {len(test_dataset)}")
    train_dataset.to_csv(outfolder + 'task/templates_fine_tuning_train.csv', index=False)
    val_dataset.to_csv(outfolder + '/task/templates_fine_tuning_val.csv', index=False)
    test_dataset.to_csv(outfolder + '/task/templates_fine_tuning_test.csv', index=False)


def generate_train_test(train_dataset_path, fine_tuning_dataset_path, test_ratio=0.2, outfolder=None):
    _generate_train_test(fine_tuning_dataset_path, test_ratio=test_ratio, outfolder=outfolder)


def repeated_file(file1, file2, outfile):
    df = pd.read_csv(file1)
    df1 = pd.read_csv(file2)
    new_df = pd.DataFrame()
    new_df['Templates'] = df1.Templates
    new_df['label'] = df1.label
    new_df['type'] = df1.type
    # df1 = df[df['label']==1]
    logging.info(f"原始df文件长度: {len(df)}")
    logging.info(f"原始new_df文件长度: {len(new_df)}")
    df = pd.concat([df, new_df], ignore_index=True)
    logging.info(df.head())
    logging.info(f"重复文件长度: {len(df)}")
    df.to_csv(outfile, index=False)


def split_train_test(df, outfolder, test_ratio=0.2, mode='test'):
    logging.info(f"原始标签为1的总数量: {len(df[df['label'] == 1])}")
    logging.info(f"原始标签为0的总数量: {len(df[df['label'] == 0])}")

    X_train, X_test, y_train, y_test = train_test_split(
        df, df['label'], test_size=test_ratio, random_state=42, stratify=df['label'])

    train_data = X_train
    test_data = X_test

    logging.info(train_data.head())
    logging.info(f"train length: {len(train_data)}")
    logging.info(f"test length: {len(test_data)}")
    logging.info(f"train标签为1的数量: {len(train_data[train_data['label'] == 1])}")
    logging.info(f"train标签为0的总数量: {len(train_data[train_data['label'] == 0])}")

    logging.info(f"test标签为1的数量: {len(test_data[test_data['label'] == 1])}")
    logging.info(f"test标签为0的总数量: {len(test_data[test_data['label'] == 0])}")

    train_data.to_csv(outfolder + 'train_with_type.csv', index=False)
    test_data.to_csv(outfolder + mode + '_with_type.csv', index=False)


def extract_frames_or_exception(file, outfile):
    """
    提取并保存traceback中的frame和exception
    """
    df = pd.read_csv(file)

    # split by [SEP]
    df['Templates_list'] = df.Templates.apply(lambda x: x.split('[SEP]'))

    df['exception'] = df.Templates_list.apply(lambda x: x[-1])
    df['Templates_list'] = df.Templates_list.apply(lambda x: x[:-1])
    df['frames'] = df.Templates_list.apply(lambda x: '[SEP]'.join(x))
    df = df.drop(columns=['Templates_list'])

    df.to_csv(outfile, index=False)


def random_sample_n(pos_n, neg_n):
    """随机挑出pos_n个正例, neg_n个负例"""
    df = pd.read_csv('./dataset/Kfold/test_fold_4.csv')
    positive_examples = df[df['label'] == 1]
    negative_examples = df[df['label'] == 0]

    positive_sample = positive_examples.sample(n=pos_n)
    negative_sample = negative_examples.sample(n=neg_n)

    df_samples = pd.concat([positive_sample, negative_sample])
    df_samples.to_csv(f'./dataset/task/sample_20.csv', index=False)  # 类别不平衡的


def replace_path(file, outfile):
    df = pd.read_csv(file)
    df['Templates'] = df.Templates.apply(lambda x: 'File [FILE],' + ','.join(x.split(',')[1:]))
    df.to_csv(outfile, index=False)
    logging.info(f"length: {len(df)}")


def remove_other_signals(file):
    df = pd.read_csv(file)

    def process_text(text):
        frames = [x for x in text.split('[SEP]')]

        def replace_slashes(match):
            # logging.info(match)
            return re.sub(r'\\+', '/', match)

        frames = [re.sub(r'File "(.*)", line', lambda match: 'File ' + replace_slashes(match.group(1)) + ', line', text)
                  for text in frames]
        return '[SEP]'.join(frames)

    df['Templates'] = tqdm(df['Templates'].apply(process_text))
    df['Templates'] = tqdm(df['Templates'].apply(lambda x: re.sub(r'\s+', ' ', x)))

    # print(df['Templates'].iloc[0])
    output_file = file.split('.')[0]
    df.to_csv(output_file + f'_remove_other_signals.csv', index=False)


def create_negative_examples(file, outfile):
    df = pd.read_csv(file)
    frames = df['frames'].tolist()
    exceptions = df['exception'].tolist()

    new_df = pd.DataFrame(columns=['Templates', 'next_label'])
    df['next_label'] = 1

    # 生成负例
    indices = list(range(len(frames)))
    negative_examples = []
    # 对每一个frame，从exception中随机采样一个，作为负例
    for i in tqdm(range(len(frames))):
        available_indices = indices.copy()
        available_indices.remove(i)
        random_index = random.choice(available_indices)
        while exceptions[random_index] == exceptions[i]:
            random_index = random.choice(available_indices)
        negative_examples.append(frames[i] + '[SEP]' + exceptions[random_index])

    # 合并
    negative_examples = pd.Series(negative_examples)
    templates = df['Templates'].tolist()
    templates.extend(negative_examples)
    new_df['Templates'] = templates

    next_label = [1] * len(frames) + [0] * len(negative_examples)
    new_df['next_label'] = next_label
    new_df.to_csv(outfile, index=False)


def remove_exceed_frame_length(data_file, max_frames_len, tokenizer_path):
    df = pd.read_csv(data_file)
    exceed_indexs = []
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    for index, traceback in tqdm(df['Templates'].items(), total=len(df)):
        frames = traceback.split('[SEP]')
        last_frame = frames[-2]
        last_frames_tokens = tokenizer.tokenize(last_frame)
        if (len(last_frames_tokens) > max_frames_len):
            exceed_indexs.append(index)
            # logging.info(f"index: {index}")
            logging.info(f"exceed_frames_tracebacks count: {len(exceed_indexs)}")
            logging.info(f"last_frames_tokens count: {len(last_frames_tokens)}")

    df = df.drop(exceed_indexs)

    logging.info(f"exceed_frames_tracebacks count: {len(exceed_indexs)}")
    logging.info(f"length new_df: {len(df)}")
    output_file = data_file.split('.')[0] + '_processed.csv'
    df.to_csv(output_file, index=False)


def remove_exceed_frame_length1(data_file, max_frames_len, tokenizer_path):
    # 初始化 tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    # 读取 CSV 文件
    df = pd.read_csv(data_file)

    # 定义一个函数来检查帧长度
    def exceeds_max_frame_len(traceback):
        frames = traceback.split('[SEP]')
        last_frame = frames[-2] if len(frames) > 1 else ''
        last_frames_tokens = tokenizer.tokenize(last_frame)
        return len(last_frames_tokens) > max_frames_len

    tqdm.pandas(desc="Processing Tracebacks")
    exceed_mask = df['Templates'].progress_apply(exceeds_max_frame_len)
    exceed_indexs = df.index[exceed_mask].tolist()
    df = df.drop(exceed_indexs)

    # 打印日志
    logging.info(f"original tracebacks length: {len(df) + len(exceed_indexs)}")
    logging.info(f"exceed_frames_tracebacks count: {len(exceed_indexs)}")
    logging.info(f"length diff: {len(df)}")

    # 保存处理后的文件
    output_file = data_file.split('.')[0] + '_processed.csv'
    df.to_csv(output_file, index=False)


def test_remove_exceed_frames(s, max_frames_len, tokenizer_path):
    for traceback in s:
        frames = traceback.split('[SEP]')
        print(f"frames length: {len(frames)}")
        last_frame = frames[-2]
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        last_frames_tokens = tokenizer.tokenize(last_frame)
        if (len(last_frames_tokens) > max_frames_len):
            print(f"last frame length: {len(last_frames_tokens)}")
            # exceed_indexs.append(traceback.index)


if __name__ == '__main__':
    pass
