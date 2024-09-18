# import sys
# sys.path.append("../")
# sys.path.append("../../")
import re
import random
import logging
import pandas as pd
from util.utils import seed_everything

logging.basicConfig(format='%(message)s', level=logging.INFO)

seed_everything()
# random.seed(42)
def _random_inplace_number(text, multi_ratio):

    pattern = r'line \d+'
    texts = []
    for _ in range(multi_ratio):
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                # 提取行号中的数字部分
                # line_number = int(re.search(r'\d+', match).group())
                # 生成随机的替换数字
                new_line_number = random.randint(1, 500)
                # 将新的行号替换原来的行号
                text = text.replace(match+',', f"line {new_line_number},")
                # print("==="*20)
                # print(text)
        texts.append(text)

    return texts


def random_inplace_number(texts, labels, save_file):
    """增强正例数据"""
    augment_text = []  
    augment_label = []

    texts_positive = [text for text, label in zip(texts, labels) if label == 1]
    texts_neg = [text for text, label in zip(texts, labels) if label == 0]
    logging.info(f"length of texts: {len(texts)}")
    logging.info(f"length of label=1: {len(texts_positive)}")
    logging.info(f"length of label=0: {len(texts_neg)}")

    for text in texts_positive:
        replaced_texts = _random_inplace_number(text, multi_ratio=3)
        augment_text += replaced_texts
        augment_label += [1]* len(replaced_texts)
    
    logging.info(f"augment_text length: {len(augment_text)}")
    logging.info(f"augment_label length: {len(augment_label)}")
    
    assert len(augment_text) == len(augment_label)

    new_texts_positive = texts_positive+augment_text
    # texts_neg = [text for text, label in zip(texts, labels) if label == 0]
    logging.info(f"length of new_texts_positive: {len(new_texts_positive)}")
    # logging.info(f"length of texts_neg: {len(texts_neg)}")
    if len(new_texts_positive)<len(texts_neg):
        diff = len(texts_neg)-len(new_texts_positive)
        logging.info(f"diff: {diff}")
        samples = random.sample(new_texts_positive, diff)
        for text in samples:
            replaced_texts = _random_inplace_number(text, multi_ratio=1)
            augment_text += replaced_texts
            augment_label += [1]* len(replaced_texts)

    new_texts_positive = texts_positive+augment_text
    logging.info(f"length of new_texts_positive: {len(new_texts_positive)}")
    logging.info(f"augment_text length: {len(augment_text)}")
    logging.info(f"augment_label length: {len(augment_label)}")
    
    assert len(augment_text) == len(augment_label)

    new_texts = texts+augment_text
    new_labels = labels+augment_label

    logging.info(f"total texts length after data augmentation: {len(new_texts)}")
    logging.info(f"total labels length after data augmentation: {len(new_labels)}")
    assert len(new_texts) == len(new_labels)
    
    df = pd.DataFrame(list(zip(new_texts, new_labels)), columns=['Templates', 'label'])
    df = df.drop_duplicates(subset=['Templates'], keep='first')
    logging.info(f"length of df_augment: {len(df)}")

    return new_texts, new_labels

