# 1. 安装spacy: pip install spacy
# 2. 下载英语模型: python -m spacy download en_core_web_sm
import spacy
import os
import json
import collections
import string
from .WordTokenizer import WordTokenizer, ENGLISH_STOP_WORDS
from typing import Union, Tuple, List, Iterable, Dict


class SpacyTokenizer(WordTokenizer):
    def __init__(self, vocab: Iterable[str] = [], stop_words: Iterable[str] = ENGLISH_STOP_WORDS,
                 do_lower_case: bool = True):
        self.word2idx = None
        self.vocab = None
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def tokenize(self, text):
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        tokens_filtered = []

        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.strip(string.punctuation)
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])

        return tokens_filtered
        # return tokens

    def get_vocab(self):
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'SpacyTokenizer_config.json'), 'w') as fOut:
            json.dump({'vocab': list(self.word2idx.keys()), 'stop_words': list(self.stop_words),
                       'do_lower_case': self.do_lower_case}, fOut)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'SpacyTokenizer_config.json'), 'r') as fIn:
            config = json.load(fIn)

        return SpacyTokenizer(**config)

