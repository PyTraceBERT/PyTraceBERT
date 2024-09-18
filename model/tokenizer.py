from transformers import BertTokenizer, AutoTokenizer
import logging

from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple
import numpy as np
import torch


logging.basicConfig(format='%(message)s', level=logging.INFO)


class WordEncodeTokenizer:
    def __init__(self, tokenizer_path, max_length=512, max_excep_len=0, max_frames_len=0):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': ["<*>", "[LOCAL]", "[FILE]", "[NUM]", "[FUNC]", "[CODE]", "[NONE]"]})

        self.max_length = max_length  
        self.max_excep_len = max_excep_len
        self.max_frames_len = max_frames_len

    def get_tokenizer(self):
        return self.tokenizer

    def process_traceback(self, traceback):
        traceback_tokens = self.tokenizer.tokenize(traceback)
        total_length = len(traceback_tokens)

        if total_length > self.max_length - 2:
            last_sep_index = len(traceback_tokens) - traceback_tokens[::-1].index("[SEP]") - 1
            frames = traceback_tokens[:last_sep_index + 1]
            exception = traceback_tokens[last_sep_index + 1:]

            assert len(frames) + len(exception) == len(traceback_tokens)

            # exception
            if len(exception) > self.max_excep_len:
                last_excep_index = max(self.max_excep_len, self.max_length - 2 - len(frames))
                exception = exception[:last_excep_index]
                total_length = len(exception) + len(frames)

            while total_length > self.max_length - 2:
                first_sep_index = frames.index("[SEP]")
                frames = frames[first_sep_index + 1:]
                total_length = len(exception) + len(frames)
            traceback_tokens = frames + exception

        return traceback_tokens

    def encode_tracebacks_with_segment(self,
                                       tracebacks: List[str],
                                       padding='longest',
                                       add_special_tokens=True,
                                       truncation=True,
                                       return_tensors='pt'
                                       ):
        input_ids = []
        attention_masks = []
        token_type_ids = []
        for traceback in tracebacks:
            traceback_tokens = self.process_traceback(traceback)
            ids = self.tokenizer.convert_tokens_to_ids(traceback_tokens)

            ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
            mask = [1] * len(ids)
            # create segment ids
            frame_segment_ids = [0] * len(ids)
            sep_positions = []
            for index, token in enumerate(ids):
                if token == 102:
                    sep_positions.append(index)

            if len(sep_positions) > 1: 
                excep_begin_pos = sep_positions[-2] + 1
                frame_segment_ids[excep_begin_pos:] = [1] * (len(ids) - excep_begin_pos)

            input_ids.append(ids)
            attention_masks.append(mask)
            token_type_ids.append(frame_segment_ids)

            logging.debug(f"input_ids: {ids}")
            logging.debug(f"attention_masks: {mask}")
            logging.debug(f"frame_segment_ids: {frame_segment_ids}")

        # padding
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = [ids + [self.tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
        padded_attention_masks = [mask + [0] * (max_length - len(mask)) for mask in attention_masks]
        padded_segment_ids = [seg + [0] * (max_length - len(seg)) for seg in token_type_ids]

        logging.debug(f"padded_input_ids: {padded_input_ids}")
        logging.debug(f"padded_attention_masks: {padded_attention_masks}")
        logging.debug(f"padded_segment_ids: {padded_segment_ids}")

        # 将列表转换为tensor
        input_ids = torch.tensor(padded_input_ids)
        attention_masks = torch.tensor(padded_attention_masks)
        token_type_ids = torch.tensor(padded_segment_ids)

        logging.debug(f"token_type_ids: {token_type_ids}")
        assert len(input_ids) == len(token_type_ids) == len(attention_masks)

        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_masks}

    def encode_tracebacks_without_segment(self,
                                          tracebacks: List[str],
                                          padding='longest',
                                          add_special_tokens=True,
                                          truncation=True,
                                          return_tensors='pt'
                                          ):
        input_ids = []
        attention_masks = []
        for traceback in tracebacks:
            logging.debug(f"traceback: {traceback}")
            traceback_tokens = self.process_traceback(traceback)
            if (len(traceback_tokens) == 0):
                continue
            ids = self.tokenizer.convert_tokens_to_ids(traceback_tokens)

            ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
            mask = [1] * len(ids)

            input_ids.append(ids)
            attention_masks.append(mask)

            logging.debug(f"input_ids: {ids}")
            logging.debug(f"attention_masks: {mask}")

        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = [ids + [self.tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
        padded_attention_masks = [mask + [0] * (max_length - len(mask)) for mask in attention_masks]

        logging.debug(f"padded_input_ids: {padded_input_ids}")
        logging.debug(f"padded_attention_masks: {padded_attention_masks}")

        input_ids = torch.tensor(padded_input_ids)
        attention_masks = torch.tensor(padded_attention_masks)
        token_type_ids = torch.zeros_like(input_ids)

        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_masks}
