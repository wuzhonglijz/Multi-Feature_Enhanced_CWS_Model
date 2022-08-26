# encoding: utf-8
"""
@author: Yuxian Meng 
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: sentence_pair_processor
@time: 2019/4/8 14:58

    这一行开始写关于本文件的说明与解释
"""


import os
import sys


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler


import csv
import json
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm


from glyce.dataset_readers.bert_data_utils import *


def read_json(file):
    data = []
    print("read json:")
    with open(file, 'r', encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            data.append(json.loads(line.strip()))
    return data


class DBQAProcessor(DataProcessor):
    """Processor for the dbqa data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class LCQMCProcessor(DataProcessor):
    """Processor for the dbqa data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1].replace(" ", "")
            text_b = line[2].replace(" ", "")
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class BQProcessor(DataProcessor):
    """Processor for the dbqa data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1].replace(" ", "")
            text_b = line[2].replace(" ", "")
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    # def _create_examples(self, lines, set_type):
    #     """Creates examples for the training and dev sets."""
    #     examples = []
    #     for (i, line) in enumerate(lines):
    #         line = json.loads(line[0])
    #         if i < 2:
    #            print("-"*10)
    #            print("check  loading example")
    #            print(line)
    #            print(type(line))
    #         guid = "%s-%s" % (set_type, i)
    #         text_a = line[1] # .replace(" ", "")
    #         text_b = line[2] # .replace(" ", "")
    #         label = line
    #         examples.append(
    #             InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    #     return examples



class XNLIProcessor(DataProcessor):
    """Processor for the xnli data set """

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
