#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# Author: Xiaoy LI 
# Last update: 2019.04.01 
# First create: 2019.04.01 
# Description:
# pos_dataset_processor.py


import os 
import sys 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3]) 
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import csv 
import logging 
import argparse 
import random 
import numpy as np 
from tqdm import tqdm 


from glyce.dataset_readers.bert_data_utils import * 




class Ctb5POSProcessor(DataProcessor):
    # process for the Ctb5 pos processor 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_test_examples(self, data_dir): 
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_labels(self):
        return ['O', 'B-FW', 'S-BA', 'B-PN', 'B-NR', 'B-M', 'M-NT', 'M-AD', 'E-P', 'M-CC', 'M-P', 'M-CD', 'S-CS', 'M-NN-SHORT', 'B-MSP', 'S-CC', 'E-SP', 'E-NN', 'B-ETC', 'S-PN', 'B-NT', 'E-FW', 'S-NT-SHORT', 'S-DER', 'B-PU', 'S-NT', 'B-AD', 'S-DT', 'E-VE', 'S-SP', 'E-IJ', 'M-CS', 'S-LB', 'B-NN', 'S-VA', 'S-ETC', 'E-JJ', 'B-P', 'M-FW', 'B-LC', 'S-MSP', 'S-AS', 'S-NN', 'E-ETC', 'B-CC', 'M-VA', 'E-ON', 'S-PU', 'E-DT', 'B-CS', 'S-IJ', 'E-PU', 'S-AD', 'S-M', 'E-LC', 'B-OD', 'S-LC', 'M-PN', 'E-NR', 'E-M', 'M-NR', 'E-VC', 'B-NN-SHORT', 'E-NT', 'E-CD', 'S-NR', 'S-VV', 'E-AD', 'B-JJ', 'B-DT', 'B-ON', 'M-DT', 'M-NN', 'S-SB', 'M-VV', 'S-DEG', 'S-ON', 'S-DEV', 'S-NR-SHORT', 'E-CC', 'M-M', 'E-NN-SHORT', 'B-VV', 'S-P', 'S-JJ', 'E-VA', 'M-JJ', 'E-VV', 'M-OD', 'B-VA', 'B-IJ', 'S-CD', 'E-CS', 'B-CD', 'B-VE', 'E-OD', 'S-OD', 'S-X', 'E-MSP', 'S-FW', 'E-PN', 'B-VC', 'M-PU', 'M-VC', 'S-VC', 'S-DEC', 'S-VE', 'B-SP']


    def _create_examples(self, lines, set_type):
        # create examples for the training and dev sets 
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue 

            text_a = line[0]
            text_b = None 

            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("ctb5pos", str(i)) 
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 



class Ctb6POSProcessor(DataProcessor):
    # process for the ctb6 pos processor 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_labels(self):
        # see base class 
        return ['O', 'E-NR', 'E-OD', 'B-SP', 'S-AS', 'M-P', 'M-JJ', 'E-IJ', 'S-OD', 'M-OD', 'M-VA', 'E-ETC', 'B-CC', 'S-MSP', 'B-LC', 'E-VA', 'E-VC', 'E-DT', 'M-VC', 'S-PN', 'E-MSP', 'M-PU', 'E-VE', 'B-DT', 'S-CC', 'S-DT', 'S-DER', 'B-AD', 'S-VV', 'S-NR', 'B-OD', 'S-VE', 'B-NN-SHORT', 'S-LB', 'S-CS', 'M-CC', 'E-PN', 'E-P', 'M-NN', 'S-DEC', 'E-PU', 'M-M', 'B-PU', 'M-PN', 'S-NN', 'B-M', 'M-DT', 'S-SB', 'B-CS', 'S-SP', 'M-CD', 'B-VE', 'S-ON', 'B-PN', 'B-P', 'S-VC', 'B-VA', 'S-FW', 'B-ON', 'S-NT-SHORT', 'E-NN-SHORT', 'M-VV', 'S-DEG', 'E-ON', 'S-NT', 'S-IJ', 'S-AD', 'M-FW', 'M-AD', 'B-CD', 'S-LC', 'E-CD', 'E-JJ', 'B-IJ', 'E-NN', 'E-SP', 'S-P', 'S-VA', 'S-ETC', 'B-VV', 'E-CS', 'S-CD', 'E-M', 'B-MSP', 'S-JJ', 'E-LC', 'S-PU', 'B-ETC', 'M-NT', 'E-CC', 'B-NN', 'S-BA', 'E-NT', 'E-AD', 'M-NR', 'B-NT', 'M-CS', 'B-JJ', 'S-M', 'S-X', 'S-DEV', 'S-NR-SHORT', 'B-NR', 'M-NN-SHORT', 'B-VC', 'E-FW', 'E-VV', 'B-FW']


    def _create_examples(self, lines, set_type):
        # create examples for the training and dev set 
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue 

            text_a = line[0]
            text_b = None 
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("ctb6pos", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Ctb9POSProcessor(DataProcessor):
    # process for the ctb6 pos processor
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_lsv(os.path.join(data_dir, "train.char.bmes")), "train")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_lsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_lsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_labels(self):
        # see base class
        #return ['O', 'E-NR', 'E-OD', 'B-SP', 'S-AS', 'M-P', 'M-JJ', 'E-IJ', 'S-OD', 'M-OD', 'M-VA', 'E-ETC', 'B-CC', 'S-MSP', 'B-LC', 'E-VA', 'E-VC', 'E-DT', 'M-VC', 'S-PN', 'E-MSP', 'M-PU', 'E-VE', 'B-DT', 'S-CC', 'S-DT', 'S-DER', 'B-AD', 'S-VV', 'S-NR', 'B-OD', 'S-VE', 'B-NN-SHORT', 'S-LB', 'S-CS', 'M-CC', 'E-PN', 'E-P', 'M-NN', 'S-DEC', 'E-PU', 'M-M', 'B-PU', 'M-PN', 'S-NN', 'B-M', 'M-DT', 'S-SB', 'B-CS', 'S-SP', 'M-CD', 'B-VE', 'S-ON', 'B-PN', 'B-P', 'S-VC', 'B-VA', 'S-FW', 'B-ON', 'S-NT-SHORT', 'E-NN-SHORT', 'M-VV', 'S-DEG', 'E-ON', 'S-NT', 'S-IJ', 'S-AD', 'M-FW', 'M-AD', 'B-CD', 'S-LC', 'E-CD', 'E-JJ', 'B-IJ', 'E-NN', 'E-SP', 'S-P', 'S-VA', 'S-ETC', 'B-VV', 'E-CS', 'S-CD', 'E-M', 'B-MSP', 'S-JJ', 'E-LC', 'S-PU', 'B-ETC', 'M-NT', 'E-CC', 'B-NN', 'S-BA', 'E-NT', 'E-AD', 'M-NR', 'B-NT', 'M-CS', 'B-JJ', 'S-M', 'S-X', 'S-DEV', 'S-NR-SHORT', 'B-NR', 'M-NN-SHORT', 'B-VC', 'E-FW', 'E-VV', 'B-FW']
        return ['O', 'B-', 'B-AD', 'B-AS', 'B-BA', 'B-CC', 'B-CD', 'B-CS', 'B-DEG', 'B-DT', 'B-EM', 'B-ETC', 'B-FW', 'B-IC', 'B-IJ', 'B-JJ', 'B-LC', 'B-M', 'B-MSP', 'B-NN', 'B-NN-SHORT', 'B-NOI', 'B-NR', 'B-NT', 'B-OD', 'B-ON', 'B-P', 'B-PN', 'B-PU', 'B-SB', 'B-SP', 'B-URL', 'B-VA', 'B-VC', 'B-VE', 'B-VV', 'B-VV-2', 'E-', 'E-AD', 'E-AS', 'E-BA', 'E-CC', 'E-CD', 'E-CS', 'E-DEG', 'E-DT', 'E-EM', 'E-ETC', 'E-FW', 'E-IC', 'E-IJ', 'E-JJ', 'E-LC', 'E-M', 'E-MSP', 'E-NN', 'E-NN-SHORT', 'E-NOI', 'E-NR', 'E-NT', 'E-OD', 'E-ON', 'E-P', 'E-PN', 'E-PU', 'E-SB', 'E-SP', 'E-URL', 'E-VA', 'E-VC', 'E-VE', 'E-VV', 'E-VV-2', 'M-AD', 'M-AS', 'M-BA', 'M-CC', 'M-CD', 'M-CS', 'M-DT', 'M-EM', 'M-ETC', 'M-FW', 'M-IC', 'M-IJ', 'M-JJ', 'M-LC', 'M-M', 'M-NN', 'M-NN-SHORT', 'M-NOI', 'M-NR', 'M-NT', 'M-OD', 'M-ON', 'M-P', 'M-PN', 'M-PU', 'M-SP', 'M-URL', 'M-VA', 'M-VC', 'M-VE', 'M-VV', 'S-AD', 'S-AS', 'S-AS-1', 'S-BA', 'S-CC', 'S-CD', 'S-CS', 'S-DEC', 'S-DEG', 'S-DER', 'S-DEV', 'S-DT', 'S-EM', 'S-ETC', 'S-FW', 'S-IC', 'S-IJ', 'S-JJ', 'S-LB', 'S-LC', 'S-M', 'S-MSP', 'S-MSP-2', 'S-NN', 'S-NOI', 'S-NR', 'S-NR-SHORT', 'S-NT', 'S-NT-SHORT', 'S-OD', 'S-ON', 'S-P', 'S-PN', 'S-PU', 'S-SB', 'S-SP', 'S-VA', 'S-VC', 'S-VE', 'S-VV', 'S-X']

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev set
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue

            text_a = line[0]
            text_b = None
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("ctb9pos", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Ud1POSProcessor(DataProcessor):
    # process for the ud1 pos processor 
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.char.bmes")), "train")


    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.char.bmes")), "dev")


    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.char.bmes")), "test")


    def get_labels(self):
        return ['O', 'B-PART', 'M-ADV', 'B-CONJ', 'E-SYM', 'S-PROPN', 'S-PUNCT', 'B-ADP', 'S-PART', 'B-PUNCT', 'B-PRON', 'E-PRON', 'B-NOUN', 'E-ADP', 'E-NOUN', 'M-SYM', 'S-ADV', 'B-AUX', 'E-VERB', 'M-NUM', 'M-VERB', 'S-ADP', 'E-AUX', 'B-X', 'E-ADV', 'E-PROPN', 'S-AUX', 'M-X', 'S-VERB', 'B-PROPN', 'M-DET', 'M-PUNCT', 'E-PUNCT', 'S-DET', 'B-SYM', 'M-ADJ', 'S-NOUN', 'S-NUM', 'B-NUM', 'E-DET', 'B-VERB', 'S-CONJ', 'M-NOUN', 'S-SYM', 'E-NUM', 'B-ADJ', 'M-PART', 'S-PRON', 'E-ADJ', 'E-X', 'M-ADP', 'E-PART', 'M-PROPN', 'M-CONJ', 'S-X', 'B-ADV', 'S-ADJ', 'E-CONJ', 'B-DET']

    def _create_examples(self, lines, set_type):
        # create examples for the training and dev set 
        examples = []
        for (i, line) in enumerate(lines):
            if line == "\n":
                continue 

            text_a = line[0]
            text_b = None 
            label = line[1]
            label = label.split(" ")
            guid = "{}_{}".format("ud1pos", str(i))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples 
