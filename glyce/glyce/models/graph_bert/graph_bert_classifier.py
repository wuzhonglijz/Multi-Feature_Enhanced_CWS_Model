#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
# Author: Jason Wang
# Based on code by: Yuxian Meng (Glyce)

import os 
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-4])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss
from glyce.layers.classifier import * 
from glyce.layers.bert_basic_model import *
from glyce.layers.graph_transformer import GraphTransformer

class GraphBertClassifier(nn.Module):
    def __init__(self, config, num_labels=2):
        super(GraphBertClassifier, self).__init__()
        self.num_labels = num_labels
        self.graph_transformer = GraphTransformer(config, num_labels=num_labels)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.classifier_sign == "single_linear":
            self.classifier = SingleLinearClassifier(config.hidden_size, self.num_labels)
        elif config.classifier_sign == "multi_nonlinear":
            self.classifier = MultiNonLinearClassifier(config.hidden_size, self.num_labels)
        else:
            raise ValueError 

    def updateEpoch(self):
        self.graph_transformer.updateEpoch()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output = self.graph_transformer(input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask)
        features_output = encoded_layers[-1]
        features_output = features_output[:, 0, :]
        features_output = self.dropout(features_output)
        logits = self.classifier(features_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
