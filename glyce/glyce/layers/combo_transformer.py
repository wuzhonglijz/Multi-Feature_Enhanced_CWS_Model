#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
# Author: Jason Wang
# Based on code by: Xiaoya Li (Glyce)

import os 
import sys 

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import json
import math
import copy
import logging
import tarfile
import tempfile
import shutil
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from glyce.layers.bert_basic_model import *
from glyce.layers.glyph_position_embed import GlyphPositionEmbedder
from glyce.layers.combo_position_embed import ComboPositionEmbedder

class ComboTransformer(nn.Module):
    def __init__(self, config, num_labels=4):
        super(ComboTransformer, self).__init__()
        self.num_labels = num_labels
        self.residual = config.residual
        self.epoch = 0
        self.training_strat = config.training_strat
        self.combo_embedder = ComboPositionEmbedder(config.glyph_config,config.graph_config,config)
        bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        self.bert_model = BertModel(bert_config)
        self.transformer_layer = BertEncoder(config.transformer_config)
        self.pooler = BertPooler(config)
        self.bert_model = self.bert_model.from_pretrained(config.glyph_config.bert_model)
        if config.bert_frozen == "true":
            print("!=!"*20)
            print("Please notice that the bert model if frozen")
            print("the loaded weights of models is ")
            print(config.glyph_config.bert_model)
            print("!-!"*20)
            for param in self.bert_model.parameters():
                param.requires_grad=False

    def updateEpoch(self):
        self.epoch+=1
        if self.training_strat=="bert-glyce-joint":
            if self.epoch==1:
                print("\nTraining Bert...")
                for param in self.bert_model.parameters():
                    param.requires_grad=True
                for param in self.combo_embedder.parameters():
                    param.requires_grad=False
            if self.epoch==6:
                print("\nTraining Combo...")
                for param in self.bert_model.parameters():
                    param.requires_grad=False
                for param in self.combo_embedder.parameters():
                    param.requires_grad=True
            if self.epoch==11:
                print("\nTraining Both...")
                for param in self.bert_model.parameters():
                    param.requires_grad=True
                for param in self.combo_embedder.parameters():
                    param.requires_grad=True

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,):
        combo_embed, glyph_cls_loss = self.combo_embedder(input_ids, token_type_ids=token_type_ids)
        sequence_output, pooled_output = self.bert_model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        context_bert_output = sequence_output[-1]
        input_features = torch.cat([combo_embed, context_bert_output], -1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * - 10000.0
        encoded_layers = self.transformer_layer(input_features, extended_attention_mask,
            output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]
        pooled_output2 = self.pooler(sequence_output)

        return encoded_layers, pooled_output2, glyph_cls_loss

    def bertForward(self, input_ids, token_type_ids=None, attention_mask=None,):
        sequence_output, pooled_output = self.bert_model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        context_bert_output = sequence_output[-1]
        return context_bert_output
    def bertComboForward(self, input_ids, token_type_ids=None, attention_mask=None,):
        combo_embed, glyph_cls_loss = self.combo_embedder(input_ids, token_type_ids=token_type_ids)
        sequence_output, pooled_output = self.bert_model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        context_bert_output = sequence_output[-1]
        input_features = torch.cat([combo_embed, context_bert_output], -1)
        return input_features