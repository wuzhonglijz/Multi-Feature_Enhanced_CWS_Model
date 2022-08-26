#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
# Modifications Copyright 2022 Zhongli WU
# Author: Jason Wang
# Based on code by: Xiaoya Li(Glyce)


import os 
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import json 
import math 
import shutil 
import tarfile 
import logging 
import tempfile
import torch, gensim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from gensim.models import KeyedVectors

from glyce.layers.bert_basic_model import * 
from glyce.utils.tokenization import BertTokenizer  
from glyce.layers.char_glyph_embedding import CharGlyphEmbedding
from glyce.layers.char_stroke_embedding import CharStrokeEmbedding
from glyce.layers.char_graph_embedding import CharGraphEmbedding

class ComboPositionEmbedder(nn.Module):
    def __init__(self, configGlyph, configGraph, parentconfig):
        super(ComboPositionEmbedder, self).__init__()
        self.position_embeddings = nn.Embedding(configGraph.max_position_embeddings, configGlyph.output_size + configGraph.output_size)

        token_tool = BertTokenizer.from_pretrained(parentconfig.bert_model, do_lower_case=False)
        idx2tokens = token_tool.ids_to_tokens 
        self.idx2tokens = idx2tokens
        self.glyph_encoder = CharGlyphEmbedding(configGlyph, idx2tokens)
        self.graph_encoder = CharGraphEmbedding(configGraph, idx2tokens)    
        #print(idx2tokens.shape)
        self.stroke_encoder = CharStrokeEmbedding(idx2tokens)
        #self.linear = nn.Linear(768, 120)  #for stroke embedding 768 version
        
        self.layer_norm = BertLayerNorm(configGlyph.output_size + configGraph.output_size, eps=1e-12)
        self.dropout = nn.Dropout(configGraph.hidden_dropout_prob)


    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1) 
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        glyph_embeddings, glyph_cls_loss = self.glyph_encoder(input_ids)
        graph_embeddings = self.graph_encoder(input_ids)
        stroke_embeddings = self.stroke_encoder(input_ids)

        embeddings = position_embeddings + torch.cat((glyph_embeddings, graph_embeddings),dim=2)  + stroke_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, glyph_cls_loss

    def position(self,input_ids,token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return self.position_embeddings(position_ids)

    def comboForward(self,input_ids,token_type_ids=None):
        glyph_embeddings, glyph_cls_loss = self.glyph_encoder(input_ids)
        graph_embeddings = self.graph_encoder(input_ids)
        stroke_embeddings = self.stroke_encoder(input_ids)
        return torch.cat((glyph_embeddings, graph_embeddings), dim=2) + stroke_embeddings

if __name__ == "__main__":
    pass 
