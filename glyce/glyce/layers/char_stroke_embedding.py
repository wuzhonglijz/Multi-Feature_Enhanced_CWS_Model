#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
# Modifications Copyright 2022 Zhongli WU

from lib2to3.pytree import convert
import os 
import sys 
import json 
import random 


root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gensim

from glyce.utils.tokenization import BertTokenizer
from glyce.utils.random_erasing import RandomErasing
from glyce.layers.mask_cross_entropy import MaskCrossEntropy
from glyce.utils.components import SubCharComponent, Highway
from glyce.utils.render import multiple_glyph_embeddings, get_font_names
from glyce.layers.bert_layernorm import BertLayerNorm
    
class CharStrokeEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(CharStrokeEmbedding, self).__init__()
        #self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.tokenizer = BertTokenizer('/home/wzl/msc/StrokeOrderEmbeddings/data/chinese_L-12_H-768_A-12/vocab.txt')
        # self.model = gensim.models.KeyedVectors.load_word2vec_format(fname='/home/wzl/msc/StrokeOrderEmbeddings/data/embeddings-2818038-final-save.txt')
        # self.stroke_tokenizer = BertTokenizer('/home/wzl/msc/StrokeOrderEmbeddings/data/stroke_vocabs.txt')
        self.model = gensim.models.KeyedVectors.load_word2vec_format(fname='/home/wzl/msc/StrokeOrderEmbeddings/data/stroke.vec')
        self.stroke_tokenizer = BertTokenizer('/home/wzl/msc/StrokeOrderEmbeddings/data/stroke_vocab.txt')
  
        
        self.add_vocab = list(self.model.index_to_key)
        weights = torch.FloatTensor(self.model.vectors) 
        self.word_embeddings = nn.Embedding.from_pretrained(weights)
        self.LayerNorm = BertLayerNorm(120, eps=1e-12)
        self.dropout = nn.Dropout(0.2)

    def convert_ids(self, input_ids):
        converted_ids = []
        for sen in input_ids:
            #print(sen)
            tokens = self.tokenizer.convert_ids_to_tokens(sen.cpu().numpy())
            tmp_list = []
            for i in tokens:
                if i in self.add_vocab:
                    tmp_list.append(self.model.key_to_index[i])
                else:
                    tmp_list.append(self.model.key_to_index['一'])
            converted_ids.append(tmp_list)
        return converted_ids
    
    def forward(self, input_ids, token_type_ids=None):
        #print(input_ids)
        converted_ids = self.convert_ids(input_ids)
        #print(self.stroke_tokenizer.convert_ids_to_tokens(converted_ids))
        #print('input_ids:', input_ids)
        #print('converted_ids:', converted_ids)
        converted_ids = torch.IntTensor(converted_ids).to('cuda')
        embeddings = self.word_embeddings(converted_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings