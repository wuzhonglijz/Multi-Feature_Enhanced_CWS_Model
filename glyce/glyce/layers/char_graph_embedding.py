#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
# Author: Jason Wang
# Based on code by: Glyce team

import os 
import sys 
import json 
import random
import pickle
import numpy as np

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import SAGEConv, Node2Vec, GCNConv, GINConv, GINEConv, GATv2Conv, BatchNorm, LayerNorm, GraphNorm
from torch_geometric.nn import global_sort_pool, global_max_pool

from glyce.utils.random_erasing import RandomErasing
from glyce.layers.mask_cross_entropy import MaskCrossEntropy
from glyce.utils.components import SubCharComponent, Highway
from glyce.utils.render import multiple_glyph_embeddings, get_font_names

def getGraphEmbeddings(idx2word,graphPath,numFeat):
    myGraphs = []
    with open(graphPath,"rb") as f:
        graphs = pickle.load(f)
    for i,zi in idx2word.items():
        if zi in graphs:
            g = graphs[zi].to("cuda" if torch.cuda.is_available() else "cpu")
            g.x = g.x.float()
            myGraphs.append(g)
        else:
            myGraphs.append(Data(x=torch.tensor([[0 for _ in range(numFeat)]],dtype=torch.float),edge_index=torch.tensor([[],[]],dtype=torch.long)).to("cuda" if torch.cuda.is_available() else "cpu"))
    return myGraphs

def myIndexSelect(emb,indices):
    selectEmb = []
    for ind in indices:
        selectEmb.append(emb[ind])
    return selectEmb

class MyGCN(nn.Module):
    def __init__(self,layer,num_features,hidden,output_features,gcn_drop,k,pool,batch_norm):
        super(MyGCN, self).__init__()
        self.gcn_drop = gcn_drop
        self.k = k
        self.pool = pool
        self.batch_norm = batch_norm

        self.gconv1 = SAGEConv(num_features,hidden)
        self.gconv2 = SAGEConv(hidden, hidden)

        if batch_norm=="batch":
            self.norm1 = BatchNorm(hidden)
            self.norm2 = BatchNorm(hidden)
        elif batch_norm=="layer":
            self.norm1 = LayerNorm(hidden)
            self.norm2 = LayerNorm(hidden)
        elif batch_norm=="graph":
            self.norm1 = GraphNorm(hidden)
            self.norm2 = GraphNorm(hidden)

        self.conv1d = nn.Conv1d(hidden, 32, 5)
        self.linear1 = nn.Linear(32 * (self.k - 5 + 1), hidden)
        self.linear2 = nn.Linear(hidden, output_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gconv1(x, edge_index)
        if self.batch_norm=="batch" or self.batch_norm=="layer" or self.batch_norm=="graph":
            x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.gcn_drop, training=self.training)
        x = self.gconv2(x, edge_index)
        if self.batch_norm=="batch" or self.batch_norm=="layer" or self.batch_norm=="graph":
            x = self.norm2(x)
        x = F.relu(x)

        if self.pool=="sort":
            x = global_sort_pool(x, batch, self.k)
            x = x.view(len(x), self.k, -1).permute(0, 2, 1)
            x = F.relu(self.conv1d(x))
            x = x.view(len(x), -1)
            x = F.relu(self.linear1(x))
            x = F.dropout(x, p=self.gcn_drop, training=self.training)
            x = self.linear2(x)
        elif self.pool=="max":
            x = global_max_pool(x, batch)
            x = F.dropout(x, p=self.gcn_drop, training=self.training)
            x = self.linear2(x)
        return x

class CharGraphEmbedding(nn.Module):
    """输入token_id，输出其对应的char embedding　glyph embedding或者两者的结合. config中的参数定义如下：
    dropout: float, dropout rate
    idx2char: dict, 单词到对应token_id的映射
    char_embsize: int, char embedding size
    glyph_embsize: int, glyph embedding size
    pretrained_char_embedding: numpy.ndarray 预训练字向量
    font_channels: int, 塞到channel里的字体的数目，如果random_fonts > 0，则代表可供随机选择的总字体数
    random_fonts: int, 每个batch都random sample　n个不同字体塞到n个channel里
    font_name: str, 形如'CJK/NotoSansCJKsc-Regular.otf'的字体名称，当font_channels=1时有效
    font_size: int, 字体大小
    use_traditional: bool, 是否用繁体字代替简体字
    font_normalize: bool, 是否对字体输入的灰度值归一化，即减去均值，除以标准差
    subchar_type: str, 是否使用拼音('pinyin')或者五笔('wubi')
    subchar_embsize: int, 拼音或五笔的embedding_size
    random_erase: bool, 是否随机block掉字体灰度图中的一小块
    num_fonts_concat: int, 将一个字对应的n个字体分别过CNN后得到的特征向量concat在一起
    glyph_cnn_type: str, 用于抽取glyph信息的cnn模型的名称
    cnn_dropout: float, glyph cnn dropout rate
    use_batch_norm: bool, 是否使用batch normalization
    use_layer_norm: bool, 是否使用layer normalization
    use_highway: bool, 是否将concat之后的向量过highway
    fc_merge: bool, 是否将concat之后的向量过全连接
    output_size: bool, 输出向量的维度
    """

    def __init__(self, model_config, idx2char=None):
        super(CharGraphEmbedding, self).__init__()
        self.config = model_config
        if idx2char is not None:
            self.config.idx2char = idx2char
        self.drop = nn.Dropout(self.config.dropout)

        if self.config.char_embsize:
            self.char_embedding = nn.Embedding(len(self.config.idx2char), self.config.char_embsize) # (4452, 13, 13)

        self.graph_embeddings = getGraphEmbeddings(self.config.idx2char,self.config.graph_path,self.config.num_features)

        self.token_size = self.config.graph_embsize
        if self.config.subchar_type:
            self.token_size += self.config.char_embsize

        if not (self.config.use_highway or self.config.yuxian_merge or self.config.fc_merge):
            assert self.token_size == self.config.output_size, '没有用后处理，token_size {}应该等于output_size {}'.format(self.token_size, self.config.output_size)

        self.graph_model = MyGCN(layer=self.config.gcn_layer,num_features=self.config.num_features, hidden=self.config.gcn_hidden, output_features=self.config.graph_embsize, gcn_drop=self.config.gcn_dropout, k=self.config.k, pool=self.config.pool, batch_norm=self.config.batch_norm)
        if self.config.pretrained_graph=="yes":
            self.graph_model.load_state_dict(torch.load(self.config.graph_dict,map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")),strict=False)

        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.token_size)
        if self.config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.token_size)

        self.init_weights()  # 模型初始化时权值也要初始化

    def forward(self, data):  # 前向传播，输入输出加dropout，data:  (seq_len, batch)
        all_embeddings = []
        glyph_loss = []
        input_data = data.view(-1)

        if self.config.char_embsize:
            all_embeddings.append(self.drop(self.char_embedding(input_data)))  # emb: (-1, emb_size)

        graph_emb = myIndexSelect(self.graph_embeddings,input_data.detach().cpu().numpy())

        graph_emb = Batch.from_data_list(graph_emb)
        graph_feat = self.graph_model(graph_emb)

        all_embeddings.append(graph_feat)

        emb = torch.cat(all_embeddings, -1)  # seql, batch, feat*2

        out_shape = list(data.size())
        out_shape.append(self.config.output_size)
        return emb.view(*out_shape)

    def init_weights(self):
        if self.config.char_embsize:
            initrange = 0.1  # (-0.1, 0.1)的均匀分布，只对embedding和最后的线性层做初始化
            self.char_embedding.weight.data.uniform_(-initrange, initrange)
            if self.config.pretrained_char_embedding:
                self.char_embedding.weight = nn.Parameter(torch.FloatTensor(self.config.pretrained_char_embedding))