# -*- coding: utf-8 -*-
# Author: Jason Wang
# Based on code by: Glyce team


import os
import sys
root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
if root_path not in sys.path:
    sys.path.insert(0, root_path)
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch_geometric
from torch_geometric.data import Data, DataLoader, Batch
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.text as mpl_text
import pandas as pd
import random
import time
from tqdm import tqdm
import argparse
import logging
import pickle
import scipy
import math
import csv
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from captum.attr import Saliency, IntegratedGradients

from glyce.models.glyce_bert.glyce_bert_classifier import GlyceBertClassifier
from glyce.models.graph_bert.graph_bert_classifier import GraphBertClassifier
from glyce.models.combo_bert.combo_bert_classifier import ComboBertClassifier
from glyce.models.glyce_bert.glyce_bert_tagger import GlyceBertTagger
from glyce.models.graph_bert.graph_bert_tagger import GraphBertTagger
from glyce.models.combo_bert.combo_bert_tagger import ComboBertTagger

from glyce.utils.tokenization import BertTokenizer
from glyce.utils.optimization import BertAdam, warmup_linear
from glyce.dataset_readers.bert_config import Config
from glyce.dataset_readers.bert_data_utils import convert_examples_to_features
from glyce.dataset_readers.bert_data_utils import *
from glyce.dataset_readers.bert_ner import *
from glyce.dataset_readers.bert_pos import *
from glyce.dataset_readers.bert_cws import *
from glyce.dataset_readers.bert_sent_pair import *
from glyce.dataset_readers.bert_single_sent import *


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",type=str,default=None)
    parser.add_argument("--config_path", default="/home/lixiaoya/dataset/", type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--max_seq_length", default=128,
                        type=int, help="the maximum total input sequence length after ")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true",
                        help="set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bert_frozen", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3306)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_name", type=str, default="model.bin")
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--classifier_sign", type=str, default="multi_nonlinear")
    parser.add_argument("--residual", type=str, default="no")
    parser.add_argument("--hidden_size",type=int,default=None)
    parser.add_argument("--font_channels", type=int, default=None)
    parser.add_argument("--font_name", type=str, default=None)
    parser.add_argument("--font_names", type=str, default=[], nargs='+', action="append")
    parser.add_argument("--font_size", type=int, default=None)
    parser.add_argument("--num_fonts_concat", type=int, default=None)
    parser.add_argument("--glyph_embsize", type=int, default=None)
    parser.add_argument("--gcn_hidden",type=int,default=None)
    parser.add_argument("--k",type=int,default=None)
    parser.add_argument("--gcn_layer",type=str,default=None)
    parser.add_argument("--graph_embsize",type=int,default=None)
    parser.add_argument("--output_size",type=int,default=None)
    parser.add_argument("--pooler_fc_size",type=int,default=None)
    parser.add_argument("--transformer_hidden_size",type=int,default=None)
    parser.add_argument("--num_features",type=int,default=None)
    parser.add_argument("--graph_path",type=str,default=None)
    parser.add_argument("--pool",type=str,default="sort")
    parser.add_argument("--training_strat",type=str,default=None)
    parser.add_argument("--transformer",type=str,default="yes")
    parser.add_argument("--graph_dict",type=str,default=None)
    parser.add_argument("--pretrained_graph",type=str,default="no")
    parser.add_argument("--batch_norm",type=str,default="no")
    args = parser.parse_args()
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
    return args

def merge_config(args_config):
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config

def load_data(config):
    if config.data_sign == "msra_ner":
        data_processor = MsraNERProcessor()
    elif config.data_sign == "resume_ner":
        data_processor = ResumeNERProcessor()
    elif config.data_sign == "ontonotes_ner":
        data_processor = OntoNotesNERProcessor()
    elif config.data_sign == "weibo_ner":
        data_processor = WeiboNERProcessor()
    elif config.data_sign == "ctb5_pos":
        data_processor = Ctb5POSProcessor()
    elif config.data_sign == "ctb6_pos":
        data_processor = Ctb6POSProcessor()
    elif config.data_sign == "ctb9_pos":
        data_processor = Ctb9POSProcessor()
    elif config.data_sign == "ud1_pos":
        data_processor = Ud1POSProcessor()
    elif config.data_sign == "ctb6_cws":
        data_processor = Ctb6CWSProcessor()
    elif config.data_sign == "pku_cws":
        data_processor = PkuCWSProcessor()
    elif config.data_sign == "msr_cws":
        data_processor = MsrCWSProcessor()
    elif config.data_sign == "cityu_cws":
        data_processor = CityuCWSProcessor()
    elif config.data_sign == "as_cws":
        data_processor = AsCWSProcessor()
    elif config.data_sign == "weibo_cws":
        data_processor = WeiboCWSProcessor()
    elif config.data_sign == "nlpcc-dbqa":
        data_processor = DBQAProcessor()
    elif config.data_sign == "bq":
        data_processor = BQProcessor()
    elif config.data_sign == "xnli":
        data_processor = XNLIProcessor()
    elif config.data_sign == "lcqmc":
        data_processor = LCQMCProcessor()
    elif config.data_sign == "fudan":
        data_processor = FuDanProcessor()
    elif config.data_sign == "chinanews":
        data_processor = ChinaNewsProcessor()
    elif config.data_sign == "ifeng":
        data_processor = ifengProcessor()
    elif config.data_sign == "chn_senti_corp":
        data_processor = ChnSentiCorpProcessor()
    else:
        raise ValueError
    label_list = data_processor.get_labels()
    return label_list

def load_model(config, label_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    model = None
    if config.model_type=="glyce_clf":
        model = GlyceBertClassifier(config,num_labels=len(label_list))
    elif config.model_type=="glyce_tag":
        model = GlyceBertTagger(config,num_labels=len(label_list))
    elif config.model_type=="graph_clf":
        model = GraphBertClassifier(config,num_labels=len(label_list))
    elif config.model_type=="graph_tag":
        model = GraphBertTagger(config,num_labels=len(label_list))
    elif config.model_type=="combo_clf":
        model = ComboBertClassifier(config,num_labels=len(label_list))
    elif config.model_type=="combo_tag":
        model = ComboBertTagger(config,num_labels=len(label_list))
    model.load_state_dict(torch.load(config.output_name,map_location=torch.device('cpu')),strict=False)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    return model

def getEmbedding(part,character,config,model,tokenizer,label_list):
    features = convert_examples_to_features(character, label_list, config.max_seq_length, tokenizer, task_sign=config.task_name)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    embedding = None
    if config.model_type=="glyce_clf" or config.model_type=="glyce_tag":
        if part=="transformer":
            embedding = model.glyph_transformer(input_ids)
            return embedding[0][-1].detach().numpy()
        elif part=="pos+char":
            embedding = model.glyph_transformer.glyph_embedder(input_ids)
            return embedding[0].detach().numpy()
        elif part=="char":
            embedding = model.glyph_transformer.glyph_embedder.glyph_encoder(input_ids)
            return embedding[0].detach().numpy()
        elif part=="pos":
            embedding = model.glyph_transformer.glyph_embedder.position(input_ids)
            return embedding.detach().numpy()
        elif part=="bert":
            embedding = model.glyph_transformer.bertForward(input_ids)
            return embedding.detach().numpy()
        elif part=="bert+pos+char":
            embedding = model.glyph_transformer.bertGlyphForward(input_ids)
            return embedding.detach().numpy()
    if config.model_type=="graph_clf" or config.model_type=="graph_tag":
        if part=="transformer":
            embedding = model.graph_transformer(input_ids)
            return embedding[0][-1].detach().numpy()
        elif part=="pos+char":
            embedding = model.graph_transformer.graph_embedder(input_ids)
            return embedding.detach().numpy()
        elif part=="char":
            embedding = model.graph_transformer.graph_embedder.graph_encoder(input_ids)
            return embedding.detach().numpy()
        elif part=="pos":
            embedding = model.graph_transformer.graph_embedder.position(input_ids)
            return embedding.detach().numpy()
        elif part=="bert":
            embedding = model.graph_transformer.bertForward(input_ids)
            return embedding.detach().numpy()
        elif part=="bert+pos+char":
            embedding = model.graph_transformer.bertGraphForward(input_ids)
            return embedding.detach().numpy()
    if config.model_type=="combo_clf" or config.model_type=="combo_tag":
        if part=="transformer":
            embedding = model.combo_transformer(input_ids)
            return embedding[0][-1].detach().numpy()
        elif part=="pos+char":
            embedding = model.combo_transformer.combo_embedder(input_ids)
            return embedding[0].detach().numpy()
        elif part=="glyce":
            embedding = model.combo_transformer.combo_embedder.glyph_encoder(input_ids)
            return embedding[0].detach().numpy()
        elif part=="graph":
            embedding = model.combo_transformer.combo_embedder.graph_encoder(input_ids)
            return embedding.detach().numpy()
        elif part=="char":
            embedding = model.combo_transformer.combo_embedder.comboForward(input_ids)
            return embedding.detach().numpy()
        elif part=="pos":
            embedding = model.combo_transformer.combo_embedder.position(input_ids)
            return embedding.detach().numpy()
        elif part=="bert":
            embedding = model.combo_transformer.bertForward(input_ids)
            return embedding.detach().numpy()
        elif part=="bert+pos+char":
            embedding = model.combo_transformer.bertComboForward(input_ids)
            return embedding.detach().numpy()

def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def compare(mode,x,y,a,b,config,model,tokenizer,label_list):
    sentA = InputExample(1,a,label="")
    sentB = InputExample(2,b,label="")
    embeddings = getEmbedding(mode,[sentA,sentB],config,model,tokenizer,label_list)
    posA = x
    posB = y
    print("Cos_Sim(%s,%s,%s):" % (sentA.text_a[posA-1] if 0<=posA-1<len(sentA.text_a) else ("[CLS]" if posA==0 else "[SEP]"),sentB.text_a[posB-1] if 0<=posB-1<len(sentB.text_a) else ("[CLS]" if posB==0 else "[SEP]"),mode),
          cosine(embeddings[0][posA],embeddings[1][posB]))

def diffPair(mode,x1,y1,x2,y2,a,b,config,model,tokenizer,label_list):
    sentA = InputExample(1,a,label="")
    sentB = InputExample(2,b,label="")
    embeddings = getEmbedding(mode,[sentA,sentB],config,model,tokenizer,label_list)
    print("Cos_Sim(%s-%s,%s-%s,%s):" % (sentA.text_a[x1-1] if 0<=x1-1<len(sentA.text_a) else ("[CLS]" if x1==0 else "[SEP]"),sentB.text_a[y1-1] if 0<=y1-1<len(sentB.text_a) else ("[CLS]" if y1==0 else "[SEP]"),sentA.text_a[x2-1] if 0<=x2-1<len(sentA.text_a) else ("[CLS]" if x2==0 else "[SEP]"),sentB.text_a[y2-1] if 0<=y2-1<len(sentB.text_a) else ("[CLS]" if y2==0 else "[SEP]"),mode),
          cosine(embeddings[0][x1]-embeddings[1][y1],embeddings[0][x2]-embeddings[1][y2])
          )
def wordDiff(mode,characters,config,model,tokenizer,label_list):
    embeddings = getEmbedding(mode,[InputExample(1,characters,label="")],config,model,tokenizer,label_list)
    print("Cos_Sim(%s-%s,%s-%s,%s): %s" % (characters[0],characters[1],characters[2],characters[3],mode,cosine(embeddings[0][1]-embeddings[0][2],embeddings[0][3]-embeddings[0][4])))
def wordSubtract(mode,characters,config,model,tokenizer,label_list):
    emb1 = getEmbedding(mode,[InputExample(1,characters[0],label="")],config,model,tokenizer,label_list)[0][1]
    emb2 = getEmbedding(mode,[InputExample(1,characters[1],label="")],config,model,tokenizer,label_list)[0][1]
    emb3 = getEmbedding(mode,[InputExample(1,characters[2],label="")],config,model,tokenizer,label_list)[0][1]
    emb4 = getEmbedding(mode,[InputExample(1,characters[3],label="")],config,model,tokenizer,label_list)[0][1]
    print("Cos_Sim(%s-%s,%s-%s,%s): %s" % (characters[0],characters[1],characters[2],characters[3],mode,cosine(emb1-emb2,emb3-emb4)))

class TextObject(object):
    def __init__(self, text, color):
        self.my_text = text
        self.my_color = color
class TextHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        fontP = font_manager.FontProperties()
        fontP.set_family('SimHei')
        patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline',
                              horizontalalignment=u'left', multialignment=None,
                              fontproperties=fontP, linespacing=None,
                              rotation_mode=None)
        handlebox.add_artist(patch)
        return patch
def tsnePlot(mode,sentences,config,model,tokenizer,label_list,colors=None,markersList=None,title=None,legendLabels=None):
    if colors is None:
        colors = ["tab:blue","tab:green","tab:red","tab:orange","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
    sent = [InputExample(i,x,label="") for i,x in enumerate(sentences)]
    embeddings = getEmbedding(mode,sent,config,model,tokenizer,label_list)
    pca = PCA(n_components=24)
    embeddingsPCA = pca.fit_transform(embeddings.reshape(-1,embeddings.shape[-1]))
    tsne = TSNE()
    embeddingsTSNE = tsne.fit_transform(embeddingsPCA).reshape(embeddings.shape[0],embeddings.shape[1],-1)
    plt.rcParams.update({'mathtext.fontset': "custom",'mathtext.sf': "SimHei", 'mathtext.rm': "SimHei", 'mathtext.cal': "SimHei", 'mathtext.tt': "SimHei", 'mathtext.it': "SimHei:italic", 'mathtext.bf': "SimHei:bold"})
    for i,s in enumerate(sent):
        if markersList is None:
            markers = [r"$CLS$"]+[r"$ "+ch+" $" for ch in s.text_a]+[r"$SEP$"] + [r"$PAD$" for _ in range(150-len(s.text_a)-2)]
        else:
            markers = markersList[i]+[r"$PAD$" for _ in range(150-len(markersList[i]))]
        for j in range(embeddingsTSNE.shape[1]):
            plt.scatter(embeddingsTSNE[i][j,0],embeddingsTSNE[i][j,1],color=colors[i],marker=markers[j],s=150)

    if title is not None:
        plt.title(title)
    textObjs = [TextObject(legendLabels[0][i],colors[i]) for i in range(len(legendLabels[0]))]
    if legendLabels is not None:
        plt.legend(textObjs, legendLabels[1],handler_map={textObj:TextHandler() for textObj in textObjs},handletextpad=0)
    plt.show()

def saliencyMapGlyph(pos,character,config,model,tokenizer,label_list):
    character = [InputExample(1,character,label="")]
    features = convert_examples_to_features(character, label_list, config.max_seq_length, tokenizer, task_sign=config.task_name)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    #Char_Glyph_Embedding
    input_data = input_ids.view(-1)
    all_embeddings = []
    glyphs = []
    for glyph_embedding in model.glyph_transformer.glyph_embedder.glyph_encoder.glyph_embeddings:
        glyph_emb = glyph_embedding.index_select(0, input_data)
        glyph_emb.requires_grad_() #glyph_embed grad
        glyphs.append(glyph_emb)
        glyph_feat = model.glyph_transformer.glyph_embedder.glyph_encoder.glyph_cnn_model(glyph_emb)
        all_embeddings.append(glyph_feat)
    emb = torch.cat(all_embeddings, -1)
    out_shape = list(input_ids.size())
    out_shape.append(model.glyph_transformer.glyph_embedder.glyph_encoder.config.output_size)
    #Glyph_Position_Embedding
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    position_embeddings = model.glyph_transformer.glyph_embedder.position_embeddings(position_ids)
    glyph_embed = position_embeddings  + emb.view(*out_shape)
    #Glyce_Transformer
    sequence_output, pooled_output = model.glyph_transformer.bert_model(input_ids, None, None, output_all_encoded_layers=True)
    context_bert_output = sequence_output[-1]
    input_features = torch.cat([glyph_embed, context_bert_output], -1)
    attention_mask = torch.ones_like(input_ids)
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=next(model.glyph_transformer.parameters()).dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * - 10000.0
    encoded_layers = model.glyph_transformer.transformer_layer(input_features, extended_attention_mask,output_all_encoded_layers=True)
    #Glyce_BERT_Tagger
    features_output = encoded_layers[-1]
    features_output = model.dropout(features_output)
    logits = model.classifier(features_output).squeeze()
    score, indices = torch.max(logits, 1)
    score[pos].backward()
    slc, _ = torch.max(torch.abs(glyphs[0].grad[pos]), dim=0)
    slc = (slc - slc.min())/(slc.max()-slc.min())
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(slc,cmap="hot")
    ax1.axis("off")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(glyphs[0][pos].detach().numpy().squeeze(),cmap="hot")
    ax2.axis("off")
    plt.show()

def myIndexSelect(emb,indices):
    selectEmb = []
    for ind in indices:
        selectEmb.append(emb[ind])
    return selectEmb

def saliencyMapGraph(pos,character,config,model,tokenizer,label_list):
    character = [InputExample(1,character,label="")]
    features = convert_examples_to_features(character, label_list, config.max_seq_length, tokenizer, task_sign=config.task_name)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    #Char_Graph_Embedding
    input_data = input_ids.view(-1)
    all_embeddings = []
    graph_emb = myIndexSelect(model.graph_transformer.graph_embedder.graph_encoder.graph_embeddings,input_data.detach().cpu().numpy())
    graph_emb[pos].x.requires_grad_() #graph_embed grad
    graph_emb2 = Batch.from_data_list(graph_emb)
    graph_feat = model.graph_transformer.graph_embedder.graph_encoder.graph_model(graph_emb2)
    all_embeddings.append(graph_feat)
    emb = torch.cat(all_embeddings, -1)
    out_shape = list(input_ids.size())
    out_shape.append(model.graph_transformer.graph_embedder.graph_encoder.config.output_size)
    #Graph_Position_Embedding
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    position_embeddings = model.graph_transformer.graph_embedder.position_embeddings(position_ids)
    graph_embed = position_embeddings + emb.view(*out_shape)
    #Graph_Transformer
    sequence_output, pooled_output = model.graph_transformer.bert_model(input_ids, None, None, output_all_encoded_layers=True)
    context_bert_output = sequence_output[-1]
    input_features = torch.cat([graph_embed, context_bert_output], -1)
    attention_mask = torch.ones_like(input_ids)
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=next(model.graph_transformer.parameters()).dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * - 10000.0
    encoded_layers = model.graph_transformer.transformer_layer(input_features, extended_attention_mask,output_all_encoded_layers=True)
    #Graph_BERT_Tagger
    features_output = encoded_layers[-1]
    features_output = model.dropout(features_output)
    logits = model.classifier(features_output).squeeze()
    score, indices = torch.max(logits, 1)
    score[pos].backward()
    slc, _ = torch.max(torch.abs(graph_emb[pos].x.grad), dim=1)
    slc = (slc - slc.min())/(slc.max()-slc.min())
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_aspect('equal')
    sGraph = Data(x=slc,edge_index=graph_emb[pos].edge_index)
    sGraphx = to_networkx(sGraph)
    gpos = {i:(graph_emb[pos].x[i,0].detach().numpy().item(),graph_emb[pos].x[i,1].detach().numpy().item()) for i in range(graph_emb[pos].x.shape[0])}
    nx.draw(sGraphx,gpos,ax=ax1,node_size=50,arrows=False,cmap="hot",node_color=sGraph.x)
    ax1.collections[0].set_edgecolor("#000000")
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_aspect('equal')
    graph = graph_emb[pos]
    graphx = to_networkx(graph)
    nx.draw(graphx,gpos,ax=ax2,node_size=50,arrows=False)
    ax2.collections[0].set_edgecolor("#000000")
    plt.show()

def main():
    args_config = args_parser()
    config = merge_config(args_config)
    label_list = load_data(config)
    model = load_model(config, label_list)
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    #T-SNE Plots
    wood = "札朴机朽权杆杈杉材村杓杖杜杞杠杨极杭杯杵杷杼松板构枇枉枋析枕林枚枝枢枪枫枯枰枳枷枸柄柏柑柘柚柜柞柠柢柩柯柱柳柵柿栀栅标栈栉栋栎栏树栓栖校栩株样核根格桁桂桃桅框桉桎桐桓桔桠桡桢档桥桦桧桩桶桿梅梆梏梓梗梢梧梭梯械梳检" \
           "棂棉棋棍棒棕棚棟棣棧棱棲棵棹棺椁椅椋植椎椒椪椭椰椹椽椿楂楊楓楔楝楞楠楣楨楫楮極楷楸楹楼概榄榆榈榉榔榕榛榜榨榫榭榱榴榷榻槁構槌槍槎槐槓槛槟槤槭槲槽槿樁樑樓標樞樟模樣横樯樱樵樸樹樺樽樾橄橇橋橘橙機橡橢橫橱橹橼檀檄檎" \
           "檐檔檜檢檬檯檳檸檻櫃櫚櫛櫥櫸櫻欄權欖"[:148]
    water = "氾汀汁汇汉汐汕汗汛汝江池污汤汨汩汪汰汲汴汶汹決汽汾沁沂沃沅沈沉沌沏沐沒沖沙沛沟没沣沥沦沧沪沫沭沮沱河沸油治沼沽沾沿況泄泊泌泓法泗泛泞泠泡波泣泥注泪泫泮泯泱泳泷泸泻泼泽泾洁洄洋洒洗洙洛洞津洩洪洮洱洲洵洶洹活洼洽派" \
            "流浃浅浇浊测济浏浑浒浓浔浙浚浜浣浦浩浪浮浯浴海浸涂涅涇消涉涌涎涓涔涕涛涝涞涟涠涡涣涤润涧涨涩涪涮涯液涵涸涼涿淀淄淅淆淇淋淌淑淒淖淘淙淚淞淡淤淦淨淩淪淫淬淮深淳淵混淹淺添清渊渍渎渐渔渗渚渙減渝渠渡渣渤渥渦温測渭港" \
            "渲渴游渺渾湃湄湊湍湖湘湛湟湧湫湮湯湾湿溃溅溉滋滞溏源準溜溝溟溢溥溧溪溫溯溱溴溶溺滁滂滄滅滇滑滓滔滕滟满滢滤滥滦滨滩漓滌滚滬滯滲滴滷滸滾滿漁漂漆漉漏演漕漠漢漣漩漪漫漬漯漱漲漳漸漾漿潆潇潋潍潑潔潘潛潜潞潟潢潤潦潭潮潰" \
            "潴潸潺潼澄澆瀕瀘瀚瀛瀝瀟瀧瀨瀰瀾灌灏灑灘灞灣"[:148]
    fire = "灯灶灼灿炀炉炊炒炔炕炖炜炫炬炮炯炳炷炸炼炽烁烂烃烊烘烙烛烟烤烦烧烨烩烬烯烷烽焊焕焖焗焙焯焰煅煉煊煌煙煜煤煥煨煩煸煽熄熔熠熵熾燃燈燉燎燒燔燜燥燦燧燭燴燻燼爆爍爐爛"
    person = "亿什仁仃仅仆仇仍仔仕他仗付仙仞仟代仨仪们仰仲件价任份仿伉伊伍伎伏伐休伕优伙伟传伢伤伦伪伫佤伯估伴伶伸伺似伽佃但佇佈位低住佐佑体佔何佗佚佛作佝佞佟你佣佩佬佯佰佳併佶佻佼使侃侄侈例侍侏侑侗供依侠侣侥侦侧侨侬侮侯侵侶侷便係促俄俊俏俐俑俗俘俚保俟俠信俨俩俪俭修俯" \
             "俳俸俺俾倆個倌倍倏們倒倔倖倘候倚倜借倡倦倩倪倫倬倭债值倾偃假偈偉偌偎偏偕做停健側偵偶俱 偷偹偻偼傁傉傍傔傕傞傟傡傤傥储傩傮傯傱傳傷傺僁僋働僑僒僖僡僣僩僪僭僳僲僶僼儂億儆儉儋儒儕儘償儡優儲儷儼"[:148]
    place = "队阡阪阮阱防阳阴阵阶阻阿陀陂附际陆陇陈陋陌降限陕陛陝陞陟陡院陣除陨险陪陰陲陳陵陶陷陸陽隅隆隈隊隋隍階随隐隔隕隘隙際障隧隨險隱隴"
    tsnePlot("char",[water,wood,fire,person,place],config,model,tokenizer,label_list,title="Combo Embeddings t-SNE Plot",legendLabels=[["氵","木","火","亻","阝"],["Water Radical","Wood Radical","Fire Radical","Person Radical","Place Radical"]])

    #Saliency Maps
    saliencyChars = "杨雷笛赶"
    for i in range(1,len(saliencyChars)+1):
        saliencyMapGraph(i,saliencyChars,config,model,tokenizer,label_list)

    #Word Subtraction Analogies
    wordSubtract("char","阴月阳日",config,model,tokenizer,label_list)
    wordSubtract("char","阴月阳房",config,model,tokenizer,label_list)

    wordSubtract("char","伙火们门",config,model,tokenizer,label_list)
    wordSubtract("char","伙火们房",config,model,tokenizer,label_list)

    wordSubtract("char","草早芋于",config,model,tokenizer,label_list)
    wordSubtract("char","草早芋房",config,model,tokenizer,label_list)

    wordSubtract("char","送关通甬",config,model,tokenizer,label_list)
    wordSubtract("char","送关通房",config,model,tokenizer,label_list)

    wordSubtract("char","邻令郊交",config,model,tokenizer,label_list)
    wordSubtract("char","邻令郊房",config,model,tokenizer,label_list)

    wordSubtract("char","痒羊疼冬",config,model,tokenizer,label_list)
    wordSubtract("char","痒羊疼房",config,model,tokenizer,label_list)

    wordSubtract("char","国玉固古",config,model,tokenizer,label_list)
    wordSubtract("char","国玉固房",config,model,tokenizer,label_list)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("Time Elapsed:",end-start)