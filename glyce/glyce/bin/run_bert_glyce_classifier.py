# -*- coding: utf-8 -*-
# Edits made by Jason Wang
# Original Author: Yuxian Meng (Glyce)

import os
import sys

root_path = "/".join(os.path.realpath(__file__).split("/")[:-3])
print("check the root_path of this repo")
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import os 
import csv 
import sys 
import torch
import random
import logging
import argparse
import numpy as np 
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, \
SequentialSampler
from glyce.utils.tokenization import BertTokenizer
from glyce.utils.optimization import BertAdam
from glyce.dataset_readers.bert_config import Config 
from glyce.dataset_readers.bert_single_sent import ChinaNewsProcessor, DianPingProcessor, JDFullProcessor, JDBinaryProcessor, FuDanProcessor, ChnSentiCorpProcessor, ifengProcessor
from glyce.dataset_readers.bert_sent_pair import DBQAProcessor, LCQMCProcessor, BQProcessor, XNLIProcessor  
from glyce.models.glyce_bert.glyce_bert_classifier import GlyceBertClassifier 
from glyce.dataset_readers.bert_data_utils import convert_examples_to_features 
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from glyce.utils.metrics.cls_evaluate_funcs import  acc, f1_measures



logging.basicConfig()
logger = logging.getLogger(__name__)



def args_parser():
    # start parser
    parser = argparse.ArgumentParser()

    # required parameters
    parser.add_argument("--config_path", default="/home/lixiaoya/dataset/", type=str)
    parser.add_argument("--data_dir", default=None, type=str, help="the input data dir")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert-large-uncased, bert-base-cased, bert-large-cased")
    parser.add_argument("--task_name", default=None, type=str)

    # # other parameters
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
    parser.add_argument("--nworkers", type=int, default=1)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--output_name", type=str, default="model.bin")
    parser.add_argument("--data_sign", type=str, default="nlpcc-dbqa")
    parser.add_argument("--classifier_sign", type=str, default="single_linear")
    parser.add_argument("--residual", type=str, default="no")
    parser.add_argument("--hidden_size",type=int,default=None)
    parser.add_argument("--font_channels", type=int, default=None)
    parser.add_argument("--font_name", type=str, default=None)
    parser.add_argument("--font_names", type=str, default=[], nargs='+', action='append')
    parser.add_argument("--font_size", type=int, default=None)
    parser.add_argument("--num_fonts_concat", type=int, default=None)
    parser.add_argument("--glyph_embsize", type=int, default=None)
    parser.add_argument("--training_strat",type=str,default=None)
    parser.add_argument("--transformer",type=str,default="yes")
    args = parser.parse_args()

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    return args


def load_data(config):
    # load some data and processor
    if config.data_sign == "nlpcc-dbqa":
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
    tokenizer = BertTokenizer.from_pretrained(config.bert_model, do_lower_case=True)

    # load data examples
    train_examples = data_processor.get_train_examples(config.data_dir)
    dev_examples = data_processor.get_dev_examples(config.data_dir)
    test_examples = data_processor.get_test_examples(config.data_dir)

    # convert data example into features
    train_features = convert_examples_to_features(train_examples, label_list, config.max_seq_length, tokenizer,
                                                  task_sign=config.task_name)
    train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    train_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    train_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
    # train_sampler = DistributedSampler(train_data)
    train_sampler = RandomSampler(train_data)

    dev_features = convert_examples_to_features(dev_examples, label_list, config.max_seq_length, tokenizer,
                                                task_sign=config.task_name)
    dev_input_ids = torch.tensor([f.input_ids for f in dev_features], dtype=torch.long)
    dev_input_mask = torch.tensor([f.input_mask for f in dev_features], dtype=torch.long)
    dev_segment_ids = torch.tensor([f.segment_ids for f in dev_features], dtype=torch.long)
    dev_label_ids = torch.tensor([f.label_id for f in dev_features], dtype=torch.long)
    dev_data = TensorDataset(dev_input_ids, dev_input_mask, dev_segment_ids, dev_label_ids)

    dev_sampler = RandomSampler(dev_data)

    test_features = convert_examples_to_features(test_examples, label_list, config.max_seq_length, tokenizer,
                                                 task_sign=config.task_name)
    test_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    test_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    test_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    test_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_data = TensorDataset(test_input_ids, test_input_mask, test_segment_ids, test_label_ids)
    # test_sampler = DistributedSampler(test_data)
    test_sampler = RandomSampler(test_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, \
                                  batch_size=config.train_batch_size, num_workers=config.nworkers)

    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, \
                                batch_size=config.dev_batch_size, num_workers=config.nworkers)

    test_dataloader = DataLoader(test_data, sampler=test_sampler, \
                                 batch_size=config.test_batch_size, num_workers=config.nworkers)

    #num_train_steps = int(len(train_examples) / config.train_batch_size * 5)
    num_train_steps = int(math.ceil(len(train_examples)/config.train_batch_size)/config.gradient_accumulation_steps)*config.num_train_epochs
    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list


def load_model(config, num_train_steps, label_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    model = GlyceBertClassifier(config, num_labels=len(label_list))

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # prepare  optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=config.warmup_proportion,
                         t_total=num_train_steps)

    return model, optimizer, device, n_gpu


def train(model, optimizer, train_dataloader, dev_dataloader, test_dataloader, config, \
          device, n_gpu, label_list):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    dev_best_acc = 0
    dev_best_precision = 0
    dev_best_recall = 0
    dev_best_f1 = 0
    dev_best_loss = 10000000000000

    test_best_acc = 0
    test_best_precision = 0
    test_best_recall = 0
    test_best_f1 = 0
    test_best_loss = 1000000000000000

    model.train()

    for idx in range(int(config.num_train_epochs)):
        model.updateEpoch()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        print("#######" * 10)
        print("EPOCH: ", str(idx))
        for step, batch in tqdm(enumerate(train_dataloader),position=0,leave=True):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, glyph_loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()
                glyph_loss = glyph_loss.mean()

            if global_step < config.glyph_warmup:
                sum_loss = loss + config.glyph_ratio * glyph_loss
            else:
                sum_loss = loss + config.glyph_ratio * glyph_loss * config.glyph_decay ** (idx + 1)

            sum_loss.backward()

            tr_loss += loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (nb_tr_steps+1) % config.checkpoint == 0:
                print("-*-" * 15)
                print("current training loss is : ")
                print("classification loss, glyph loss")
                print(loss.item(), glyph_loss.item())
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model,
                                                                        dev_dataloader,
                                                                        config, device,
                                                                        n_gpu, label_list,
                                                                        eval_sign="dev")
                print("......" * 10)
                print("DEV: loss, precision, recall, f1, acc")
                print(tmp_dev_loss, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1, tmp_dev_acc)

                if tmp_dev_f1 > dev_best_f1 or tmp_dev_acc > dev_best_acc:
                    dev_best_acc = tmp_dev_acc
                    dev_best_loss = tmp_dev_loss
                    dev_best_precision = tmp_dev_prec
                    dev_best_recall = tmp_dev_rec
                    dev_best_f1 = tmp_dev_f1

                    tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model,
                                                                               test_dataloader,
                                                                               config,
                                                                               device,
                                                                               n_gpu,
                                                                               label_list,
                                                                               eval_sign="test")
                    print("......" * 10)
                    print("TEST: loss, precision, recall, f1, acc")
                    print(tmp_test_loss, tmp_test_prec, tmp_test_rec, tmp_test_f1, tmp_test_acc)

                    if tmp_test_f1 > test_best_f1 or tmp_test_acc > test_best_acc:
                        test_best_acc = tmp_test_acc
                        test_best_loss = tmp_test_loss
                        test_best_precision = tmp_test_prec
                        test_best_recall = tmp_test_rec
                        test_best_f1 = tmp_test_f1

                        # export model
                        if config.export_model:
                            model_to_save = model.module if hasattr(model, "module") else model
                            output_model_file = config.output_name
                            torch.save(model_to_save.state_dict(), output_model_file)

                print("-*-" * 15)

    tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model,
                                                                                            test_dataloader,
                                                                                            config,
                                                                                            device,
                                                                                            n_gpu,
                                                                                            label_list,
                                                                                            eval_sign="test")
    print("......" * 10)
    print("TEST: loss, precision, recall, f1, acc")
    print(tmp_test_loss, tmp_test_prec, tmp_test_rec, tmp_test_f1, tmp_test_acc)

    if tmp_test_f1 > test_best_f1 or tmp_test_acc > test_best_acc:
        test_best_acc = tmp_test_acc
        test_best_loss = tmp_test_loss
        test_best_precision = tmp_test_prec
        test_best_recall = tmp_test_rec
        test_best_f1 = tmp_test_f1

        # export model
        if config.export_model:
            model_to_save = model.module if hasattr(model, "module") else model
            output_model_file = config.output_name
            torch.save(model_to_save.state_dict(), output_model_file)

    # export a trained model
    model_to_save = model
    output_model_file = config.output_name
    if config.export_model:
        torch.save(model_to_save.state_dict(), output_model_file)

    print("=&=" * 15)
    print("DEV: current best loss, precision, recall, f1, acc")
    print(dev_best_loss, dev_best_precision, dev_best_recall, dev_best_f1, dev_best_acc)
    print("TEST: current best loss, precision, recall, f1, acc")
    print(test_best_loss, test_best_precision, test_best_recall, test_best_f1, test_best_acc)
    print("=&=" * 15)


def eval_checkpoint(model_object, eval_dataloader, config, \
                    device, n_gpu, label_list, eval_sign="dev"):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    model_object.eval()

    idx2label = {i: label for i, label in enumerate(label_list)}

    eval_loss = 0
    eval_accuracy = []
    eval_f1 = []
    eval_recall = []
    eval_precision = []
    logits_all = []
    labels_all = []
    eval_steps = 0

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,position=0,leave=True):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss, glyph_loss = model_object(input_ids, segment_ids, input_mask, label_ids)
            logits, glyph_loss = model_object(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to("cpu").numpy()
        input_mask = input_mask.to("cpu").numpy()
        logits = np.argmax(logits, axis=-1)

        input_mask = input_mask.tolist()


        eval_loss += tmp_eval_loss.mean().item()


        logits_all.extend(logits.tolist())
        labels_all.extend(label_ids.tolist())
        eval_steps += 1

    average_loss = round(eval_loss / eval_steps, 4)
    eval_accuracy = float(accuracy_score(y_true=labels_all, y_pred=logits_all))
    matric = f1_measures(preds=logits_all, labels=labels_all)
    eval_f1 = float(matric["f1"])
    eval_recall = float(matric["recall"])
    eval_precision = float(matric["precision"])

    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1


def merge_config(args_config):
    model_config_path = args_config.config_path
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config


def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, device, n_gpu = load_model(config, num_train_steps, label_list)
    train(model, optimizer, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)



if __name__ == "__main__":
    main()
