# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 命令行解释器
parser = argparse.ArgumentParser(description='Chinese Text Classification')
# 这里是解析命令行参数 
# parser.add_argument('--model', type=str, default='bert',required=False, help='choose a model: Bert, ERNIE')
# 解析结果存在args变量里面
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    args.model='bert'
    args.model='bert'
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.bert_path1))
    train(config, model, train_iter, dev_iter, test_iter)
