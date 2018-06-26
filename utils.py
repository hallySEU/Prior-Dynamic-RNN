#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ utils.py
 Author @ huangjunheng
 Create date @ 2018-06-24 12:26:27
 Description @ 
"""

import torch
import jieba


def file2array(filename):
    """
    file to array
    :param filename: 
    :return: 
    """
    ret_array = []
    fr = open(filename)
    for line in fr:
        line = line.rstrip('\n')
        ret_array.append(line)

    return ret_array


def read_sentiment_file(sentiment_file):
    """
    
    :param sentiment_file: 
    :return: 
    """
    sentiment_dict = {}
    fr = open(sentiment_file)
    for line in fr:
        line = line.rstrip('\n')
        tokens = line.split('&&&')
        sentiment_word = tokens[0].decode('utf8')
        sentiment_dict[sentiment_word] = [float(tokens[1]), float(tokens[2])]

    return sentiment_dict


def cal_model_para(filename):
    """
    根据数据计算模型的参数
    1. 最大sequence长度: max_seq_len
    2. 单个输入特征的维度: input_size
    3. label的维度，几分类就几个维度: num_class
    :param filename: 
    :return: 
    """
    max_seq_len = -1
    fr = open(filename)
    for i, line in enumerate(fr):
        line = line.rstrip('\n')
        data_split = line.split('&&&')
        feature_data_list = data_split[0].split('\t')

        if i == 0:
            input_size = len(feature_data_list[0].split('#'))
            num_class = len(data_split[1].split('\t'))

        cur_seq_len = len(feature_data_list)
        if cur_seq_len > max_seq_len:
            max_seq_len = cur_seq_len

    if max_seq_len % 10 != 0:
        max_seq_len = ((max_seq_len / 10) + 1) * 10

    print 'According to "%s", seq_max_len is set to %d, ' \
          'input_size is set to %d, num_class is set to %d.' \
          % (filename, max_seq_len, input_size, num_class)
    return max_seq_len, input_size, num_class


def asym_kl_divergence(p, q):
    """
    cal kl 
    :return: 
    """
    kl_item = p * torch.log(p / q)
    kl_divergence = kl_item.sum()

    return kl_divergence


def sym_kl_divergence(p, q):
    """
    cal sym kl divergence
    :param p: 
    :param q: 
    :return: 
    """
    kl_divergence = (asym_kl_divergence(p, q) + asym_kl_divergence(q, p)) / 2.0

    return kl_divergence


def cal_word_frequency(filename):
    """
    
    :param filename: 
    :return: 
    """
    line_array = file2array(filename)
    word_frequency_dict = {}
    for line in line_array:
        tokens = jieba.lcut(line)
        for token in tokens:
            token = token
            if token not in word_frequency_dict:
                word_frequency_dict[token] = 0

            word_frequency_dict[token] += 1

    word_frequency_dict = sorted(word_frequency_dict.items(), key=lambda x: x[1], reverse=True)

    for k, v in word_frequency_dict:
        print k, v


if __name__ == '__main__':
    # read_sentiment_file('data/sentiment_dict.txt')

    cal_word_frequency('data/v3/origin_sex_data/sex_results.txt')