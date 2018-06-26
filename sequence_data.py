#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ data_processor.py
 Author @ huangjunheng
 Create date @ 2018-06-23 18:06:27
 Description @ 
"""
import numpy as np
import torch

import utils


class SequenceData(object):
    """
    数据处理
    """

    def __init__(self, filename, max_seq_len=5):
        self.batch_id = 0
        self.filename = filename
        self.data, self.labels, self.seqlen, self.words = self.load_data(filename, max_seq_len)

    def next(self, batch_size):
        """ 
        获取全量数据(长度为n_samples)中的批量数据(长度为batch_size)
         e.g. n_samples = 100, batch_size = 16, batch_num = 7(6+1), last_batch_size = 4
        Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_index = min(self.batch_id + batch_size, len(self.data))

        batch_data = (self.data[self.batch_id: batch_index])
        batch_labels = (self.labels[self.batch_id: batch_index])
        batch_seqlen = (self.seqlen[self.batch_id: batch_index])
        batch_word = (self.words[self.batch_id: batch_index])

        self.batch_id = batch_index

        x_tensor = torch.from_numpy(np.array(batch_data, dtype=np.float32))
        y_tensor = torch.from_numpy(np.array(batch_labels, dtype=np.int64))
        seq_len_tensor = torch.from_numpy(np.array(batch_seqlen, dtype=np.int64))

        return x_tensor, y_tensor, seq_len_tensor, batch_word

    def get_all_data(self):
        """
        
        :return: 
        """
        x_tensor = torch.from_numpy(np.array(self.data, dtype=np.float32))
        y_tensor = torch.from_numpy(np.array(self.labels, dtype=np.int64))
        seq_len_tensor = torch.from_numpy(np.array(self.seqlen, dtype=np.int64))

        return x_tensor, y_tensor, seq_len_tensor, self.words

    def load_data(self, filename, max_seq_len):
        """
        加载数据
        :return: 
        """
        fr = open(filename)
        datas = []
        labels = []
        seqlen = []
        words = []
        # line_list = ['1#3\t2#5\t3#7&1\t0', '3#3\t3#3\t5#5\t7#7&0\t1']
        for line in fr:
            line = line.rstrip('\n')
            data_split = line.split('&&&')
            feature_data_list = data_split[0].split('\t')
            cur_seq_len = len(feature_data_list)
            seqlen.append(cur_seq_len)

            input_size = len(feature_data_list[0].split('#'))
            s = [[float(i) for i in item.split('#')] for item in feature_data_list]
            s += [[0.] * input_size for i in range(max_seq_len - cur_seq_len)]
            datas.append(s)

            if len(data_split) > 1:  # 区分训练与预测
                label_data_list = data_split[1].split('\t')
                labels.append([float(item) for item in label_data_list])

                words.append(data_split[2].split('\t'))

        return datas, labels, seqlen, words

    def test(self):
        """
        test func
        :return: 
        """
        max_seq_len, input_size, num_class = utils.cal_model_para(self.filename)


if __name__ == '__main__':
    filename = 'data/training_test_data/test_data.txt'
    batch_size = 3

    max_seq_len, input_size, num_class = utils.cal_model_para(filename)
    loader = SequenceData(filename, max_seq_len)

    loader.next(batch_size)










