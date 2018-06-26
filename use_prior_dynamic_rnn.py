#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
 Filename @ use_prior_dynamic_rnn.py
 Author @ huangjunheng
 Create date @ 2018-06-23 16:23:27
 Description @ 
"""
import torch
import torch.nn as nn

import utils
import torch.nn.functional as F

from sequence_data import SequenceData
from config import Config


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, seq_len):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # # Decode the hidden state of the last time step
        last_step_index_list = (seq_len - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
        last_hidden_outputs = out.gather(1, last_step_index_list).squeeze()

        last_out = self.fc(last_hidden_outputs)
        all_out = self.fc(out)

        return last_out, all_out


class UsePriorDynamicRNN(object):
    """
        数据处理
        """
    def __init__(self):

        self.config = Config()
        self.sentiment_dict = utils.read_sentiment_file(self.config.sentiment_file)
        # Hyper-parameters

        self.max_seq_len, input_size, num_classes = utils.cal_model_para(self.config.training_file)

        self.model = RNN(input_size, self.config.num_hidden,
                         self.config.num_layers, num_classes).to(device)

        self.loss_and_optimizer()

    def loss_and_optimizer(self):
        """
        Loss and optimizer
        :return: 
        """
        # self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def cross_entropy_loss(self, y_predict, y):
        """
        only 2-d
        sum {- y * log(y_predict)}
        :return: 
        """
        n_sample = y_predict.shape[0]
        n_dim = y_predict.shape[1]

        y_tensor = torch.zeros(n_sample, n_dim, dtype=torch.long)
        for i in range(n_sample):
            y_tensor[i, y[i]] = 1

        softmax_y_predict = F.softmax(y_predict, dim=1)

        loss_item = -1 * y_tensor.float() * torch.log(softmax_y_predict)
        loss = torch.mean(loss_item.sum(dim=1))

        return loss

    def regularized_loss(self, all_outputs, seq_len, words):
        """
        
        :param all_outputs: 
        :return: 
        """
        ret_loss = 0.0
        M = 0.0  #可调参数
        for timestep_outputs, valid_len, word_list in zip(all_outputs, seq_len, words):
            timestep_outputs = F.softmax(timestep_outputs, dim=1)
            pre_dist = None
            for i in range(valid_len):
                if i == 0:
                    pre_dist = timestep_outputs[i]
                    continue
                cur_dist = timestep_outputs[i]
                cur_word = word_list[i].decode('utf8')
                if cur_word in self.sentiment_dict:
                    # print cur_word.encode('utf8'), self.sentiment_dict[cur_word]
                    pre_dist = pre_dist + torch.Tensor(self.sentiment_dict[cur_word])
                    pre_dist = F.softmax(pre_dist, dim=0)
                    # print pre_dist, cur_dist, utils.sym_kl_divergence(pre_dist, cur_dist) - M
                    regularzied_loss = torch.max(torch.Tensor([0]), utils.sym_kl_divergence(pre_dist, cur_dist) - M)
                    ret_loss += regularzied_loss

        return ret_loss

    def train(self):
        """
        train
        :return: 
        """
        print 'Start training model.'
        training_set = SequenceData(filename=self.config.training_file, max_seq_len=self.max_seq_len)
        for i in range(self.config.training_steps):
            batch_x, batch_y, batch_seqlen, batch_words = training_set.next(self.config.batch_size)

            batch_x = batch_x.to(device)
            _, batch_y = torch.max(batch_y, 1) # 元组第一个维度为最大值，第二个维度为最大值的索引
            batch_y = batch_y.to(device)
            batch_seqlen = batch_seqlen.to(device)

            # Forward pass
            outputs, all_outputs = self.model(batch_x, batch_seqlen)
            # loss = self.cross_entropy_loss(outputs, batch_y)
            loss = self.cross_entropy_loss(outputs, batch_y) + \
                   self.regularized_loss(all_outputs, batch_seqlen, batch_words)
            # loss = self.criterion(outputs, batch_y)

            # Backward and optimize
            self.optimizer.zero_grad()  # 清空梯度缓存
            loss.backward()  # 反向传播，计算梯度
            self.optimizer.step()  # 利用梯度更新模型参数

            if (i + 1) % 100 == 0:
                print 'Step [{}/{}], Loss: {:.4f}'\
                    .format(i + 1, self.config.training_steps, loss.item())

        # Save the model checkpoint
        print 'Start saving model to "%s".' % self.config.save_model_path
        torch.save(self.model.state_dict(), self.config.save_model_path)

    def test(self, load_model=False):
        """
        test
        :param load_model: 
        :return: 
        """
        if load_model:
            print 'Start loading model from "%s"' % self.config.load_model_path
            self.model.load_state_dict(torch.load(self.config.load_model_path))

        test_set = SequenceData(filename=self.config.test_file, max_seq_len=self.max_seq_len)

        with torch.no_grad():
            correct = 0
            total = 0
            features, labels, seqlen, words = test_set.get_all_data()

            features = features.to(device)
            _, labels = torch.max(labels, 1)
            labels = labels.to(device)
            seqlen = seqlen.to(device)

            outputs, all_outputs = self.model(features, seqlen)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print 'Test Accuracy of the model: {} %'.format(100 * correct / total)

    def main(self):
        """
        main
        :return: 
        """
        self.train()
        self.test(load_model=True)

if __name__ == '__main__':
    rnn = UsePriorDynamicRNN()
    rnn.main()



