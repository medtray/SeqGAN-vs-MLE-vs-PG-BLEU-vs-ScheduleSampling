# -*- coding:utf-8 -*-

import os
import random
import math
import copy

import tqdm
from torchnlp.metrics import get_moses_multi_bleu

import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class bleu_Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate,refrences,vocab):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.refrences=refrences
        self.vocab=vocab

    def get_reward(self, x, num):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        full_hypo = []
        for line in x:
            phrase = []
            for char in line:
                phrase.append(self.vocab[char])

            #full_hypo.append(' '.join(phrase))
            full_hypo.append(phrase)

        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                samples = self.own_model.sample(batch_size, seq_len, data)
                #pred = discriminator(samples)
                #pred = pred.cpu().data[:,1].numpy()

                hypotheses = []
                for line in samples:
                    phrase = []
                    for char in line:
                        phrase.append(self.vocab[char])

                    #hypotheses.append(' '.join(phrase))
                    hypotheses.append(phrase)



                pred=[]

                for sentence in hypotheses:
                    pred.append(nltk.translate.bleu_score.sentence_bleu(self.refrences, sentence))
                    #pred.append(get_moses_multi_bleu([sentence], self.refrences, lowercase=True))

                pred=np.array(pred)

                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = []

            for sentence in full_hypo:
                #pred.append(get_moses_multi_bleu([sentence], self.refrences, lowercase=True))
                pred.append(nltk.translate.bleu_score.sentence_bleu(self.refrences, sentence))

            pred = np.array(pred)
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred

            print ('done %d out of %d' %(i,num))
        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
