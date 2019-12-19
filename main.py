# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

from torchnlp.metrics import get_moses_multi_bleu
import re
from collections import Counter
import pickle
from bleu_rollout import bleu_Rollout
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import os
import random
import math
import pandas as pd
import argparse
import tqdm

from matplotlib import pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy
import torch.nn.functional as F

from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
opt = parser.parse_args()
#opt.cuda='cuda'
print(opt)

# Basic Training Paramters
#SEED = 88
BATCH_SIZE = 64
TOTAL_BATCH = 200
GENERATED_NUM = 500
POSITIVE_FILE = 'real.data'
NEGATIVE_FILE = 'gene.data'
EVAL_FILE = 'eval.data'
VOCAB_SIZE = 30
PRE_EPOCH_NUM = 2

epochs_for_mle_model=600

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 20

# Discriminator Parameters
d_emb_dim = 64
d_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
d_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

d_dropout = 0.75
d_num_class = 2

perf_dict={}
true_loss=[]
generator_loss=[]
disc_loss=[]

perf_dict_pgbleu={}

def generate_samples(model, batch_size, generated_num, output_file):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)

def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:#tqdm(
        #data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)


def train_ss(model, data_iter, criterion, optimizer,start_ss_pred,p):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:  # tqdm(
        # data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)

        h, c = model.init_hidden(len(data))
        x=data[:,0:start_ss_pred]
        emb = model.emb(x)
        output, (h, c) = model.lstm(emb, (h, c))
        pred = F.softmax(model.lin(output[:,-1,:]), dim=1)

        predicted = pred.multinomial(1)


        for i in range(start_ss_pred,g_sequence_len+1):
            if random.random()>p:
                inputs = torch.unsqueeze(data[:, i], 1)
            else:
                inputs = predicted

            emb = model.emb(inputs)
            output2, (h, c) = model.lstm(emb, (h, c))
            pred = F.softmax(model.lin(output2[:, -1, :]), dim=1)
            #predicted = pred.multinomial(1)
            predicted=torch.unsqueeze(pred.argmax(1),1)
            output=torch.cat((output,output2),dim=1)

        pred = model.softmax(model.lin(output.contiguous().view(-1, model.hidden_dim)))

        loss = criterion(pred, target)
        total_loss += loss.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)


def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    with torch.no_grad():
        for (data, target) in data_iter:#tqdm(
            #data_iter, mininterval=2, desc=' - Training', leave=False):
            data = Variable(data)
            target = Variable(target)
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            target = target.contiguous().view(-1)
            pred = model.forward(data)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_words += data.size(0) * data.size(1)
        data_iter.reset()

    assert total_words > 0  # Otherwise NullpointerException
    return math.exp(total_loss / total_words)

class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""
    def __init__(self):
        super(GANLoss, self).__init__()

    def forward(self, prob, target, reward):
        """
        Args:
            prob: (N, C), torch Variable
            target : (N, ), torch Variable
            reward : (N, ), torch Variable
        """
        N = target.size(0)
        C = prob.size(1)
        one_hot = torch.zeros((N, C))
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        one_hot.scatter_(1, target.data.view((-1,1)), 1)
        one_hot = one_hot.type(torch.ByteTensor)
        one_hot = Variable(one_hot)
        if prob.is_cuda:
            one_hot = one_hot.cuda()
        loss = torch.masked_select(prob, one_hot)
        loss = loss * reward
        loss =  -torch.sum(loss)
        return loss


target_lstm = TargetLSTM(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
if opt.cuda:
    target_lstm = target_lstm.cuda()
# Generate toy data using target lstm
print('Generating data ...')
generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE)
# Load data from file
gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
original_generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)


def MLE(epochs,generator_):

    generator_loss_mle=[]
    true_loss_mle=[]
    generator = copy.deepcopy(generator_)
    if opt.cuda:
        generator = generator.cuda()

    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()

    for epoch in range(epochs):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))
        generator_loss_mle.append(loss)
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        true_loss_mle.append(loss)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

    return true_loss_mle,generator_loss_mle,generator


def SS(epochs,start_ss,p,generator_,start_ss_pred):

    nb_epochs_mle=start_ss

    generator_loss_mle = []
    true_loss_mle = []
    generator = copy.deepcopy(generator_)
    if opt.cuda:
        generator = generator.cuda()

    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()

    for epoch in range(nb_epochs_mle):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f' % (epoch, loss))
        generator_loss_mle.append(loss)
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        true_loss_mle.append(loss)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

    p=0
    ii=0
    for epoch in range(nb_epochs_mle,epochs):
        ii+=1
        p=ii*0.002
        loss = train_ss(generator, gen_data_iter, gen_criterion, gen_optimizer,start_ss_pred,p)

        print('Epoch [%d] Model Loss: %f' % (epoch, loss))
        generator_loss_mle.append(loss)
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        true_loss_mle.append(loss)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

    return true_loss_mle, generator_loss_mle, generator



def main(generator_):
    #random.seed(SEED)
    #np.random.seed(SEED)

    nb_batch_per_epoch=int(GENERATED_NUM / BATCH_SIZE)

    # Define Networks
    generator = copy.deepcopy(generator_)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    #target_lstm = TargetLSTM(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        #target_lstm = target_lstm.cuda()
    # Generate toy data using target lstm
    #print('Generating data ...')
    #generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE)

    # Load data from file
    #gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)

    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))
        generator_loss.append(loss)
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        true_loss.append(loss)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Discriminator ...')
    for epoch in range(5):
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for _ in range(3):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            disc_loss.append(loss)
            print('Epoch [%d], loss: %f' % (epoch, loss))
    # Adversarial Training
    rollout = Rollout(generator, 0.8)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()

    # gen_criterion = nn.NLLLoss(reduction='sum')
    # if opt.cuda:
    #     gen_criterion = gen_criterion.cuda()
    # dis_criterion = nn.NLLLoss(reduction='sum')
    # dis_optimizer = optim.Adam(discriminator.parameters())
    # if opt.cuda:
    #     dis_criterion = dis_criterion.cuda()

    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        for it in range(1):
            print(it)
            samples = generator.sample(BATCH_SIZE, g_sequence_len)
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 16, discriminator)
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if opt.cuda:
                rewards = rewards.cuda()
            prob = generator.forward(inputs)
            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
            loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
            true_loss.append(loss)
            print('Batch [%d] True Loss: %f' % (total_batch, loss))
            loss_gen = eval_epoch(generator, gen_data_iter, gen_criterion)
            print('Epoch [%d] Model Loss: %f' % (total_batch, loss_gen))
            generator_loss.append(loss_gen)
        rollout.update_params()

        for _ in range(4):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for _ in range(2):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
                disc_loss.append(loss)

    perf_dict['true_loss']=true_loss
    perf_dict['generator_loss'] = generator_loss
    perf_dict['disc_loss'] = disc_loss
    np.save('perf_dict', perf_dict)

    # plt.figure(1)
    # plt.plot(true_loss)
    # plt.figure(2)
    # plt.plot(generator_loss)
    # plt.figure(3)
    # plt.plot(disc_loss)
    # plt.show()





class WordIndexer:
    """Transform a dataset of text to a list of index of words."""

    def __init__(self, min_word_occurences=10, oov_word="OOV"):
        """ min_word_occurrences: integer, the minimum frequency of the word to keep.
            oov_word: string, a special string for out-of-vocabulary words.
        """
        self.oov_word = oov_word
        self.min_word_occurences = min_word_occurences
        self.word_to_index = {oov_word: 0}
        self.index_to_word = [oov_word]
        self.word_occurrences = {}

        # for retaining only words
        self.re_words = re.compile(r"\b[a-zA-Z]{2,}\b")

    def get_word_index(self, word, add_new_word=True):
        """ Find the index of a word.

            word: string, the query word.
            add_new_word: if true, if the word has no entry, assign a new integer index to word.
                            if false, return the index of the oov_word

            return: index of the query word
        """
        try:
            return self.word_to_index[word]
        except:
            if add_new_word:
                index = self.n_words
                self.word_to_index[word] = index
                self.index_to_word.append(word)
                return index
            else:
                return self.word_to_index[self.oov_word]

    def fit_transform(self, texts, use_existing_indexer=False):

        stop_words = set(stopwords.words('english'))

        l_words = []
        for sentence in texts:
            inter = sentence.lower()
            inter2 = inter.split(' ')
            inter3 = ' '.join([word for word in inter2 if word not in stop_words])
            l_words.append(self.re_words.findall(inter3))

        ## when finish, print out the number of sentences.
        print(f'number of sentences: {len(l_words)}')

        mapped_word_to_indices = []

        if use_existing_indexer:
            for sentence in l_words:
                inter = [self.get_word_index(word, False) for word in sentence]
                mapped_word_to_indices.append(inter)
            # return the mapped word indices here

        if not use_existing_indexer:

            all_words = []
            for sentence in l_words:
                all_words += sentence

            unique_words = Counter(all_words)

            self.word_occurrences = {}
            for word, nb in unique_words.items():
                if nb >= self.min_word_occurences:
                    self.word_occurrences[word] = nb

            for sentence in l_words:
                inter = [self.get_word_index(word, True) if word in self.word_occurrences else self.word_to_index[
                    self.oov_word] for word in sentence]
                mapped_word_to_indices.append(inter)

        return mapped_word_to_indices

    @property
    def n_words(self):
        """ return: the vocabulary size
        """
        return len(self.word_to_index)

class AmazonReviewGloveDataset(Dataset):
    def __init__(self, path, indexer=None, min_word_occurences=10):
        """ Load the reviews from a csv file. One row is one review.
                See train_small.csv for the format.

            path: path to the csv file containing the reviews and their ratings
            indexer: if None, build a new WordIndexer, otherwise use the one passed in.
            right_window: integer, how large the window is to get context words.
                        Looking into the right hand side of the center word only.
            min_word_occurrences: integer, the minimum frequency of the word to keep.
        """

        # Step 1: tokenize the first field of each row into sentences
        #         (e.g. using nltk.tokenize.sent_tokenize).
        ## Your codes go here

        data = pd.read_csv(path)
        data = data[pd.notnull(data['reviewText'])]
        reviews = data['reviewText']
        sentences = []
        for i, review in enumerate(reviews):
            inter = nltk.tokenize.sent_tokenize(review)
            sentences += inter

        # Step 2: pass the list of all sentences to WordIndexer.
        # turn list of sentences into list of lists of word indices in the sentences.
        # Keep the word ordering.
        print('Indexing the corpus...')
        ## Your codes go here

        if indexer:
            self.indexer = indexer
            mapped_word_to_indices = self.indexer.fit_transform(sentences, True)
        else:
            self.indexer = WordIndexer(min_word_occurences, "OOV")
            mapped_word_to_indices = self.indexer.fit_transform(sentences, False)

        print('Done indexing the corpus.')


def PG_BLEU(generator_):

    def read_file(data_file):
        with open(data_file, 'r') as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(' ')
            l = [int(s) for s in l]
            lis.append(l)
        return lis



    lines=read_file(POSITIVE_FILE)
    train_dataset = AmazonReviewGloveDataset('train_small.csv')
    vocab=train_dataset.indexer.index_to_word[1::]
    del train_dataset

    refrences=[]
    for line in lines:
        phrase=[]
        for char in line:
            phrase.append(vocab[char])

        #refrences.append(' '.join(phrase))
        refrences.append(phrase)

    hypotheses = []
    for line in lines[:3]:
        phrase = []
        for char in line:
            phrase.append(vocab[char])

        #hypotheses.append(' '.join(phrase))
        hypotheses.append(phrase)

    BLEUscore = nltk.translate.bleu_score.sentence_bleu(refrences, hypotheses[0])

    nb_batch_per_epoch=int(GENERATED_NUM / BATCH_SIZE)

    # Define Networks
    generator = copy.deepcopy(generator_)
    discriminator = Discriminator(d_num_class, VOCAB_SIZE, d_emb_dim, d_filter_sizes, d_num_filters, d_dropout)
    #target_lstm = TargetLSTM(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    if opt.cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        #target_lstm = target_lstm.cuda()
    # Generate toy data using target lstm
    #print('Generating data ...')
    #generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE)

    # Load data from file
    #gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)

    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(reduction='sum')
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(40):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f'% (epoch, loss))
        generator_loss.append(loss)
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
        loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
        true_loss.append(loss)
        print('Epoch [%d] True Loss: %f' % (epoch, loss))

    # Pretrain Discriminator
    dis_criterion = nn.NLLLoss(reduction='sum')
    dis_optimizer = optim.Adam(discriminator.parameters())
    if opt.cuda:
        dis_criterion = dis_criterion.cuda()
    print('Pretrain Discriminator ...')
    for epoch in range(1):
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
        dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
        for _ in range(1):
            loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
            disc_loss.append(loss)
            print('Epoch [%d], loss: %f' % (epoch, loss))
    # Adversarial Training
    rollout = bleu_Rollout(generator, 0.8,refrences,vocab)
    print('#####################################################')
    print('Start Adeversatial Training...\n')
    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_gan_loss = gen_gan_loss.cuda()

    for total_batch in range(TOTAL_BATCH):
        ## Train the generator for one step
        #nb_batch_per_epoch
        for it in range(1):
            #print(it)
            samples = generator.sample(BATCH_SIZE, g_sequence_len)
            # construct the input to the genrator, add zeros before samples and delete the last column
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor)
            if samples.is_cuda:
                zeros = zeros.cuda()
            inputs = Variable(torch.cat([zeros, samples.data], dim = 1)[:, :-1].contiguous())
            targets = Variable(samples.data).contiguous().view((-1,))
            # calculate the reward
            rewards = rollout.get_reward(samples, 2)
            rewards = Variable(torch.Tensor(rewards))
            rewards = torch.exp(rewards).contiguous().view((-1,))
            if opt.cuda:
                rewards = rewards.cuda()
            prob = generator.forward(inputs)
            loss = gen_gan_loss(prob, targets, rewards)
            gen_gan_optm.zero_grad()
            loss.backward()
            gen_gan_optm.step()

        if total_batch % 1 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
            eval_iter = GenDataIter(EVAL_FILE, BATCH_SIZE)
            loss = eval_epoch(target_lstm, eval_iter, gen_criterion)
            true_loss.append(loss)
            print('Batch [%d] True Loss: %f' % (total_batch, loss))
            loss_gen = eval_epoch(generator, gen_data_iter, gen_criterion)
            print('Epoch [%d] Model Loss: %f' % (total_batch, loss_gen))
            generator_loss.append(loss_gen)
        rollout.update_params()

        for _ in range(4):
            generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)
            dis_data_iter = DisDataIter(POSITIVE_FILE, NEGATIVE_FILE, BATCH_SIZE)
            for _ in range(2):
                loss = train_epoch(discriminator, dis_data_iter, dis_criterion, dis_optimizer)
                disc_loss.append(loss)

    perf_dict_pgbleu['true_loss']=true_loss
    perf_dict_pgbleu['generator_loss'] = generator_loss
    perf_dict_pgbleu['disc_loss'] = disc_loss
    np.save('perf_dict', perf_dict)





if __name__ == '__main__':

    perf_dict_ss = {}
    true_loss_ss, generator_loss_ss, _=SS(epochs_for_mle_model, 10, 0.5, original_generator,1)
    perf_dict_ss['true_loss'] = true_loss_ss
    perf_dict_ss['generator_loss'] = generator_loss_ss
    np.save('perf_dict_ss', perf_dict_ss)
    # #
    perf_dict_mle={}
    true_loss_mle, generator_loss_mle,_=MLE(epochs_for_mle_model,original_generator)
    perf_dict_mle['true_loss'] = true_loss_mle
    perf_dict_mle['generator_loss'] = generator_loss_mle
    np.save('perf_dict_mle', perf_dict_mle)
    main(original_generator)
    PG_BLEU(original_generator)
