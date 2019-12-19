# -*- coding:utf-8 -*-

# custom import
import sys

#sys.path.insert(0, '../core')

from process_real_data import *
#
import os
import random
import math
import argparse
import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from generator import Generator
from discriminator import Discriminator
from target_lstm import TargetLSTM
from rollout import Rollout
from data_iter import GenDataIter, DisDataIter
from torchnlp.metrics import get_moses_multi_bleu
from matplotlib import pyplot as plt

# ================== Parameter Definition =================

parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--cuda', action='store', default=None, type=int)
parser.add_argument('--test', action='store_true',default=True)
opt = parser.parse_args()

# Basic Training Paramters
SEED = 88
BATCH_SIZE = 10
TOTAL_BATCH = 1
GENERATED_NUM = 100
ROOT_PATH = 'experiments_mle/'
POSITIVE_FILE = ROOT_PATH + 'real.data'
TEST_FILE = ROOT_PATH + 'test.data'
NEGATIVE_FILE = ROOT_PATH + 'gene.data'
DEBUG_FILE = ROOT_PATH + 'debug.data'
EVAL_FILE = ROOT_PATH + 'eval.data'
VOCAB_SIZE = 5000
PRE_EPOCH_NUM = 2000
CHECKPOINT_PATH = ROOT_PATH + 'checkpoints/'

try:
    os.makedirs(CHECKPOINT_PATH)
except OSError:
    print('Directory already exists!')

if opt.cuda is not None and opt.cuda >= 0:
    torch.cuda.set_device(opt.cuda)
    opt.cuda = True

# Genrator Parameters
g_emb_dim = 32
g_hidden_dim = 32
g_sequence_len = 50

track_training=1



def demo():
    metadata = load_vocab(CHECKPOINT_PATH)
    idx_to_word=metadata['idx_to_word']
    word_to_idx=metadata['word_to_idx']
    VOCAB_SIZE=metadata['vocab_size']
    test_iter = GenDataIter(TEST_FILE, BATCH_SIZE)
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)
    if opt.cuda:
        generator = generator.cuda()
    generator.load_state_dict(torch.load(CHECKPOINT_PATH + 'generator_mle.model'))
    #test_predict(generator, test_iter, idx_to_word)
    generate_samples(generator, BATCH_SIZE, GENERATED_NUM, NEGATIVE_FILE)

    show_some_generated_sequences(idx_to_word, 100, NEGATIVE_FILE)


def get_word(s, idx_to_words=None):
    if idx_to_words == None:
        return str(s)
    return idx_to_words[int(s)]


def generate_samples(model, batch_size, generated_num, output_file, idx_to_word=None):
    samples = []
    for _ in range(int(generated_num / batch_size)):
        sample = model.sample(batch_size, g_sequence_len).cpu().data.numpy().tolist()
        # for each_sen in sample:
        #    generate_sentence_from_id(idx_to_word, each_sen, DEBUG_FILE, header = 'REAL ---')
        samples.extend(sample)
    with open(output_file, 'w') as fout:
        for sample in samples:
            string = ' '.join([str(s) for s in sample])
            fout.write('%s\n' % string)


def train_epoch(model, data_iter, criterion, optimizer):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:  # tqdm(
        # data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data)
        target = Variable(target)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        if len(pred.shape) > 2:
            pred = torch.reshape(pred, (pred.shape[0] * pred.shape[1], -1))
        loss = criterion(pred, target)
        total_loss += loss.data.item()
        total_words += data.size(0) * data.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    data_iter.reset()
    return math.exp(total_loss / total_words)


def eval_epoch(model, data_iter, criterion):
    total_loss = 0.
    total_words = 0.
    for (data, target) in data_iter:  # tqdm(
        # data_iter, mininterval=2, desc=' - Training', leave=False):
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        pred = model.forward(data)
        loss = criterion(pred, target)
        total_loss += loss.data.item()
        total_words += data.size(0) * data.size(1)
    data_iter.reset()
    return math.exp(total_loss / total_words)


def test_predict(model, data_iter, idx_to_word, train_mode=False):
    #data_iter.reset()
    for (data, target) in data_iter:
        data = Variable(data, volatile=True)
        target = Variable(target, volatile=True)
        if opt.cuda:
            data, target = data.cuda(), target.cuda()
        target = target.contiguous().view(-1)
        prob = model.forward(data)
        mini_batch = prob.shape[0]
        if len(prob.shape) > 2:
            prob = torch.reshape(prob, (prob.shape[0] * prob.shape[1], -1))
        else:
            mini_batch /= g_sequence_len
            mini_batch = int(mini_batch)
        predictions = torch.max(prob, dim=1)[1]
        predictions = predictions.view(mini_batch, -1)
        # print('PRED SHAPE:' , predictions.shape)
        for each_sen in list(predictions):
            sentence=generate_sentence_from_id(idx_to_word, each_sen)
            sentence=' '.join(sentence)
            print('Sample Test Output:',sentence)
        sys.stdout.flush()
        if train_mode:
            break
    data_iter.reset()

def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        lis.append(l)
    return lis

def show_some_generated_sequences(idx_to_word,nb_sentences,gen_file):
    lis=read_file(gen_file)
    all_indices=np.arange(len(lis))
    random.shuffle(all_indices)
    lis_to_show=[l for index,l in enumerate(lis) if index in all_indices[:nb_sentences]]
    for each_sen in lis_to_show:
        sentence = generate_sentence_from_id(idx_to_word, each_sen)
        sentence = ' '.join(sentence)
        print('Sample Test Output:', sentence)
    sys.stdout.flush()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    track_blue = []

    # Build up dataset
    s_train, s_test = load_from_big_file('obama_speech',g_sequence_len)
    # idx_to_word: List of id to word
    # word_to_idx: Dictionary mapping word to id
    idx_to_word, word_to_idx = fetch_vocab(s_train, s_train, s_test)
    # input_seq, target_seq = prepare_data(DATA_GERMAN, DATA_ENGLISH, word_to_idx)

    global VOCAB_SIZE
    VOCAB_SIZE = len(idx_to_word)
    save_vocab(CHECKPOINT_PATH + 'metadata.data', idx_to_word, word_to_idx, VOCAB_SIZE, g_emb_dim, g_hidden_dim)

    print('VOCAB SIZE:', VOCAB_SIZE)
    # Define Networks
    generator = Generator(VOCAB_SIZE, g_emb_dim, g_hidden_dim, opt.cuda)

    if opt.cuda:
        generator = generator.cuda()

    # Generate toy data using target lstm
    print('Generating data ...')

    # Generate samples either from sentences file or lstm
    # Sentences file will be structured input sentences
    # LSTM based is BOG approach
    generate_real_data('obama_speech', BATCH_SIZE, GENERATED_NUM, idx_to_word, word_to_idx,
                       POSITIVE_FILE, TEST_FILE)
    # generate_samples(target_lstm, BATCH_SIZE, GENERATED_NUM, POSITIVE_FILE, idx_to_word)
    # generate_samples(target_lstm, BATCH_SIZE, 10, TEST_FILE, idx_to_word)
    # Create Test data iterator for testing
    test_iter = GenDataIter(TEST_FILE, BATCH_SIZE)
    #test_predict(generator, test_iter, idx_to_word, train_mode=True)

    # Load data from file
    gen_data_iter = GenDataIter(POSITIVE_FILE, BATCH_SIZE)
    lines = read_file(POSITIVE_FILE)

    refrences = []
    for line in lines:
        phrase = []
        for char in line:
            phrase.append(idx_to_word[char])

        refrences.append(' '.join(phrase))
        #refrences.append(phrase)



    # Pretrain Generator using MLE
    gen_criterion = nn.NLLLoss(size_average=False)
    gen_optimizer = optim.Adam(generator.parameters())
    if opt.cuda:
        gen_criterion = gen_criterion.cuda()
    print('Pretrain with MLE ...')
    for epoch in range(PRE_EPOCH_NUM):
        loss = train_epoch(generator, gen_data_iter, gen_criterion, gen_optimizer)
        print('Epoch [%d] Model Loss: %f' % (epoch, loss))
        sys.stdout.flush()
        generate_samples(generator, BATCH_SIZE, GENERATED_NUM, EVAL_FILE)
        if track_training:
            lines = read_file(EVAL_FILE)
            hypotheses = []
            for line in lines:
                phrase = []
                for char in line:
                    phrase.append(idx_to_word[char])

                hypotheses.append(' '.join(phrase))
                #hypotheses.append(phrase)

            bleu_score=get_moses_multi_bleu(hypotheses, refrences, lowercase=True)
            track_blue.append(bleu_score)
            print(track_blue)


    torch.save(generator.state_dict(), CHECKPOINT_PATH + 'generator_mle.model')
    track_blue=np.array(track_blue)
    np.save(ROOT_PATH+'track_blue_mle3.npy',track_blue)


    plt.plot(track_blue)
    plt.show()


if __name__ == '__main__':
    if opt.test:
        demo()
        exit()
    main()