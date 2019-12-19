import glob
from sklearn.model_selection import train_test_split
import numpy as np

import collections
import itertools

import torch

from torch import nn
from torch import optim
import sys

from nltk.translate.bleu_score import corpus_bleu

NUMBER_OF_SENTENCES = 1000
g_sequence_len = 50

Token = collections.namedtuple("Token", ["index", "word"])
SOS = Token(0, "<sos>")
EOS = Token(1, "<eos>")
PAD = Token(2, "<pad>")


# Helper for interactive demo
def pad_sentences(sentence, sentence_len):
    words = sentence.split(" ")
    if len(words) > sentence_len:
        # keep only 10 words
        words = words[:sentence_len]
    else:
        for i in range(sentence_len - len(words)):
            words.append(PAD.word)

    return words


# Convert the sentence to word ids
def get_ids(sentence, idx_to_word, word_to_idx, VOCAB_SIZE):
    sentence_ids = []

    for word in sentence:
        if word != PAD.word:
            flag = 1
            break

    if flag == 1:

        for word in sentence:
            if word.lower() not in word_to_idx:
                # PAD when unknown word found
                sentence_ids.append(SOS.index)
            elif word.lower():
                sentence_ids.append(word_to_idx[word.lower()])
    else:
        sentence_ids.append(SOS.word)
        for word in sentence[1:]:
            sentence_ids.append(PAD.word)

    return sentence_ids


def load_from_big_file(file,nb_words_to_use):
    s = []

    with open(file) as f:
        lines = f.readlines()

        for line in lines[:NUMBER_OF_SENTENCES]:
            line = line.strip()
            line = line.rstrip(".")
            words = line.split()
            for i in range(len(words)):
                words[i] = words[i].strip(',"')
            # if len(words) >= nb_words_to_use-1:
            #     sent = " ".join(words[:nb_words_to_use-1])
            #     sent += " ."
            # else:
            #     sent = " ".join(words)
            #     sent += " ."
            #     sent += (" " + PAD.word) * (nb_words_to_use-1 - len(words))

            if len(words)>=nb_words_to_use:
                sent = " ".join(words[:nb_words_to_use])
            elif len(words)<nb_words_to_use:
                sent = " ".join(words)
                sent += (" " + PAD.word) * (nb_words_to_use - len(words))



            s.append(sent)

    s_train, s_test = train_test_split(s, shuffle=True, test_size=0.1, random_state=42)
    return s_train, s_test


def fetch_vocab(DATA_GERMAN, DATA_ENGLISH, DATA_GERMAN2):  # -> typing.Tuple[typing.List[str], typing.Dict[str, int]]:
    """Determines the vocabulary, and provides mappings from indices to words and vice versa.

    Returns:
        tuple: A pair of mappings, index-to-word and word-to-index.
    """
    # gather all (lower-cased) words that appear in the data
    all_words = set()
    for sentence in itertools.chain(DATA_GERMAN, DATA_ENGLISH, DATA_GERMAN2):
        all_words.update(word.lower() for word in sentence.split(" ") if word != PAD.word)

        # create mapping from index to word
    idx_to_word = [SOS.word, EOS.word, PAD.word] + list(sorted(all_words))

    # create mapping from word to index
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

    return idx_to_word, word_to_idx


def generate_sentence_from_id(idx_to_word, input_ids, file_name=None, header=''):
    sentence = []
    if file_name:
        out_file = open(file_name, 'a')
        out_file.write(header + ':')

    sep = ''
    for id in input_ids:
        sentence.append(idx_to_word[id])
        if file_name:
            out_file.write(sep + idx_to_word[id])
            sep = ' '
    if file_name:
        out_file.write('\n')
        out_file.close()
    return sentence


def generate_file_from_sentence(sentences, out_file, word_to_idx, generated_num=0):
    if generated_num:
        generated_index = np.random.choice(len(sentences), generated_num)
    else:
        generated_index = np.arange(0, len(sentences))

    out_file = open(out_file, "w")
    for i in generated_index:
        sent = sentences[i].split(' ')
        new_sent_id = []
        sep = ''
        for word in sent:
            out_file.write(sep + str(word_to_idx[word.lower()]))
            sep = ' '
        out_file.write('\n')


def generate_real_data(input_file, batch_size, generated_num, idx_to_word, word_to_idx, train_file, test_file=None):
    train_sen, test_sen = load_from_big_file(input_file,g_sequence_len)

    generate_file_from_sentence(train_sen, train_file, word_to_idx, generated_num)
    if test_file:
        generate_file_from_sentence(test_sen, test_file, word_to_idx)


def save_vocab(checkpoint, idx_to_word, word_to_idx, vocab_size, g_emb_dim=None, g_hidden_dim=None,
               g_sequence_len=None):
    """
    out_file = open(checkpoint+'idx_to_word.pkl', "wb")
    pickle.dump(idx_to_word, out_file)
    out_file.close()

    out_file = open(checkpoint+'word_to_idx.pkl', "wb")
    pickle.dump(word_to_idx, out_file)
    out_file.close()
    out_file = open(checkpoint+'vocab_size.pkl', "wb")
    pickle.dump(vocab_size, out_file)
    out_file.close()
    """
    metadata = {}
    metadata['idx_to_word'] = idx_to_word
    metadata['word_to_idx'] = word_to_idx
    metadata['vocab_size'] = vocab_size
    metadata['g_emb_dim'] = g_emb_dim
    metadata['g_hidden_dim'] = g_hidden_dim
    metadata['g_sequence_len'] = g_sequence_len
    torch.save(metadata, checkpoint)


def load_vocab(checkpoint):
    metadata = torch.load(checkpoint + 'metadata.data')
    return metadata