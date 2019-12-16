import sys
import time
from tqdm import *
from glob import glob
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def load_dataset(data_dir, dataset_name, sub_data_size,sub_dataset):
    train_files = glob(os.path.join(data_dir, dataset_name, "questions",
                                  sub_dataset,"*.question"))  # read all training dataset file name
    
    sub_train_files = train_files[:sub_data_size]  # select part of sub_data to use
    train_d = []
    train_q = []
    d_len = []
    q_len = []
    max_idx = len(sub_train_files)
    y = []
    for idx, fname in tqdm(enumerate(sub_train_files)):  # read each sub_data file 
        with open(fname) as f:
            (_, document, question, answer, _), data_idx, data_max_idx = f.read().split("\n\n"), idx, max_idx 
            d = [str(d) for d in document.split()]  # read each document word in each file
            q = [str(q) for q in question.split()]  # read each query word in each file
            train_d.append(d)
            train_q.append(q)
            y.append(answer)
            d_len.append(len(train_d[-1]))
            q_len.append(len(train_q[-1]))
    return train_d, train_q, d_len, q_len, y   # All training document, query, document length, query length and answer(like @entity)


def build_dict(sentences, max_words=50000):
     
    #Build a dictionary for the words in each training sentences including document and query.
    #Only the max_words ones are kept and the remaining will be mapped to <UNK>.
     
    word_count = Counter()
    for sent in sentences:
        for w in sent:
            word_count[w] += 1

    ls = word_count.most_common(max_words) 
    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        print(key)
    print('...')
    for key in ls[-5:]:
        print(key)

    # here leave 0 to UNK
    # here leave 1 to delimiter liek |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}


def load_glove_weights(glove_dir, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.' + str(embd_dim) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index)) 
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def get_minibatches(n, mb_size, shuffle=False):
    idx_list = np.arange(0, n, mb_size)
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    for i in idx_list:
        minibatches.append(np.arange(i, min(n, i+mb_size)))
    return minibatches


def prepare_data(seqs):
    lengths = [len(seq) for seq in seqs]
    n_samples = len(seqs)
    max_len = np.max(lengths)
    x = np.zeros((n_samples, max_len)).astype('int32')
    x_mask = np.zeros((n_samples, max_len)).astype(float)
    for idx, seq in enumerate(seqs):
        x[idx, :lengths[idx]] = seq   # padding all sample for each minibatch
        x_mask[idx, :lengths[idx]] = 1.0
    return x, x_mask


def vectorize(doc, query, ans, word_dict, entity_dict,
              sort_by_len=True, verbose=True):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(doc), len(entity_dict))).astype(float)
    in_y = []
    for idx, (d, q, a) in enumerate(zip(doc, query, ans)):
        assert (a in d)
        seq1 = [word_dict[w] if w in word_dict else 0 for w in d] #  here 0 for unk
        seq2 = [word_dict[w] if w in word_dict else 0 for w in q]
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            in_l[idx, [entity_dict[w] for w in d if w in entity_dict]] = 1.0
            in_y.append(entity_dict[a] if a in entity_dict else 0)
        if verbose and (idx % 1000 == 0):
            print('vectorize: Vectorization: processed %d / %d' % (idx, len(doc)))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        # sort by the document length
        sorted_index = len_argsort(in_x1)
        in_x1 = [in_x1[i] for i in sorted_index]
        in_x2 = [in_x2[i] for i in sorted_index]
        in_l = in_l[sorted_index]
        in_y = [in_y[i] for i in sorted_index]

    return in_x1, in_x2, in_l, in_y


def gen_examples(x1, x2, l, y, batch_size):
    """
        Divide examples into batches of size `batch_size`.
    """
    minibatches = get_minibatches(len(x1), batch_size) # get the index of each sample 
    all_ex = []
    for minibatch in minibatches:
        mb_x1 = [x1[t] for t in minibatch]
        mb_x2 = [x2[t] for t in minibatch]
        mb_l = l[minibatch]
        mb_y = [y[t] for t in minibatch]
        mb_x1, mb_mask1 = prepare_data(mb_x1)
        mb_x2, mb_mask2 = prepare_data(mb_x2)
        all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y))
    return all_ex
