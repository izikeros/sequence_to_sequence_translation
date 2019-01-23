# -*- coding: utf-8 -*-
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from nltk import FreqDist


def split_text_to_array_of_sentences(x_data, y_data, max_len, num_sent=None):
    """Splitting raw text into array of sentences"""
    # Splitting raw text into array of sequences
    x_sentence_array = [text_to_word_sequence(x)[::-1]
         for x, y in zip(x_data.split('\n'), y_data.split('\n')) if
         0 < len(x) <= max_len and 0 < len(y) <= max_len
         ]

    y_sentence_array = [text_to_word_sequence(y)
         for x, y in zip(x_data.split('\n'), y_data.split('\n')) if
         0 < len(x) <= max_len and 0 < len(y) <= max_len
         ]

    # limit number of sentences if needed
    if num_sent:
        x_sentence_array = x_sentence_array[0:num_sent]
        y_sentence_array = y_sentence_array[0:num_sent]
    return x_sentence_array, y_sentence_array


def create_vocabulary(sequences_array, vocab_size):
    """Creating the vocabulary set with the most common words"""
    target = FreqDist(np.hstack(sequences_array))
    vocab = target.most_common(vocab_size - 1)
    return vocab


def convert_words_to_indices(sent_array, word_to_idx):
    """Converting each word to its index value"""
    for i, sentence in enumerate(sent_array):
        for j, word in enumerate(sentence):
            if word in word_to_idx:
                sent_array[i][j] = word_to_idx[word]
            else:
                sent_array[i][j] = word_to_idx['UNK']
    return sent_array


def create_mapping(vocab):
    """Creating an array of words from the vocabulary set,

        use this array as index-to-word dictionary
     """
    idx_to_word = [word[0] for word in vocab]
    # Adding the word "ZERO" to the beginning of the array
    idx_to_word.insert(0, 'ZERO')
    # Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
    idx_to_word.append('UNK')
    # Creating the word-to-index dictionary from the array created above
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    return word_to_idx, idx_to_word


def one_hot_encoding(word_sentences, max_len, word_to_idx):
    """Vectorize each sequence"""
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_idx)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences
