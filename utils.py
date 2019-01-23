# -*- coding: utf-8 -*-
from collections import Counter

import matplotlib.pyplot as plt
from keras.preprocessing.text import text_to_word_sequence

from preporcessing import (split_text_to_array_of_sentences,
                           convert_words_to_indices,
                           create_vocabulary,
                           create_mapping)


def load_from_file(file_name, gdrive=None):
    """ load training data from file"""
    f = open('data/' + file_name, 'r')
    data = f.read()
    f.close()
    return data


def display_data_info(in_sentences, out_sentences,
                      in_lang='English', out_lang='French'):
    """ display short stats on training data"""
    in_sentences = in_sentences.split('\n')
    out_sentences = out_sentences.split('\n')

    in_words = [word for sentence in in_sentences for word in
                sentence.split()]
    out_words = [word for sentence in out_sentences for word in
                 sentence.split()]

    print(f'How many sentences? {len(in_sentences)}')

    print(f'How many {in_lang} words? {len(in_words)}')
    print(f'How many unique {in_lang} words? {len(Counter(in_words))}')
    print(f'10 Most common words in the {in_lang} dataset:')
    print('"' + '" "'.join(
        list(zip(*Counter(in_words).most_common(10)))[0]) + '"')
    print()
    print(f'How many {out_lang} words? {len(out_words)}')
    print(f'How many unique {out_lang} words? {len(Counter(out_words))}')
    print(f'10 Most common words in the {out_lang} dataset:')
    print(f'"' + '" "'.join(
        list(zip(*Counter(out_words).most_common(10)))[0]) + '"')


def load_train_data(file_source_lang, file_target_lang,
                    load_method=load_from_file, drive=None):
    """Reading raw text from source and destination files"""
    print("- Loading training data")
    source_data = load_method(file_source_lang, drive)
    target_data = load_method(file_target_lang, drive)
    return source_data, target_data


def preprocess_data(sent_array, vocab_size):
    print("- creating vocabulary")
    vocab = create_vocabulary(sent_array, vocab_size)

    print("- creating mappings")
    word_to_idx, idx_to_word = create_mapping(vocab)

    print("- converting words to indices")
    sent_array = convert_words_to_indices(sent_array, word_to_idx)
    vocab_len = len(vocab) + 2
    return sent_array, vocab_len, word_to_idx, idx_to_word


def load_test_data(source, word_to_idx, max_len, load_method=load_from_file):
    data = load_method(source)

    sent_array = [text_to_word_sequence(x)[::-1] for x in data.split('\n') if
                  0 < len(x) <= max_len]
    for i, sentence in enumerate(sent_array):
        for j, word in enumerate(sentence):
            if word in word_to_idx:
                sent_array[i][j] = word_to_idx[word]
            else:
                sent_array[i][j] = word_to_idx['UNK']
    return sent_array


def plot_history(history):
    # Plot training (& validation) accuracy values
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('model_accuracy.png')

    # Plot training (& validation) loss values
    plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('model_loss.png')
