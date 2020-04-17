# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
from keras.layers import (
    Activation, TimeDistributed, Dense, RepeatVector, Embedding
)
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences


from preporcessing import one_hot_encoding
from utils import load_test_data, plot_history


def create_model(x_vocab_len, x_max_len, y_vocab_len, y_max_len, hidden_size,
                 num_layers):

    model = Sequential()

    # Encoder
    model.add(Embedding(
        x_vocab_len, 1000, input_length=x_max_len, mask_zero=True)
    )
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(y_max_len))

    # Decoder
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train(X, Y, y_word_to_ix, y_max_len, saved_weights, model, conf):
    k_start = 1
    # If any trained weight was found, then load them into the model
    if len(saved_weights) > 0:
        print('[INFO] Saved weights found, loading...')
        epoch = saved_weights[
                saved_weights.rfind('_') + 1:saved_weights.rfind('.')]
        model.load_weights(saved_weights)
        k_start = int(epoch) + 1

    for k in range(k_start, conf['EPOCHS'] + 1):
        # Shuffling the training data every epoch to avoid local minima
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        # Training 1000 sequences at a time
        for i in range(0, len(X), 1000):
            if i + 1000 >= len(X):
                i_end = len(X)
            else:
                i_end = i + 1000
            y_sequences = one_hot_encoding(Y[i:i_end], y_max_len,
                                           y_word_to_ix)

            print(f'[INFO] Training model: epoch {k}th {i}/{len(X)} samples')
            history = model.fit(X[i:i_end], y_sequences, batch_size=conf[
                'BATCH_SIZE'],
                                epochs=1, verbose=2)

        # actions on epoch finalization
        model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))


def run_test(saved_weights, model, conf, x_word_to_idx, x_max_len,
             y_idx_to_word, num=None):
    # Only performing test if there is any saved weights
    if len(saved_weights) == 0:
        print("The network hasn't been trained! Program will exit...")
        sys.exit()
    else:
        print(" - loading test data")
        x_test = load_test_data('test', x_word_to_idx, conf['MAX_LEN'])
        if num:
            x_test = x_test[0:num]
        x_test = pad_sequences(x_test, maxlen=x_max_len, dtype='int32')
        print(" - loading model")
        model.load_weights(saved_weights)

        print(" - calculating predictions")
        predictions = np.argmax(model.predict(x_test), axis=2)
        sequences = []
        print(" - processing")
        for prediction in predictions:
            sequence = ' '.join(
                [y_idx_to_word[index] for index in prediction if index > 0])
            print(sequence)
            sequences.append(sequence)
        np.savetxt('test_result', sequences, fmt='%s')


def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if
                       'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    print("found checkpoint(s):")
    print(modified_time)
    return checkpoint_file[int(np.argmax(modified_time))]
