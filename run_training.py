from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model

from network import create_model, find_checkpoint_file, train
from preporcessing import split_text_to_array_of_sentences
from utils import preprocess_data, load_train_data, load_from_file


# import configuration for this experiment
from config import long_train_conf as conf

# Loading input sequences, output sequences
print('Loading data...')
x, y = load_train_data(file_source_lang='small_vocab_en',
                       file_target_lang='small_vocab_fr',
                       load_method=load_from_file,
                       )

print("Splitting to sentences, reversing word order in input sentences")
x, y = split_text_to_array_of_sentences(x, y,
                                        conf['MAX_LEN'],
                                        conf['NUM_SENT']
                                        )

# create mappings
x_sent_array, x_vocab_len, x_word_to_idx, x_idx_to_word = preprocess_data(
    x, conf['VOCAB_SIZE']
)
y_sent_array, y_vocab_len, y_word_to_idx, y_idx_to_word = preprocess_data(
    y, conf['VOCAB_SIZE']
)

# Find the length of the longest sequence
x_max_len = max([len(sentence) for sentence in x_sent_array])
y_max_len = max([len(sentence) for sentence in y_sent_array])

# Padding zeros to make all sequences have a same length with the longest one
print('Zero padding...')
X = pad_sequences(x_sent_array, maxlen=x_max_len, dtype='int32')
y = pad_sequences(y_sent_array, maxlen=y_max_len, dtype='int32')

# Creating the network model
print('Compiling model...')
model = create_model(x_vocab_len, x_max_len,
                     y_vocab_len, y_max_len,
                     conf['HIDDEN_DIM'], conf['LAYER_NUM'])
plot_model(model, to_file='./images/model.png')
# Finding trained weights of previous epoch if any
saved_weights = find_checkpoint_file('.')

saved_weights = []
train(X, y, y_word_to_idx, y_max_len, saved_weights, model, conf)
