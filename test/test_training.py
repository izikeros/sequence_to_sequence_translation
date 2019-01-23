from keras.preprocessing.sequence import pad_sequences

from network import create_model, find_checkpoint_file, train
from utils import preprocess_data, load_train_data, load_from_file

from config import test_conf as conf


# Loading input sequences, output sequences
print('Loading data...')
x, y = load_train_data(file_source_lang='small_vocab_en',
                       file_target_lang='small_vocab_fr',
                       load_method=load_from_file,
                       )

# create mappings
x_sent_array, x_vocab_len, x_word_to_idx, x_idx_to_word = preprocess_data(
    x, conf['MAX_LEN'], conf['VOCAB_SIZE'], conf['NUM_SENT'])
y_sent_array, y_vocab_len, y_word_to_idx, y_idx_to_word = preprocess_data(
    y, conf['MAX_LEN'], conf['VOCAB_SIZE'], conf['NUM_SENT'])

# Find the length of the longest sequence
x_max_len = max([len(sentence) for sentence in x_sent_array])
y_max_len = max([len(sentence) for sentence in y_sent_array])

# Padding zeros to make all sequences have a same length with the longest one
print('Zero padding...')
X = pad_sequences(x_sent_array, maxlen=x_max_len, dtype='int32')
y = pad_sequences(y_sent_array, maxlen=y_max_len, dtype='int32')

# Creating the network model
print('Compiling model...')
model = create_model(x_vocab_len, x_max_len, y_vocab_len, y_max_len,
                     conf['HIDDEN_DIM'], conf['LAYER_NUM'])

# Finding trained weights of previous epoch if any
saved_weights = find_checkpoint_file('.')

saved_weights = []
train(X, y, y_word_to_idx, y_max_len, saved_weights, model, conf)

