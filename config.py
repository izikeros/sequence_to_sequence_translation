long_train_conf = {
    'MAX_LEN': 200,
    'VOCAB_SIZE': 250,  # 20000
    'BATCH_SIZE': 100,  # 100
    'LAYER_NUM': 3,
    'HIDDEN_DIM': 20,  # 1000
    'EPOCHS': 500,
    'NUM_SENT': None  # None means all
}

short_conf = {
    'MAX_LEN': 200,
    'VOCAB_SIZE': 100,  # 20000
    'BATCH_SIZE': 100,  # 100
    'LAYER_NUM': 3,
    'HIDDEN_DIM': 10,  # 1000
    'EPOCHS': 50,
    'NUM_SENT': 100000
}

test_conf = {
    'MAX_LEN': 200,
    'VOCAB_SIZE': 100,  # 20000
    'BATCH_SIZE': 100,  # 100
    'LAYER_NUM': 3,
    'HIDDEN_DIM': 10,  # 1000
    'EPOCHS': 4,
    'NUM_SENT': 1100
}
