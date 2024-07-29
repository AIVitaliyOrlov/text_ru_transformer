from data_set.preprocessing import BPETokenizer

SPECIAL_TOKENS = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
END_OF_WORD  = '</w>'

TRAIN_DATA_PATH = 'D:\\DataSet\\row_data\\tokenize_train'
#TRAIN_DATA_PATH = 'D:\\DataSet\\row_data\\tt_test'
TARGET_DATA_PATH = 'D:\\DataSet\\row_data\\tokenize_target'

tokenizer = BPETokenizer(SPECIAL_TOKENS, END_OF_WORD, 50000)
tokenizer.train(TRAIN_DATA_PATH, [10000, 15000, 20000, 25000, 30000, 35000, 40000], TARGET_DATA_PATH)

#tokenizer = BPETokenizer(SPECIAL_TOKENS, END_OF_WORD, 200)
#tokenizer.train(TRAIN_DATA_PATH, [50, 100, 150], TARGET_DATA_PATH)