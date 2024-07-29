import torchvision.transforms as transforms
from easydict import EasyDict

data_cfg = EasyDict()

data_cfg.data_set = EasyDict()
data_cfg.data_set.name = 'TextDataset'
data_cfg.data_set.special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
data_cfg.data_set.end_of_word = '</w>'
data_cfg.data_set.end_of_line = '<|end|>'
#data_cfg.data_set.end_of_line = '<|endoftext|>'
data_cfg.data_set.tokenizer_path = 'data\\tokenizer\\vocabulary_25000.data'
data_cfg.data_set.path_to_raw_data = 'data\\raw'
data_cfg.data_set.preprocessed_data_path='data\\preprocessed'
data_cfg.data_set.preprocessed_data_statistic_template='data\\preprocessed\\statistic_%s.pickle'



