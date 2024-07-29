import pickle

import unicodedata
from tqdm import tqdm
import regex as re

from data_set.preprocessing import BPETokenizer

INPUT_DATA = 'qwdqwdqwwdqwwdqwwdqwd'
INPUT_FILE = 'D:\\DataSet\\packs\\part_0.txt'
END_OF_SEQ = '<|end|>'

class Tokenizer():
    def __init__(self, special_tokens=None, split_pattern=None):
        self.token2id = {}
        self.id2token = {}
        self.split_pattern = ''
        self.special_tokens = special_tokens
        self._init_vocabulary(split_pattern =split_pattern)

    def _init_vocabulary(self, id2token=None, token2id=None, split_pattern=None):
        self.split_pattern = split_pattern
        if all((id2token, token2id)):
            self.id2token, self.token2id = id2token, token2id

        else:
            vocab_init_value = '0123456789йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ,.?!;:'
            vocab = {value: idx for idx, value in enumerate(list(vocab_init_value))}
            self.id2token = {i: token for i, token in enumerate(vocab)}
            self.token2id = {token: i for i, token in enumerate(vocab)}

    def save_to_file(self, file_path_prefix: str):
        file_path = file_path_prefix + str(len(self.token2id)) + '.tokens'
        with open(file_path, 'wb') as f:
            pickle.dump({
                'split_pattern': self.split_pattern,
                'id2token': self.id2token,
                'token2id': self.token2id}, f)

    def load_from_file(self, path):
        """Loads vocabulary state."""
        with open(path, 'rb') as f:
            vocab_mappings = pickle.load(f)
        self._init_vocabulary(vocab_mappings['id2token'],
                              vocab_mappings['token2id'],
                              vocab_mappings['split_pattern'])


    def _get_statistic(self, ids, counter):
        for pair in zip(ids, ids[1:]):
            counter[pair] = counter.get(pair, 0) + 1
        return counter

    def _merge(self, raw_ids, join_pair, new_id):
        join_result = []
        i = 0
        while i < len(raw_ids):
            if raw_ids[i] == join_pair[0] and i < len(raw_ids) - 1 and raw_ids[i + 1] == join_pair[1]:
                join_result.append(new_id)
                i += 2
            else:
                join_result.append(raw_ids[i])
                i += 1
        return join_result
    def add_token(self, new_token:str):
        index = len(self.token2id)
        self.token2id[new_token] = index
        self.id2token[index] = new_token
    def train(self, text, num_merges):
        text_chunks = re.findall(self.split_pattern, text)
        tokens_words = [list(ch) for ch in text_chunks]
        for i in range(num_merges):
            stats = {}
            for chunk_ids in tokens_words:
                self._get_statistic(chunk_ids, stats)
            pair = max(stats, key=stats.get)

            if pair[0] not in self.token2id:
                self.add_token(pair[0])

            if pair[1] not in self.token2id:
                self.add_token(pair[1])

            new_token = pair[0]+pair[1]

            self.add_token(new_token)
            tokens_words = [self._merge(chunk_ids, pair, new_token) for chunk_ids in tokens_words]
            print('Join >'+new_token)
    def encode(self, text:str):
        encoded = []
        tokens = []
        words = [tuple(word)+(self.end_of_word,) for word in text.strip().split()]
        for word in words:
            i = 0
            while i < len(word):
                unknown = True
                for j in range(len(word), i, -1):
                    subword = ''.join(word[i:j])
                    if subword in self.token2id:
                        encoded.append(self.token2id[subword])
                        tokens.append(subword)
                        i = j - 1
                        unknown = False
                        break
                i += 1
                if unknown :
                    encoded.append(self.token2id['[UNK]'])
        return encoded

    def decode(self, tokens: list[int])->str:
        decoded = ''.join([self.id2token[t] for t in tokens])

        return decoded

INPUT_FILE = 'D:\\DataSet\\row_data\\tokenize_train\\8911456.fb2_data'
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
#tokenizer = Tokenizer({}, GPT4_SPLIT_PATTERN)
#tokenizer.load_from_file('token\\token_voc183.tokens')
SPECIAL_TOKENS = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
END_OF_WORD  = '</w>'
tokenize = BPETokenizer(SPECIAL_TOKENS, END_OF_WORD, vocabulary_size=100)

with open(INPUT_FILE, 'r', encoding='UTF-8') as f:
#    content = f.read()
#    tokenizer.train(content, 100)
#    tokenizer.save_to_file('token\\token_voc')
#    lines_text = f.read()
    tokenize.train(INPUT_FILE)
#    tokens = tokenizer.encode(lines_text)
#    decode_value = tokenizer.decode(tokens)
#    print(decode_value == lines_text)



#with tqdm(total=100000) as pbar:
#    with open(INPUT_FILE, 'r', encoding='UTF-8') as f:
#        content = ''
#        for line in f.readlines():
#            line = line.strip()
#            if line != END_OF_SEQ:
#                content += line
#            else:
#               tokenizer.load_statistics(content)
#                content = ''
#                pbar.update()

#print(tokenizer.common_statistic)
