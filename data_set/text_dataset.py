import os

from torch.utils.data import Dataset

from data_set.preprocessing import BPETokenizer
from utils.common_functions import write_file, read_file
from utils.enums import SetType


class TextDataset(Dataset):

    def __init__(self, config, set_type: SetType):
        self.config = config
        self.set_type = set_type
        self.current_dataset_ids = []
        self.data_statistic = []
        self._init_preprocessors()
        self._prepare_data()


    def _prepare_data(self):
        statistic_file_path = self.config.preprocessed_data_statistic_template % self.set_type.name
        if not os.path.exists(statistic_file_path):
            raw_data_path = os.path.join(
                self.config.path_to_raw_data, self.set_type.name
            )

            prepared_data_path = os.path.join(
                self.config.preprocessed_data_path, self.set_type.name
            )
            os.makedirs(prepared_data_path, exist_ok=True)

            statistic = {}
            idx = 0
            for root, dir, file in os.walk(raw_data_path):
                if len(file) > 0:
                    for filename in file:
                        print(f'start_prepare_data from {filename}')
                        file_path = os.path.join(root, filename)
                        prepared_date = []
                        prepared_file_name = filename + '.pickle'
                        statistic[prepared_file_name] = []
                        with open(file_path, 'r', encoding='utf-8') as data_file:
                            line = ''
                            for tmp_line in data_file:
                                tmp_line = tmp_line.strip()
                                if tmp_line == self.config.end_of_line:
                                    if len(line) > 0:
                                        idx += 1
                                        tokens = self.tokenizer.encode(line)
                                        prepared_date.append({'id': idx, 'tokens': tokens})
                                        statistic[prepared_file_name].append(idx)
                                        line = ''
                                else:
                                    if len(line) > 0:
                                        line += ' '
                                    line += tmp_line

                        #save prepared data
                        prepared_data_file_path = os.path.join(prepared_data_path, prepared_file_name)
                        write_file(prepared_date, prepared_data_file_path)

            #save statistic file
            print('end_of_save statistic')
            write_file(statistic, statistic_file_path)

    def _get_dataset(self, idx):
        if idx in self.current_dataset_ids:
            return self.data_set

        for file_name in self.data_statistic:
            if idx in self.data_statistic[file_name]:
                self.current_dataset_ids = self.data_statistic[file_name]
                prepared_data_path = os.path.join(
                    self.config.preprocessed_data_path, self.set_type.name
                )
                prepared_data_file_path = os.path.join(prepared_data_path, file_name)
                self.data_set = read_file(prepared_data_file_path)
                break
        return self.data_set

    def _init_preprocessors(self):
        self.tokenizer = BPETokenizer(self.config.special_tokens, self.config.end_of_word)
        self.tokenizer.load(self.config.tokenizer_path)
        # check if data already preprocessed
        statistic_file_path = self.config.preprocessed_data_statistic_template % self.set_type.name
        if not os.path.exists(statistic_file_path):
            self._prepare_data()

        self.data_statistic = read_file(statistic_file_path)

    def get_vocabulary_size(self):
      return self.tokenizer.get_vocab_size()

    def __len__(self):
        total_cont = 0
        for file_id in self.data_statistic:
            total_cont += len(self.data_statistic[file_id])
        return total_cont

    def get_ids_group(self):
        id_groups = []
        for file_id in self.data_statistic:
            id_groups.append(self.data_statistic[file_id])
        return id_groups

    def __getitem__(self, idx: int):
        # load dataset
        data_set = self._get_dataset(idx)

        sample_data = {
            'id': data_set[idx]['id'],
            'tokens': data_set[idx]['tokens']
        }
        return sample_data
