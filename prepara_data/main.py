import concurrent
import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process

from fb2_parser import FB2_Parser
from os import listdir
from os.path import isfile, join

from utils import save_pre_build_data


min_seq_len = 150
max_seq_len = 1000

ROOT_PATH = 'D:\\DataSet\\row_data\\filatov'
#ROOT_PATH = 'C:\\dataSet\\row_data'
#ROOT_FB2_PATH = join(ROOT_PATH, 'fb2')
ROOT_FB2_PATH = join(ROOT_PATH, 'row_data_fb2')

#ROOT_PRE_BUILD_PATH = join(ROOT_PATH, 'pre_build')
ROOT_PRE_BUILD_PATH = join(ROOT_PATH, 'pre_'+str(min_seq_len)+'_'+str(max_seq_len))


STATISTIC_FILE_PATH = join(ROOT_PATH, 'statistics.csv')
DATA_LINE_SEPARATOR = '<|end|>'



def parse_file(data):
    file_path = data['file_path']
    target_path = data['target_file_path']
    parser = FB2_Parser(max_sec_len=max_seq_len, min_sec_len=min_seq_len)
    parser.parse_doc(file_path)
    p_len = len(parser.result_parsing)
    save_pre_build_data(target_path, parser.result_parsing, DATA_LINE_SEPARATOR)
    print(f'parsed {p_len} lines : {file_path}')
    return p_len

data_rows = []
files_list = []

#для конвертации файл - файл
def run_parse():
    for folder in listdir(ROOT_FB2_PATH):
        data_folder_path = join(ROOT_FB2_PATH, folder)
        for filename in listdir(data_folder_path):
            file_path = join(data_folder_path, filename)
            if isfile(file_path):
                # if file does not processed before
                target_folder =join(ROOT_PRE_BUILD_PATH, folder)
                if not os.path.exists(target_folder):
                    os.mkdir(target_folder)
                target_file_path = join(target_folder, filename+'_data')
                if not os.path.exists(target_file_path):
                    files_list.append({'file_path' : file_path, 'target_file_path' :target_file_path})

    with ProcessPoolExecutor(max_workers=10) as executer:
        print('init>>' + file_path)
        data_precess_feature = executer.map(parse_file, files_list)

    for future in concurrent.futures.as_completed(data_precess_feature):
        print(future.result())




if __name__ == '__main__':
    run_parse()
    print('END')


