import os


def save_pre_build_data(file_path, data, separator):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(item)
            file.write('\n'+separator+'\n')
    return file_path