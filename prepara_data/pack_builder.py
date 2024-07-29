import os
from multiprocessing import Process

SOURCE_DIR = 'D:\\DataSet\\row_data\\classic_fb2\\pre_150_1000'
TARGET_DIR = 'D:\\DataSet\\row_data\\classic_fb2\\pack_150_1000'

total_min_lines=0
def store_part(p_index, data, total_min_lines):
    with open(os.path.join(TARGET_DIR, 'part_c_'+str(p_index)+'.txt'), 'w', encoding='utf-8') as part_file:
        totol = 0
        for item in data:
            if len(str(item)) < 200 :
                #print(item)
                totol+=1
                total_min_lines+=1
            #if item.find('———') != -1:
                #print(item)

            part_file.write(item)
            part_file.write('\n<|end|>\n')
        print(f'part{part_index} >>  total {totol}')

TOTAL_ROWS_BY_PART = 10000
END_OF_SEQ = '<|end|>'

row_buffer = []
part_index = 0

for root, dir, file in  os.walk(SOURCE_DIR):
    if len(file) >0:
        file_path = os.path.join(root, file[0])
        line = ''
        with open(file_path, encoding='utf-8') as f:
            for data_line in f:
                data_line = data_line.strip()
                if data_line != END_OF_SEQ:
                    line += data_line + ' '
                else:
                    row_buffer.append(line.strip())
                    line = ''

                if len(row_buffer) >= TOTAL_ROWS_BY_PART:
                    store_part(part_index, row_buffer, total_min_lines)
                    row_buffer = []
                    part_index += 1

store_part(part_index, row_buffer, total_min_lines)

print(f'total_min_lines >>>> {total_min_lines}')