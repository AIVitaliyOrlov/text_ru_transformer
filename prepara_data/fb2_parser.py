import os
import lxml
from bs4 import BeautifulSoup as Soup

class FB2_Parser():
    def __init__(self, min_sec_len=200, max_sec_len=2000):
        self.max_seq_len = max_sec_len
        self.min_seq_len = min_sec_len
        self.result_parsing = []

    def parse_doc(self, doc_path):
        if os.path.exists(doc_path):
            try:
                with open(doc_path, encoding='utf-8') as doc_file:
                    soup = Soup(doc_file.read(), features="html.parser")
                    for body in soup.find_all('body'):
                        self.parse_body(body)
            except:
                try:
                    with open(doc_path, encoding='windows-1251') as doc_file:
                        content = doc_file.read()
                        soup = Soup(content, features="html.parser")
                        for body in soup.find_all('body'):
                            self.parse_body(body)
                except :
                    print(f'error in parsing file{doc_path}')

    def parse_body(self, body):
        for section in body.find_all('section', recursive=False):
            self.parse_sections(section)

    def parse_sections(self, section):

        sub_section = section.find_all('section', recursive=False)
        if len(sub_section) > 0 :
            for sub_section in sub_section:
                self.parse_sections(sub_section)
        else:
            self.parse_single_section(section)


    def parse_single_section(self, section):

        tmp_line = ''
        paragraph_list = section.find_all('p', recursive=False)
        for paragraph in paragraph_list:
            new_p = self.clean_text(paragraph)
            if not self.is_skip(new_p):
                if self.is_can_join(tmp_line, new_p):
                    tmp_line += new_p
                else:
                    self.add_line_to_result(tmp_line)
                    tmp_line = new_p

        self.add_line_to_result(tmp_line)

        # for poem
        for poem in section.find_all('poem', recursive=False):
            tmp_line = ''
            for stanza in poem.find_all('stanza', recursive=False):
                tmp_line += self.clean_text(stanza)
            self.add_line_to_result(tmp_line)

    def add_line_to_result(self, line):
        if not self.is_skip(line):
            if len(line) > self.max_seq_len*1.5:
                while len(line) > self.max_seq_len*1.5:
                    index_for_split = line.rfind('.', 0, self.max_seq_len)
                    if index_for_split == -1:
                        index_for_split = line.rfind('!', 0, self.max_seq_len)

                    if index_for_split == -1:
                        index_for_split = line.rfind(';', 0, self.max_seq_len)

                    if index_for_split == -1:
                        index_for_split = line.rfind(',', 0, self.max_seq_len)

                    if index_for_split == -1:
                        index_for_split = line.rfind(' ', 0, self.max_seq_len)

                    self.result_parsing.append(line[:index_for_split+1].strip())
                    line = line[index_for_split+1:]

                self.result_parsing.append(line.strip())
            elif len(line) > self.min_seq_len:
                self.result_parsing.append(line.strip())

    def clean_text(self, paragraph):
        [x.extract() for x in paragraph.findAll('a')]
        text = paragraph.text.strip()
        return (text.replace(u'\xa0', u' ')
                .replace(u'\n', u' ')
                .replace(u'\t', u' ')
                .replace(u"\u2003", u'')
                .replace('. . . . . . . . . . . . . . .', ''))

    def is_can_join(self, str1, str2):

        len1 = len(str1)
        len2 = len(str2)
        #return len2 > self.min_seq_len and (len1 + len1 < self.max_seq_len)
        return len1 < self.max_seq_len

    def is_skip(self, data_line):
        skip_text = [
            'Текст предоставлен ООО',
            '—————',
            '———',
            'Литрес',
            'ЛитРес',
            'Прочитайте эту книгу целиком',
            'WebMoney',
            'QIWI',
            'PayPal',
            'Конец ознакомительного фрагмента'
        ]

        for skip in skip_text:
            if data_line.find(skip) != -1:
                return True
        return False