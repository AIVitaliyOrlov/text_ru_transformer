from bs4 import BeautifulSoup as Soup
from lxml.html.soupparser import fromstring

from fb2_parser import FB2_Parser

doc_path ='C:\\dataSet\\row_data\\avidreaders.ru\\unzip\\12-ulev-ili-legenda-o-tampuke.html\\avidreaders.ru__12-ulev-ili-legenda-o-tampuke.fb2'


parser = FB2_Parser(max_sec_len=2000, min_sec_len=150)
parser.parse_doc(doc_path)

for line in parser.result_parsing:
    if len(line) < 200:
        print(line)

with open(doc_path, encoding='windows-1251') as doc_file:
    content = doc_file.read()
    #soup = fromstring(content)
    soup = Soup(content, features="html.parser")
    bodys = soup.find_all('body')
    print(bodys)