import os

from bs4 import BeautifulSoup
import requests
#import urllib.request

#graph|password

ROOT_DATA_PATH = 'row_data'
def create_dirs(path_list) :
    allPath = ROOT_DATA_PATH
    for path in path_list:
        if path != '' and path != 'https:':
            allPath = os.path.join(allPath, path)
            os.makedirs(allPath, exist_ok=True)

    return allPath




#base_domain = 'https://fictionbook.ru'
base_url = 'https://avidreaders.ru/genre/detektivy/'
session = requests.Session()


headers_get = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'Host': 'avidreaders.ru',
        'Origin': 'https://avidreaders.ru',
        'Referer': 'https://avidreaders.ru/genre/',
        'Sec-Ch-Ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'User-Name': 'root',
        'User-Roles': 'root'
}

# do login
#login_page = session.get(url='https://fictionbook.ru/pages/login/', verify=False, allow_redirects=True, headers=headers_get)
#login_page_soup = BeautifulSoup(login_page.content, 'html.parser')

#login_action_url = ''

auth_data = {
    'pre_action' : 'login',
    'ref_url' : '',
    'login' : 'graph',
    'pwd' : 'pwd'
}

headers_login = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
        'Accept-Language': 'ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7',
        'Cache-Control': 'max-age=0',
        'Connection': 'keep-alive',
        'Content-Type' : 'application/x-www-form-urlencoded',
        'Host': 'avidreaders.ru_01',
         'Origin': 'https://avidreaders.ru',
        'Referer': 'https://avidreaders.ru/genre/',
        'Sec-Ch-Ua': '"Not/A)Brand";v="8", "Chromium";v="126", "Google Chrome";v="126"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-User': '?1',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        'User-Name': 'root',
        'User-Roles': 'root'
}

#do_login = session.post('https://fictionbook.ru/', verify=False, allow_redirects=True, headers=headers_login, data_set=auth_data)

for p_index in range(59, 1000):
    session = requests.Session()
    print('start crawling page ' + str(p_index))
    pageUrl = base_url + str(p_index)
    main_page = session.get(url=pageUrl, verify=False, allow_redirects=True, headers=headers_get)
    main_soup = BeautifulSoup(main_page.content, 'html.parser')
    for book_div in main_soup.find_all('div', class_='card_info'):
        link = book_div.find('a', class_='btn')['href']
        book_page = session.get(url=link, verify=False, allow_redirects=True, headers=headers_get)
        book_soup = BeautifulSoup(book_page.content, 'html.parser')
        for download_link_box in book_soup.find_all('div', class_='format_download'):
            for links in download_link_box.find_all('a', class_='btn'):
                download_link = links.attrs['href']
                if download_link.endswith('fb2'):
                    try:
                        download_page = session.get(url=download_link, verify=False, allow_redirects=True, headers=headers_get)
                        download_page_soup = BeautifulSoup(download_page.content, 'html.parser')
                        d_box = download_page_soup.find('div', class_='dnld-info')
                        result_download_link = d_box.find('a')['href']

                        result_download_link.split('/')
                        dir_path = create_dirs(link.split('/'))
                        file_data = session.get(url=result_download_link, verify=False, allow_redirects=True, headers=headers_get )

                        with open(os.path.join(dir_path, 'data_set.fb2.zip'), 'wb') as file:
                           file.write(file_data.content)

                        print(f'file {result_download_link} download complete {p_index}' )
                        break
                    except:
                        print('error downloading')




