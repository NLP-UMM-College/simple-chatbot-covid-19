from bs4 import BeautifulSoup
import requests
from requests_html import HTMLSession

link = requests.get('https://kawalcovid19.id/faq')

soup = BeautifulSoup(link.content, 'html.parser').get_text(separator='\n')

with open('data2.txt', 'w', encoding='utf-8') as f:
    f.write(soup)
