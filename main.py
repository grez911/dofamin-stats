#!/usr/bin/env python3
# Project page: https://github.com/grez911/dofamin-stats

import requests
import sys

try:
  from bs4 import BeautifulSoup
except:
  print('Please, install BeautifulSoup4. Do this: pip install beautifulsoup4')
  sys.exit(1)

page = requests.get("https://dofamin.org/").content.decode('UTF-8')
soup = BeautifulSoup(page, 'html.parser')

soup.find_all('div', class_="nofap-ranks__element nofap-ranks__element_period")
