#!/usr/bin/env python3
# Project page: https://github.com/grez911/dofamin-stats

import requests
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup

def get_data():
  '''
  Get data from the site.

  Inputs nothing.
  
  Output:
  data - dictionary with days as keys and values as number of people.
  '''
  page = 0
  data = dict()
  while True:
    page += 1
    print(f"Collecting data from page #{page}...")
    html = requests.get(f"https://dofamin.org/index.php?v=Main&page={page}")
    html = html.content.decode('UTF-8')
    soup = BeautifulSoup(html, 'html.parser')
    divs = soup.find_all('div', class_="nofap-ranks__element "
                                     + "nofap-ranks__element_period")
    if len(divs) == 0:
      break
    for div in divs:
      days = int(div.text.split()[0])
      data.setdefault(days, 0)
      data[days] += 1
  return data

def prepare(data):
  '''
  Make lists for x and y axis for plotting.

  Input:
  data - dictionary with days as keys and values as number of people.

  Output:
  x - numpy array with values for x axis;
  y - numpy array with values for y axis.
  '''
  x = []
  y = []
  for days in sorted(data):
    x.append(days)
    y.append(data[days])
  x = np.array(x)
  y = np.array(y)
  return x, y

def plot(x, y):
  '''
  Create plot.png.

  Input:
  x - numpy array with values for x axis;
  y - numpy array with values for y axis.

  Outputs nothing.
  '''
  print("Making plot.png...")
  t = np.arange(1, 500, 1)
  fig, ax = plt.subplots()
  ax.semilogx(x, y)
  ax.semilogx(t, 60/t)
  ax.set(xlabel='nofap time (days)', ylabel='count (humans)')
  ax.grid(True, which='both')
  fig.savefig("plot.png")

def main():
  '''
  Main function.
  Inputs and outputs nothing.
  '''
  x, y = prepare(get_data())
  plot(x, y)
  total = np.sum(y)
  avg = np.sum(x * y) / total
  print("-----")
  print(f"Total number of people: {total}")
  print(f"Average nofap time: {avg:.1f} days")

if __name__ == '__main__':
  main()
