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
    html = html.content.decode('UTF-8', errors='ignore')
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
  for day in range(0, max(data) + 1):
    x.append(day)
    if day in data:
      y.append(data[day])
    else:
      y.append(0)
  # Add half a day because on the sitedisplayed floor values,
  # e.g. "2 days" means time between 2 and 3 days, so we are
  # taking an average.
  x = np.array(x) + 0.5
  y = np.array(y)
  y = y / np.sum(y)
  return x, y

def mse(y1, y2):
  '''
  Mean square error.

  Input:
  y1 - first numpy array;
  y2 - second numpy array.

  Output a mean square error.
  '''
  return ((y1 - y2) ** 2).mean()

def find_curve(x, y, learning_rate, epochs):
  '''
  Find distribution function via gradient descend.

  Input:
  x - numpy array with values for x axis (actual data);
  y - numpy array with values for y axis (actual data);
  learning_rate - learning rate;
  epochs - number of epochs (we are doing a full batch training).

  Output:
  xc - numpy array with values for x axis (fitted curve).
  yc - numpy array with values for y axis (fitted curve).
  '''
  print("Making gradient descent...")
  m = len(x) # Number of training points.
  a = 0.206
  b = -0.258
  c = 0.0583
  d = 0.0922
  h = 0
  k = 0
  for i in range(epochs+1):
    if i % 1000 == 0:
      yc = a * np.exp((b * np.exp(c * x) + d) * x + h) + k
      loss = mse(y, yc)
      print(f"Epoch {i}: loss = {loss}")
      print(f"a = {a}, b = {b}, c = {c}, d = {d}, h = {h}, k = {k}")
    da = np.sum(2/m * (yc - y) * np.exp(x * (b * np.exp(c * x) + d) +h))
    db = np.sum(2/m * (yc - y) * a * x * np.exp(x * (b * np.exp(c * x) + d) + c * x + h))
    dc = np.sum(2/m * (yc - y) * a * b * x**2 * np.exp(x * (b * np.exp(c * x) + d) + c * x + h))
    dd = np.sum(2/m * (yc - y) * a * x * np.exp(x * (b * np.exp(c * x) + d) + h))
    dh = np.sum(2/m * (yc - y) * a * np.exp(x * (b * np.exp(c * x) + d) + h))
    dk = np.sum(2/m * (yc - y))
    a -= da * learning_rate
    b -= db * learning_rate
    c -= dc * learning_rate
    d -= dd * learning_rate
#    h -= dh * learning_rate
#    k -= dk * learning_rate
  #a = np.around(a, 3)
  #b = np.around(b, 3)
  print("-------")
  print("Found solution for the function y(x) = a * e^((b * e^(c * x) + d) * x + h) + k:")
  print(f"a = {a}")
  print(f"b = {b}")
  print(f"c = {c}")
  print(f"d = {d}")
  print(f"h = {h}")
  print(f"k = {k}")
  xc = np.logspace(np.log10(0.5), np.log10(max(x)))
  yc = a * np.exp((b * np.exp(c * xc) + d) * xc + h) + k
  return xc, yc, (a, b, c, d, h, k)

def plot(x, y):
  '''
  Draw plot.

  Outputs nothing.
  '''
  fig, ax = plt.subplots()
  ax.semilogx(x, y, 'b.')
  xc, yc, params = find_curve(x, y, learning_rate=100 * 10**(-3), epochs=1 * 10**5)
  ax.semilogx(xc, yc, '#ff7f0e')#, label=f'${params[0]}e^{{{params[1]}x}}$')
  plt.title("Nofap time distribution")
  plt.xlabel("nofap time (days)")
  plt.ylabel("proportion of people")
  ax.grid()
  plt.legend()
  plt.show()

def main():
  '''
  Main function.
  Inputs and outputs nothing.
  '''
  plt.rc('font', size=16)
#  data = get_data()

  import pickle
#  pickle.dump(data, open("data.p", "wb"))
  data = pickle.load(open("data.p", "rb" ))
  x, y = prepare(data)
  plot(x, y)

if __name__ == '__main__':
  main()
