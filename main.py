#!/usr/bin/env python3
# Project page: https://github.com/grez911/dofamin-stats

import requests
import pickle
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

def find_curve(x, y):
  '''
  Find distribution function via gradient descend.

  Input:
  x - numpy array with values for x axis (actual data);
  y - numpy array with values for y axis (actual data).

  Output:
  xc - numpy array with values for x axis (fitted curve);
  yc - numpy array with values for y axis (fitted curve).
  '''
  #ax.semilogx(t, 60/t)
  print("Making gradient descent...")
  #xc = np.arange(0.5, 500.5, 1)
  m = len(x) # Number of training points.
  learning_rate = 0.001

  # a = 1
  # b = 1
  # c = 0
  # for i in range(100000):
    # yc = a * np.exp(-b * (x - c))
    # if i % 1000 == 0:
      # print(f"Step {i}: loss = {mse(yc, y)}")




  Output:
  xc - numpy array with values for x axis (fitted curve);
  yc - numpy array with values for y axis (fitted curve).
  yc - numpy array with values for y axis (fitted curve).
  '''
  #ax.semilogx(t, 60/t)
  print("Making gradient descent...")
  #xc = np.arange(0.5, 500.5, 1)
  m = len(x) # Number of training points.
  learning_rate = 0.001

  # a = 1
  # b = 1
  # c = 0
  # for i in range(100000):
    # yc = a * np.exp(-b * (x - c))
    # if i % 1000 == 0:
      # print(f"Step {i}: loss = {mse(yc, y)}")
      # print(a,b,c)
    # da = np.sum(2/m * (yc - y) * (np.exp(-b * (x - c))))
    # db = np.sum(-2/m * (yc - y) * a * (x - c) * np.exp(-b * (x - c)))
    # dc = np.sum(2/m * (yc - y) * a * b * np.exp(-b * (x - c)))
    # a -= da * learning_rate
    # b -= db * learning_rate
    # c -= dc * learning_rate

  # a = 0.9
  # b = 1
  # c = 1
  # d = 0.01
  # for i in range(200000):
    # yc = a**(b * x + c) + d
    # if i % 2000 == 0:
      # print(f"Step {i}: loss = {mse(yc, y)}")
    # da = np.sum(2/m * (yc - y) * (b * x + c) * a**(b * x + c - 1))
    # db = np.sum(2/m * (yc - y) * x * np.log(a) * a**(b * x + c))
    # dc = np.sum(2/m * (yc - y) * np.log(a) * a**(b * x + c))
    # dd = np.sum(2/m * (yc - y))
    # a -= da * learning_rate
    # b -= db * learning_rate
    # c -= dc * learning_rate
    # d -= dd * learning_rate

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
  # for days in sorted(data):
    # x.append(days)
    # y.append(data[days])
  # x = np.array(x) + 0.5
  # y = np.array(y)
  for day in range(0, 500):
    x.append(days)
    if day in data:
      y.append(data[days])
    else:
      y.append(0)
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

def find_curve(x, y):
  '''
  Find distribution function via gradient descend.

  Input:
  x - numpy array with values for x axis (actual data);
  y - numpy array with values for y axis (actual data).

  Output:
  xc - numpy array with values for x axis (fitted curve);
  yc - numpy array with values for y axis (fitted curve).
  '''
  #ax.semilogx(t, 60/t)
  print("Making gradient descent...")
  #xc = np.arange(0.5, 500.5, 1)
  m = len(x) # Number of training points.
  learning_rate = 0.001

  # a = 1
  # b = 1
  # c = 0
  # for i in range(100000):
    # yc = a * np.exp(-b * (x - c))
    # if i % 1000 == 0:
      # print(f"Step {i}: loss = {mse(yc, y)}")
      # print(a,b,c)
    # da = np.sum(2/m * (yc - y) * (np.exp(-b * (x - c))))
    # db = np.sum(-2/m * (yc - y) * a * (x - c) * np.exp(-b * (x - c)))
    # dc = np.sum(2/m * (yc - y) * a * b * np.exp(-b * (x - c)))
    # a -= da * learning_rate
    # b -= db * learning_rate
    # c -= dc * learning_rate

  # a = 0.9
  # b = 1
  # c = 1
  # d = 0.01
  # for i in range(200000):
    # yc = a**(b * x + c) + d
  data - dictionary with days as keys and values as number of people.
      break
      break
      break
    for div in divs:
      days = int(div.text.split()[0])
      data.setdefault(days, 0)
      data[days] += 1
  return data

def prepare(data):
  '''
#!/usr/bin/env python3
# Project page: https://github.com/grez911/dofamin-stats

import requests
import pickle
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
  for day in range(0, max(data) + 1):
    x.append(day)
    if day in data:
      y.append(data[day])
    else:
      y.append(0)
  # Add half a day because in the table on site displayed floor values.
  # So such result as "2 days" actually between 2 and 3 days.
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

def find_curve(x, y):
  '''
  Find distribution function via gradient descend.

  Input:
  x - numpy array with values for x axis (actual data);
  y - numpy array with values for y axis (actual data).

  Output:
  xc - numpy array with values for x axis (fitted curve);
  yc - numpy array with values for y axis (fitted curve).
  '''
  #ax.semilogx(t, 60/t)
  print("Making gradient descent...")
  #xc = np.arange(0.5, 500.5, 1)
  m = len(x) # Number of training points.
  learning_rate = 0.001

  # a = 1
  # b = 1
  # c = 0
  # for i in range(100000):
    # yc = a * np.exp(-b * (x - c))
    # if i % 1000 == 0:
      # print(f"Step {i}: loss = {mse(yc, y)}")
      # print(a,b,c)
    # da = np.sum(2/m * (yc - y) * (np.exp(-b * (x - c))))
    # db = np.sum(-2/m * (yc - y) * a * (x - c) * np.exp(-b * (x - c)))
    # dc = np.sum(2/m * (yc - y) * a * b * np.exp(-b * (x - c)))
    # a -= da * learning_rate
    # b -= db * learning_rate
    # c -= dc * learning_rate

  # a = 0.9
  # b = 1
  # c = 1
  # d = 0.01
  # for i in range(200000):
    # yc = a**(b * x + c) + d
    # if i % 2000 == 0:
      # print(f"Step {i}: loss = {mse(yc, y)}")
    # da = np.sum(2/m * (yc - y) * (b * x + c) * a**(b * x + c - 1))
    # db = np.sum(2/m * (yc - y) * x * np.log(a) * a**(b * x + c))
    # dc = np.sum(2/m * (yc - y) * np.log(a) * a**(b * x + c))

      # print(a,b,c)
    # da = np.sum(2/m * (yc - y) * (np.exp(-b * (x - c))))
    # db = np.sum(-2/m * (yc - y) * a * (x - c) * np.exp(-b * (x - c))
    # dc = np.sum(2/m * (yc - y) * a * b * np.exp(-b * (x - c)))
    # a -= da * learning_rate
    # b -= db * learning_rate
    # c -= dc * learning_rate

  a = 0.9
  b = 1
  c = 1
  d = 0.01
  yc = a**(b * x + c) + d
  return x, yc

def plot(x, y):
  '''
  Create plot.png.

  Input:
  x - numpy array with values for x axis;
  y - numpy array with values for y axis.

  Outputs nothing.
  '''
  fig, ax = plt.subplots()
  ax.semilogx(x, y, 'b.')
  xc, yc = find_curve(x, y)
  ax.semilogx(xc, yc)
  ax.set(xlabel='nofap time (days)', ylabel='proportion of people',
         title='Nofap time distribution')
  #ax.grid(True, which='both')
  ax.grid()
  fig.savefig("plot.png")
  plt.show()

def main():
  '''
  Main function.
  Inputs and outputs nothing.
  '''
  #pickle.dump(prepare(get_data()), open("data.p", 'wb'))
  x, y = pickle.load(open("data.p", 'rb'))
  plot(x, y)
  #total = np.sum(y)
  #avg = np.sum(x * y) / total
  #print("-----")
  #print(f"Total number of people: {total}")
  #print(f"Average nofap time: {avg:.1f} days")

if __name__ == '__main__':
  main()
