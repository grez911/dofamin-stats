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

  y = fill_gaps(y)
  y = np.log(y)
  return x, y

def fill_gaps(y):
  '''
  Replace zeros in `y` with mean values. E.g.
  [8, 4, 0, 2, 0, 0, 0, 1] will become
  [8, 4, 1, 1, 0.25, 0.25, 0.25, 0.25].
  
  Input:
  y - array with zeros.
  
  Output:
  y - array without zeros.
  '''
  last_nonzero = len(y) - 1
  counter = 1
  for i in reversed(range(len(y))):
    if y[i] != 0:
      y[i+1:last_nonzero+1] = y[last_nonzero] / counter
      counter = 1
      last_nonzero = i
    else:
      counter += 1
  return(y)

def mse(y1, y2):
  '''
  Mean square error.

  Input:
  y1 - first numpy array;
  y2 - second numpy array.

  Output a mean square error.
  '''
  return ((y1 - y2) ** 2).mean()

def mae(y_pred, y_true):
  '''
  A mean absolute error.

  Input:
  y_pred - predicted values;
  y_true - true values.
  
  Output a mean absolute error.
  '''
  return np.abs(y_pred - y_true).mean()

def dmae(y_pred, y_true):
  '''
  Derivative of a mean absolute error.

  Input:
  y_pred - predicted values;
  y_true - true values.
    
  Output derivatives of a mean absolute error.
  '''
  return (y_pred > y_true) * 2 - 1

def find_curve(x, y, learning_rate):
  '''
  Find distribution function via gradient descend.

  Input:
  x - numpy array with values for x axis (actual data);
  y - numpy array with values for y axis (actual data);
  learning_rate - learning rate;

  Output:
  xc - numpy array with values for x axis (fitted curve).
  yc - numpy array with values for y axis (fitted curve).
  '''
  print("Making gradient descent...")
  m = len(x) # Number of training points.

  p = np.random.randn(7) * 0.1
  p[6] = -10
  dp = np.zeros(7)
  eps = 10**(-8)
  M = np.zeros(7)
  R = np.zeros(7)
  beta1 = 0.9
  beta2 = 0.999

  # a = 3.279210148249056
  # b = -0.006483609824476167
  # c = 0.5965040704470368
  # d = -10.612307583296714
  i = 0
  try:
    while True:
      # yc = a * np.exp(b * x + c) + d
      yc = p[0] * np.exp((p[1] * np.exp(p[2] * x + p[3]) + p[4]) * x + p[5]) + p[6]
      if i % 1000 == 0:
        #loss = mae(yc, y)
        loss = mse(yc, y)
        print(f"Epoch {i}: loss = {loss}")
        # print(f"a = {a}, b = {b}, c = {c}, d = {d}")
        print(p[0], p[1], p[2], p[3], p[4], p[5], p[6])
  
      # da = np.sum(2/m * (yc - y) * (np.exp(b * x + c)))
      # db = np.sum(2/m * (yc - y) * a * x * np.exp(b * x + c))
      # dc = np.sum(2/m * (yc - y) * a * b * np.exp(b * (x + c)))
      # dd = np.sum(2/m * (yc - y))
  
      dp[0] = np.sum(2/m * (yc - y) * np.exp(x * (p[1] * np.exp(p[2] * x + p[3]) + p[4]) + p[5]))
      dp[1] = np.sum(2/m * (yc - y) * p[0] * x * np.exp(x * (p[1] * np.exp(p[2] * x + p[3]) + p[4]) + p[2] * x + p[3] + p[5]))
      dp[2] = np.sum(2/m * (yc - y) * p[0] * p[1] * x**2 * np.exp(x * (p[1] * np.exp(p[2] * x + p[3]) + p[4]) + p[2] * x + p[3] + p[5]))
      dp[3] = np.sum(2/m * (yc - y) * p[0] * p[1] * x * np.exp(x * (p[1] * np.exp(p[2] * x + p[3]) + p[4]) + p[2] * x + p[3] + p[5]))
      dp[4] = np.sum(2/m * (yc - y) * p[0] * x * np.exp(x * (p[1] * np.exp(p[2] * x + p[3]) + p[4]) + p[5]))
      dp[5] = np.sum(2/m * (yc - y) * p[0] * np.exp(x * (p[1] * np.exp(p[2] * x + p[3]) + p[4]) + p[5]))
      dp[6] = np.sum(2/m * (yc - y))
  
      # da = np.sum(2/m * dmae(yc, y) * np.exp(x * (b * np.exp(c * x + d) + h) + k))
      # #db = np.sum(2/m * dmae(yc, y) * a * x * np.exp(x * (b * np.exp(c * x + d) + h) + c * x + d + k))
      # #dc = np.sum(2/m * dmae(yc, y) * a * b * x**2 * np.exp(x * (b * np.exp(c * x + d) + h) + c * x + d + k))
      # #dd = np.sum(2/m * dmae(yc, y) * a * b * x * np.exp(x * (b * np.exp(c * x + d) + h) + c * x + d + k))
      # dh = np.sum(2/m * dmae(yc, y) * a * x * np.exp(x * (b * np.exp(c * x + d) + h) + k))
      # dk = np.sum(2/m * dmae(yc, y) * a * np.exp(x * (b * np.exp(c * x + d) + h) + k))
      # #dp = np.sum(2/m * dmae(yc, y))
    
      M = beta1 * M + (1.0 - beta1) * dp
      R = beta2 * R + (1.0 - beta2) * dp**2
      m_hat = M / (1.0 - beta1**(i+1))
      r_hat = R / (1.0 - beta2**(i+1))
      p -= learning_rate * m_hat / (np.sqrt(r_hat) + eps)
      i += 1
  except KeyboardInterrupt:


    # a -= da * learning_rate * 100000
    # b -= db * learning_rate * 100000
    # c -= dc * learning_rate * 10000
    # d -= dd * learning_rate * 100000
    # h -= dh * learning_rate
    # k -= dk * learning_rate * 1000
    # p -= dp * learning_rate * 1000

  #a = np.around(a, 3)
  #b = np.around(b, 3)
    print("-------")
    # print("Found solution for the function y(x) = a * np.exp(b * x + c) + d:")
    #print("Found solution for the function y(x) = a * e^((b * e^(c * x + d) + h) * x + k) + p:")
    print(f"p[0] = {p[0]}")
    print(f"p[1] = {p[1]}")
    print(f"p[2] = {p[2]}")
    print(f"p[3] = {p[3]}")
    print(f"p[4] = {p[4]}")
    print(f"p[5] = {p[5]}")
    print(f"p[6] = {p[6]}")
    xc = np.logspace(np.log10(0.5), np.log10(max(x)))
    #yc = a * np.exp(b * xc + c) + d
    yc = p[0] * np.exp((p[1] * np.exp(p[2] * xc + p[3]) + p[4]) * xc + p[5]) + p[6]
    #return xc, yc, (a, b, c, d)
    return xc, yc, p

def plot(x, y):
  '''
  Draw plot.

  Outputs nothing.
  '''
  fig, ax = plt.subplots()
  #ax.plot(x, y, 'b.')
  ax.semilogx(x, y, 'b.')
  xc, yc, params = find_curve(x, y, learning_rate=1 * 10**(-3))
  ax.semilogx(xc, yc, '#ff7f0e')#, label=f'${params[0]}e^{{{params[1]}x}}$')
  #ax.plot(xc, yc, '#ff7f0e')
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
