#!/usr/bin/env python

import numpy as np 
from scipy.io import loadmat
import colorsys
import pickle

gscv = pickle.load(open('model.dat', 'rb'))
data = loadmat("/classes/ece2720/fpp/test_x_only.mat")
test_x = data.get("test_x")

rows = test_x.shape[0]
cols = test_x.shape[1]
dim = test_x.shape[2]
num_samples = 10000

x_hsv = np.empty((rows, cols, dim, num_samples))
for i in range(num_samples):
    for row in range(28):
        for col in range(28):
            r = int(test_x[row, col, 0, i]) / 255.
            g = int(test_x[row, col, 1, i]) / 255. 
            b = int(test_x[row, col, 2, i]) / 255.
            nir = int(test_x[row, col, 3, i]) / 255.
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            x_hsv[row, col, 0, i] = h 
            x_hsv[row, col, 1, i] = s 
            x_hsv[row, col, 2, i] = v 
            x_hsv[row, col, 3, i] = nir 

x_hsv_avg = np.empty((8, num_samples))
for i in range(num_samples):
    values_lst = [] 
    values_lst.append(np.mean(x_hsv[:, :, 0, i]))
    values_lst.append(np.mean(x_hsv[:, :, 1, i]))
    values_lst.append(np.mean(x_hsv[:, :, 2, i]))
    values_lst.append(np.mean(x_hsv[:, :, 3, i]))
    values_lst.append(np.std(x_hsv[:, :, 0, i]))
    values_lst.append(np.std(x_hsv[:, :, 1, i]))
    values_lst.append(np.std(x_hsv[:, :, 2, i]))
    values_lst.append(np.std(x_hsv[:, :, 3, i]))
    for j in range(8):
        x_hsv_avg[j, i] = values_lst[j]  

y_pred = gscv.predict(x_hsv_avg.T)

L = ['barren land', 'trees', 'grassland', 'none']
s = ','.join([L[t] for t in y_pred])
f = open('landuse.csv', 'w')
f.write(s)
f.close()





