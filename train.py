#!/usr/bin/env python

import numpy as np 
from scipy.io import loadmat
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
# import matplotlib
# matplotlib.use('PDF')
# import matplotlib.pyplot as plt
import pandas as pd
import colorsys
import pickle

data = loadmat("/classes/ece2720/fpp/sat-4-full.mat")
test_x = data.get("test_x")
test_y = data.get("test_y")
train_x = data.get("train_x")
train_y = data.get("train_y")

train_y_shaped = []
test_y_shaped = []

for row in train_y.T:
    if (row[0] == 1):
        train_y_shaped.append(0)
    elif (row[1] == 1):
        train_y_shaped.append(1)
    elif (row[2] == 1):
        train_y_shaped.append(2)
    else:
        train_y_shaped.append(3)

for row in test_y.T:
    if (row[0] == 1): 
        test_y_shaped.append(0)
    elif (row[1] == 1): 
        test_y_shaped.append(1)
    elif (row[2] == 1): 
        test_y_shaped.append(2)
    else: 
        test_y_shaped.append(3)

# barren_im = []
# trees_im = []
# grassland_im = []
# none_im = []

# def sort_images():
#     has_1_elt = [False, False, False, False]
#     for i in range(400000):
#         if train_y_shaped[i] == 0 and has_1_elt[0] == False:
#             barren_im.append(train_x[:, :, :, i])
#             has_1_elt[0] = True
#         elif train_y_shaped[i] == 1 and has_1_elt[1] == False:
#             trees_im.append(train_x[:, :, :, i])
#             has_1_elt[1] = True
#         elif train_y_shaped[i] == 2 and has_1_elt[2] == False:
#             grassland_im.append(train_x[:, :, :, i])
#             has_1_elt[2] = True
#         elif train_y_shaped[i] == 3 and has_1_elt[3] == False:
#             none_im.append(train_x[:, :, :, i])
#             has_1_elt[3] = True
#         else:
#             pass

#         if all(has_1_elt) == True:
#             return
#     return

# # Returns NoneType, but does not matter
# s = sort_images()

# plt.imshow(np.squeeze(barren_im[0][:, :, 0:3]).astype(float))
# plt.title("Barren Land")
# plt.savefig('barren_land.pdf')
# plt.close()
# plt.imshow(np.squeeze(trees_im[0][:, :, 0:3]).astype(float))
# plt.title("Trees")
# plt.savefig('trees.pdf')
# plt.close()
# plt.imshow(np.squeeze(grassland_im[0][:, :, 0:3]).astype(float))
# plt.title("Grassland")
# plt.savefig('grassland.pdf')
# plt.close()
# plt.imshow(np.squeeze(none_im[0][:, :, 0:3]).astype(float))
# plt.title("None")
# plt.savefig('none.pdf')
# plt.close()


train_x_shaped = train_x.reshape(3136, 400000).T
test_x_shaped = test_x.reshape(3136, 100000).T

num_samples = 20000
# random_samples = train_x_shaped[np.random.choice(train_x_shaped.shape[0], 20000, replace=False), :]
test_samples = 4000
# random_test_samples = test_x_shaped[np.random.choice(test_x_shaped.shape[0], 4000, replace=False), :]
rows = train_x.shape[0]
cols = train_x.shape[1]
dims = train_x.shape[2]

x_hsv = np.empty((rows, cols, dims, num_samples))
x_hsv_test = np.empty((rows, cols, dims, test_samples))

for i in range(num_samples):
    for row in range(rows):
        for col in range(cols):
            r = int(train_x[row, col, 0, i]) / 255.
            g = int(train_x[row, col, 1, i]) / 255. 
            b = int(train_x[row, col, 2, i]) / 255.
            nir = int(train_x[row, col, 3, i]) / 255.
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            x_hsv[row, col, 0, i] = h 
            x_hsv[row, col, 1, i] = s 
            x_hsv[row, col, 2, i] = v 
            x_hsv[row, col, 3, i] = nir 

for i in range(test_samples):
    for row in range(rows):
        for col in range(cols):
            r = int(test_x[row, col, 0, i]) / 255.
            g = int(test_x[row, col, 1, i]) / 255. 
            b = int(test_x[row, col, 2, i]) / 255.
            nir = int(test_x[row, col, 3, i]) / 255.
            h, s, v = colorsys.rgb_to_hsv(r,g,b)
            x_hsv_test[row, col, 0, i] = h 
            x_hsv_test[row, col, 1, i] = s 
            x_hsv_test[row, col, 2, i] = v 
            x_hsv_test[row, col, 3, i] = nir

# Average h, s, v, nir for each image
x_hsv_avg = np.empty((8, num_samples))
x_hsv_avg_test = np.empty((8, test_samples))
for i in range(num_samples):
    new_x_vals = [] 
    new_x_vals.append(np.mean(x_hsv[:, :, 0, i]))
    new_x_vals.append(np.mean(x_hsv[:, :, 1, i]))
    new_x_vals.append(np.mean(x_hsv[:, :, 2, i]))
    new_x_vals.append(np.mean(x_hsv[:, :, 3, i]))
    new_x_vals.append(np.std(x_hsv[:, :, 0, i]))
    new_x_vals.append(np.std(x_hsv[:, :, 1, i]))
    new_x_vals.append(np.std(x_hsv[:, :, 2, i]))
    new_x_vals.append(np.std(x_hsv[:, :, 3, i]))
    for j in range(8):
        x_hsv_avg[j, i] = new_x_vals[j]

for i in range(test_samples):
    new_x_vals = [] 
    new_x_vals.append(np.mean(x_hsv_test[:, :, 0, i]))
    new_x_vals.append(np.mean(x_hsv_test[:, :, 1, i]))
    new_x_vals.append(np.mean(x_hsv_test[:, :, 2, i]))
    new_x_vals.append(np.mean(x_hsv_test[:, :, 3, i]))
    new_x_vals.append(np.std(x_hsv_test[:, :, 0, i]))
    new_x_vals.append(np.std(x_hsv_test[:, :, 1, i]))
    new_x_vals.append(np.std(x_hsv_test[:, :, 2, i]))
    new_x_vals.append(np.std(x_hsv_test[:, :, 3, i]))
    for j in range(8):
        x_hsv_avg_test[j, i] = new_x_vals[j] 
x_hsv_avg_test = x_hsv_avg_test.T      

svc = SVC(kernel='rbf')

# These are what I ran locally, but for the ECE Linux Server, 
# I only train on the best two parameters
# c_vals = [1., 5., 10., 50., 100., 200.]
# gamma_vals = [0.01, 0.1, 0.2, 0.5, 1., 2.]

c_vals = [200.]
gamma_vals = [2.]
gscv = GridSearchCV(svc,
            dict(C=c_vals,
                 gamma=gamma_vals),
                 cv=3,
                 n_jobs=1)

gscv.fit(x_hsv_avg.T, train_y_shaped[:num_samples])

pickle.dump(gscv, open('model.dat', 'wb'))

# Testing Other Algorithms. I run these locally, but not on ECE linux cluster

# clfLinSVC = LinearSVC()
# clfLinSVC.fit(x_hsv_avg.T, train_y_shaped[:num_samples])
# print("Linear SVC: " + str(clfLinSVC.score(x_hsv_avg_test, test_y_shaped[:test_samples])))

# clfLogis = LogisticRegression()
# clfLogis.fit(x_hsv_avg.T, train_y_shaped[:num_samples])
# print("Logistic Regression: " + str(clfLogis.score(x_hsv_avg_test, test_y_shaped[:test_samples])))

# clfRidge = Ridge()
# clfRidge.fit(x_hsv_avg.T, train_y_shaped[:num_samples])
# print("Ridge Regression: " + str(clfRidge.score(x_hsv_avg_test, test_y_shaped[:test_samples])))

# clfLasso = Lasso()
# clfLasso.fit(x_hsv_avg.T, train_y_shaped[:num_samples])
# print("Lasso Regression: " + str(clfLasso.score(x_hsv_avg_test, test_y_shaped[:test_samples])))

# gscv = pickle.load(open('model.dat', 'rb'))

results = pd.DataFrame(gscv.cv_results_)
# c1_vals = []
# c5_vals = []
# c10_vals = []
# c50_vals = []
# c100_vals = []
# c200_vals = []
# Cs = []

# def set_y_vals():
#     for i in range(len(c_vals)*len(gamma_vals)):
#         score = results.iloc[i].get("mean_test_score")
#         if i in range(0, 6):
#             c1_vals.append(score)
#         elif i in range(6, 12):
#             c5_vals.append(score)
#         elif i in range(12, 18):
#             c10_vals.append(score)
#         elif i in  range(18, 24):
#             c50_vals.append(score)
#         elif i in range(24, 30):
#             c100_vals.append(score)
#         else:
#             c200_vals.append(score)
        
#     Cs.append(c1_vals)
#     Cs.append(c5_vals)
#     Cs.append(c10_vals)
#     Cs.append(c50_vals)
#     Cs.append(c100_vals)
#     Cs.append(c200_vals)
#     return Cs

# y = set_y_vals()

# plt.plot(gamma_vals, y[0],label = "C: 1.0")
# plt.plot(gamma_vals, y[1], label = "C: 5.0")
# plt.plot(gamma_vals, y[2], label = "C: 10.0")
# plt.plot(gamma_vals, y[3], label = "C: 50.0")
# plt.plot(gamma_vals, y[4], label = "C: 100.0")
# plt.plot(gamma_vals, y[5], label = "C: 200.0")
# plt.legend(loc = 'lower right')
# plt.title('Scores of Grid Search Models')
# plt.xlabel('Gamma')
# plt.ylabel('Mean score')
# plt.savefig('model_scores.pdf')
# plt.close()

# I run these locally, but not for ECE linux:
# lin_score = clfLinSVC.score(x_hsv_avg_test, test_y_shaped[:test_samples])
# log_score = clfLogis.score(x_hsv_avg_test, test_y_shaped[:test_samples])
# ridge_score = clfRidge.score(x_hsv_avg_test, test_y_shaped[:test_samples])
# lasso_score = clfLasso.score(x_hsv_avg_test, test_y_shaped[:test_samples])
# score_vals = np.array([lin_score, log_score, ridge_score, lasso_score])
# x = np.arange(len(score_vals))
# ticks = ("LinearSVM", "Logistic", "Ridge", "Lasso")

# plt.bar(x, score_vals, align = 'center')
# plt.xticks(x, ticks)
# plt.ylabel("Score")
# plt.title("Scores of Other Classifiers")
# plt.savefig('other_models.pdf')
# plt.close()
