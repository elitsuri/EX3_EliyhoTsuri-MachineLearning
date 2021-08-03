# file: ex3_1
# Name: Eliyho Tsuri
# Id: 201610672
# ====================================================================
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
# --------------------------- nudge dataset --------------------------
# This produces a dataset 5 times bigger than the original one,
# by moving the 8x8 images in X around by 1px to left, right, down, up
def nudge_dataset(X, Y):
    my_vector = [[[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]
    shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel() 
    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in my_vector]) 
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y
# -------------------------- def_plt_window -------------------------
# This function defind the window and show the results
def def_plt_window(range,pipe):
    plt.plot(range, pipe)
    plt.plot([log] * max(range), label="Logistic Regression precision")
    plt.title('precision vs number of components')
    plt.xlabel('precision')
    plt.ylabel('number of components')
    plt.xlim(min(range), max(range))
    plt.ylim(min(pipe))
    plt.legend(loc='lower right', frameon=False)
    plt.show()
    plt.plot(range, pipe)
    plt.title('running time vs number of components')
    plt.xlabel('Running time')
    plt.ylabel('Number of components')
    plt.xlim(min(range), max(range))
    plt.ylim(min(pipe))
    plt.show()   
# -------------------------- print_and_show -------------------------
# This function getting the parameters and print and show it on the terminal 
# window and show the results on the window we defind it mean the picture 
def print_and_show(logistic, my_pipe, pipe_time, plot_range):
    logistic = linear_model.LogisticRegression(C = 100.0)
    logistic.fit(X_train, Y_train)
    precision = metrics.precision_score(Y_test, logistic.predict(X_test), average = 'weighted')
    range = [pow(i, 2) for i in range(2, plot_range)]
    def_plt_window(range,my_pipe)
# --------------------------- def window -----------------------------
# This function defind the window for show the picture of results
def def_window(j):
      plt.figure(figsize=(5.2, 5))
      for i, comp in enumerate(rbm.components_):
          plt.subplot(j, j, i + 1)
          plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,interpolation = 'nearest')                    
          plt.xticks(())
          plt.yticks(())
      plt.suptitle(str(pow(j,2)) + ' components extracted by RBM', fontsize = 16)
      plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
# ----------------------------- run ex1 ------------------------------
# This is the main function of the program the function 
# runing the ex1 and respon on the program and do the demands
def run_ex1():
     for j in range(2, 21):
          rbm.learning_rate = 0.06
          rbm.n_iter = 20
          rbm.n_components = pow(j,2)
          logistic.C = 6000.0
          start_time = time.time()
          classifier.fit(X_train, Y_train)
          my_time.append(time.time() - start_time)
          my_precision.append(metrics.precision_score(Y_test, classifier.predict(X_test), average='weighted'))
          def_window(j)                
          plt.show()
# =============================== Main ===============================
digits = datasets.load_digits()
my_precision = []
my_time = []
lr_precision = []
X = np.asarray(digits.data, 'float32')
X, Y = nudge_dataset(X, digits.target)
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001) 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state = 0, verbose = True)
classifier = Pipeline(steps = [('rbm', rbm), ('logistic', logistic)])
run_ex1()
print_and_show(lr_precision, my_precision, my_time, 21)