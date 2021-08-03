# file: ex3_2
# Name: Eliyho Tsuri
# Id: 201610672

"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""
from __future__ import print_function
import logging
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_lfw_people
print(__doc__)
# ------------------------- print_and_show ---------------------------
# This function print and show the names and the picturs of the peopleS
def print_and_show(variances):
    x = [i * 100 for i in variances]
    print(variances[0])
    bar_range = range(150)
    print(bar_range)
    plt.plot(bar_range, x)
    plt.title('Variances vs. PC #')
    plt.xlabel('PC#')
    plt.ylabel('Variances[%]')
    plt.xlim(min(bar_range), max(bar_range))
    plt.ylim(min(x))
    plt.show()
    print("Projecting the input data on the eigenfaces orthonormal basis")
# ------------------------- plot_gallery -------------------------
# This function defind the gallery of the program the window and the names
def plot_gallery(images, titles, h, w, n_row=6, n_col=7):
    plt.figure(figsize=(1.45 * n_col, 1.5 * n_row))
    plt.subplots_adjust(bottom=0.03, left=.01, right=.99, top=.93, hspace=.36)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)       
        plt.xticks(())
        plt.yticks(())
    plt.show()
# --------------------------- def_title ------------------------------
# This function defind the titels of the windows on the program
def def_title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
# ---------------------- print_main_table ------------------------
def print_main_table(n_samples,n_classes):
     print("Total dataset size:")
     print("n_samples: %d" % n_samples)
     print("n_features: %d" % n_samples)
     print("n_classes: %d" % n_classes)
# ---------------------- print_titels ------------------------
# This function print the titels on the terminal
def print_titels():
     print('----------------------------------------------------------------------------------')
     print('Train a SVM classification model on the original data')
     print("Fitting the classifier to the training set\n\n")
     t0 = time()
     print("The grid Search fit, original data, done in %0.3fs" % (time() - t0))
     print("Best estimator found by grid search:")
     print("Predicting people's names on the test set, original data")
# ============================= Main =============================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
print_main_table(n_samples,n_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
n_components = 150
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
print("done in %0.3fs" % (time() - t0))
eigenfaces = pca.components_.reshape((n_components, h, w))
print_and_show(pca.explained_variance_ratio_) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True).fit(X_train)
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))
print("Fitting the classifier to the training set\n\n")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("\n\nSearch fit done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))             
plot_gallery(X_test, [def_title(y_pred, y_test, target_names, i)for i in range(y_pred.shape[0])]  , h, w)
plot_gallery(eigenfaces, ["eigenface %d" % i for i in range(eigenfaces.shape[0])], h, w)
print_titels()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))