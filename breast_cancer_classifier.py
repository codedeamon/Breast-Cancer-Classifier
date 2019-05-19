import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
import os
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.classification import _prf_divide

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import np_version
from sklearn.utils.multiclass import unique_labels
from numpy import bincount
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.classification import _prf_divide
from sklearn.feature_selection import SelectKBest, f_classif

RANDOM_STATE = 14


def convert_class(item):
	if item == "car":
		return 1
	elif item == "fad":
		return 2
	elif item == "mas":
		return 3
	elif item == "gla":
		return 4
	elif item == "con":
		return 5
	elif item == "adi":
		return 6


def bincount(X, weights=None, minlength=None):
    """Replacing np.bincount in numpy < 1.6 to provide minlength."""
    result = np.bincount(X, weights)
    if len(result) >= minlength:
        return result
    out = np.zeros(minlength, np.int)
    out[:len(result)] = result
    return out

if np_version[:2] < (1, 6):
    bincount = bincount
else:
    bincount = np.bincount

def g_mean(y_true, y_pred, labels=None, correction=0.01):
    present_labels = unique_labels(y_true, y_pred)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,assume_unique=True)])

    le = LabelEncoder()
    le.fit(labels)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    sorted_labels = le.classes_

    # labels are now from 0 to len(labels) - 1 -> use bincount
    tp = y_true == y_pred
    tp_bins = y_true[tp]

    if len(tp_bins):
        tp_sum = bincount(tp_bins, weights=None, minlength=len(labels))
    else:
        # Pathological case
        true_sum = tp_sum = np.zeros(len(labels))

    if len(y_true):
        true_sum = bincount(y_true, weights=None, minlength=len(labels))

    # Retain only selected labels
    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]

    recall = _prf_divide(tp_sum, true_sum, "recall", "true", None, "recall")
    recall[recall == 0] = correction

    return sp.stats.mstats.gmean(recall)




'''
================================================================================================
Normalizing the data
================================================================================================
'''
# normalizeData function normalizes our data values
def normalizeData(filenameIn, filenameOut):
	myInput = pd.read_excel(filenameIn, 1, converters = {'Class':convert_class})

	#normalizing
	myInput.ix[:, 2:] = myInput.ix[:, 2:].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

	#myInput.to_excel(filenameOut, index=False)
	return myInput


my_norm_dta = normalizeData("BreastTissue.xlsx", "normalized.xlsx")

# lets define our feature data and the target data
data = my_norm_dta.ix[:, 2:]
target = my_norm_dta.ix[:, 1]


# with KFold we will shuffle the data randomly and then split it into 5 folds
k_fold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# here we make our scoring metric: geometric mean, which is defined above
scoring = make_scorer(g_mean)

#========================== 3 : linear SVM implementation ======================================
c_scores = []
max_score = 0
max_C = 1


# here we search for the best C value, using linear kernel
for i in range(1, 200, 5):
	clf = svm.SVC(kernel='linear', C=i)
	score = cross_val_score(clf, data, target, cv=k_fold, scoring=scoring)
	ms = score.mean()
	#print("the g_mean score of C = ", i, " is ", ms)
	c_scores.append(ms)
	if ms > max_score:
		max_score = ms
		max_C = i

print("scores are ", c_scores)
print("max score was ", max_score, " with C = ", max_C)

plt.figure(1)
plt.plot(range(1,200,5),c_scores)
plt.xlabel('C Values for SVM linear')
plt.ylabel('Geometric Mean Score')
plt.show()


# now lets search for the best gamma value
gamma_scores = []
max_score = 0
best_gamma = 0.5

gamma = 0.5
# here we search for the best gamma value, using rbf kernel
while gamma <= 10:
	clf = svm.SVC(kernel='rbf', gamma=gamma, C=max_C)
	score = cross_val_score(clf, data, target, cv=k_fold, scoring=scoring)
	ms = score.mean()
	#print("the g_mean score of gamma = ", gamma, " is ", ms)
	gamma_scores.append(ms)
	if ms > max_score:
		max_score = ms
		best_gamma = gamma
	gamma += 0.5

print("scores are ", gamma_scores)
print("max score was ", max_score, " with gamma = ", best_gamma)

plt.figure(2)
plt.plot(np.arange(0,10, 0.5), gamma_scores)
plt.xlabel('Gamma Values for SVM RBF')
plt.ylabel('Geometric Mean Score')
plt.show()

# ======================= KNN Classifier =======================================================
k_n = 3
best_k = 3
max_k_score = 0
k_scores = []
while k_n < 16:
	knn = KNeighborsClassifier(n_neighbors=k_n)
	score = cross_val_score(knn, data, target, cv=k_fold, scoring=scoring)
	ms = score.mean()
	#print("the g_mean score of knn for k =  ", k_n, " is ", ms)
	k_scores.append(ms)
	if ms > max_k_score:
		max_k_score = ms
		best_k = k_n
	k_n += 1

print("knn mean scores are ", k_scores)
print("max score was ", max_k_score, " with k = ", best_k)

plt.figure(3)
plt.plot(range(3,16), k_scores)
plt.xlabel('K Values for KNN')
plt.ylabel('Mean Score')
plt.show()

#====================== Gaussian Naive Bayes Classifier =========================================
gnb = GaussianNB()
score = cross_val_score(gnb, data, target, cv=k_fold, scoring=scoring)
ms = score.mean()
print("the mean score of Naive Bayes is ", ms)



'''
=============================================================================================

Now let's implement Student t-test for each characteristic

=============================================================================================
'''
del my_norm_dta['Case #']

featureSelector = SelectKBest(f_classif, k=4)
Xtrunc = featureSelector.fit_transform(data, target)
print(Xtrunc)

k_n = 3
best_k = 3
max_k_score = 0
k_scores = []
while k_n < 16:
	knn = KNeighborsClassifier(n_neighbors=k_n)
	score = cross_val_score(knn, Xtrunc, target, cv=k_fold, scoring=scoring)
	ms = score.mean()
	#print("the g_mean score of knn for k =  ", k_n, " is ", ms)
	k_scores.append(ms)
	if ms > max_k_score:
		max_k_score = ms
		best_k = k_n
	k_n += 1

print("knn mean scores are ", k_scores)
print("max score was ", max_k_score, " with k = ", best_k)

plt.figure(4)
plt.plot(range(3,16), k_scores)
plt.xlabel('K Values for KNN')
plt.ylabel('Mean Score')
plt.show()
