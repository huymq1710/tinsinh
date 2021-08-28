import numpy as np
import sklearn
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
# from imblearn.under_sampling import RandomUnderSampler
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pickle


file_pos = pd.read_csv(
    "/tinsinh/svm-and-desition-tree/train_pos-R_fscore_sorted.csv")
X1 = file_pos.to_numpy()
X1 = X1[:, 1:]
y1 = np.ones(X1.shape[0])
file_neg = pd.read_csv(
    "/tinsinh/svm-and-desition-tree/train_neg-R_fscore_sorted.csv")
X0 = file_neg.to_numpy()
X0 = X0[:, 1:]
y0 = np.zeros(X0.shape[0])
X_train = np.concatenate((X0, X1), axis=0)
y_train = np.concatenate((y0, y1))
print(X_train.shape)
print(y_train.shape)
test_pos = pd.read_csv("/tinsinh/svm-and-desition-tree/test_pos-R_fscore_sorted.csv")
test_neg = pd.read_csv("/tinsinh/svm-and-desition-tree/test_neg-R_fscore_sorted.csv")
X_test0 = test_neg.to_numpy()[:, 1:]
X_test1 = test_pos.to_numpy()[:, 1:]
y_test0 = np.zeros(X_test0.shape[0])
y_test1 = np.ones(X_test1.shape[0])
X_test = np.concatenate((X_test0, X_test1), axis=0)
y_test = np.concatenate((y_test0, y_test1))
model = SVC(kernel='rbf', C=800, class_weight={0: X1.shape[0]/X_train.shape[0], 1: X0.shape[0]/X_train.shape[0]})
model.fit(X_train, y_train)
y_pred1 = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred1))
print(classification_report(y_test, y_pred1))
print(roc_auc_score(y_test, y_pred1))

# save the model to disk
#filename = 'model_R.sav'
#pickle.dump(model, open(filename, 'wb'))