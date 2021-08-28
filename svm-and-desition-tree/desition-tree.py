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
    "/tinsinh/svm-and-desition-tree/train_pos-T_fscore_sorted.csv")
X1 = file_pos.to_numpy()
X1 = X1[:, 1:]
y1 = np.ones(X1.shape[0])
file_neg = pd.read_csv(
    "/tinsinh/svm-and-desition-tree/train_neg-T_fscore_sorted.csv")
X0 = file_neg.to_numpy()
X0 = X0[:, 1:]
y0 = np.zeros(X0.shape[0])
X_train = np.concatenate((X0, X1), axis=0)
y_train = np.concatenate((y0, y1))
print(X_train.shape)
print(y_train.shape)
test_pos = pd.read_csv("/tinsinh/svm-and-desition-tree/test_pos-T_fscore_sorted.csv")
test_neg = pd.read_csv("/tinsinh/svm-and-desition-tree/test_neg-T_fscore_sorted.csv")
X_test0 = test_neg.to_numpy()[:, 1:]
X_test1 = test_pos.to_numpy()[:, 1:]
y_test0 = np.zeros(X_test0.shape[0])
y_test1 = np.ones(X_test1.shape[0])
X_test = np.concatenate((X_test0, X_test1), axis=0)
y_test = np.concatenate((y_test0, y_test1))
# model = SVC(kernel='rbf', C=50, class_weight={0: X1.shape[0]/X_train.shape[0], 1: X0.shape[0]/X_train.shape[0]})
# model.fit(X_train, y_train)
# y_pred1 = model.predict(X_test)
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred1))
# print(classification_report(y_test, y_pred1))
# print(roc_auc_score(y_test, y_pred1))

# save the model to disk
#filename = 'model_R.sav'
#pickle.dump(model, open(filename, 'wb'))
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

tmp = [10,20,30,40,50,100,300,500,700,900,1000,1500,2000]
tmp1 = [10,15,20,25,30,35,40,45,50,55,60,65,70,100]
for x in tmp1:
  print(x)
  bag_clf = BaggingClassifier(
      DecisionTreeClassifier(), n_estimators=x,
      max_samples=200, bootstrap=False, random_state=40)
  bag_clf.fit(X_train, y_train)
  y_pred = bag_clf.predict(X_test)
  from sklearn.metrics import accuracy_score
 # print(accuracy_score(y_test, y_pred))
  print(classification_report(y_test, y_pred))
  print(roc_auc_score(y_test, y_pred))