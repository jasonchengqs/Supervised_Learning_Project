#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qisen
"""
########################################
## Import supporting config and helpers
########################################
import digit_config
from digit_helpers import *

def multiclass_roc_auc_score(truth, pred, average="macro"):

    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)

    return roc_auc_score(truth, pred, average=average)

########################################
## Loading data and some EDA
########################################
data=pd.read_csv("sampled_digit.csv", sep='\t')
print('>>>>> raw dataset info:')
print("shape of raw dataset:", data.shape)
# print(data.columns.values)

print(">>>>> missing data check in raw data:")
# print(data.isnull().sum())
# labels = data.pop('label')

########################################
## Pre-processing
########################################
# Estimate missing Age data from Name
# data = data.sample(5000)
labels = data.pop('label')
X_train, X_test, y_train, y_test= train_test_split(data, labels, test_size=0.3, random_state=0)

pca=PCA(n_components=30, whiten=True)
X_train_pca=pca.fit_transform(X_train)
X_test_pca=pca.transform(X_test)
var_ratio = sum(pca.explained_variance_ratio_)
print("Explained variance: {:.3%}".format(var_ratio))

X_train = X_train_pca
X_test = X_test_pca

print("final trainning:", X_train.shape)
########################################
## Model setup
########################################
# clf_svm = SVC(kernel="rbf", C =0.1, gamma=0.05, probability=False)
clf_svm = SVC(kernel="linear", verbose=True)
# clf_svm_rbf = SVC(kernel="rbf", verbose=True)
clf_svm_rbf = SVC(kernel="rbf", C =0.1, gamma=0.05, probability=False)
########################################
## Model training
########################################
classifiers = [clf_svm, clf_svm_rbf]

log_cols = ["clf", "accuracy", "log-loss"]
log = pd.DataFrame(columns=log_cols)
for clf in classifiers:
  s1=time.time()
  clf.fit(X_train_pca, y_train)
  e1=time.time()
  name = clf.__class__.__name__
  print("="*30)
  print(name)
  print("trian_time:", e1-s1)

  print('****Results****')
  s2=time.time()
  train_predictions = clf.predict(X_test_pca)
  e2=time.time()
  print("test_time:", e2-s2)

  acc = accuracy_score(y_test, train_predictions)
  print("Accuracy: {:.4%}".format(acc))
  acc = f1_score(y_test, train_predictions, average='macro')
  print("F1: {:.4}".format(acc))
  acc = multiclass_roc_auc_score(y_test, train_predictions, average='macro')
  print("AUROC: {:.4}".format(acc))

print("="*30)

#######################################
# Parameter tuning 
#######################################

scores = ['accuracy']
sk = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
# param_grid = {'C': [2**x for x in range(-5,6)], 'gamma': [2**x for x in range(-5,6)]}
param_grid = {'C': [0.1], 'gamma': [0.05]}

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=10,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full test set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    # clf_knn = KNeighborsClassifier(clf.best_params_['n_neighbors'])
    # clf_knn.fit(X_train, y_train)
    # predictions = clf_knn.predict(X_test)
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy: {:.4%}".format(acc))
    print()

########################################
## ploting for Parameter tuning 
########################################

param_range = [0.001, 0.1, 1, 5 ,10]
# param_range = [0.005, 0.05, 0.1 , 0.5]
train_scores, val_scores = validation_curve(estimator=SVC(gamma = 0.05), X=X_train, y=y_train, param_name="C", \
  param_range=param_range, cv=10, verbose=1, scoring='accuracy')
# train_scores, val_scores = validation_curve(estimator=SVC(C = 1), X=X_train, y=y_train, param_name="gamma", \
#   param_range=param_range, cv=10, verbose=1, scoring='accuracy')
plt.figure()
plt.title("CV: Tuning of C")
# plt.title("CV: Tuning of gamma")
plt.xlabel("C")
# plt.xlabel("Gamma")
plt.ylabel("Accuracy")
plt.ylim(0,1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)
plt.grid()
plt.fill_between(param_range, train_scores_mean - train_scores_std,\
                     train_scores_mean + train_scores_std, alpha=0.1,\
                     color="r")
plt.fill_between(param_range, val_scores_mean - val_scores_std,\
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.plot(param_range, train_scores_mean, 'o-', color="r",\
         label="Training score")
plt.plot(param_range, val_scores_mean, 'o-', color="g",\
         label="Cross-validation score")

plt.legend(loc="best")


#######################################
# testing
#######################################

kernel='rbf'
C=0.1
gamma=0.05

train_sizes, train_scores, val_scores = learning_curve(estimator=SVC(kernel=kernel, C=C, gamma=gamma), \
  X=X_train, y=y_train, train_sizes=[0.1, 0.3, 0.6, 1], cv=10, verbose=1, scoring='neg_mean_squared_error', shuffle=True)

plt.figure()
plt.title("Error: SVM(kernel="+kernel+",C="+str(C)+")")
plt.xlabel("Training examples")
plt.ylabel("MSE error")
plt.ylim(0,3)
train_scores_mean = np.mean(-train_scores, axis=1)
train_scores_std = np.std(-train_scores, axis=1)
val_scores_mean = np.mean(-val_scores, axis=1)
val_scores_std = np.std(-val_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,\
                     train_scores_mean + train_scores_std, alpha=0.1,\
                     color="r")
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,\
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",\
         label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g",\
         label="Cross-validation score")

plt.legend(loc="best") 

train_sizes, train_scores, val_scores = learning_curve(estimator=SVC(kernel=kernel, C=1, gamma=gamma), \
  X= X_train, y=y_train, train_sizes=[0.1, 0.3, 0.6, 1], cv=10, verbose=1, shuffle=True)

plt.figure()
plt.title("Accuracy: SVM(kernel="+kernel+",C="+str(C)+")")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.ylim(0,1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,\
                     train_scores_mean + train_scores_std, alpha=0.1,\
                     color="r")
plt.fill_between(train_sizes, val_scores_mean - val_scores_std,\
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",\
         label="Training score (rbf)")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g",\
         label="Cross-validation score (rbf)")
plt.legend(loc="best")

# train_sizes, train_scores, val_scores = learning_curve(estimator=SVC(kernel='linear', C=C), \
#   X= X_train, y=y_train, train_sizes=[0.1, 0.3, 0.6, 1], cv=10, verbose=1, shuffle=True)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# val_scores_mean = np.mean(val_scores, axis=1)
# val_scores_std = np.std(val_scores, axis=1)
# plt.grid()
# plt.fill_between(train_sizes, train_scores_mean - train_scores_std,\
#                      train_scores_mean + train_scores_std, alpha=0.1,\
#                      color="m")
# plt.fill_between(train_sizes, val_scores_mean - val_scores_std,\
#                  val_scores_mean + val_scores_std, alpha=0.1, color="b")
# plt.plot(train_sizes, train_scores_mean, 'o--', color="m",\
#          label="Training score (linear)")
# plt.plot(train_sizes, val_scores_mean, 'o--', color="b",\
#          label="Cross-validation score (linear)")
# plt.legend(loc="best")
plt.show()