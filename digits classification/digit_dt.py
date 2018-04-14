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
X_train, X_test, y_train, y_test= train_test_split(data.iloc[:,1:], labels, test_size=0.3, random_state=0)

# pca=PCA(n_components=30, whiten=True)
# X_train_pca=pca.fit_transform(X_train)
# X_test_pca=pca.transform(X_test)
# var_ratio = sum(pca.explained_variance_ratio_)
# print("Explained variance: {:.3%}".format(var_ratio))

# X_train = X_train_pca
# X_test = X_test_pca

print("final trainning:", X_train.shape)

########################################
## Model setup
########################################
clf_dt_org = DecisionTreeClassifier()
# clf_dt = DecisionTreeClassifier(min_impurity_decrease=5e-6, max_depth=20)
clf_dt = DecisionTreeClassifier(min_samples_split=4, max_depth=10)
########################################
## Model training
########################################
classifiers = [clf_dt_org, clf_dt]

log_cols = ["clf", "accuracy", "log-loss"]
log = pd.DataFrame(columns=log_cols)
for clf in classifiers:
  s1=time.time()
  clf.fit(X_train, y_train)
  e1=time.time()
  name = clf.__class__.__name__
  print("="*30)
  print(name)
  print("trian_time:", e1-s1)

  print('****Results****')
  s2=time.time()
  train_predictions = clf.predict(X_test)
  e2=time.time()
  print("test_time:", e2-s2)

  acc = accuracy_score(y_test, train_predictions)
  print("Accuracy: {:.4%}".format(acc))
  acc = f1_score(y_test, train_predictions, average='macro')
  print("F1: {:.4}".format(acc))
  acc = multiclass_roc_auc_score(y_test, train_predictions, average='macro')
  print("AUROC: {:.4}".format(acc))

print("="*30)

########################################
## Parameter tuning 
########################################

scores = ['accuracy']
sk = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
param_grid = {'min_samples_split': [4], 'max_depth': [10]}
# param_grid = {'min_samples_leaf': [0.01]}
# param_grid = {'max_depth': [2**x for x in range(1,10)]}
# param_grid = {'min_impurity_decrease': [6**x for x in range(-15,-12)]}
# param_grid = {'min_impurity_decrease': [8e-5,8.5e-5,9e-5]}
# param_grid = {'min_impurity_decrease': [8.5e-5], 'max_depth':[30]}
# param_grid = {'min_impurity_decrease': [5e-4], 'max_depth': [20]}
for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(DecisionTreeClassifier(criterion="gini"), param_grid, cv=10,
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

# params = clf.best_params_
# print(params)
# clf_dt.set_params(**params)
# clf_dt.fit(X_train, y_train)
# y_pred = clf_dt.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy: {:.4%}".format(acc))

# dot_data = export_graphviz(clf_dt, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("tree")

#######################################
# testing
#######################################

mss=4
mid = 5e-4
md = 10

# train_sizes, train_scores, val_scores = learning_curve(\
  # estimator=DecisionTreeClassifier(min_impurity_decrease=mid, max_depth=md), \
  # X=X_train, y=y_train, train_sizes=[0.1, 0.3, 0.6, 1], cv=10, verbose=1, scoring='neg_mean_squared_error', shuffle=True)
train_sizes, train_scores, val_scores = learning_curve(\
  estimator=DecisionTreeClassifier(min_samples_split=mss, max_depth=md), \
  X=X_train, y=y_train, train_sizes=[0.1, 0.3, 0.6, 1], cv=10, verbose=1, scoring='neg_mean_squared_error', shuffle=True)
print (val_scores)
plt.figure()
plt.title("Error: DT")
plt.xlabel("Training examples")
plt.ylabel("MSE error")
plt.ylim(0,5)
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

# train_sizes, train_scores, val_scores = learning_curve(\
#   estimator=DecisionTreeClassifier(min_impurity_decrease=mid, max_depth=md), \
#   X= X_train, y=y_train, train_sizes=[0.1, 0.3, 0.6, 1], cv=10, verbose=1, shuffle=True)
train_sizes, train_scores, val_scores = learning_curve(\
  estimator=DecisionTreeClassifier(min_samples_split=mss, max_depth=md), \
  X=X_train, y=y_train, train_sizes=[0.1, 0.3, 0.6, 1], cv=10, verbose=1, scoring='accuracy', shuffle=True)

plt.figure()
plt.title("Accuracy: DT")
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
         label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g",\
         label="Cross-validation score")
plt.legend(loc="best")

plt.show()