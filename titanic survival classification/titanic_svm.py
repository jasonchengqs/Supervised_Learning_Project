#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qisen
"""
########################################
## Import supporting config and helpers
########################################
import titanic_config
from titanic_helpers import *

########################################
## Loading data and some EDA
########################################
data = pd.read_csv("titanic_data.csv")
print('>>>>> raw dataset info:')
print("shape of raw dataset:", data.shape)
print(data.columns.values)

print(">>>>> missing data check in raw data:")
print(data.isnull().sum())


########################################
## Pre-processing
########################################
# Estimate missing Age data from Name
data['Initial']=0
for i in data:
    data['Initial']=data.Name.str.extract('([A-Za-z]+)\.')
data['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
data.loc[(data.Age.isnull())&(data.Initial=='Mr'),'Age']=33
data.loc[(data.Age.isnull())&(data.Initial=='Mrs'),'Age']=36
data.loc[(data.Age.isnull())&(data.Initial=='Master'),'Age']=5
data.loc[(data.Age.isnull())&(data.Initial=='Miss'),'Age']=22
data.loc[(data.Age.isnull())&(data.Initial=='Other'),'Age']=46

# Encode continuous Age data into Age_band
data['Age_band']=0
data.loc[data['Age']<=16,'Age_band']=0
data.loc[(data['Age']>16)&(data['Age']<=32),'Age_band']=1
data.loc[(data['Age']>32)&(data['Age']<=48),'Age_band']=2
data.loc[(data['Age']>48)&(data['Age']<=64),'Age_band']=3
data.loc[data['Age']>64,'Age_band']=4

# Fill in missing Embarked data
data['Embarked'].fillna('S',inplace=True)

# Convert Parch and SibSp into Family_Size and Alone 
data['Family_Size']=0
data['Family_Size']=data['Parch']+data['SibSp']#family size
data['Alone']=0
data.loc[data.Family_Size==0,'Alone']=1#Alone

# Encode continuous Fare data into Fare_cat
data['Fare_Range']=pd.qcut(data['Fare'],4)
data['Fare_cat']=0
data.loc[data['Fare']<=7.91,'Fare_cat']=0
data.loc[(data['Fare']>7.91)&(data['Fare']<=14.454),'Fare_cat']=1
data.loc[(data['Fare']>14.454)&(data['Fare']<=31),'Fare_cat']=2
data.loc[(data['Fare']>31)&(data['Fare']<=513),'Fare_cat']=3

# Encode categorical data into numerical data
data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'],[0,1,2,3,4],inplace=True)

data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId','Parch','SibSp'],axis=1,inplace=True)
print(">>>>> features after pre-processing:")
print(data.columns.values)


########################################
## Assemble features and split data
########################################
#full_X = pd.concat([sex, embarked, pclass, age, fare, \
#                    title, cabin, ticket, family], axis = 1)
data_y = data['Survived']
data_X = data.iloc[:,1:]
data_X = (data_X-data_X.mean())/data_X.std()
# print(data_X.head())
# print(data_y.head())
# print(len(data_y))

sss = StratifiedShuffleSplit(1, test_size=0.25, random_state=0)
for train_index, test_index in sss.split(data.iloc[:,1:], data['Survived']):
  X_train, X_test = data_X.values[train_index], data_X.values[test_index]
  y_train, y_test = data_y.values[train_index], data_y.values[test_index]

# X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.20, stratify = data_y)
# print(X_train[1])
# print(X_train.shape)

########################################
## Model setup
########################################
clf_svm = SVC(kernel="linear", C=5, verbose=True)
clf_svm_rbf = SVC(kernel="rbf", C=5, gamma=0.05, verbose=True)
clf_svm_rbf = SVC(kernel="rbf")
########################################
## Model training
########################################
classifiers = [clf_svm, clf_svm_rbf]

for clf in classifiers:
    s1=time.time()
    clf.fit(X_train, y_train)
    e1=time.time()
    name = clf.__class__.__name__
    print("="*30)
    print(name)
    print("train_time:", e1-s1)
    print('****Results****')
    s2=time.time()
    train_predictions = clf.predict(X_test)
    e2=time.time()
    print("test_time:", e2-s2)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    acc = f1_score(y_test, train_predictions)
    print("F1: {:.4}".format(acc))
    acc = roc_auc_score(y_test, train_predictions)
    print("AUROC: {:.4}".format(acc))

print("="*30)

########################################
## Parameter tuning 
########################################

scores = ['roc_auc']
sk = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# param_grid = {'C': [0.001, 0.1, 1, 5, 10], 'gamma': [0.001, 0.005, 0.01, 0.05, 1, 5, 10]}
param_grid = {'C': [5], 'gamma': [0.05]}

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=sk,
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

# param_range = [0.001, 0.1, 1, 5 ,10, 20]
param_range = [0.005, 0.025, 0.05, 0.075, 0.1 , 0.2, 0.5]
# train_scores, val_scores = validation_curve(estimator=SVC(gamma = 0.05), X=X_train, y=y_train, param_name="C", \
#   param_range=param_range, cv=10, verbose=1, scoring='roc_auc')
train_scores, val_scores = validation_curve(estimator=SVC(C = 5), X=X_train, y=y_train, param_name="gamma", \
  param_range=param_range, cv=10, verbose=1, scoring='roc_auc')
plt.figure()
# plt.title("CV: Tuning of C")
plt.title("CV: Tuning of gamma")
# plt.xlabel("C")
plt.xlabel("Gamma")
plt.ylabel("AUROC")
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
C=5
gamma=0.05

train_sizes, train_scores, val_scores = learning_curve(estimator=SVC(kernel=kernel, C=C, gamma=gamma), \
  X=X_train, y=y_train, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=10, verbose=1, scoring='neg_mean_squared_error', shuffle=True)

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

train_sizes, train_scores, val_scores = learning_curve(estimator=SVC(kernel=kernel, C=C, gamma=gamma), \
  X= X_train, y=y_train, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=10, verbose=1, shuffle=True)

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
#   X= X_train, y=y_train, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1], cv=10, verbose=1, shuffle=True)
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

plt.show()
