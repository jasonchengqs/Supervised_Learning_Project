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
acc_n = []
err_n = []
val_acc_n = []
val_err_n = []
total = len(data_y)

lr = 0.0005

for ratio in [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, \
0.6, 0.6, 0.6, 0.8, 0.8, 0.8]:
	sss = StratifiedShuffleSplit(1, test_size=1-ratio, random_state=0)
	for use_index, hold_index in sss.split(data.iloc[:,1:], data['Survived']):
	  X_use, X_hold = data_X.values[use_index], data_X.values[hold_index]
	  y_use, y_hold = data_y.values[use_index], data_y.values[hold_index]

	sss = StratifiedShuffleSplit(1, test_size=0.25, random_state=0)
	for train_index, test_index in sss.split(X_use, y_use):
	  X_train, X_test = X_use[train_index], X_use[test_index]
	  y_train, y_test = y_use[train_index], y_use[test_index]

	# for train_index, val_index in sss.split(X_train, y_train):
	#   X_train, X_val = X_train[train_index], X_train[val_index]
	#   y_train, y_val = y_train[train_index], y_train[val_index]

	# X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.20, stratify = data_y)
	print(X_train[1])
	print("into trainning:", X_train.shape)


########################################
## Model training 
########################################
	in_dim = X_train.shape[1]
	print("in_dim:", in_dim)

	model=models.Sequential()
	optimizer = opt.Adam(lr=lr)

	model.add(layers.Dense(output_dim=16, activation='relu', \
	      input_dim=in_dim, kernel_initializer='glorot_uniform'))
	model.add(layers.Dropout(0.3))
	model.add(layers.Dense(output_dim=32, activation='relu', kernel_initializer='glorot_uniform'))
	model.add(layers.Dropout(0.3))
	model.add(layers.Dense(output_dim=1, activation='sigmoid'))

	model.compile(loss=losses.binary_crossentropy, optimizer=optimizer, \
	  metrics=[metrics.binary_accuracy]) 

	history = model.fit(X_train, y_train, validation_split=0.2, \
	  epochs=20, batch_size=16, verbose=0)

	acc = history.history['binary_accuracy']
	val_acc = history.history['val_binary_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']

	acc_n.append(acc[-1])
	val_acc_n.append(val_acc[-1])
	err_n.append(loss[-1])
	val_err_n.append(val_loss[-1])

	# plt.figure()
	# epochs = range(1, len(acc) + 1)
	# plt.plot(epochs, loss, 'bo', label='Training loss')
	# plt.plot(epochs, val_loss,'b', label='Validation loss')

	# plt.legend()
	# plt.xlabel('Epochs')
	# plt.ylabel('Loss')
	# plt.show()

	# plt.clf()
	# plt.figure()
	# epochs = range(1, len(acc) + 1)
	# plt.plot(epochs, acc, 'bo', label='Training Acc')
	# plt.plot(epochs, val_acc,'b', label='Validation Acc')
	# plt.legend()
	# plt.xlabel('Epochs')
	# plt.ylabel('Acc')
	# plt.show()

	y_prob = model.predict(X_test)
	bi_func = np.vectorize(lambda x: 1 if x>=0.5 else 0)
	y_pred = bi_func(y_prob)
	# print(y_pred)
	acc = accuracy_score(y_test, y_pred)

	print("Accuracy: {:.4%}".format(acc))

# print ("acc_cv", acc_n)
plt.figure()
plt.title("Accuracy: NN")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.ylim(0,1)
train_scores_mean = [np.mean(acc_n[x:x+3]) for x in [0, 3, 6, 9]]
train_scores_std = [np.std(acc_n[x:x+3]) for x in [0, 3, 6, 9]]
val_scores_mean = [np.mean(val_acc_n[x:x+3]) for x in [0, 3, 6, 9]]
val_scores_std = [np.std(val_acc_n[x:x+3]) for x in [0, 3, 6, 9]]
plt.grid()
train_sizes = [int(total*x) for x in [0.2, 0.4, 0.6, 0.8]]
plt.fill_between(train_sizes, [x-y for x,y in zip(train_scores_mean, train_scores_std)],\
                     [x+y for x,y in zip(train_scores_mean, train_scores_std)], alpha=0.1,\
                     color="r")
plt.fill_between(train_sizes, [x-y for x,y in zip(val_scores_mean, val_scores_std)],\
                 [x+y for x,y in zip(val_scores_mean, val_scores_std)], alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",\
         label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g",\
         label="Cross-validation score")
plt.legend(loc="best")



plt.figure()
plt.title("Error: NN")
plt.xlabel("Training examples")
plt.ylabel("Error")
# plt.ylim(0,5)
train_scores_mean = [np.mean(err_n[x:x+3]) for x in [0, 3, 6, 9]]
train_scores_std = [np.std(err_n[x:x+3]) for x in [0, 3, 6, 9]]
val_scores_mean = [np.mean(val_err_n[x:x+3]) for x in [0, 3, 6, 9]]
val_scores_std = [np.std(val_err_n[x:x+3]) for x in [0, 3, 6, 9]]
plt.grid()
train_sizes = [int(total*x) for x in [0.2, 0.4, 0.6, 0.8]]
plt.fill_between(train_sizes, [x-y for x,y in zip(train_scores_mean, train_scores_std)],\
                     [x+y for x,y in zip(train_scores_mean, train_scores_std)], alpha=0.1,\
                     color="r")
plt.fill_between(train_sizes, [x-y for x,y in zip(val_scores_mean, val_scores_std)],\
                 [x+y for x,y in zip(val_scores_mean, val_scores_std)], alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",\
         label="Training score")
plt.plot(train_sizes, val_scores_mean, 'o-', color="g",\
         label="Cross-validation score")
plt.legend(loc="best")


#######################################
# testing
#######################################

sss = StratifiedShuffleSplit(1, test_size=0.25, random_state=0)
for train_index, test_index in sss.split(data.iloc[:,1:], data['Survived']):
  X_train, X_test = data_X.values[train_index], data_X.values[test_index]
  y_train, y_test = data_y.values[train_index], data_y.values[test_index]


model=models.Sequential()
optimizer = opt.Adam(lr=lr)

model.add(layers.Dense(output_dim=16, activation='relu', \
      input_dim=in_dim, kernel_initializer='glorot_uniform'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(output_dim=32, activation='relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(output_dim=1, activation='sigmoid'))

model.compile(loss=losses.binary_crossentropy, optimizer=optimizer, \
  metrics=[metrics.binary_accuracy]) 

s1 = time.time()
history = model.fit(X_train, y_train, validation_split=0.2, \
  epochs=20, batch_size=16, verbose=1)
e1 = time.time()
print("training time:", e1-s1)

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss, 'm', label='Training loss')
plt.plot(epochs, val_loss,'b', label='Validation loss')

plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
# plt.show()

plt.figure()
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'm', label='Training Acc')
plt.plot(epochs, val_acc,'b', label='Validation Acc')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.grid()
# plt.show()

s2 = time.time()
y_prob = model.predict(X_test)
e2 = time.time()
print("testing time:", e2-s2)

bi_func = np.vectorize(lambda x: 1 if x>=0.5 else 0)
y_pred = bi_func(y_prob)
# print(y_pred)
acc = accuracy_score(y_test, y_pred)

print("Accuracy: {:.4%}".format(acc))


plt.show()
