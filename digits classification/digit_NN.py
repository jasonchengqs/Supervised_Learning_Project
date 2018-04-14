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
# print(data.iloc[:,1:].head())
X_train, X_test, y_train, y_test= train_test_split(data.iloc[:,1:], labels, test_size=0.3, random_state=0)
# X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.3, random_state=0)
# print('1',X_train.shape)
# print(X_train.head())
# print(labels)

# mean_x = X_train.mean()
# std_x = X_train.std()

# def normalization(x, mean_x, std_x):
#     return (x-mean_x)/std_x
# X_train = normalization(X_train, mean_x, std_x)
# X_test = normalization(X_test, mean_x, std_x)
X_train = X_train/255
X_test = X_test/255

print('2',X_train.head())
X_train = X_train.values
X_test = X_test.values
# print('3',X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)

# g = plt.imshow(X_train[0][:,:,0])

# pca=PCA(n_components=50, whiten=True)
# X_train_pca=pca.fit_transform(X_train)
# X_test_pca=pca.transform(X_test)
# X_val_pca=pca.transform(X_val)
# var_ratio = sum(pca.explained_variance_ratio_)
# print("Explained variance: {:.3%}".format(var_ratio))

# X_train = X_train_pca
# X_test = X_test_pca
# X_val = X_val_pca

print("final trainning:", X_train.shape)


########################################
## Model training
########################################
input_shape = (28, 28, 1)
num_classes = 10
optimizer = opt.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model=models.Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu', padding='same',\
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,\
              optimizer=optimizer,\
              metrics=['accuracy'])
s1 = time.time()
history = model.fit(X_train, y_train, validation_split=0.2, \
  epochs=30, batch_size=16, verbose=1)
e1 = time.time()
print("training time:", e1-s1)

acc = history.history['acc']
val_acc = history.history['val_acc']
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
# plt.show()s

plt.figure()
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'm', label='Training Acc')
plt.plot(epochs, val_acc,'b', label='Validation Acc')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.grid()
plt.show()

s2 = time.time()
y_prob = model.predict(X_test)
e2 = time.time()
print("testing time:", e2-s2)

print(y_prob)
print(y_prob.shape)
# bi_func = np.vectorize(lambda x: 1 if x>=0.5 else 0)
y_pred = np.argmax(y_prob, axis=1)
print(y_pred)
print(y_test)
y_pred = to_categorical(y_pred, num_classes = 10)
acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.4%}".format(acc))

# optimizer = opt.Adam()

# model.add(layers.Dense(output_dim=16, activation='relu', \
#       input_dim=in_dim, kernel_initializer='glorot_uniform'))
# model.add(layers.Dropout(0.3))
# model.add(layers.Dense(output_dim=32, activation='relu', kernel_initializer='glorot_uniform'))
# model.add(layers.Dropout(0.3))
# model.add(layers.Dense(output_dim=1, activation='sigmoid'))

# model.compile(loss=losses.categorical_crossentropy, optimizer=optimizer, \
#   metrics=[metrics.accuracy]) 

# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), \
#   epochs=200, batch_size=100, verbose=0)

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss,'b', label='Validation loss')

# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# # plt.show()

# plt.clf()

# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training Acc')
# plt.plot(epochs, val_acc,'b', label='Validation Acc')
# plt.legend()
# plt.xlabel('Epochs')
# plt.ylabel('Acc')
# # plt.show()

# y_prob = model.predict(X_test)
# print(y_prob)
# # bi_func = np.vectorize(lambda x: 1 if x>=0.5 else 0)
# y_pred = y_prob.argmax(axis=1)
# # print(y_pred)
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy: {:.4%}".format(acc))






# classifiers = [clf_gb]

# log_cols = ["clf", "accuracy", "log-loss"]
# log = pd.DataFrame(columns=log_cols)
# for clf in classifiers:
#   clf.fit(X_train, y_train)
#   name = clf.__class__.__name__
#   print("="*30)
#   print(name)

#   print('****Results****')
#   train_predictions = clf.predict(X_test)
#   acc = accuracy_score(y_test, train_predictions)
#   print("Accuracy: {:.4%}".format(acc))

#   train_predictions = clf.predict_proba(X_test)
#   ll = log_loss(y_test, train_predictions)
#   print("Log Loss: {}".format(ll))

#   log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
#   log = log.append(log_entry)

# print("="*30)


# scores = ['roc_auc']
# sk = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
# # param_grid = {'max_depth': [3,4,5,6,7,8], 'learning_rate':[0.19]}
# param_grid = {'min_samples_split': [60], 'learning_rate':[0.1]}
# # param_grid = {'min_samples_split': range(17,63)}
# # param_grid = {'max_depth': [2**x for x in range(1,10)]}
# # param_grid = {'min_impurity_split': [2**x for x in range(-7,-1)]}

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(GradientBoostingClassifier(), param_grid, cv=10,
#                        scoring='%s' % score)
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full test set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     # clf_knn = KNeighborsClassifier(clf.best_params_['n_neighbors'])
#     # clf_knn.fit(X_train, y_train)
#     # predictions = clf_knn.predict(X_test)
#     acc = accuracy_score(y_true, y_pred)
#     print("Accuracy: {:.4%}".format(acc))
#     print()