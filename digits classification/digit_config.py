#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:16:40 2017

@author: qisen
"""
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import time

# Handle table-like data and matrices
import numpy as np
import pandas as pd


# Modelling Algorithms
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

import keras.models as models
import keras.layers as layers
import keras.optimizers as opt
from keras import losses, metrics
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization

# Modeling helper functions
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, normalize, scale
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve, learning_curve

from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics import accuracy_score, log_loss, roc_curve

# Visulisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
plt.style.use('fivethirtyeight')
# import graphviz

# Configure visualisations
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# Configure working directory, terminal display
import os
# clear screen of terminal
print ('\n'*1)
print ('====================', \
       'HERE DIGITS PRJ CODE START', \
       '====================\n')

# display current directory 
cwd = os.getcwd()
print ('>>>>> current directory:', cwd, '\n')

