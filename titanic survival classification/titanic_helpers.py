#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:10:47 2017

@author: qisen
"""
from titanic_config import *

def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution(df, var, label, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue = label , aspect = 4 , row = row , col = col)
    facet.map(sns.kdeplot, var, shade = True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()
    plt.show(block=False)
    
def plot_volindist(df, var, label, **kwargs):
    cat = kwargs.get('cat', None)
    fig, (ax0, ax1, ax2) = plt.subplots(1,3,figsize=(12,5), sharey = True, \
          gridspec_kw = {'width_ratios':[2, 1, 1]})
    sns.violinplot(x = cat, y = var, hue = label, data = df, \
                          palette = 'muted', split = True, ax = ax0)
    df[cat] = 'All'
    sns.violinplot(x = cat, y = var, hue = label, data = df, ax = ax1, \
                          palette = 'muted', split = True)
    sns.violinplot(x = cat, y = var, data = df, ax = ax2, \
                          palette = 'muted', split = True)
    plt.show(block=False)


def plot_categories(df, cat, label, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, size = 5, aspect = 1, row = row, col = col)
    facet.map(sns.barplot, cat, label)
    facet.add_legend()
    plt.show(block=False)

def plot_correlation_map(df):
    corr = df.corr()
    _ , ax = plt.subplots(figsize =(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square = True, 
        cbar_kws = {'shrink': .9}, 
        ax=ax, 
        annot = True, 
        annot_kws = {'fontsize': 15}
    )
    plt.show(block=False)

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier(random_state = 99)
    tree.fit(X, y)
    plot_model_var_imp(tree, X, y)
    
def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame( 
        model.feature_importances_, 
        columns = ['Importance'], 
        index = X.columns 
    )
    imp = imp.sort_values(['Importance'], ascending = True)
    imp[ : ].plot(kind = 'barh', figsize = (10,10))
    plt.show(block = False)
    
    print ('Model score: ', model.score(X, y), '\n')