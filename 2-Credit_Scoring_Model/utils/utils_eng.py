#import libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

    
# a function to fit/predict and show metrics

def modelFitCrossValPredMetrics(model, train, train_labels):
    '''
    model : instantiate the model first
    train : data to fit
    train_labels : labels
    
    we use cross val predict so we do not a validation set, we can use only  the training 
    set
    
    '''

    train_pred = cross_val_predict(model, train, train_labels, cv=3)
    
    print("Confusion Matrix \n \n", confusion_matrix(train_labels, train_pred))
    print("\n")
    print("Precision : ", precision_score(train_labels, train_pred))
    print("\n")
    print("Recall : " , recall_score(train_labels, train_pred))
    print("\n")
    print("F_Score : " ,f1_score(train_labels, train_pred))
    
    
# feature importance
def plot_feature_importances(df,col):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better. 
    
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest) 
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 15))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:col]))), 
            df['importance_normalized'].head(col), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:col]))))
    ax.set_yticklabels(df['feature'].head(col))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df


# PCA 

def pcaDataFrame(df):
    ''' take a quantitative dataframe
        return pca.components_, pca instance, and df of coord. ind and PCAs
    '''
    # PCA libraries
  
    scaler=StandardScaler()
    n_components=df.shape[1]
    colz=['PC_'+ str(i) for i in range(1,n_components+1)]
    dex=df.index
    dexPca=df.columns
    X = df.values
    X_scaled=scaler.fit_transform(X)
    pca = PCA(n_components)
    df_v=pca.fit_transform(X_scaled)
    return  pd.DataFrame(pca.components_, index=colz, columns=dexPca), \
                       pca, pd.DataFrame(df_v, index=dex, columns=colz)


# One Hot Encoding
def oneHotEncoding(df):
    # Create a label encoder object
    le = LabelEncoder()

    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
        # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                # Train on the training data
                #le.fit(df[col])
                # Transform both training and testing data
                df[col] = le.fit_transform(df[col])
    return df



def ordinalEncoding(df):
    
    ordinal_encoder = OrdinalEncoder()
    return pd.DataFrame(ordinal_encoder.fit_transform(df), index=df.index, columns=df.columns)



# correlation list with respect to one variable for quanti variables

def corrByVar(df,var):
    ''' return a list of variables that are significantly correlated with the given variable
    parameters :
         df: dataframe
         var: var to be used for correlation computation
    '''
    D={}
    for i in df.columns.tolist():
        if i==var : 
            continue
        else:
            D[i]=np.corrcoef(df[var], df[i])[0][1]
    d=pd.DataFrame.from_dict(D, orient='index', columns=['corrz']).sort_values('corrz', ascending=False)
    print("\nVariables that are very correlated with", var)
    #.index.tolist()
    return d[(d['corrz'] < -0.3) | (d['corrz'] > 0.3) ]

# return list instead
def corrByVarList(df,var):
    ''' return a list of variables that are significantly correlated with the given variable
    parameters :
         df: dataframe
         var: var to be used for correlation computation
    '''
    D={}
    for i in df.columns.tolist():
        if i==var : 
            continue
        else:
            D[i]=np.corrcoef(df[var], df[i])[0][1]
    d=pd.DataFrame.from_dict(D, orient='index', columns=['corrz']).sort_values('corrz', ascending=False)
    print("\nVariables that are very correlated with", var)
    #.index.tolist()
    return d[(d['corrz'] < -0.3) | (d['corrz'] > 0.3) ].index.tolist()


def display_scree_plot_coud(pca):
   
    
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(15,8))
    #plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
def display_scree_plot_bar(pca):
    
    scree = pca.explained_variance_ratio_*100
    plt.figure(figsize=(15,8))
    plt.bar(np.arange(len(scree))+1, scree)
    #plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
    
# group of functions to frequence encode a categorial variable
# In can be done using this 
def count_mappings(df, variable):
    return df[variable].value_counts().to_dict()

def frequency_mappings(df, variable):
    return (df[variable].value_counts() / len(df)).to_dict()

def encode(df, variable, mapping):
    df[variable] = df[variable].map(mapping)
