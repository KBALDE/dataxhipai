# read a file from a directory

#import pandas as pd

def readCsvFileFromDir(directory, dataFile):
    ''' directory: is the path for data location
        dataFile: the csv file name
    '''
    import pandas as pd
    import os
    return pd.read_csv(os.path.join(directory,dataFile))


# first stats
def firstStats(df):
    import pandas as pd
    return pd.DataFrame(df.shape, index=['nbRows', 'nbColumns'], 
             columns=['stats']).T.merge(pd.DataFrame(df.dtypes.value_counts(), 
                                                     columns=['stats']).T,
                                        right_index=True, left_index=True).T



# Function to calculate missing values by column
def missingValues(df):
    
    import pandas as pd
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
        
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns=mis_val_table_ren_columns.sort_values('% of Total Values', ascending=False).round(1)
        
    return mis_val_table_ren_columns


# investigate all variables
def describeData(df):
    for x in df.columns:
        if  (df[x].dtypes=='float64') | (df[x].dtypes=='int64'):
            print("\n La distribution de la variable {}\n".format(x) , df[x].describe())
        else :
            print("\n La frequence de la variable {}\n".format(x), df[x].value_counts()[:10])
            
# put NaN to mean and unknown            
def manageMissingData(df):
    ''' manage missing data by data type and outliers 
        how: for quali, you are not going to invent data
          for quanti, let's just fill in the median for now
          remove outliers using envelop method from sklean
        return: a fully clean dataframe
        
    '''
    import pandas as pd
    # fill unknown for quali
    df_obj=df.select_dtypes(['object']).fillna('Unknown')
    #fill median for numeric columns
    df_flo=df.select_dtypes(['int', 'float']).apply(pd.to_numeric, errors='coerce')
    df_flo.fillna(df_flo.mean(), inplace=True)
    df_flo.merge(df_obj, on=df_flo.index, how='left')
    return df_flo.merge(df_obj, on=df_flo.index, how='left')
          
            
# KNN Imputer            
def knnImputer(df_nan, n_neighbors):
    import pandas as pd
    from sklearn.impute import KNNImputer
    colz=df_nan.columns
    dex=df_nan.index
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform', metric='nan_euclidean')
    return pd.DataFrame(imputer.fit_transform(df_nan), index=df_nan.index, columns=df_nan.columns)
