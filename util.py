"""
Util for processing the data.
Authors: Mia and Charlie
Date: May 8th 2024
"""
import pandas as pd

import optparse
import sys
import numpy as np

def process_txt(csv_name):
    """ 
    Helper function to process a text file specified by txt_name into encoded text
    """
    
    df = pd.read_csv(csv_name, na_filter= True, header=0)
    df = df.drop(columns=["HPSS_ACCESS", "GEOID10", "OBJECTID", "Shape__Area", "Shape__Length", "PCT_POVERTY"]) 
    # drop due to the label being "moderate to high" too complicated to binarize. 
  
    # convert binary columns to number binary 
    for col in df.columns:
        if df[col][1] == 'Yes' or df[col][1] == 'No':
            df[col] = (df[col] == 'Yes').astype(int)

    df = df.dropna() # drop na values. 
   
    # shuffle the dataframe
    df = df.sample(frac= 1)

    return df

def split_data(df, label_col):
    """
    split the data to features and labels given label title
    """

    y = df[label_col]
    X = df.loc[:, df.columns != label_col]
    
    return X, y


