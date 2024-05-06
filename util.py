"""
Util for processing the data.
Authors: Mia and Charlie
Date: 
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
    print(df.columns)
    df = df.drop(columns=["HPSS_ACCESS", "GEOID10", "OBJECTID", "Shape__Area", "Shape__Length", "PCT_POVERTY", "HPSS_ACCESS"]) # drop due to the label being "moderate to high" too complicated to binarize. 
    print(df.columns)
    for col in df.columns:
        # print(col)
        # column_name = str(col)
        # print(df[col][1] == 'Yes' or df[col][1] == 'No' )
        if df[col][1] == 'Yes' or df[col][1] == 'No':
            df[col] = (df[col] == 'Yes').astype(int)
    # print(df.shape)
    df = df.dropna() # drop na values. 
    # shuffle the datafram

    print(df)
    df = df.sample(frac= 1)
    print(df)

    return df

def split_data(df, label_col):

    y = df[label_col]
    print(y)
    X = df.loc[:, df.columns != label_col]
    

    return y, X

# def parse_args():
#     """Parse command line arguments (train and test arff files)."""
#     parser = optparse.OptionParser(description='run emsemble method')

#     parser.add_option('-r', '--train_filename', type='string', help='path to' +\
#         ' train arff file')
#     parser.add_option('-e', '--test_filename', type='string', help='path to' +\
#         ' test arff file')
#     parser.add_option('-l', '--label_col', type='string', help='CSV column to treat as label')


#     (opts, args) = parser.parse_args()

#     # mandatories = ['label_col']
#     # for m in mandatories:
#     #     if not opts.__dict__[m]:
#     #         print('mandatory option ' + m + ' is missing\n')
#     #         parser.print_help()
#     #         sys.exit()

#     return opts

