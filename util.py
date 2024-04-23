"""
Util for processing the data.
Authors: Mia and Charlie
Date: 
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import optparse
import sys

def process_txt(csv_name):
    """ 
    Helper function to process a text file specified by txt_name into encoded text
    """
    opts = parse_args()
    label_col = opts.label_col
    df = pd.read_csv(csv_name)
    y = df[label_col]
    X = df.loc[:, df.columns != label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=10000).batch(64)
    # val_dset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
    # test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
    

    # splitting the dataset into train, val, and test subsets 
    # train_index = int(df.shape[0]*0.8)
    # val_index = int(df.shape[0]*0.9)

    # train = encode_txt[:train_index]
    # val = encode_txt[train_index:val_index]
    # test = encode_txt[val_index:]

    #train = train.shuffle()

    return X_train, X_test, y_train, y_test

def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run emsemble method')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    parser.add_option('-l', '--label_col', type='string', help='CSV column to treat as label')


    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename', 'label_col',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

