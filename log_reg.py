#import pandas
import pandas as pd
import util

def process_data(csv_name):
    #col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
    # load dataset

    label_col = util.opt
    food_data = pd.read_csv(csv_name, header=0)

    feature_cols = []
    label_col = food_data[]

    X = pima[feature_cols]
