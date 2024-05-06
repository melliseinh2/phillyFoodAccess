import sys
import optparse
from sklearn.datasets import fetch_openml, load_breast_cancer, fetch_20newsgroups_vectorized
from sklearn.model_selection import StratifiedKFold, GridSearchCV, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import utils
from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import svm
import util


def main():

    original_df = util.process_txt("NeighborhoodFoodRetail.csv")
    # print(df)
    # X, y = util.split_data(df, i)

    label_col = ["SUPERMARKET_ACCESS", "HIGH_POVERTY"] 

    svc_clf = SVC()

    for i in label_col:
        print(original_df)
        y, X = util.split_data(original_df, i)
        y = y.to_numpy()
        X = X.to_numpy()

    ## generate_plots(f'{"SVC"}_{opts.dataset}.csv', "SVM", opts.dataset )
    ## generate_plots(f'{"RF"}_{opts.dataset}.csv', "RF", opts.dataset )

    # if running from scratch, run below. if not, uncomment two lines above and run that instead. 
    #   this helps in testing, as i dont have to go through the whole dataset everytime. just use csv output instead. 

        print("----------------")
        print(f'SVC Results, Label: {i}')

        svc_clf = SVC()

        svc_params_used = [pow(10, -5), pow(10, -4),pow(10, -3), pow(10, -2),pow(10, -1),1, 10]

        svc_train_scores, svc_test_scores = get_curve(svc_clf,svc_params_used, X, y, "gamma")

        df = pd.DataFrame({"params_used": svc_params_used, "train": svc_train_scores, "test": svc_test_scores})
        df.to_csv(f'{"SVC"}_{i}.csv')

        generate_plots(f'{"SVC"}_{i}.csv', "SVM", i )

    pass

def generate_plots(CSV_name, clf_name, label_name ):
    df = pd.read_csv(CSV_name)

    params_used = list(df["params_used"])
    train_scores = list(df["train"])
    test_scores = list(df["test"])

    plt.clf() 

    # plot train and test scores on same graph
    fig1, ax1 = plt.subplots()
    ax1.plot(params_used, train_scores, 'bo-', label="train")
    ax1.plot(params_used, test_scores, 'ro-', label = "test")
    ax1.legend()

    # change x ticks and label according to classifer

    ax1.set_xscale('log')
    ax1.set_xticks(params_used)
    plt.xlabel("gamma")
    # change y ticks and label
    plt.yticks(np.arange(0, 1, step=0.2))
    plt.ylabel("accuracy")


    plt.title(f'{clf_name} Learning Curves for Label: {label_name}')

    # output plot
    plt.savefig(f'{label_name}_{clf_name}.png')
    plt.clf() 
    


def get_curve(clf, params_used, X, y, name):
    """
    generate validation curve. 
    parameters: clf (classifer used), params_used(list of parameters choosen), X, y, name (str version of classfier name)
    output: final_train(rounded and averaged train scores), final_test(rounded and averaged test scores)
    """
    train_scores, test_scores = validation_curve(clf, X, y, param_name=name, param_range=params_used, cv=3)

    final_train = []
    final_test = []

    print(f'{name}, Train Score, Test Score')

    # iterate through all parameters used. 
    for i in range(len(params_used)):
        final_train.append(round(mean(train_scores[i]), 4)) # round to 4 digits
        final_test.append(round(mean(test_scores[i]), 4))
        print(f'{params_used[i]} {round(mean(train_scores[i]), 4)} {round(mean(test_scores[i]), 4)}')
    return final_train, final_test


if __name__ == "__main__":
    main()