"""
Run classifers on given datasets and compare test/train scores. 
Author: Mia Ellis-Einhorn
"""
import sys
import optparse
from sklearn.datasets import fetch_openml, load_breast_cancer, fetch_20newsgroups_vectorized
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import utils
import util
import pandas as pd

import log_reg

def main():
    df = util.process_txt("NeighborhoodFoodRetail.csv")
    # X, y = util.split_data(df, i)
    best_params_all = []

    label_col = ["SUPERMARKET_ACCESS", "HIGH_POVERTY"] 

    svc_clf = SVC()

    svc_params = {"C": [1, 10, 100, 1000], "kernel": ['rbf'], "gamma": [pow(10, -4),pow(10, -3), pow(10, -2),pow(10, -1),1]}

    for i in label_col:
        y, X = util.split_data(df, i)
        y = y.to_numpy()
        X = X.to_numpy()
        print("----------------")
        print(f'SVC Results, Label: {i}')
        curr_label = i

        test_results, best_params, y_pred, y_test = run_tune_test(svc_clf, svc_params, X, y)
        
        print("Fold, Test Accuracy")
        for i in range(len(test_results)):
            print(f'{i+1}, {test_results[i]}')
            # print("predictions:")
            # print(y_pred[i])

        svc_df = pd.DataFrame({'parameters_used': best_params, 'test results': test_results})
        # print("best_results")
        best_index = test_results.index(max(test_results))
        best_params = best_params[best_index]
        best_params_all.append(best_params)
        # best_result = y_pred[best_index]
        # print(best_result)
        print(svc_df)
        # print(svc_clf.classes_)
        # print(i)

        # log_reg.create_cm(y_test[best_index], best_result, svc_clf, "SVM", curr_label)

    return best_params_all



def run_tune_test(learner, params, X, y):
    """
    method to run given classifer
    parameters: learner (classifier), params (possible params to use), X, y
    output: test_accuracy (list of accuracy scores for test data), best_params (list of parameters choosen)
    """

    test_accuracy = []
    best_params = []
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True) # create k stratified folds
    y_pred = []
    all_y_test = []
    
    # iterate through folds. 
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(type(X))
        print(type(y))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = GridSearchCV(learner, param_grid = params)
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        test_accuracy.append(test_score)
        best_params.append(clf.best_params_)
        y_pred.append(clf.predict(X_test))
        all_y_test.append(y_test)
        print(f"Fold {i}: \n {clf.best_params_}")
        print(f'Training Score: {train_score}')
        
    return test_accuracy, best_params, y_pred, all_y_test


if __name__ == "__main__":
    main()


