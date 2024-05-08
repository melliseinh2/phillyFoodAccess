"""
Main Driver class to run SVM and Logistic Regression models
Author: Mia Ellis-Einhorn and Charlie Crawford
"""
import pandas as pd
import util
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import svm

from sklearn import svm


def run_model(clf, clf_name, X_train, X_test, y_train, y_test, label_name, kernel = None):
    """
    method to train a given classifier and run confusion matrix code. 
    inputs: clf (generic classifer), clf_name (string name of clf), label_name (name of current label)
    """
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Generate and output accuracy score
    accuracyLog = accuracy_score(y_test, y_pred)
    print(f'{clf_name} {kernel} accuracy {label_name}:', accuracyLog*100, "%")


    # Confusion Matrix generation
    create_cm(y_test, y_pred, clf_name , label_name, kernel)


    # Output weight vector for logistic regression / linear kernel SVM. 
    if clf_name == "log_reg" or kernel == "linear":
        weight_vector = clf.coef_[0]
        # print(weight_vector)
        print("Weight Vector Values:")
        for i in range(len(list(X_test.columns))):
            print(f'Feature: {X_test.columns[i]} \n Weight: {round(weight_vector[i], 3)}')
    
    return 0

def create_cm(y_test, y_pred, clf_name, label_name, kernel = ""):
    """
    generate confusion matrix and output to png
    """
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    kernel_txt = ""
    if kernel:
        kernel_txt = f' Kernel: {kernel}'
    else:
        kernel = ""
    plt.title(f'{clf_name} predicting {label_name} {kernel_txt}')
    plt.savefig(f'cm_{clf_name}_{label_name}_{kernel}.png')
    return cnf_matrix


def main():
    label_col = ["SUPERMARKET_ACCESS", "HIGH_POVERTY"] 

    df = util.process_txt("NeighborhoodFoodRetail.csv")

    for i in label_col: # iterate through both labels. 
        X, y = util.split_data(df, i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

        print(f'Logistic Regression Results for Label {i}')
        clf = LogisticRegression(random_state=16, max_iter=500)
        run_model( clf, "log_reg", X_train, X_test, y_train, y_test, i)

        for j in ['linear', 'rbf']:
            print(f'SVM Results for Label {i} and Kernel {j}')
            clf = svm.SVC(kernel=j)
            run_model( clf, "SVM", X_train, X_test, y_train, y_test, i, j)

    
if __name__ == "__main__":
    main()
