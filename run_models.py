#import pandas
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

import tensorflow as tf

# def train_step(model):
#     #col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
#     # load dataset
    

    # label_col = util.opt

def run_model(clf, clf_name, X_train, X_test, y_train, y_test, label_name):
    
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # prob_pos = clf.predict_proba(X_test)

    accuracyLog = accuracy_score(y_test, y_pred)
    print(f'{clf_name} accuracy {label_name}:', accuracyLog*100, "%")

    create_cm(y_test, y_pred, clf, clf_name , label_name)

    return 0

def create_cm(y_test, y_pred, clf, clf_name, label_name):
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix)
    disp.plot()
    plt.title(f'{clf_name} predicting {label_name}')
    plt.savefig(f'cm_{clf_name}_{label_name}.png')
    return cnf_matrix

    # train_step(logreg, X_train, X_test, y_train,

def visual(prob_pos_A, prob_pos_B, labels):
    plt.clf()
    plt.scatter(prob_pos_A, prob_pos_B)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title("Probability of Positive by Label Predicted")

    plt.savefig("probpos.png")

def main():
    # test two layer function
    # opts = util.parse_args()
    label_col = ["SUPERMARKET_ACCESS", "HIGH_POVERTY"] #opts.label_col
    # prob_pos = [] 
    df = util.process_txt("NeighborhoodFoodRetail.csv")

    for i in label_col:
        # logreg = LogisticRegression(random_state=16, max_iter=500)
        X, y = util.split_data(df, i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)
        # print(X_train)
        # print(y_train)
        print(f'Logistic Regression Results for Label {i}')
        clf = LogisticRegression(random_state=16, max_iter=500)
        run_model( clf, "log_reg", X_train, X_test, y_train, y_test, i)
        print(f'SVM Results for Label {i}')
        clf = svm.SVC()
        run_model( clf, "SVM", X_train, X_test, y_train, y_test, i)
        # print(f'FC NN Results for Label {i}')
        # train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)

        # model = tf.keras.models.Sequential([
        #     # tf.keras.layers.Flatten(input_shape=(28, 28)),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(10)
        # ])

        # predictions = model(X_train[:1]).numpy()

        # print(predictions)
        # print(tf.nn.softmax(predictions).numpy())


    # visual(prob_pos[0], prob_pos[1], label_col)

    
if __name__ == "__main__":
    main()
