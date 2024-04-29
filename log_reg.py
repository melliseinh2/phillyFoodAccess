#import pandas
import pandas as pd
import util
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# def train_step(model):
#     #col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
#     # load dataset
    

    # label_col = util.opt

def run_model(csv_name, label_name):
    X_train, X_test, y_train, y_test = util.process_txt(csv_name, label_name)
    logreg = LogisticRegression(random_state=16, max_iter=500)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    accuracyLog = accuracy_score(y_test, y_pred)
    print("Logistic Regression accuracy: ", accuracyLog*100, "%")

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=logreg.classes_)
    disp.plot()
    plt.title(f'Logistic Regression predicting {label_name}')
    plt.savefig(f'cm_logreg_{label_name}.png')


    # train_step(logreg, X_train, X_test, y_train,


def main():
    # test two layer function
    opts = util.parse_args()
    label_col = opts.label_col
    run_model("NeighborhoodFoodRetail.csv", label_col)

if __name__ == "__main__":
    main()
