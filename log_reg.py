#import pandas
import pandas as pd
import util
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# def train_step(model):
#     #col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
#     # load dataset
    

    # label_col = util.opt

def run_model(csv_name):
    X_train, X_test, y_train, y_test = util.process_txt(csv_name)
    logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)

    # train_step(logreg, X_train, X_test, y_train,


def main():
    # test two layer function
    run_model("NeighborhoodFoodRetail.csv")

if __name__ == "__main__":
    main()
