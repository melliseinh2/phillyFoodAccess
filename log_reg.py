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

def run_model(X_train, X_test, y_train, y_test, label_name):
    
    logreg = LogisticRegression(random_state=16, max_iter=500)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    prob_pos = logreg.predict_proba(X_test)

    accuracyLog = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression accuracy {label_name}:', accuracyLog*100, "%")

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=logreg.classes_)
    disp.plot()
    plt.title(f'Logistic Regression predicting {label_name}')
    plt.savefig(f'cm_logreg_{label_name}.png')

    return prob_pos


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
    prob_pos = [] 
    df = util.process_txt("NeighborhoodFoodRetail.csv")
    for i in label_col:
        X_train, X_test, y_train, y_test = util.split_data(df, i)
        prob_pos.append(run_model(X_train, X_test, y_train, y_test, i))

    visual(prob_pos[0], prob_pos[1], label_col)

    
if __name__ == "__main__":
    main()
