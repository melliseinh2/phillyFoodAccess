# phillyFoodAccess
CS360 final project -- Food Access and Poverty in Philadelphia 

Authors: Mia Ellis-Einhorn and Charlie Crawford

## Description:
This codebase is set up to run a Logistic Regression model as well as Support Vector Machine on the Philadelphia Neighborhood Food Retail dataset. The models attempt to predict one of two labels wihtin the dataset, "SUPERMARKET_ACCESS" or "HIGH_POVERTY". The code will output six confusion matrices total, one of each label for LR, SVM with RBF kernel, and SVM with linear kernel, as well as testing accuracies for each and a breakdown of the most influential features by weight. 

## Environment:
Python: v.3.7.7

Sklearn: scikit-learn-1.0.2

Pandas: v.1.0.5

Matplotlib: v.3.2.2 

## How to run the code:

```
download dataset (neighborhood food retail csv) from: https://opendataphilly.org/datasets/neighborhood-food-retail/ 
python3 run_philly_food.py -d 
```

