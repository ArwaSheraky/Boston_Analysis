# Boston_Analysis
This is the final project of CEBD-1160 course, based on Boston housing dataset.

| Name | Date |
|:-------|:---------------|
| Arwa Sheraky | 28-3-2019 |

-----

### Resources
This repository includes:

- Python script for boston-housing-data analysis: `boston_analysis.py`
- Results figures: `Figures/`
- Dockerfile for experiment: `Dockerfile`
- runtime-instructions: `RUNME.md`

-----

## Research Question
Knowing the average prices of houses in suburbs of Boston from the 1970 and the features that could affect them, could we predict the average prices of new houses, having these 13 features?

### Abstract

- opportunity (what data do we have)
- challenge (what is the "problem" we could solve with this dataset)
- action (how will we try to solve this problem/answer this question)
- resolution (what did we end up producing)

### Introduction
The dataset used in this project, is publically shared on scikit-learn [datasets](https://scikit-learn.org/stable/datasets/index.html#boston-dataset), which could be explicitly imported into any python app, from `sklearn` library.

### Methods

By applying different regressors, comparing their performance using [R-squared](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) and [MSE](https://en.wikipedia.org/wiki/Mean_squared_error), we can find the best one to solve the problem.

The used regressors are:
- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression).
- [Bayesian Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge).
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso).
- [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor).

After choosing the best regressor, we can use Cross Validation, to identify the best k-fold, that predicts the prices as close as possible.

### Results

![Models Comparison](./Figures/Regression_Models.png)

| Linear Regression | Bayesian Ridge | Lasso | Gradient Boosting |
|:---------------|:---------------|:---------------|:---------------|
| MSE | - | - | - | - |
| R-squared | - | - | - | - |


As shown in the figure and performance table, **Gradient Boosting Regressor** was the best model to predict the average prices as close to the real values as possible, with minimum MSE and maximum R-Squared.
The algorithm of the Gradient Boosting regressor can be found [here](https://en.wikipedia.org/wiki/Gradient_boosting#Algorithm), and how the regressor works on python is on scikit-learn documentation, mentioned above.

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

-------
