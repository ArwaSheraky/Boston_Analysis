#!usr/bin/env python

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

# 1. Load & Visualize data
boston_data = load_boston()
df = pd.DataFrame(data=boston_data['data'],
                  columns=boston_data['feature_names'])
df["MEDV"] = boston_data['target']


# 2. Plotting:
sns.set(color_codes=True)

# 2.1 Create Directories to save figs
if not (os.path.exists('./Figures')):
    os.makedirs('./Figures')
    os.makedirs('./Figures/Cols-Histograms')
    os.makedirs('./Figures/Cols-Scatters')
    os.makedirs('./Figures/multiple_features_plotly')

# 2.2 Plot & calculate some overviews
# 2.2.2 Pairplot
print("Creating overview!")
sns.pairplot(df)
plt.savefig("./Figures/Pairplot.png")
plt.close()

# 2.2.3 Correlation matrix
correlation_matrix = df.corr().round(2)
plt.figure(figsize=(20, 15))
sns.heatmap(data=correlation_matrix, annot=True)
plt.savefig("./Figures/Correlation_Matrix.png")
plt.close()

# 2.2.4 Max & Min Corr. to MEDV
medv_corr = correlation_matrix.iloc[13, :-1]
maxcor_col = medv_corr.idxmax()
mincor_col = medv_corr.idxmin()

print("Max Correlation with MEDV: {0}, Corr. value = {1}".format(
    maxcor_col, max(medv_corr)))
print("Min Correlation with MEDV: {0}, Corr. value = {1}".format(
    mincor_col, min(medv_corr)))

# 2.3 Plot Features

# 2.3.1 Histogram for each col.
print("Creating plots!")

sorted_df = df.sort_values("MEDV")
sorted_df = sorted_df.reset_index(drop=True)


for col in df:

    idx = df.columns.get_loc(col)

    sns.distplot(df[col].values, rug=False, bins=25).set_title(
        "Histogram of {0}".format(col))
    plt.savefig(
        "./Figures/Cols-Histograms/{0}_{1}.png".format(idx, col), dpi=100)
    plt.close()

    # 2.2.3 Scatterplot and a regression line for each 2 columns
    for nxt_col in df.iloc[:, idx+1:]:

        sns.regplot(df[col], df[nxt_col], color='r')
        plt.xlabel('Value of {0}'.format(col))
        plt.ylabel('Value of {0}'.format(nxt_col))
        plt.title('Scatter plot of {0} and {1}'.format(col, nxt_col))

        plt.savefig(
            "./Figures/Cols-Scatters/{0}_{1}_{2}".format(idx, col, nxt_col), dpi=200)
        plt.close()

    # 2.3.3 Scatterplot for +3 features (MedV vs LSTAT,RM,col)
    if(col == maxcor_col or col == mincor_col or col == 'MEDV'):
        continue

    trace0 = go.Scatter(
        x=sorted_df['MEDV'],
        y=sorted_df[maxcor_col],
        mode='lines',
        name=maxcor_col
    )

    trace1 = go.Scatter(
        x=sorted_df['MEDV'],
        y=sorted_df[mincor_col],
        mode='lines',
        name=mincor_col
    )

    trace2 = go.Scatter(
        x=sorted_df['MEDV'],
        y=sorted_df[col],
        mode='lines',
        opacity=0.8,
        name=col
    )

    data = [trace0, trace1, trace2]
    layout = go.Layout(
        title='MEDV vs {0}, {1}, {2}'.format(maxcor_col, mincor_col, col),
        yaxis=dict(title='MEDV'),
        xaxis=dict(title='{0}, {1}, {2}'.format(maxcor_col, mincor_col, col)),
        plot_bgcolor="#f3f3f3"
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename="./Figures/multiple_features_plotly/{0}_{1}.html".format(
        idx, col), auto_open=False)

# 3. Apply Regressorss
print("Creating and fitting Regression Model!")

# 3.1 Split the data into training and testing
df_train, df_test, medv_train, medv_test = train_test_split(boston_data["data"],
                                                            boston_data["target"], random_state=5)

# 3.2 Linear Regression

# 3.2.1 Make a model and fit the values
lr_reg = linear_model.LinearRegression()
lr_reg.fit(df_train, medv_train)

predicted_medv = lr_reg.predict(df_test)
expected_medv = medv_test

# 3.2.2 Linear Regression performance

lr_mse = round(mean_squared_error(expected_medv, predicted_medv), 3)
lr_r2 = round(r2_score(expected_medv, predicted_medv), 5)

plt.figure(figsize=(16, 9), dpi=200)
plt.subplot(2, 2, 1)
sns.regplot(expected_medv, predicted_medv, color='g')
plt.ylabel('Predicted Value')
plt.title(
    'Linear Regression.\nMSE= {0} , R-Squared= {1}'.format(lr_mse, lr_r2))

# 3.3 Bayesian Ridge Linear Regression

# 3.3.1 Make a model and fit the values
br_reg = linear_model.BayesianRidge()
br_reg.fit(df_train, medv_train)

predicted_medv = br_reg.predict(df_test)

# 3.3.2 Model performance
br_mse = round(mean_squared_error(expected_medv, predicted_medv), 3)
br_r2 = round(r2_score(expected_medv, predicted_medv), 5)

plt.subplot(2, 2, 2)
sns.regplot(expected_medv, predicted_medv, color='red')
plt.title(
    'Bayesian Ridge Linear Regression.\nMSE= {0} , R-Squared= {1}'.format(br_mse, br_r2))

# 3.4 Lasso

# 3.4.1 Creating a model and fit it
lasso_reg = linear_model.LassoLars(alpha=.1)
lasso_reg.fit(df_train, medv_train)

predicted_medv = lasso_reg.predict(df_test)

# 3.4.2 Model performance
lasso_mse = round(mean_squared_error(expected_medv, predicted_medv), 3)
lasso_r2 = round(r2_score(expected_medv, predicted_medv), 5)

plt.subplot(2, 2, 3)
sns.regplot(expected_medv, predicted_medv, color='orange')
plt.xlabel('Expected Value')
plt.ylabel('Predicted Value')
plt.title(
    'Lasso Linear Regression.\nMSE= {0} , R-Squared= {1}'.format(lasso_mse, lasso_r2))

# 3.5 Gradient boosted tree

# 3.5.1 Make a model and fit the values
gb_reg = GradientBoostingRegressor(loss='ls')
gb_reg.fit(df_train, medv_train)

predicted_medv = gb_reg.predict(df_test)

# 3.5.2 Gradient Boosting performance
gb_mse = round(mean_squared_error(expected_medv, predicted_medv), 3)
gb_r2 = round(r2_score(expected_medv, predicted_medv), 5)

plt.subplot(2, 2, 4)
sns.regplot(expected_medv, predicted_medv, color='b')
plt.xlabel('Expected Value')
plt.title(
    'Gradient Boosting.\nMSE= {0} , R-Squared= {1}'.format(gb_mse, gb_r2))
plt.tight_layout()
plt.savefig("./Figures/1_Regression_Models.png")
plt.close()


# 4. Find/Build the best regressor/Model
d = {'Model': ['Linear Regression', 'Bayesian Ridge', 'Lasso', 'Gradient Boosting'],
     'Variable': [lr_reg, br_reg, lasso_reg, gb_reg],
     'MSE': [lr_mse, br_mse, lasso_mse, gb_mse],
     'R-Squared': [lr_r2, br_r2, lasso_r2, gb_r2]
     }

results_df = pd.DataFrame(data=d)
print(results_df)

# Find the minimum MSE
min_error_df = results_df.sort_values(by=['MSE'])
min_error_df = min_error_df.reset_index(drop=True)

best_regressor = min_error_df.loc[0, "Model"]
print("Best Regressor: ", best_regressor)

# 5 Apply Cross Validation
# 5.1 Choose different values of k-fold
k_models_mse = []
k_fold = [3, 5, 7, 10, 15, 20]

colors = sns.color_palette("Paired")
plt.figure(figsize=(16, 9), dpi=200)

# 5.2 Foreach k in k_fold;
# 5.2.1 create a cross validation model
# 5.2.2 calculate MSE and save it to a list
# 5.2.3 Plot expected vs predicted values

for idx, k in enumerate(k_fold):

    k_model = cross_val_predict(
        min_error_df.loc[0, "Variable"], df_test, medv_test, cv=k)
    k_model_mse = round(mean_squared_error(expected_medv, k_model), 3)
    k_models_mse.append(k_model_mse)

    plt.subplot(2, 3, idx+1)
    sns.regplot(x=expected_medv, y=k_model, color=colors[idx])

    plt.xlabel("Expected")
    plt.ylabel("Predicted")
    plt.title("K = {0} \n MSE= {1}".format(k, k_model_mse))

plt.tight_layout()
plt.savefig("./Figures/2_Gradient_Boosting_CV.png")
plt.close()

# 5.3 Choose best k (Minimum MSE)
min_mse_idx = k_models_mse.index(min(k_models_mse))
best_k = k_fold[min_mse_idx]
print("Best k: ", best_k)

# 6. Build the final Model
final_model = cross_val_predict(
    min_error_df.loc[0, "Variable"], df_test, medv_test, cv=best_k)

sns.regplot(x=expected_medv, y=final_model, color='purple')
plt.xlabel("Expected")
plt.ylabel("Predicted")
plt.title("Final Model: Gradient Boosting.\n CV, k= {0}".format(best_k))

plt.savefig("./Figures/3_Final_Model.png", dpi=200)
plt.close()

print("-------------- FINISHED! --------------")
