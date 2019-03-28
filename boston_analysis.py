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

#1. Load & Visualize data
boston_data = load_boston()
df = pd.DataFrame(data= boston_data['data'], columns= boston_data['feature_names'])
df["MEDV"] = boston_data['target']


#2. Plotting:
sns.set(color_codes=True)

#2.1 Create Directories to save figs
if not (os.path.exists('./Figures')):
    os.makedirs('./Figures')
    os.makedirs('./Figures/Cols-Histograms')
    os.makedirs('./Figures/Cols-Scatters')
    os.makedirs('./Figures/multiple_features_plotly')

#2.2 Plot & calculate some overviews
#2.2.2 Pairplot
print("Creating overview!")
sns.pairplot(df)
plt.savefig("./Figures/Pairplot.png")
plt.close()

#2.2.3 Correlation matrix
correlation_matrix = df.corr().round(2)
plt.figure(figsize=(20, 15))
sns.heatmap(data=correlation_matrix, annot=True)
plt.savefig("./Figures/Correlation_Matrix.png")
plt.close()

#2.2.4 Max & Min Corr. to MEDV
medv_corr = correlation_matrix.iloc[13, :-1]
maxcor_col = medv_corr.idxmax()
mincor_col = medv_corr.idxmin()
print("Max Correlation with MEDV: {0}, Corr. value = {1}".format(
    maxcor_col, max(medv_corr)))
print("Min Correlation with MEDV: {0}, Corr. value = {1}".format(
    mincor_col, min(medv_corr)))

#2.3 Plot Features

#2.3.1 Histogram for each col.
print("Creating histograms and scatter plots!")

for col in df:
    idx = df.columns.get_loc(col)
    sns.distplot(df[col].values,rug=False,bins=50).set_title("Histogram of {0}".format(col))
    plt.savefig("./Figures/Cols-Histograms/{0}_{1}.png".format(idx,col), dpi=100)
    plt.close()

#2.3.2 Scatterplot and a regression line for each column with 'MEDV'
    if (col == 'MEDV'):
        continue
    sns.regplot(df[col], df['MEDV'], color='r')
    plt.xlabel('Value of {0}'.format(col))
    plt.ylabel('Value of MEDV')
    plt.title('Scatter plot of {0} and MEDV'.format(col))
    plt.savefig("./Figures/Cols-Scatters/{0}_{1}_MEDV".format(idx,col), dpi=200)
    plt.close()

#2.3.3 Scatterplot for +3 features
print("Creating plots for 4 features!")
sorted_df = df.sort_values("MEDV")
sorted_df = sorted_df.reset_index(drop=True)

for col in sorted_df:
    if(col == maxcor_col or col == mincor_col or col == 'MEDV'):
        continue
    idx = df.columns.get_loc(col)
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
        title='MEDV vs {0}, {1}, {2}'.format(
            maxcor_col, mincor_col, col),
        yaxis=dict(title='MEDV'),
        xaxis=dict(title='{0}, {1}, {2}'.format(
            maxcor_col, mincor_col, col)),
        plot_bgcolor="#f3f3f3"
        )
fig = go.Figure(data=data, layout=layout)
plot(fig, filename="./Figures/multiple_features_plotly/{0}_{1}.html".format(idx, col), auto_open=False)

#3. Apply Regressorss
print("Creating and fitting Regression Model!")

#3.1 Split the data into training and testing
df_train, df_test, medv_train, medv_test = train_test_split(boston_data["data"],
 boston_data["target"], shuffle = False, test_size=0.3)

#3.2 Linear Regression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

#3.2.1 Make a model and fit the values
lr_reg = linear_model.LinearRegression()
lr_reg.fit(df_train, medv_train)

predicted_medv = lr_reg.predict(df_test)
expected_medv = medv_test

#3.2.2 Linear Regression performance
from sklearn.metrics import r2_score

lr_mse = round(mean_squared_error(expected_medv, predicted_medv),3)
lr_r2 = round(r2_score(expected_medv, predicted_medv),5)

plt.figure(figsize=(16, 9), dpi=200)
plt.subplot(2, 2, 1)
sns.regplot(expected_medv, predicted_medv, color='g')
plt.ylabel('Predicted Value')
plt.title('Linear Regression.\nMSE= {0} , R-Squared= {1}'.format(lr_mse,lr_r2))

#3.3 Bayesian Ridge Linear Regression

#3.3.1 Make a model and fit the values
br_reg = linear_model.BayesianRidge()
br_reg.fit(df_train, medv_train)

predicted_medv = br_reg.predict(df_test)

#3.3.2 Model performance
br_mse =  round(mean_squared_error(expected_medv, predicted_medv),3)
br_r2 = round(r2_score(expected_medv, predicted_medv),5)

plt.subplot(2, 2, 2)
sns.regplot(expected_medv, predicted_medv, color='red')
plt.title('Bayesian Ridge Linear Regression.\nMSE= {0} , R-Squared= {1}'.format(br_mse, br_r2))

#3.4 Lasso

#3.4.1 Creating a model and fit it
lasso_reg = linear_model.LassoLars(alpha=.1)
lasso_reg.fit(df_train, medv_train)

predicted_medv = lasso_reg.predict(df_test)

#3.4.2 Model performance
lasso_mse =  round(mean_squared_error(expected_medv, predicted_medv),3)
lasso_r2 = round(r2_score(expected_medv, predicted_medv),5)

plt.subplot(2, 2, 3)
sns.regplot(expected_medv, predicted_medv, color='orange')
plt.xlabel('Expected Value')
plt.ylabel('Predicted Value')
plt.title('Lasso Linear Regression.\nMSE= {0} , R-Squared= {1}'.format(lasso_mse, lasso_r2))

#3.5 Gradient boosted tree
from sklearn.ensemble import GradientBoostingRegressor

#3.5.1 Make a model and fit the values
gb_reg= GradientBoostingRegressor(loss='ls')
gb_reg.fit(df_train, medv_train)

predicted_medv = gb_reg.predict(df_test)

#3.5.2 Gradient Boosting performance
gb_mse = round(mean_squared_error(expected_medv, predicted_medv),3)
gb_r2 = round(r2_score(expected_medv, predicted_medv),5)

plt.subplot(2, 2, 4)
sns.regplot(expected_medv, predicted_medv, color='b')
plt.xlabel('Expected Value')
plt.title('Gradient Boosting.\nMSE= {0} , R-Squared= {1}'.format(gb_mse,gb_r2))
plt.tight_layout()
plt.savefig("./Figures/Regression_Models.png")
plt.close()

d = {'Model':['Linear Regression', 'Bayesian Ridge' ,'Lasso', 'Gradient Boosting'],
    'Variable': [lr_reg, br_reg, lasso_reg, gb_reg],
    'MSE': [lr_mse, br_mse, lasso_mse, gb_mse],
    'R-Squared': [lr_r2, br_r2, lasso_r2, gb_r2]
    }

results_df = pd.DataFrame(data=d)
print(results_df)

#4. Choose the best regressor
#4.1 Minimum MSE or Maximum R2
min_error_df = results_df.sort_values(by=['MSE'])
min_error_df = min_error_df.reset_index(drop=True)

best_regressor = min_error_df.loc[0,"Model"]
print("Best Regressor: ", best_regressor)

#4.2 Apply Cross Validation

from sklearn.model_selection import cross_val_predict

cv3_predicted_medv = cross_val_predict(min_error_df.loc[0,"Variable"], df_test, medv_test, cv=3)
cv5_predicted_medv = cross_val_predict(min_error_df.loc[0,"Variable"], df_test, medv_test, cv=5)
cv10_predicted_medv = cross_val_predict(min_error_df.loc[0,"Variable"], df_test, medv_test, cv=10)
cv20_predicted_medv = cross_val_predict(min_error_df.loc[0,"Variable"], df_test, medv_test, cv=20)

cvn_models = [cv3_predicted_medv, cv5_predicted_medv, cv10_predicted_medv, cv20_predicted_medv]
cvn=[3,5,10,20]
plt.figure(figsize=(16, 9), dpi=200)

for idx,model in enumerate(cvn_models):

    cv_mse =  round(mean_squared_error(expected_medv, model),3)
    
    plt.subplot(2, 2, idx+1)
    sns.regplot(x=expected_medv ,y=model, color='purple')
    
    plt.xlabel("Expected")
    plt.ylabel("Predicted")
    plt.title("CV = {0} \n MSE= {1}".format(cvn[idx], cv_mse))


plt.tight_layout()
plt.savefig("./Figures/gradient_boosting_cv.png")
plt.close()

print("-------------- FINISHED! --------------")
