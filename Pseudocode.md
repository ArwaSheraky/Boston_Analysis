* 0. Load data
* 1. Visualize the data as a dataframe
* 2. Plotting
    * 2.1. Create Directories to save figs
    * 2.2. Plot & calculate some overviews
        * 2.2.1. Pairplot
        * 2.2.2. Correlation matrix
        * 2.2.3. Calculate Max & Min Corr. to MEDV

    * 2.3. Plot features
        * 2.3.1. Histogram for each col.
        * 2.3.2. Scatterplot and a regression line for each 2 columns
        * 2.3.3. Scatterplot for +3 features

* 3. Apply Regressors
    * 3.1. Split the data into training and testing
    * 3.2. For each Regression model: (Linear, Bayesian Ridge, Lasso, Gradient Boosting)
        * 3.2.1. Build a model and fit the values
        * 3.2.2. Calculate performance (RMSE, R-Score)
        * 3.2.3. Plot Expected vs Predicted MEDV
    
    *3.3. Chooe the best regressor, with minimum MSE.

* 4. Apply Cross Validation to the regressor
    * 4.1. Choose different k-fold
    * 4.2. Foreach k:
    ```
        * 4.2.1. Build a Cross Validation model, with cv=k
        * 4.2.2. Calculate MSE, save it to a list
        * 4.3.3. Plot expected vs predicted values for this model
    ```
    * 4.3 Choose the best k (Minimum MSE)

* 5. Build the final Model
    * 5.1. Using previous results:
        * Regressor: `Gradient Boosting`.
        * CV: k = 10.
    * 5.2. Calculate the accuracy of the model.
    * 5.3. Plot the final model.