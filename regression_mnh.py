import math
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from voice_features import vf_load, norm_vf, stand_vf
from mnh import MNH_load, df_mnh_vf

data_dir = "/Users/buehlmar/local/projects/playground/audio_pathology/MA Christos"


def load_data(data_dir): # Load CAMP data

    df_vf, pat = vf_load()

    ## Load demographics of patients
    path_demo_csv = os.path.join(data_dir, "PID_info.csv")
    df_demo = pd.read_csv(path_demo_csv)

    ## Load MNH data
    df_MNH, mnh_X = MNH_load(data_dir)

    return df_vf, pat, df_demo, df_MNH



## Simple LR or PR
def SLR_manual(X,y):
    ## Using the MSE as performance metric of our model
    def MSE(X, theta, y):
        SE = 0
        for i in range(len(X)):
            SE += (theta.T.dot(X[i]) - y[i])**2 # h(x) = theta.T.dot(X[i])
        return SE/len(X)

    ## Trying out different values for theta_1
    X_b = np.c_[np.ones((len(X), 1)), X] # add x0=1 to account for bias term
    thetas = np.linspace(0, 1.0, 100)
    MSEs = [MSE(X_b, np.array([0, theta]), y) for theta in thetas]

    plt.figure(figsize=(8, 5))
    plt.plot(thetas, MSEs)
    plt.xlabel('theta_1')
    plt.ylabel('MSE')
    plt.show()

    ## Find the best theta_1
    id_min = MSEs.index(np.min(MSEs))
    print("Best theta:", thetas[id_min])

    ## Computing the optimal theta directly using the normal equation
    X_b = np.c_[np.ones((len(X),1)), X] # add x0=1 to account for bias term
    theta_hat = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    print("Best theta:", theta_hat)

    ## Make predictions using our found parameters
    X_new = np.array([X.max()/2])
    X_new_b = np.c_[np.ones((1,1)), X_new] # again, add x0=1 to account for bias term
    y_prediction = X_new_b.dot(theta_hat)
    print("Predicted value:", y_prediction)

    ##Plot
    X_plt_1 = [X.min()]
    X_plt_2 = [X.max()]
    X_plt_1_a = np.array(X_plt_1)
    X_plt_1_a = np.c_[np.ones((1,1)), X_plt_1_a] # again, add x0=1 to account for bias term
    y_plt_1 = X_plt_1_a.dot(theta_hat)
    X_plt_2_a = np.array(X_plt_2)
    X_plt_2_a = np.c_[np.ones((1,1)), X_plt_2_a] # again, add x0=1 to account for bias term
    y_plt_2 = X_plt_2_a.dot(theta_hat)
    X_plt = [X_plt_1, X_plt_2]
    y_plt = [y_plt_1.tolist()[0], y_plt_2.tolist()[0]]
    plt.figure(figsize=(8, 5))
    plt.plot(X, y, '.', X_plt, y_plt, 'r-')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend(['input data', 'linear fit'])
    plt.show()

def SLR_sklearn(X,y,f, sex, age_min, age_max, R_2_min):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    ## Transform data
    # poly = PolynomialFeatures(degree=max_poly_degr, include_bias=False)
    # poly.fit(X_train)
    # X_train_tf = poly.transform(X_train)
    # X_train = X_train_tf
    # X_test_tf = poly.transform(X_test)
    # X_test = X_test_tf

    lin_reg = LinearRegression() # create an instance of the LinearRegression class
    lin_reg.fit(X_train, y_train) # fit training data

    #coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
    #print(coeff_df)
    # print('Intercept: \n', lin_reg.intercept_)
    # print('Coefficients: \n', lin_reg.coef_)
    y_pred_train = lin_reg.predict(X_train)
    y_pred_test = lin_reg.predict(X_test)

    #df_result_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
    #print("Train",df_result_train)
    #df_result_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    #print("Test", df_result_test)

    ## Compute the MSE for the training set and the test set (i.e. the training error and the test error)
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    MAE_test = mean_absolute_error(y_test, y_pred_test)

    RMSE_train = r2_score(y_train, y_pred_train)
    RMSE_test = r2_score(y_test, y_pred_test)

    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)

    def err(y_true, y_pred):
        oe = np.std((y_true - y_pred) ** 2, axis=0)
        return oe

    print("Linear Regression")
    print("MSE training set:", MSE_train, err(y_train, y_pred_train), len(X_train))
    print("MSE test set:", MSE_test, err(y_test, y_pred_test), len(X_test))

def SLR_sklearn_old(X,y,f, sex, age_min, age_max, R_2_min):
    lin_reg = LinearRegression() # create an instance of the LinearRegression class
    lin_reg.fit(X, y) # call the .fit method to compute the optimal parameters (no need to manually add bias)
    # print("Theta_0:", lin_reg.intercept_)
    # print("Theta_1:", lin_reg.coef_)

    ## Make a prediction using the trained regressor
    X_ex = (X.max()-X.min())/2
    X_new = [[X_ex]] # shape (m x 1)
    y_prediction_sklearn = lin_reg.predict(X_new)
    # print("Predicted value for %f:" %X_ex, y_prediction_sklearn)

    R_2 = lin_reg.score(X, y) # computes the R^2 score (R^2=1 means perfect fit)
    # print("R^2:", R_2)

    if R_2 > R_2_min:
        # print(e,f)
        ##Plot
        X_plt = [[X.min()], [X.max()]]
        y_plt = lin_reg.predict(X_plt)
        plt.figure(figsize=(8, 5))
        plt.plot(X, y, '.', X_plt, y_plt, 'r-')
        plt.title("Sex %d, Age %d - %d, R^2 = %1.2f" %(sex, age_min, age_max, R_2))
        plt.xlabel('X = %s' %f)
        plt.ylabel('y = Mac New Heart')
        plt.legend(['input data', 'linear fit'])
        plt.show()

def SLR_sklearn_scikit(X,y):
    # Split the data into training/testing sets
    X_train = X[:-20]
    X_test = X[-20:]

    # Split the targets into training/testing sets
    y_train = y[:-20]
    y_test = y[-20:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

    # Plot outputs
    plt.scatter(X_test, y_test, color="black")
    plt.plot(X_test, y_pred, color="blue", linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()

def PR_sklearn(X,y,f, sex, age_min, age_max, R_2_min, max_poly_degr):
    lin_reg = LinearRegression() # create an instance of the LinearRegression class
    X_plt = np.linspace(X.min(),X.max(),100) # create data to be predicted
    X_plt = np.expand_dims(X_plt, axis=1)
    scores = []
    degrees = [*range(1, max_poly_degr+1, 1)]
    s = int(math.ceil(math.sqrt(max_poly_degr)))
    for degree in degrees:
        # create polynomial feature and fit model
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_tf = poly.fit_transform(X)
        lin_reg.fit(X_tf, y)

        ## Compute the R^2-scores for the models
        scores.append(lin_reg.score(X_tf, y))

    if max(scores) > R_2_min:
        plt.figure(figsize=(15, 8))
        for degree in degrees:
            plt.subplot(s, s, degree)

            # create polynomial feature and fit model
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_tf = poly.fit_transform(X)
            lin_reg.fit(X_tf, y)
            R_2 = lin_reg.score(X_tf, y)

            # make predictions and plot data
            X_plt_tf = poly.transform(X_plt)
            y_plt = lin_reg.predict(X_plt_tf)

            plt.plot(X, y, '.', X_plt, y_plt, 'r-')
            plt.xlabel('X = %s' %f)
            plt.ylabel('y = mnh')
            #plt.ylim([-200, 300])
            plt.legend(['input data', 'polynomial fit'])
            plt.title("polynomial feature degree=%d, R^2 = %1.2f" %(degree, R_2))
            plt.tight_layout()
            plt.suptitle("Sex %d, Age %d - %d" %(sex, age_min, age_max))
        plt.show()

        ## Plot the R^2-scores for the models
        # plt.figure(figsize=(6, 5))
        # plt.plot(degrees, scores,'o-')
        # plt.xlabel('Highest degree of polynomial feature')
        # plt.ylabel('$R^2$ score')
        # plt.show()

        ## Retrieve the parameters and the degree of the unknown polynomial function using the insight from above
        poly = PolynomialFeatures(degree=degrees[scores.index(max(scores))], include_bias=False)
        X_tf = poly.fit_transform(X)
        lin_reg.fit(X_tf, y)
        print(lin_reg.coef_)
        print(lin_reg.intercept_)

def PR_ridge(X,y,f, sex, age_min, age_max, R_2_min, max_poly_degr):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

    poly = PolynomialFeatures(degree=max_poly_degr, include_bias=False)
    poly.fit(X_train)
    X_train_tf = poly.transform(X_train)
    X_test_tf = poly.transform(X_test)

    print(X_train.shape)
    print(X_train_tf.shape)

    ## Unregularized linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_tf, y_train)

    y_pred_train = lin_reg.predict(X_train_tf)
    y_pred_test = lin_reg.predict(X_test_tf)

    ## Use the provided function plot_model() to plot the predicted y-values (targets) of the training set against the ground truth
    def plot_model(X, y, y_pred, title='', show_plot=True): # Function to plot model prediction against ground truth
        plt.scatter(X, y) # X: Matrix of training or test set instances; y: Vector of training or test set targets
        order = np.argsort(X.ravel())
        plt.plot(X[order], y_pred[order], 'r', linewidth=2) # y_pred: Vector of predicted targets
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(title)
        if show_plot: # bool, draw plot using plt.show(). Disable for use in subplots.
            plt.show()

    plt.figure(figsize=(8, 5))
    plot_model(X_train, y_train, y_pred_train, title='unregularized model')

    ## Training error and test error
    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    print("MSE training set:", MSE_train)
    print("MSE test set:", MSE_test)

    # Ridge
    plt.figure(figsize=(15,8))
    i = 1
    alphas = [0, 0.01, 0.1, 1e6]
    for alpha in alphas:
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train_tf, y_train)
        y_pred_train = ridge_reg.predict(X_train_tf)
        plt.subplot(2, 2, i)
        plot_model(X_train, y_train, y_pred_train, title="alpha={}".format(alpha), show_plot=False)
        i+=1
    plt.show()

    ## Compute the MSE for the training set and the test set using Ridge regression. Plot the results in function of the regularization hyperparameter alpha
    alphas = np.linspace(0.0001, 0.003, 1000)

    MSEs_train = []
    MSEs_test = []
    for alpha in alphas:
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train_tf, y_train)
        y_pred_train = ridge_reg.predict(X_train_tf)
        y_pred_test = ridge_reg.predict(X_test_tf)
        MSE_train = mean_squared_error(y_train, y_pred_train)
        MSE_test = mean_squared_error(y_test, y_pred_test)
        MSEs_train.append(MSE_train)
        MSEs_test.append(MSE_test)

    idmin = np.where(MSEs_test == min(MSEs_test))

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, MSEs_train, alphas, MSEs_test)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('MSE')
    plt.legend(['training error', 'test error'])
    plt.title("Best alpha: {:.5f}".format(alphas[idmin][0]))
    plt.show()

    # Using alpha=0.003
    ridge_reg = Ridge(alpha=0.003)
    ridge_reg.fit(X_train_tf, y_train)
    y_pred_train = ridge_reg.predict(X_train_tf)
    y_pred_test = ridge_reg.predict(X_test_tf)

    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    print("MSE training set:", MSE_train)
    print("MSE test set:", MSE_test)

    plt.figure(figsize=(8, 5))
    plot_model(X_train, y_train, y_pred_train, title="optimal model")

def PR_lasso(X,y,f, sex, age_min, age_max, R_2_min, max_poly_degr):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

    poly = PolynomialFeatures(degree=max_poly_degr, include_bias=False)
    poly.fit(X_train)
    X_train_tf = poly.transform(X_train)
    X_test_tf = poly.transform(X_test)

    print(X_train.shape)
    print(X_train_tf.shape)

    ## Unregularized linear regression model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_tf, y_train)

    y_pred_train = lin_reg.predict(X_train_tf)
    y_pred_test = lin_reg.predict(X_test_tf)

    ## Use the provided function plot_model() to plot the predicted y-values (targets) of the training set against the ground truth
    def plot_model(X, y, y_pred, title='', show_plot=True): # Function to plot model prediction against ground truth
        plt.scatter(X, y) # X: Matrix of training or test set instances; y: Vector of training or test set targets
        order = np.argsort(X.ravel())
        plt.plot(X[order], y_pred[order], 'r', linewidth=2) # y_pred: Vector of predicted targets
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(title)
        if show_plot: # bool, draw plot using plt.show(). Disable for use in subplots.
            plt.show()

    plt.figure(figsize=(8, 5))
    plot_model(X_train, y_train, y_pred_train, title='unregularized model')

    ## Training error and test error
    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    print("MSE training set:", MSE_train)
    print("MSE test set:", MSE_test)

    # Lasso
    plt.figure(figsize=(15,8))
    i = 1
    alphas = [0.0001, 0.001, 0.01, 1]
    for alpha in alphas:
        lasso_reg = Lasso(alpha=alpha, tol=0.01)
        lasso_reg.fit(X_train_tf, y_train)
        y_pred_train = lasso_reg.predict(X_train_tf)
        plt.subplot(2, 2, i)
        plot_model(X_train, y_train, y_pred_train, title="alpha={}".format(alpha), show_plot=False)
        i+=1
    plt.show()

    ## Observe the coefficients of your Lasso model as you increase the amount of regularization.
    alpha = 0.0001
    lasso_reg = Lasso(alpha=alpha, tol=0.01)
    lasso_reg.fit(X_train_tf, y_train)
    print("Coefficients alpha =", alpha)
    print(lasso_reg.coef_)

    alpha = 0.01
    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X_train_tf, y_train)
    print("Coefficients alpha =", alpha)
    print(lasso_reg.coef_)

    alpha = 0.1
    lasso_reg = Lasso(alpha=alpha)
    lasso_reg.fit(X_train_tf, y_train)
    print("Coefficients alpha =", alpha)
    print(lasso_reg.coef_)

def SLR_sklearn_rf(X,y,f, sex, age_min, age_max, R_2_min):
    lin_reg = RandomForestRegressor() # create an instance of the LinearRegression class
    lin_reg.fit(X, y) # call the .fit method to compute the optimal parameters (no need to manually add bias)
    # print("Theta_0:", lin_reg.intercept_)
    # print("Theta_1:", lin_reg.coef_)

    ## Make a prediction using the trained regressor
    X_ex = (X.max()-X.min())/2
    X_new = [[X_ex]] # shape (m x 1)
    y_prediction_sklearn = lin_reg.predict(X_new)
    # print("Predicted value for %f:" %X_ex, y_prediction_sklearn)

    R_2 = lin_reg.score(X, y) # computes the R^2 score (R^2=1 means perfect fit)
    # print("R^2:", R_2)

    if R_2 > R_2_min:
        # print(e,f)
        ##Plot
        X_plt = [[X.min()], [X.max()]]
        y_plt = lin_reg.predict(X_plt)
        plt.figure(figsize=(8, 5))
        plt.plot(X, y, '.', X_plt, y_plt, 'r-')
        plt.title("Sex %d, Age %d - %d, R^2 = %1.2f" %(sex, age_min, age_max, R_2))
        plt.xlabel('X = %s' %f)
        plt.ylabel('y = Mac New Heart')
        plt.legend(['input data', 'linear fit'])
        plt.show()


## Multiple LR
def MLR_sklearn(X,y,f, sex, age_min, age_max, R_2_min):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    ## Transform data
    # poly = PolynomialFeatures(degree=max_poly_degr, include_bias=False)
    # poly.fit(X_train)
    # X_train_tf = poly.transform(X_train)
    # X_train = X_train_tf
    # X_test_tf = poly.transform(X_test)
    # X_test = X_test_tf

    lin_reg = LinearRegression() # create an instance of the LinearRegression class
    lin_reg.fit(X_train, y_train) # fit training data

    #coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
    #print(coeff_df)
    # print('Intercept: \n', lin_reg.intercept_)
    # print('Coefficients: \n', lin_reg.coef_)
    y_pred_train = lin_reg.predict(X_train)
    y_pred_test = lin_reg.predict(X_test)

    df_result_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
    #print("Train",df_result_train)
    df_result_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    #print("Test", df_result_test)

    ## Compute the MSE for the training set and the test set (i.e. the training error and the test error)
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    MAE_test = mean_absolute_error(y_test, y_pred_test)

    RMSE_train = r2_score(y_train, y_pred_train)
    RMSE_test = r2_score(y_test, y_pred_test)

    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)

    def err(y_true, y_pred):
        oe = np.std((y_true - y_pred) ** 2, axis=0)
        return oe

    print("MSE training set:", MSE_train, err(y_train, y_pred_train), len(X_train))
    print("MSE test set:", MSE_test, err(y_test, y_pred_test), len(X_test))

def MLR_ridge(X,y,f, sex, age_min, age_max, R_2_min):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    ## Transform data
    poly = PolynomialFeatures(degree=max_poly_degr, include_bias=False)
    poly.fit(X_train)
    X_train_tf = poly.transform(X_train)
    X_train = X_train_tf
    X_test_tf = poly.transform(X_test)
    X_test = X_test_tf

    ## Ridge
    ## Compute the MSE for the training set and the test set using Ridge regression. Plot the results in function of the regularization hyperparameter alpha
    alphas = np.linspace(0.0001, 0.003, 1000)

    MSEs_train = []
    MSEs_test = []
    for alpha in alphas:
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train_tf, y_train)
        y_pred_train = ridge_reg.predict(X_train_tf)
        y_pred_test = ridge_reg.predict(X_test_tf)
        MSE_train = mean_squared_error(y_train, y_pred_train)
        MSE_test = mean_squared_error(y_test, y_pred_test)
        MSEs_train.append(MSE_train)
        MSEs_test.append(MSE_test)

    idmin = np.where(MSEs_test == min(MSEs_test))

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, MSEs_train, alphas, MSEs_test)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('MSE')
    plt.legend(['training error', 'test error'])
    plt.title("Best alpha: {:.5f}".format(alphas[idmin][0]))
    plt.show()

    ## Use best alpha
    alpha = 0.003
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X_train_tf, y_train)
    y_pred_train = ridge_reg.predict(X_train_tf)
    y_pred_test = ridge_reg.predict(X_test_tf)

    df_result_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
    print(df_result_train)
    df_result_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    print(df_result_test)

    ## Compute the MSE for the training set and the test set (i.e. the training error and the test error)
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    MAE_test = mean_absolute_error(y_test, y_pred_test)

    RMSE_train = r2_score(y_train, y_pred_train)
    RMSE_test = r2_score(y_test, y_pred_test)

    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    print("MSE training set:", MSE_train)
    print("MSE test set:", MSE_test)

def MLR_lasso(X,y,f, sex, age_min, age_max, R_2_min):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    ## Use PolynomialFeatures to transform your dataset using a 20-degree polynomial
    poly = PolynomialFeatures(degree=max_poly_degr, include_bias=False)
    poly.fit(X_train)
    X_train_tf = poly.transform(X_train)
    X_train = X_train_tf
    X_test_tf = poly.transform(X_test)
    X_test = X_test_tf

    # Lasso
    alpha = 0.0001
    lasso_reg = Lasso(alpha=alpha, tol=0.01)
    lasso_reg.fit(X_train_tf, y_train)
    print("Coefficients alpha =", alpha)
    print(lasso_reg.coef_)

    alpha = 0.01
    lasso_reg = Lasso(alpha=alpha, tol=0.01)
    lasso_reg.fit(X_train_tf, y_train)
    print("Coefficients alpha =", alpha)
    print(lasso_reg.coef_)

    alpha = 0.1
    lasso_reg = Lasso(alpha=alpha, tol=0.01)
    lasso_reg.fit(X_train_tf, y_train)
    print("Coefficients alpha =", alpha)
    print(lasso_reg.coef_)

    y_pred_train = lasso_reg.predict(X_train_tf)
    y_pred_test = lasso_reg.predict(X_test_tf)

    df_result_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
    print(df_result_train)
    df_result_test = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})
    print(df_result_test)

    ## Compute the MSE for the training set and the test set (i.e. the training error and the test error)
    MAE_train = mean_absolute_error(y_train, y_pred_train)
    MAE_test = mean_absolute_error(y_test, y_pred_test)

    RMSE_train = r2_score(y_train, y_pred_train)
    RMSE_test = r2_score(y_test, y_pred_test)

    MSE_train = mean_squared_error(y_train, y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    print("MSE training set:", MSE_train)
    print("MSE test set:", MSE_test)


    scores_dict_train = calculate_scores(y_pred=y_pred_train, y_true=y_train)
# feat = 88 # feature number according to excel table
# feat = feat - 2
# print(df.columns[feat])

start = time.time()
# np.random.seed(0)
# X = np.array([df.loc[:,df.columns[feat]]]).T #(1000, 1)
# y = np.array([df['Pleasure']]).T
# SLR_sklearn(X,y)

#Plot X,y
# plt.figure(figsize=(8, 5))
# plt.plot(X, y, '.')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.show()
age_min = 20
age_max = 90
sex = 1

R_2_min = 0.01
max_poly_degr = 4

patients_checked = []
np.random.seed(0)

f0 = "F0semitoneFrom27.5Hz_sma3nz_amean"
f1 = "F1frequency_sma3nz_amean"
f2 = "F2frequency_sma3nz_amean"
f3 = "F3frequency_sma3nz_amean"
ji = "jitterLocal_sma3nz_amean"
shi = "shimmerLocaldB_sma3nz_amean"
mfcc1 = "mfcc1_sma3_amean"
mfcc2 = "mfcc2_sma3_amean"
mfcc3 = "mfcc3_sma3_amean"
mfcc4 = "mfcc4_sma3_amean"



df_vf, pat, df_demo, df_MNH = load_data()
for sex in range(2,3):# = 1 # 1=male, 2=female
    for age_min in range(20,70,20):
        if age_min >= 60:
            age_max = age_min + 19 + 11
        else:
            age_max = age_min + 19
        age_min = df_demo['age'].min()
        age_max = df_demo['age'].max()

    df_group = load_group(df_vf, df_demo, sex, age_min, age_max)
    mv = df_mnh_vf(df_MNH, df_vf, df_group)
    mv = mv.dropna(subset=['mnh'])
    mv.set_index('PID', inplace=True)
    #norm_vf(mv, list(mv.columns[4:]))
    voice_features = [f0, ji, shi, mfcc1]

    ## Simple LR or PR
    for f in voice_features:
        print(f)
        np.random.seed(0)
        X = np.array([mv.loc[:,f]]).T
        y = np.array([mv['mnh']]).T
        SLR_sklearn(X,y,f, sex, age_min, age_max, R_2_min)
        #SLR_sklearn_scikit(X,y)#,f, sex, age_min, age_max, R_2_min)
        #PR_sklearn(X,y,f, sex, age_min, age_max, R_2_min, max_poly_degr)
        #PR_ridge(X, y, f, sex, age_min, age_max, R_2_min, max_poly_degr)
        #PR_lasso(X, y, f, sex, age_min, age_max, R_2_min, max_poly_degr)
        #SLR_sklearn_rf(X,y,f, sex, age_min, age_max, R_2_min)

    ## Multiple LR
    # vf2feat = list([mv.columns[4]]) + list([mv.columns[14]]) #+ [mv.columns[24]] + [mv.columns[26] + [mv.columns[28]]] + [mv.columns[30]] + [mv.columns[32]] + [mv.columns[34]] + [mv.columns[36]] + [mv.columns[44]] + [mv.columns[50]] + [mv.columns[52]] + [mv.columns[56]] + [mv.columns[58]])
    # vf2feat = mv.columns[4:6]

    X = mv[voice_features]
    y = mv.values[:,3]
    #MLR_sklearn(X,y,f, sex, age_min, age_max, R_2_min)
    #MLR_ridge(X,y,f, sex, age_min, age_max, R_2_min)
    #MLR_lasso(X,y,f, sex, age_min, age_max, R_2_min)
    MLR_sklearn_rf(X,y,f, sex, age_min, age_max, R_2_min)

end = time.time()
#plt.show()

#end = time.time()
print('\n')
print((end-start)/60)
