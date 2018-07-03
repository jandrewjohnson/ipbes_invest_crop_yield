import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import hazelbean as hb
L = hb.get_logger()



def do_crop_types_ML(**kw):

    ### - - - - - - - - -
    ### Load dataset
    ### - - - - - - - - -
    L.info('Loading data')
    baseline_df = pd.read_csv('../IPBES project/intermediate/baseline_regression_data.csv')
    L.info('Data loaded')
    ### - - - - - - - - -
    ### Cleaning dataset
    ### - - - - - - - - -

    ### - - - - - - - - - - - - - - - -
    ### Feature Engineering / selection
    ### - - - - - - - - - - - - - - - -

    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    #linridge = Ridge(alpha=20.0).fit(X_train_scaled, y_train)

    ### - - - - - - - - -
    ### Train/test split
    ### - - - - - - - - -

    x = baseline_df.drop(['calories_per_cell'], axis=1)
    y = baseline_df['calories_per_cell']

    X_train, X_test, y_train, y_test = train_test_split(x, y)

    # Feature normalization
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ### - - - - - - - - - - - - - -
    ###           MODELS
    ### - - - - - - - - - - - - - -

    L.info('KNeighborsRegressor (scaled)')

    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_score= knn.score(X_test_scaled, y_test)
    print('KNN CLASSIFIER with ',5,' nearest neighbors - - - ')
    print('KNN Score on training set: ', knn.score(X_train_scaled, y_train))
    print('KNN Score on test set: ', knn_score)



    L.info('LinearRegression')

    linreg = LinearRegression().fit(X_train, y_train)
    print('R-squared score (training): {:.3f}'
          .format(linreg.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}'
          .format(linreg.score(X_test, y_test)))



    L.info('LinearRegression (scaled)')

    linreg = LinearRegression().fit(X_train_scaled, y_train)
    print('R-squared score (training): {:.3f}'
          .format(linreg.score(X_train_scaled, y_train)))
    print('R-squared score (test): {:.3f}'
          .format(linreg.score(X_test_scaled, y_test)))



    L.info('Ridge Regression ')
    from sklearn.linear_model import Ridge

    linridge = Ridge(alpha=20.0).fit(X_train, y_train)
    print('R-squared score (training): {:.3f}'
          .format(linridge.score(X_train, y_train)))
    print('R-squared score (test): {:.3f}'
          .format(linridge.score(X_test, y_test)))

    from sklearn.preprocessing import PolynomialFeatures




    L.info('Polynomial Regression ')
    from sklearn.linear_model import LinearRegression
    linlasso = Lasso(alpha=2.0, max_iter=10000).fit(X_train_scaled, y_train)
    print('R-squared score (training): {:.3f}'
          .format(linlasso.score(X_train_scaled, y_train)))
    print('R-squared score (test): {:.3f}\n'
          .format(linlasso.score(X_test_scaled, y_test)))
    ## XG Boost ?

    ## SVM

    L.info('Done')

do_crop_types_ML()