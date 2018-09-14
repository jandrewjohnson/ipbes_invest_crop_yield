
# coding: utf-8

# # Imports

# In[19]:


import pandas as pd
import numpy as np
from scipy import stats

 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import xgboost as xgb
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import permutation_test_score

#from mpl_toolkits.basemap import Basemap
import hazelbean as hb

#import seaborn as sns
#import matplotlib.pyplot as plt

#pd.set_option('display.max_columns', 500) pd.set_option('display.max_rows', 500)
import pickle


# # Load dataset

# ## Load data

# In[5]:


def load_data(subset=True):
    ''' subset takes values of : True/float/False
            True : (sampling of 2% by default)
            Float : Fraction to sample (e.g 0.10 for 10%)
            False : returns df, X_validation, y_validation
    '''
    
    #L.info('Loading data')
    crop_types_df = pd.read_csv('../ipbes_invest_crop_yield_project/intermediate/aggregate_crops_by_type/aggregated_crop_data.csv')
    df_land = pd.read_csv('../ipbes_invest_crop_yield_project/intermediate/create_baseline_regression_data/baseline_regression_data.csv')
    #L.info('Data loaded')

    df = crop_types_df.merge(df_land,how='outer',on='pixel_id')
    #L.info('Data merged')

    #Remove cal_per_ha per crop type for now
    df = df.drop(labels=['c3_annual_calories_per_ha', 'c3_perennial_calories_per_ha',
           'c4_annual_calories_per_ha', 'c4_perennial_calories_per_ha',
           'nitrogen_fixer_calories_per_ha'], axis=1)

    #Remove helper columns (not features)
    df = df.drop(labels=['Unnamed: 0', 'country_ids',
           'ha_per_cell_5m'], axis=1)

    # Rename cols
    df = df.rename(columns={'bio12': 'precip', 'bio1': 'temperature',
                                'minutes_to_market_5m': 'min_to_market',
                                'gdp_per_capita_2000_5m': 'gdp_per_capita',
                                'gdp_2000': 'gdp'})
    # Encode properly NaNs
    df['slope'] = df['slope'].replace({0: np.nan})  # 143 NaN in 'slope' variable
    for soil_var in ['workability_index', 'toxicity_index', 'rooting_conditions_index', 'oxygen_availability_index',
                     'nutrient_retention_index', 'nutrient_availability_index', 'excess_salts_index']:
        df[soil_var] = df[soil_var].replace({255: np.nan})
        
    # Drop NaN
    df = df.dropna()
    df = df[df['calories_per_ha'] != 0]    
    
    #Encode climate zones (as str)
    climate_zones_map = {1:'Af',2:'Am',3:'Aw',
                     5:'BWk',4:'BWh',7:'BSk',6:'BSh',
                     14:'Cfa',15:'Cfb',16:'Cfc',8:'Csa',
                     9:'Csb',10:'Csc',11:'Cwa',12:'Cwb',13:'Cwc',
                     25:'Dfa',26:'Dfb',27:'Dfc',28:'Dfd',17:'Dsa',18:'Dsb',19:'Dsc',
                     20:'Dsd',21:'Dwa',22:'Dwb',23:'Dwc',24:'Dwd',
                     30:'EF',29:'ET'}
    df['climate_zones'] = df['climate_zones'].map(climate_zones_map)
    
    # Encode climate zones as dummies
    climate_dummies_df = pd.get_dummies(df['climate_zones'])
    for col in climate_dummies_df.columns:
        climate_dummies_df = climate_dummies_df.rename({col:str('climatezone_'+col)},axis=1)
    
    df = df.merge(climate_dummies_df, right_index=True,left_index=True)
    df = df.drop('climate_zones',axis=1)
    
    # Lat/Lon
    df['sin_lon'] = df['lon'].apply(lambda x:np.sin(np.radians(x)))
    df = df.drop('lon',axis=1)
    #df['sin_lat'] = df['lat'].apply(lambda x:np.sin(np.radians(x)))
    
    # Log some skewed variables
    df['calories_per_ha'] = df['calories_per_ha'].apply(lambda x: np.log(x) if x != 0 else 0)

    for col in ['gdp_per_capita','altitude', 'min_to_market', 'gpw_population']:
        df[str('log_'+col)] = df[col].apply(lambda x: np.log(x) if x != 0 else 0)
        df = df.drop(col,axis=1)
        
        
    # Slope
    df['slope'] = df['slope'].apply(lambda x:x-90)
    
    # Encode properly NaNs
    df['slope'] = df['slope'].replace({0: np.nan})  # 143 NaN in 'slope' variable
    for soil_var in ['workability_index', 'toxicity_index', 'rooting_conditions_index', 'oxygen_availability_index',
                     'nutrient_retention_index', 'nutrient_availability_index', 'excess_salts_index']:
        df[soil_var] = df[soil_var].replace({255: np.nan})
        
        
    
    # Cols to drop
    for col in ['pixel_id_float', 'land_mask']:
        df = df.drop(col,axis=1)
        
    if subset==True:
        #subset = 0.02 if type(subset) is bool else subset
        df = df.sample(frac=0.02, replace=False, weights=None, random_state=None, axis=0)

    elif subset==False: #Save validation data
        x = df.drop(['calories_per_ha'], axis=1)
        y = df['calories_per_ha']
        X, X_validation, Y, y_validation = train_test_split(x, y)
        df = X.merge(pd.DataFrame(Y),how='outer',left_index=True,right_index=True)
    
    #Set index 'pixel_id'
    df = df.set_index('pixel_id')
    
    y_val = pd.DataFrame(X_validation['pixel_id']).merge(pd.DataFrame(y_validation),how='outer',left_index=True,right_index=True)
    y_validation = y_val.set_index('pixel_id')
    X_validation = X_validation.set_index('pixel_id')

    if subset==True:
        return df
    
    elif subset==False:
        return df, X_validation, y_validation


# In[142]:


df, X_validation, y_validation = load_data(subset=False)


# In[148]:


columns_without_climatezones = df.columns
for col in df.columns:
    if "climatezone" in col: 
        columns_without_climatezones = columns_without_climatezones.drop([col])


# # Utilities functions

# ## Functions to make regressions

# In[5]:


def do_regression(regression,dataframe):
    ##Must make dummies for categorical variable climate_zone
    # dataframe = pd.get_dummies(dataframe, columns=['climate_zone'])
    # Or just drop column if don't want dummies: x = x.drop(['climate_zone'], axis=1)

    x = dataframe.drop(['calories_per_ha'], axis=1)
    y = dataframe['calories_per_ha']

    ### Cross validation scores
    r2_scores = cross_val_score(regression, x, y, cv=5,scoring='r2')
    mse_scores = cross_val_score(regression, x, y, cv=5, scoring='neg_mean_squared_error')
    mae_scores = 0 #cross_val_score(regression, x, y, cv=5, scoring='neg_mean_absolute_error')
    
    return [np.mean(r2_scores),np.mean(mse_scores),np.mean(mae_scores)]


# In[6]:


def compare_predictions(regression,dataframe,show_df=True,show_plot=True):
    x = dataframe.drop(['calories_per_ha','climate_zones'], axis=1)
    y = dataframe['calories_per_ha']
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    reg = regression.fit(X_train, y_train)
    y_predicted = reg.predict(X_test)

    compare = pd.DataFrame()
    compare['y_test'] = y_test
    compare['predicted'] = y_predicted

    if show_plot == True:
        ax = compare.plot.scatter(x='y_test',y='predicted',s=0.5)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
        
    return compare


# In[129]:




def run_regressions_and_save_results(model, regression, dataframe, features_selection,
                                     results_df,parameters_dict=None,inputs = None):
    '''features_selection = 'all', 'all_but_climzones', 'RFE', 'RFE_but_climzones   '''

    # - - - - - - - - - - - - - - 
    # Features = All
    # - - - - - - - - - - - - - - 
    
    if features_selection == 'all':
        print('    with all features...')
        Features = 'All w/ climzones'
        
        scores = do_regression(regression,dataframe)
        
        print('    R2_score : '+str(scores[0]))
        print('    MSE_score : '+str(scores[1]))
        R2_score = scores[0]
        MSE_score = scores[1]
        results_df = results_df.append({'Model': model,
                                        'num_features':len(dataframe.columns)-1,'Features':Features,
                                        'params':parameters_dict,
                                        'R2':R2_score,'MSE':MSE_score},ignore_index=True)
        
        
    # - - - - - - - - - - - - - - -
    # Features = All but climzones
    # - - - - - - - - - - - - - - -
    
    elif features_selection == 'all_but_climzones':
        dataframe = dataframe[columns_without_climatezones]
        
        print('    with all features but climate zones...')
        Features = 'All w/o climzones'
        
        scores = do_regression(regression,dataframe)
        
        print('    R2_score : '+str(scores[0]))
        print('    MSE_score : '+str(scores[1]))
        R2_score = scores[0]
        MSE_score = scores[1]
        results_df = results_df.append({'Model': model,
                                        'num_features':len(dataframe.columns)-1,'Features':Features,
                                        'params':parameters_dict,
                                    'R2':R2_score,'MSE':MSE_score},ignore_index=True)
           
    
    # - - - - - - - - - - - - - - -
    # Features = SUBSET
    # - - - - - - - - - - - - - - -
    
    # if features_selection = 'subset':
    ## TODO
    
    
    # - - - - - - - - - - - - - - - - - - - - - - -
    # Features = RFE selected (with climate zones)
    # - - - - - - - - - - - - - - - - - - - - - - -
    
    elif features_selection == 'RFE':
        for num_features in range (5,30):

            print(' RFE with '+str(num_features)+ ' features ...')

            ## RFE - Features selection
            selector = RFE(regression, num_features, step=1)
            x = dataframe.drop(['calories_per_ha'], axis=1) 
            y = dataframe['calories_per_ha']
            X, X_test, Y, Y_test = train_test_split(x, y)
            X_RFE = selector.fit_transform(X,Y)
            features_selected = [X.columns[feature_pos] for feature_pos in selector.get_support(indices=True)]
            
            # Do regression and append results to results_df
            scores = do_regression(regression,dataframe[(features_selected + ['calories_per_ha'])])
            print('    R2_score : '+str(scores[0]))
            print('    MSE_score : '+str(scores[1]))
            R2_score = scores[0]
            MSE_score = scores[1]
            results_df = results_df.append({'Model': model,
                                    'num_features':num_features,'Features':features_selected,
                                     'params':parameters_dict,
                                    'R2':R2_score,'MSE':MSE_score},ignore_index=True)
        
        
    # - - - - - - - - - - - - - - - - - - - - - - 
    # Features = RFE selected (w/o climate zones)
    # - - - - - - - - - - - - - - - - - - - - - - 
    
    elif features_selection == 'RFE_but_climzones':
        
        dataframe = dataframe[columns_without_climatezones]
        
        for num_features in range (5,30):

            print('RFE (no climzones) with '+str(num_features)+ ' features ...')

            ## RFE - Features selection
            selector = RFE(regression, num_features, step=1)
            x = dataframe.drop(['calories_per_ha'], axis=1) 
            y = dataframe['calories_per_ha']
            X, X_test, Y, Y_test = train_test_split(x, y)
            X_RFE = selector.fit_transform(X,Y)
            features_selected = [X.columns[feature_pos] for feature_pos in selector.get_support(indices=True)]
   
            # Do regression and append results to results_df
            scores = do_regression(regression,dataframe[(features_selected + ['calories_per_ha'])])
            print('    R2_score : '+str(scores[0]))
            print('    MSE_score : '+str(scores[1]))
            R2_score = scores[0]
            MSE_score = scores[1]
            results_df = results_df.append({'Model': model,
                                    'num_features':num_features,'Features':features_selected,
                                    'params':parameters_dict,
                                    'R2':R2_score,'MSE':MSE_score},ignore_index=True)
            
    elif features_selection == 'RFE_8_20':
        
        
        for num_features in [8,20]:

            print('RFE with '+str(num_features)+ ' features ...')

            ## RFE - Features selection
            selector = RFE(regression, num_features, step=1)
            x = dataframe.drop(['calories_per_ha'], axis=1) 
            y = dataframe['calories_per_ha']
            X, X_test, Y, Y_test = train_test_split(x, y)
            X_RFE = selector.fit_transform(X,Y)
            features_selected = [X.columns[feature_pos] for feature_pos in selector.get_support(indices=True)]
   
            # Do regression and append results to results_df
            scores = do_regression(regression,dataframe[(features_selected + ['calories_per_ha'])])
            print('    R2_score : '+str(scores[0]))
            print('    MSE_score : '+str(scores[1]))
            R2_score = scores[0]
            MSE_score = scores[1]
            results_df = results_df.append({'Model': model,
                                    'num_features':num_features,'Features':features_selected,
                                    'params':parameters_dict,
                                    'R2':R2_score,'MSE':MSE_score,
                                    'inputs' : inputs},ignore_index=True)
    
        
    return(results_df)


# # Modeling runs

# In[ ]:


def make_polynomial(df, degree = 2, interaction_terms = True):
#'''Returns a new dataFrame with added polynomial terms degree >= 2'''


    if interaction_terms == False:
        x = df.drop(['calories_per_ha'], axis=1)
        y = df['calories_per_ha']

        for deg in range(2,degree+1):
            for col in x.columns:
                x[str(col+'^'+str(deg))] = x[col].apply(lambda x:x**deg)
        
        Poly_df = x.merge(pd.DataFrame(y),right_index=True,left_index=True)

    if interaction_terms == True:
        x = df.drop(['calories_per_ha'], axis=1)
        y = df['calories_per_ha'].reset_index()

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X2 = poly.fit_transform(x)

        Poly_df = pd.DataFrame(data = np.concatenate((y.as_matrix(),X2),axis=1),
                               columns = ['pixel_id','calories_per_ha'] +
                               poly.get_feature_names((df.drop(['calories_per_ha'], axis=1)).columns))

        Poly_df = Poly_df.set_index('pixel_id')

    return Poly_df


# ## Get Validation R2 and save models

# In[ ]:


# Save the model to disk (once fitted)
def save_model(filename):
    pickle.dump(regression, open(filename, 'wb'))


# In[ ]:


simple_subset = ['slope','lat','sin_lon','log_altitude',
                  'workability_index',
                  'log_gpw_population',
                  'temp_avg','precip',
                  'calories_per_ha']

simple_subset_gdp = ['slope','lat','sin_lon','log_altitude',
                  'workability_index',
                  'gdp','log_gdp_per_capita','log_gpw_population',
                  'temp_avg','precip',
                  'calories_per_ha']


# In[ ]:


df_sub = df[simple_subset]
df_sub_gdp = df[simple_subset_gdp]
df_wo_cz = df[columns_without_climatezones]


# In[ ]:


validation_R2 = pd.DataFrame(columns=['Model','Validation_R2'])

#### SIMPLE SUBSET

# # # - - - XGB - - - # # #
model = 'xgb3'
best_parameters3 = {'colsample_bytree': 0.85, 'learning_rate': 0.02, 'max_depth': 10,
 'min_child_weight': 3, 'n_estimators': 700, 'nthread': 4,
 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.85}
regression = xgb.XGBRegressor(**best_parameters3)

X_train = df_sub.drop(['calories_per_ha'], axis=1)
y_train = df_sub['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'_simplesubset.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Linear - - - # # #

model = 'Linear'
regression = LinearRegression()

X_train = df_sub.drop(['calories_per_ha'], axis=1)
y_train = df_sub['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'_simplesubset.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Lasso - - - # # #

model = 'Lasso'
best_params_Lasso = {'alpha': 0.001, 'max_iter': 1000, 'selection': 'random', 'tol': 0.0001}
regression = Lasso(**best_params_Lasso)

X_train = df_sub.drop(['calories_per_ha'], axis=1)
y_train = df_sub['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'_simplesubset.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Ridge - - - # # #

model = 'Ridge'
best_paramsR = {'alpha': 0.001, 'max_iter': 1000, 'tol': 1e-06}
regression = Ridge(**best_paramsR)

X_train = df_sub.drop(['calories_per_ha'], axis=1)
y_train = df_sub['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'_simplesubset.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Poly - - - # # #

model = 'Poly'
best_parameters3 = {'colsample_bytree': 0.85, 'learning_rate': 0.02, 'max_depth': 10,
 'min_child_weight': 3, 'n_estimators': 700, 'nthread': 4,
 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.85}
regression = LinearRegression()

df_sub2 = make_polynomial(df_sub)
X_train = df_sub2.drop(['calories_per_ha'], axis=1)
y_train = df_sub2['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'_simplesubset.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)


validation_R2.to_csv('../ipbes_invest_crop_yield_project/output/R2_Validation/R2_validation_allinputs')


# In[ ]:


validation_R2 = pd.DataFrame(columns=['Model','Validation_R2'])

#### SUBSET GDP

# # # - - - XGB - - - # # #
model = 'xgb3'
best_parameters3 = {'colsample_bytree': 0.85, 'learning_rate': 0.02, 'max_depth': 10,
 'min_child_weight': 3, 'n_estimators': 700, 'nthread': 4,
 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.85}
xgb.XGBRegressor(**best_parameters3)

X_train = df_sub_gdp.drop(['calories_per_ha'], axis=1)
y_train = df_sub_gdp['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'simplesubset_gdp.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Linear - - - # # #

model = 'Linear'
regression = LinearRegression()

X_train = df_sub_gdp.drop(['calories_per_ha'], axis=1)
y_train = df_sub_gdp['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'simplesubset_gdp.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Lasso - - - # # #

model = 'Lasso'
best_params_Lasso = {'alpha': 0.001, 'max_iter': 1000, 'selection': 'random', 'tol': 0.0001}
regression = Lasso(**best_params_Lasso)

X_train = df_sub_gdp.drop(['calories_per_ha'], axis=1)
y_train = df_sub_gdp['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'simplesubset_gdp.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Ridge - - - # # #

model = 'Ridge'
best_paramsR = {'alpha': 0.001, 'max_iter': 1000, 'tol': 1e-06}
regression = Ridge(**best_paramsR)

X_train = df_sub_gdp.drop(['calories_per_ha'], axis=1)
y_train = df_sub_gdp['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'simplesubset_gdp.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Poly - - - # # #

model = 'Poly'
best_parameters3 = {'colsample_bytree': 0.85, 'learning_rate': 0.02, 'max_depth': 10,
 'min_child_weight': 3, 'n_estimators': 700, 'nthread': 4,
 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.85}
regression = LinearRegression()

df_sub_gdp2 = make_polynomial(df_sub_gdp)
X_train = df_sub_gdp2.drop(['calories_per_ha'], axis=1)
y_train = df_sub_gdp2['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'simplesubset_gdp.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)


validation_R2.to_csv('../ipbes_invest_crop_yield_project/output/R2_Validation/R2_validation_allinputs')


# In[ ]:


validation_R2 = pd.DataFrame(columns=['Model','Validation_R2'])


#### W/O CLIMATEZONES

# # # - - - XGB - - - # # #
model = 'xgb3'
best_parameters3 = {'colsample_bytree': 0.85, 'learning_rate': 0.02, 'max_depth': 10,
 'min_child_weight': 3, 'n_estimators': 700, 'nthread': 4,
 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.85}
xgb.XGBRegressor(**best_parameters3)

X_train = df_wo_cz.drop(['calories_per_ha'], axis=1)
y_train = df_wo_cz['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'wo_climzones.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Linear - - - # # #

model = 'Linear'
regression = LinearRegression()

X_train = df_wo_cz.drop(['calories_per_ha'], axis=1)
y_train = df_wo_cz['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'wo_climzones.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Lasso - - - # # #

model = 'Lasso'
best_params_Lasso = {'alpha': 0.001, 'max_iter': 1000, 'selection': 'random', 'tol': 0.0001}
regression = Lasso(**best_params_Lasso)

X_train = df_wo_cz.drop(['calories_per_ha'], axis=1)
y_train = df_wo_cz['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'wo_climzones.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Ridge - - - # # #

model = 'Ridge'
best_paramsR = {'alpha': 0.001, 'max_iter': 1000, 'tol': 1e-06}
regression = Ridge(**best_paramsR)

X_train = df_wo_cz.drop(['calories_per_ha'], axis=1)
y_train = df_wo_cz['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'wo_climzones.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)

# # # - - - Poly - - - # # #

model = 'Poly'
best_parameters3 = {'colsample_bytree': 0.85, 'learning_rate': 0.02, 'max_depth': 10,
 'min_child_weight': 3, 'n_estimators': 700, 'nthread': 4,
 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.85}
regression = LinearRegression()

df_wo_cz2 = make_polynomial(df_wo_cz)
X_train = df_wo_cz2.drop(['calories_per_ha'], axis=1)
y_train = df_wo_cz2['calories_per_ha']
regression.fit(X_train, y_train)
save_model('../ipbes_invest_crop_yield_project/output/Models/'+model+'wo_climzones.sav')
y_predicted = regression.predict(X_validation)

R2_validation = sklearn.metrics.r2_score(y_validation, y_predicted)
validation_R2 = validation_R2.append({'Model':model,'Validation_R2':R2_validation},ignore_index=True)


validation_R2.to_csv('../ipbes_invest_crop_yield_project/output/R2_Validation/R2_validation_allinputs')

