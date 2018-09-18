import os, math
from collections import OrderedDict

import hazelbean as hb
import numpy as np
import pandas as pd

import math
from scipy import stats

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures


from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

from scipy import stats

import xgboost as xgb

L = hb.get_logger('data_prep_v3')



# Utilities
def convert_af_to_1d_df(af):
    array = af.data.flatten()
    df = pd.DataFrame(array)
    return df


def concatenate_dfs_horizontally(df_list, column_headers=None):
    """
    Append horizontally, based on index.
    """

    df = pd.concat(df_list, axis=1)

    if column_headers:
        df.columns = column_headers
    return df


def setup_dirs(p):
    L.debug('Making default dirs.')

    p.input_dir = os.path.join(p.project_dir, 'input')
    p.intermediate_dir = os.path.join(p.project_dir, 'intermediate')
    p.run_dir = os.path.join(p.project_dir, 'intermediate')
    p.output_dir = os.path.join(p.project_dir, 'output')


    dirs = [p.project_dir, p.input_dir, p.intermediate_dir, p.run_dir, p.output_dir]
    hb.create_dirs(dirs)


def create_land_mask():
    countries_af = hb.ArrayFrame('../ipbes_invest_crop_yield_project/input/Cartographic/country_ids.tif')
    df = convert_af_to_1d_df(countries_af)
    df['land_mask'] = df[0].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(0, axis=1)
    return df


def plot_col(df, col_name, shape=(2160, 4320)):
    m = np.array(df[col_name])
    bm = Basemap()
    im = bm.imshow(np.flipud(m.reshape(shape)))
    bm.drawcoastlines(linewidth=0.15, color='0.1')
    cbar = plt.colorbar(im, orientation='horizontal')
    plt.title(col_name)
    plt.show()


def link_base_data(p):
    
    # Cartographic
    p.country_names_path = os.path.join(p.input_dir, 'cartographic/country_names.csv')
    p.country_ids_raster_path = os.path.join(p.input_dir, 'cartographic/country_ids.tif')    #
    p.ha_per_cell_5m_path = os.path.join(p.input_dir, 'cartographic/ha_per_cell_5m.tif')

    # Climate
    #   Worclim v1
    #p.precip_path = os.path.join(p.input_dir, 'Climate/worldclim/bio12.bil')
    p.temperature_path = os.path.join(p.input_dir, 'climate/worldclim/bio1.bil')
    #   Worldclim v2
    p.temp_avg_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_avg.tif')                   #wc2.0_bio_5m_01
    p.temp_diurnalrange_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_diurnalrange.tif')   #wc2.0_bio_5m_02
    p.temp_isothermality_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_isothermality.tif')  #wc2.0_bio_5m_03
    p.temp_seasonality_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_seasonality.tif')    #wc2.0_bio_5m_04
    p.temp_annualmax_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_annualmax.tif')      #wc2.0_bio_5m_05
    p.temp_annualmin_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_annualmin.tif')      #...
    p.temp_annualrange_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_annualrange.tif')
    #p.temp_wettestq_path = os.path.join(p.input_dir, 'Climate/worldclim2/wc2.0_bio_5m_08.tif')
    #p.temp_dryestq_path = os.path.join(p.input_dir, 'Climate/worldclim2/wc2.0_bio_5m_09.tif')
    #p.temp_warmestq_path = os.path.join(p.input_dir, 'Climate/worldclim2/wc2.0_bio_5m_10.tif')
    #p.temp_coldestq_path = os.path.join(p.input_dir, 'Climate/worldclim2/wc2.0_bio_5m_11.tif')
    p.precip_path = os.path.join(p.input_dir, 'Climate/worldclim2/precip.tif')
    p.precip_wet_mth_path = os.path.join(p.input_dir, 'Climate/worldclim2/precip_wet_mth.tif')
    p.precip_dry_mth_path = os.path.join(p.input_dir, 'Climate/worldclim2/precip_dry_mth.tif')
    p.precip_seasonality_path = os.path.join(p.input_dir, 'Climate/worldclim2/precip_seasonality.tif')
    #p.precip_wettestq_path = os.path.join(p.input_dir, 'Climate/worldclim2/wc2.0_bio_5m_16.tif')
    #p.precip_dryestq_path = os.path.join(p.input_dir, 'Climate/worldclim2/wc2.0_bio_5m_17.tif')
    #p.precip_warmestq_path = os.path.join(p.input_dir, 'Climate/worldclim2/wc2.0_bio_5m_18.tif')
    #p.precip_coldestq_path = os.path.join(p.input_dir, 'Climate/worldclim2/wc2.0_bio_5m_19.tif')

    #   Climate zones
    p.climate_zones_path = os.path.join(p.input_dir, 'climate/Koeppen-Geiger/climate_zones.tif')

    # Topography
    p.slope_path = os.path.join(p.input_dir, "topography/worldclim/slope.tif")
    p.altitude_path = os.path.join(p.input_dir, "topography/worldclim/altitude.tif")

    # Soil
    p.workability_index_path = os.path.join(p.input_dir, 'soil', 'gaez', "workability_index.tif")
    p.toxicity_index_path = os.path.join(p.input_dir, 'soil', 'gaez', "toxicity_index.tif")
    p.rooting_conditions_index_path = os.path.join(p.input_dir, 'soil', 'gaez', "rooting_conditions_index.tif")
    # p.rainfed_land_percent_path = os.path.join(p.input_dir, 'soil', 'gaez', "rainfed_land_percent.tif")  #
    p.protected_areas_index_path = os.path.join(p.input_dir, 'soil', 'gaez', "protected_areas_index.tif")
    p.oxygen_availability_index_path = os.path.join(p.input_dir, 'soil', 'gaez', "oxygen_availability_index.tif")
    p.nutrient_retention_index_path = os.path.join(p.input_dir, 'soil', 'gaez', "nutrient_retention_index.tif")
    p.nutrient_availability_index_path = os.path.join(p.input_dir, 'soil', 'gaez', "nutrient_availability_index.tif")
    p.irrigated_land_percent_path = os.path.join(p.input_dir, 'soil', 'gaez', "irrigated_land_percent.tif")
    p.excess_salts_index_path = os.path.join(p.input_dir, 'soil', 'gaez', "excess_salts_index.tif")
    # p.cultivated_land_percent_path = os.path.join(p.input_dir, 'soil', 'gaez', "cultivated_land_percent.tif")
    p.crop_suitability_path = os.path.join(p.input_dir, 'soil', 'gaez', "crop_suitability.tif")

    # Demographic
    p.gdp_2000_path = os.path.join(p.input_dir, 'demographic/worldbank/gdp_2000.tif')
    p.gdp_gecon = os.path.join(p.input_dir, 'demographic/nordhaus/gdp_per_capita_2000_5m.tif')
    p.minutes_to_market_path = os.path.join(p.input_dir, 'demographic/jrc/minutes_to_market_5m.tif')
    p.pop_path = os.path.join(p.input_dir, 'demographic/ciesin', 'gpw_population.tif')



def create_baseline_regression_data(p):
    p.baseline_regression_data_path = os.path.join(p.cur_dir, 'baseline_regression_data.csv')
    # Iterate through input_paths adding them.  Currently also fixes fertilizer nan issues.
    af_names_list = []
    dfs_list = []
    paths_to_add = [
        # p.country_names_path,
        p.country_ids_raster_path,
        p.ha_per_cell_5m_path,
        #p.precip_path,
        #p.temperature_path,
        p.slope_path,
        p.altitude_path,
        p.workability_index_path,
        p.toxicity_index_path,
        p.rooting_conditions_index_path,
        # p.rainfed_land_percent_path,
        p.protected_areas_index_path,
        p.oxygen_availability_index_path,
        p.nutrient_retention_index_path,
        p.nutrient_availability_index_path,
        # p.irrigated_land_percent_path,
        p.excess_salts_index_path,
        # p.cultivated_land_percent_path,
        # p.crop_suitability_path,
        p.gdp_2000_path,
        p.gdp_gecon,
        p.minutes_to_market_path,
        p.pop_path,
        p.climate_zones_path,

        p.temp_avg_path,
        p.temp_diurnalrange_path,
        p.temp_isothermality_path,
        p.temp_seasonality_path,
        p.temp_annualmax_path,
        p.temp_annualmin_path,
        p.temp_annualrange_path,
        #p.temp_wettestq_path,
        #p.temp_dryestq_path,
        #p.temp_warmestq_path,
        #p.temp_coldestq_path,
        p.precip_path,
        p.precip_wet_mth_path,
        p.precip_dry_mth_path,
        p.precip_seasonality_path,
        # p.precip_wettestq_path,
        # p.precip_dryestq_path,
        # p.precip_warmestq_path,
        # p.precip_coldestq_path
    ]

    if p.run_this:
        match_af = hb.ArrayFrame(paths_to_add[0])
        for path in paths_to_add:
            name = hb.explode_path(path)['file_root']
            af = hb.ArrayFrame(path)
            af_names_list.append(name)
            df = convert_af_to_1d_df(af)
            dfs_list.append(df)

        L.info('Concatenating all dataframes.')
        df = concatenate_dfs_horizontally(dfs_list, af_names_list)
        df[df < 0] = 0.0

        # Get rid of the oceans cells
        df['pixel_id'] = df.index
        df['pixel_id_float'] = df['pixel_id'].astype('float')
        land_mask = create_land_mask()
        df = df.merge(land_mask, right_index=True, left_on='pixel_id')
        df_land = df[df['land_mask']==1]

        df_land = df_land.dropna()

        df_land['lon'] = ((df['pixel_id_float'] % 4320.)/4320 - .5) * 360.0
        df_land['lat'] = ((df['pixel_id_float'] / 4320.).round()/2160 - .5) * 180.

        df_land.to_csv(p.baseline_regression_data_path)


def aggregate_crops_by_type(p):
    """CMIP6 and the land-use harmonization project have centered on 5 crop types: c3 annual, c3 perennial, c4 annual, c4 perennial, nitrogen fixer
    Aggregate the 15 crops to those four categories by modifying the baseline_regression_data."""

    p.aggregated_crop_data_csv_path = os.path.join(p.cur_dir, 'aggregated_crop_data.csv')
    baseline_regression_data_df = pd.read_csv(p.baseline_regression_data_path, index_col='pixel_id')

    vars_names_to_aggregate = [
        # 'production_value_per_ha',
        # 'calories_per_ha',
        'calories_per_ha_masked',
        # 'yield_per_ha'
        # 'proportion_cultivated',
        # 'PotassiumApplication_Rate',
        # 'PhosphorusApplication_Rate',
        # 'NitrogenApplication_Rate',
    ]

    crop_membership = OrderedDict()
    crop_membership['c3_annual'] = [
        'aniseetc',
        'artichoke',
        'asparagus',
        'bambara',
        'barley',
        'buckwheat',
        'cabbage',
        'canaryseed',
        'carob',
        'carrot',
        'cassava',
        'cauliflower',
        'cerealnes',
        'chestnut',
        'cinnamon',
        'cucumberetc',
        'currant',
        'date',
        'eggplant',
        'fonio',
        'garlic',
        'ginger',
        'mixedgrain',
        'hazelnut',
        'hempseed',
        'hop',
        'kapokseed',
        'linseed',
        'mango',
        'mate',
        'mustard',
        'nutmeg',
        'okra',
        'onion',
        'greenonion',
        'peppermint',
        'potato',
        'pumpkinetc',
        'pyrethrum',
        'ramie',
        'rapeseed',
        'rice',
        'safflower',
        'sisal',
        'sorghumfor',
        'sourcherry',
        'spinach',
        'sugarbeet',
        'sunflower',
        'taro',
        'tobacco',
        'tomato',
        'triticale',
        'tung',
        'vanilla',
        'vetch',
        'walnut',
        'watermelon',
        'wheat',
        'yam',
        'yautia',

    ]
    crop_membership['c3_perennial'] = [
        'almond',
        'apple',
        'apricot',
        'areca',
        'avocado',
        'banana',
        'blueberry',
        'brazil',
        'cashewapple',
        'cashew',
        'cherry',
        'chicory',
        'chilleetc',
        'citrusnes',
        'clove',
        'cocoa',
        'coconut',
        'coffee',
        'cotton',
        'cranberry',
        'fig',
        'flax',
        'grapefruitetc',
        'grape',
        'jute',
        'karite',
        'kiwi',
        'kolanut',
        'lemonlime',
        'lettuce',
        'abaca',
        'melonetc',
        'melonseed',
        'oats',
        'oilpalm',
        'oilseedfor',
        'olive',
        'orange',
        'papaya',
        'peachetc',
        'pear',
        'pepper',
        'persimmon',
        'pineapple',
        'pistachio',
        'plantain',
        'plum',
        'poppy',
        'quince',
        'quinoa',
        'rasberry',
        'rubber',
        'rye',
        'stonefruitnes',
        'strawberry',
        'stringbean',
        'sweetpotato',
        'tangetc',
        'tea',
    ]
    crop_membership['c4_annual'] = [
        'maize',
        'millet',
        'sorghum',
    ]
    crop_membership['c4_perennial'] = [
        'greencorn',
        'sugarcane',

    ]
    crop_membership['nitrogen_fixer'] = [
        'bean',
        'greenbean',
        'soybean',
        'chickpea',
        'clover',
        'cowpea',
        'groundnut',
        'lupin',
        'pea',
        'greenpea',
        'pigeonpea',
        'lentil',
        'legumenes',
        'broadbean',
        'castor',

    ]

    p.crop_types = [
        'c3_annual',
        'c3_perennial',
        'c4_annual',
        'c4_perennial',
        'nitrogen_fixer',
    ]

    if p.run_this:
        # Create a DF of zeros, ready to hold the summed results for each crop type. Indix given will  be from baseline_regression_data_df so that spatial indices match.
        crop_specific_df = pd.DataFrame(0,index=baseline_regression_data_df.index,columns=['solo_column'])

        crop_types_df = pd.DataFrame(0,index=baseline_regression_data_df.index,columns=[crop_type + '_calories_per_ha' for crop_type in p.crop_types])

        # Iterate through crop_types
        for crop_type, crops in crop_membership.items():

            L.info('Aggregating ' + str(crop_type) + ' ' + str(crops))
            crop_type_col_name = crop_type + '_calories_per_ha'


            # iterate through crops
            for crop in crops:
                 crop_col_name = crop + '_calories_per_ha'
                 #crop_specific_df[crop_col_name] = np.zeros(len(baseline_regression_data_df.index))
                 crop_specific_df[crop_col_name] = crop_specific_df['solo_column']

                 input_crop_file_name = crop + '_calories_per_ha_masked'

                 input_path = os.path.join(p.input_dir, 'Crop/crop_calories', input_crop_file_name + '.tif')
                 af = hb.ArrayFrame(input_path)
                 crop_specific_df[crop_col_name] = convert_af_to_1d_df(af)[0]


                 crop_types_df[crop_type_col_name] += crop_specific_df[crop_col_name]

            # To be fixed for weird NoData too high values in inputs files: (JUSTIN?)
            # crop_types_df[output_col_name][crop_specific_df[output_col_name] > 1e+12] = 0.0


        crop_types_df['calories_per_ha'] = sum(crop_types_df[crop_type_cal_per_ha] for crop_type_cal_per_ha in [crop_type + '_calories_per_ha' for crop_type in p.crop_types])
        crop_types_df.to_csv(p.aggregated_crop_data_csv_path)


#def merge_full_baseline_data():
    # Actually let's do that when we load the datasets in the next script?
    # merge baseline_df and crop_types_df on 'pixel_id' colmn

def load_data(p):

    if p.run_this:
        crop_types_df = pd.read_csv(p.aggregated_crop_data_csv_path)
        df_land = pd.read_csv(p.baseline_regression_data_path)

        df = df_land.merge(crop_types_df,how='outer',on='pixel_id')

        if p.subset==True:
            df = df.sample(frac=0.02, replace=False, weights=None, random_state=None, axis=0)

        elif p.subset==False: #Save validation data
            x = df.drop(['calories_per_ha'], axis=1)
            y = df['calories_per_ha']

            X, X_validation, Y, y_validation = train_test_split(x, y)

            df = X.merge(pd.DataFrame(Y),how='outer',left_index=True,right_index=True)

        elif p.subset is None: # CAREFUL FOOL ONLY DO THIS FOR PLOTTING BECAUSE LEAKAGE
            pass

        # Remove cal_per_ha per crop type for now
        df = df.drop(labels=['c3_annual_calories_per_ha', 'c3_perennial_calories_per_ha',
                             'c4_annual_calories_per_ha', 'c4_perennial_calories_per_ha',
                             'nitrogen_fixer_calories_per_ha'], axis=1)

        # Remove helper columns (not features)
        df = df.drop(labels=['Unnamed: 0', 'country_ids',
                             'ha_per_cell_5m'], axis=1)

        # Rename cols
        df = df.rename(columns={'bio12': 'precip', 'bio1': 'temperature',
                                'minutes_to_market_5m': 'min_to_market',
                                'gdp_per_capita_2000_5m': 'gdp_per_capita',
                                'gdp_2000': 'gdp'})

        # Encode Climate zones as Strings
        climate_zones_map = {1: 'Af', 2: 'Am', 3: 'Aw',
                             5: 'BWk', 4: 'BWh', 7: 'BSk', 6: 'BSh',
                             14: 'Cfa', 15: 'Cfb', 16: 'Cfc', 8: 'Csa',
                             9: 'Csb', 10: 'Csc', 11: 'Cwa', 12: 'Cwb', 13: 'Cwc',
                             25: 'Dfa', 26: 'Dfb', 27: 'Dfc', 28: 'Dfd', 17: 'Dsa', 18: 'Dsb', 19: 'Dsc',
                             20: 'Dsd', 21: 'Dwa', 22: 'Dwb', 23: 'Dwc', 24: 'Dwd',
                             30: 'EF', 29: 'ET'}

        df['climate_zones'] = df['climate_zones'].map(climate_zones_map) # TODO Why was it commented?

        # Encode climate zones as dummies
        climate_dummies_df = pd.get_dummies(df['climate_zones'])
        for col in climate_dummies_df.columns:
            climate_dummies_df = climate_dummies_df.rename({col: str('climatezone_' + col)}, axis=1)

        df = df.merge(climate_dummies_df, right_index=True, left_index=True)
        df = df.drop('climate_zones', axis=1)

        # Log some skewed variables
        df['calories_per_ha'] = df['calories_per_ha'].apply(lambda x: np.log(x) if x != 0 else 0)

        for col in ['gdp_per_capita','altitude', 'min_to_market', 'gpw_population']:
            df[str('log_'+col)] = df[col].apply(lambda x: np.log(x) if x != 0 else 0)

        # TODO figure out how to encode soil variables better?

        # Add precip_annualrange
        df['precip_annualrange'] = df['precip_wet_mth'] - df['precip_dry_mth']

        # Lat/Lon
        df['sin_lon'] = df['lon'].apply(lambda x:np.sin(np.radians(x)))

        # Encode properly NaNs
        df['slope'] = df['slope'].replace({0: np.nan})  # 143 NaN in 'slope' variable
        for soil_var in ['workability_index', 'toxicity_index', 'rooting_conditions_index',
                         'oxygen_availability_index', 'protected_areas_index',
                         'nutrient_retention_index', 'nutrient_availability_index', 'excess_salts_index']:
            df[soil_var] = df[soil_var].replace({255: np.nan})

        # Drop NaNs rows and cells with no ag
        df = df.dropna()
        df = df[df['calories_per_ha'] != 0]

        # df.set_index('pixel_id') ## TODO Why is this commented out ?

        p.df = df

        match_af = hb.ArrayFrame(p.country_ids_raster_path)
        zeros_array = np.zeros(match_af.size)
        p.full_df = pd.DataFrame(zeros_array)
        p.full_df = pd.merge(p.full_df, p.df, left_index=True, right_on='pixel_id', how='outer')

def data_transformation(p,how):
    df = p.df

    if p.run_this:
        dfTransformed = pd.DataFrame.copy(df)

        if how =='log':
            dfTransformed['calories_per_ha'] = np.log(dfTransformed['calories_per_ha'])

        elif how =='bin':
            dfTransformed = pd.cut(df['calories_per_ha'], bins=5, labels=[1, 2, 3, 4, 5]) ##Not sure about this -- to do Charlie

        # elif how =='logbin':
        #     dfLogBin['calories_per_cell'] = pd.cut(dfLogBin['calories_per_cell'], 5, labels=[1, 2, 3, 4, 5]) ##Not sure about this -- to do Charlie
    return dfTransformed

def create_results_table(p):
    #if p.run_this:
    p.results = pd.DataFrame(columns=['Model', 'num_features', 'Features', 'MSE', 'R2'])


def do_regression(regression,dataframe):
    ##Must make dummies for categorical variable climate_zone
    # dataframe = pd.get_dummies(dataframe, columns=['climate_zone'])
    # Or just drop column if don't want dummies: x = x.drop(['climate_zone'], axis=1)

    x = dataframe.drop(['calories_per_ha'], axis=1)
    y = dataframe['calories_per_ha']

    ### Cross validation scores
    r2_scores = cross_val_score(regression, x, y, cv=10,scoring='r2')
    mse_scores = cross_val_score(regression, x, y, cv=10, scoring='neg_mean_squared_error')
    mae_scores = cross_val_score(regression, x, y, cv=10, scoring='neg_mean_absolute_error')

    return [np.mean(r2_scores), np.mean(mse_scores), np.mean(mae_scores)]


def do_regression_RFE_features(p, model, regression, dataframe,min_features, max_features):
    for num_features in range(min_features,  max_features):
        L.info('   with ' + str(num_features) + ' features ...')

        ## RFE - Features selection
        selector = RFE(regression, num_features, step=1)

        x = dataframe.drop(['calories_per_ha'], axis=1)  ## TODO climate_zones encode One Hot Encoder
        y = dataframe['calories_per_ha']
        X, X_test, Y, Y_test = train_test_split(x, y)

        X_RFE = selector.fit_transform(X, Y)

        features_selected = [X.columns[feature_pos] for feature_pos in selector.get_support(indices=True)]
        features_selected.append('calories_per_ha')
        scores = do_regression(regression, dataframe[features_selected])
        R2_score = scores[0]
        MSE_score = scores[1]
        MAE_score = scores[2]

        p.results = p.results.append({'Model': model,
                                  'num_features': num_features,
                                  'Features': features_selected,
                                  'R2': R2_score,
                                  'MSE': MSE_score},
                                 ignore_index=True)

        L.info('   ... done')


def make_polynomial(df, degree=2, interaction_terms=True):
    # '''Returns a new dataFrame with added polynomial terms degree >= 2'''

    if interaction_terms == False:
        x = df.drop(['calories_per_ha'], axis=1)
        y = df['calories_per_ha']

        for deg in range(2, degree + 1):
            for col in x.columns:
                x[str(col + '^' + str(deg))] = x[col].apply(lambda x: x ** deg)

        Poly_df = x.merge(pd.DataFrame(y), right_index=True, left_index=True)

    if interaction_terms == True:
        x = df.drop(['calories_per_ha'], axis=1)
        y = df['calories_per_ha'].reset_index()

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X2 = poly.fit_transform(x)

        Poly_df = pd.DataFrame(data=np.concatenate((y.as_matrix(), X2), axis=1),
                               columns=['pixel_id', 'calories_per_ha'] +
                                       poly.get_feature_names(
                                           (df.drop(['calories_per_ha'], axis=1)).columns))

        Poly_df = Poly_df.set_index('pixel_id')

    return Poly_df


def run_polynomial_regressions(p):
    if p.run_this:
        regression = LinearRegression()

        for interaction_terms in [False, True]:
            for degree in [2, 3]:
                L.info('Running Polynomial of degree ' + str(degree) + interaction_terms * (' with interaction terms'))

                model = str('Poly deg' + str(degree) + interaction_terms * ' w/ interact')

                dataframe = make_polynomial(p.df,2,interaction_terms=False)

                do_regression_RFE_features(p,model,regression,dataframe,5,6)

                ## Add here do_regression_all_features TODO fct do_regression_all_feature
                ## Add here do_regression_my_subset TODO fct  do_regression_features_subset

def run_linear_regressions(p):
    if p.run_this:
        regression = LinearRegression()
        model = 'Linear Reg'

        L.info('Running Linear Regression')

        do_regression_RFE_features(p,model, regression, p.df, 5, 6)

def run_tree_based_models(p):
    if p.run_this:
        regression = xgb.XGBRegressor()
        model = 'XGBoost (default parameters)'

        L.info('Running XGBoost (default params)')

        do_regression_RFE_features(p, model, regression, p.df, 5, 6)



def compare_predictions(regression,dataframe,show_df=True,show_plot=True):
    x = dataframe.drop(['calories_per_ha'], axis=1)
    y = dataframe['calories_per_ha']
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    reg = regression.fit(X_train, y_train)
    y_predicted = reg.predict(X_test)

    compare = pd.DataFrame()
    compare['y_test'] = y_test
    compare['predicted'] = y_predicted

    if show_df == True:
        print(compare)

    if show_plot == True:
        ax = compare.plot.scatter(x='y_test',y='predicted',s=0.5)
        ax.plot(ax.get_xlim(), ax.get_xlim(), ls="--", c=".3")
        plt.show()


def visualize_data(p):
    df_land = pd.read_csv(p.baseline_regression_data_path) ## TODO: If we load data, this reads csv twive (time consuming) optimize?
    match_af = hb.ArrayFrame(p.country_ids_raster_path)
    zeros_array = np.zeros(match_af.size)
    zeros_df = pd.DataFrame(zeros_array)
    full_df = pd.merge(zeros_df, df_land, left_index=True, right_on='pixel_id', how='outer')

    #plot_col(agg_df, 'lat')
    #plot_col(agg_df, 'lon')
    plot_col(full_df, 'lat')
    plot_col(full_df, 'lon')

    # plot_col(p.full_df, 'log_gdp_per_capita')
    # plot_col(p.full_df, 'climate_zones')
    # plot_col(p.full_df, 'log_precip')
    # plot_col(p.full_df, 'log_altitude')
    # plot_col(p.full_df, 'log_gdp')
    # plot_col(p.full_df, 'log_min_to_market')
    plot_col(p.full_df, 'slope')
    plot_col(p.full_df, 'lon_sin')
    plot_col(p.full_df, 'lat_sin')
    plot_col(p.full_df, 'lat_sin')




main = 'here'
if __name__ =='__main__':
    p = hb.ProjectFlow('../ipbes_invest_crop_yield_project')

    p.subset = True # True for subsetting data, None ONLY FOR PLOTTING BECAUSE LEAKAGE

    setup_dirs_task = p.add_task(setup_dirs)

    link_base_data_task = p.add_task(link_base_data)
    create_baseline_regression_data_task = p.add_task(create_baseline_regression_data)
    aggregate_crops_by_type_task = p.add_task(aggregate_crops_by_type)
    load_data_task = p.add_task(load_data)
    visualize_data_task = p.add_task(visualize_data)
    create_results_table_task = p.add_task(create_results_table)
    run_linear_regressions_task = p.add_task(run_linear_regressions)
    run_polynomial_regressions_task = p.add_task(run_polynomial_regressions)
    #run_tree_based_models_task = p.add_task(run_tree_based_models)

    setup_dirs_task.run = 1
    link_base_data_task.run = 1
    create_baseline_regression_data_task.run = 1
    aggregate_crops_by_type_task.run = 1
    load_data_task.run = 1
    visualize_data_task.run = 1
    create_results_table_task.run = 1
    run_linear_regressions_task.run = 1
    run_polynomial_regressions_task.run = 0
    #run_tree_based_models_task.run = 1

    setup_dirs_task.skip_existing = 1
    link_base_data_task.skip_existing = 1
    create_baseline_regression_data_task.skip_existing = 0
    aggregate_crops_by_type_task.skip_existing = 1
    load_data_task.skip_existing = 0 # TODO: if skipped, p.df doesn't exist -- I have to re-load everytime :( what would be a good solution? Justin?
    visualize_data_task.skip_existing = 1
    create_results_table_task.skip_existing = 1 # = 0 to reset the results table



    p.execute()



