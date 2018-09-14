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

# import xgboost as xgb

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


def create_land_mask():
    countries_af = hb.ArrayFrame('../ipbes_invest_crop_yield_project/input/Cartographic/country_ids.tif')
    df = convert_af_to_1d_df(countries_af)
    df['land_mask'] = df[0].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(0, axis=1)
    return df


def setup_dirs(p):
    L.debug('Making default dirs.')

    p.input_dir = os.path.join(p.project_dir, 'input')
    p.intermediate_dir = os.path.join(p.project_dir, 'intermediate')
    p.run_dir = os.path.join(p.project_dir, 'intermediate')
    p.output_dir = os.path.join(p.project_dir, 'output')

    dirs = [p.project_dir, p.input_dir, p.intermediate_dir, p.run_dir, p.output_dir]
    hb.create_dirs(dirs)


def link_base_data(p):
    # Cartographic
    p.country_names_path = os.path.join(p.input_dir, 'cartographic/country_names.csv')
    p.country_ids_raster_path = os.path.join(p.input_dir, 'cartographic/country_ids.tif')  #
    p.ha_per_cell_5m_path = os.path.join(p.input_dir, 'cartographic/ha_per_cell_5m.tif')

    # Climate
    #   Worclim v1
    # p.precip_path = os.path.join(p.input_dir, 'Climate/worldclim/bio12.bil')
    p.temperature_path = os.path.join(p.input_dir, 'climate/worldclim/bio1.bil')
    #   Worldclim v2
    p.temp_avg_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_avg.tif')  # wc2.0_bio_5m_01
    p.temp_diurnalrange_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_diurnalrange.tif')  # wc2.0_bio_5m_02
    p.temp_isothermality_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_isothermality.tif')  # wc2.0_bio_5m_03
    p.temp_seasonality_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_seasonality.tif')  # wc2.0_bio_5m_04
    p.temp_annualmax_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_annualmax.tif')  # wc2.0_bio_5m_05
    p.temp_annualmin_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_annualmin.tif')  # ...
    p.temp_annualrange_path = os.path.join(p.input_dir, 'Climate/worldclim2/temp_annualrange.tif')
    p.precip_path = os.path.join(p.input_dir, 'Climate/worldclim2/precip.tif')
    p.precip_wet_mth_path = os.path.join(p.input_dir, 'Climate/worldclim2/precip_wet_mth.tif')
    p.precip_dry_mth_path = os.path.join(p.input_dir, 'Climate/worldclim2/precip_dry_mth.tif')
    p.precip_seasonality_path = os.path.join(p.input_dir, 'Climate/worldclim2/precip_seasonality.tif')

    p.ar5_ensemble_dir = os.path.join(p.input_dir, 'Climate', 'ar5_ensemble')






def create_climate_regression_data(p):
    p.climate_regression_data_path = os.path.join(p.cur_dir, 'climate_regression_data.csv')
    # Iterate through input_paths adding them.  Currently also fixes fertilizer nan issues.
    af_names_list = []
    dfs_list = []

    p.rcp_strings = ['45', '85']  # RCP4.5 and 8.4
    p.var_strings = ['bi']  # bi is the bioclimatic vars. otherws are all precip or temp sspecific
    p.year_strings = ['50', '70'] #2050 2070
    p.bi_variable_ids = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19']
    p.bi_file_names = [
        "1_Annual_Mean_Temperature_compressed.tif"
        "2_Mean_Diurnal_Range_compressed.tif"
        "3_Isothermality_compressed.tif"
        "4_Temperature_Seasonality_compressed.tif"
        "5_Max_Temperature_of_Warmest_Month_compressed.tif"
        "6_Min_Temperature_of_Coldest_Month_compressed.tif"
        "7_Temperature_Annual_Range_compressed.tif"
        "8_Mean_Temperature_of_Wettest_Quarter_compressed.tif"
        "9_Mean_Temperature_of_Driest_Quarter_compressed.tif"
        "10_Mean_Temperature_of_Warmest_Quarter_compressed.tif"
        "11_Mean_Temperature_of_Coldest_Quarter_compressed.tif"
        "12_Annual_Precipitation_compressed.tif"
        "13_Precipitation_of_Wettest_Month_compressed.tif"
        "14_Precipitation_of_Driest_Month_compressed.tif"
        "15_Precipitation_Seasonality_compressed.tif"
        "16_Precipitation_of_Wettest_Quarter_compressed.tif"
        "17_Precipitation_of_Driest_Quarter_compressed.tif"
        "18_Precipitation_of_Warmest_Quarter_compressed.tif"
        "19_Precipitation_of_Coldest_Quarter_compressed.tif"
    ]

    paths_to_add = [
        p.country_ids_raster_path,
        p.ha_per_cell_5m_path,
    ]


    for rcp_string in p.rcp_strings:
        for var_string in p.var_strings:
            for year_string in p.year_strings:
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "1_Annual_Mean_Temperature_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "2_Mean_Diurnal_Range_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "3_Isothermality_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "4_Temperature_Seasonality_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "5_Max_Temperature_of_Warmest_Month_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "6_Min_Temperature_of_Coldest_Month_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "12_Annual_Precipitation_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "13_Precipitation_of_Wettest_Month_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "14_Precipitation_of_Driest_Month_compressed.tif"))
                paths_to_add.append(os.path.join(p.ar5_ensemble_dir, 'ensemble_mean_' + rcp_string + var_string + year_string + "15_Precipitation_Seasonality_compressed.tif"))

                # TODO p.temp_annualrange_path, # couln'dt find this so it will have to be calculated at load time.
    hb.pp('paths_to_add', paths_to_add)
    if p.run_this:
        match_af = hb.ArrayFrame(paths_to_add[0])
        for path in paths_to_add:
            print('exploding', path)
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
        df_land = df[df['land_mask'] == 1]

        df_land = df_land.dropna()

        # df_land['lon'] = ((df['pixel_id_float'] % 4320.) / 4320 - .5) * 360.0
        # df_land['lat'] = ((df['pixel_id_float'] / 4320.).round() / 2160 - .5) * 180.
        print('p.climate_regression_data_path', p.climate_regression_data_path)
        df_land.to_csv(p.climate_regression_data_path)


main = 'here'
if __name__ == '__main__':
    p = hb.ProjectFlow('../ipbes_invest_crop_yield_project')

    setup_dirs_task = p.add_task(setup_dirs)

    link_base_data_task = p.add_task(link_base_data)
    create_climate_regression_data_task = p.add_task(create_climate_regression_data)

    setup_dirs_task.run = 1
    link_base_data_task.run = 1
    create_climate_regression_data_task.run = 1

    setup_dirs_task.skip_existing = 0
    link_base_data_task.skip_existing = 0
    create_climate_regression_data_task.skip_existing = 0

    p.execute()































