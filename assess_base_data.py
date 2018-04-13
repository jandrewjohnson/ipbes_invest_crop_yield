import os,  sys
from collections import OrderedDict

import pandas as pd
import numpy as np

import hazelbean as hb
import numdal as nd
import geoecon as ge

L = hb.get_logger()

def get_default_kw(**kw):
    # This required function is called outside of this code's scope to create a kwargs dicitonary. It will be passed to each
    # step in the projects logic, potentially modified, then passed to the next.
    L.debug('Getting default keywords.')
    if not kw:
        kw = OrderedDict()
    if type(kw) is not OrderedDict:
        kw = OrderedDict(kw)

    ### These should be the only lines that need editing for a new project.
    kw['project_name'] = kw.get('project_name', 'ipbes')  # Name of the project being run. A project is a specific implementation of the repository's code to some input data relative to the workspace_dir.
    kw['project_dir'] = kw.get('project_dir', os.path.join('c:/onedrive/projects', 'ipbes'))  # This is the ONLY absolute path and it is specific to the researcher and the researcher's current project.
    kw['repository_dir'] = 'ipbes_0.1'  # This is the only dir that will be under Version Control. Don't put code anywhere else.

    ### Generic non-project-specific dir links.
    kw['base_data_dir'] = kw.get('base_data_dir', hb.BASE_DATA_DIR)
    kw['bulk_data_dir'] = kw.get('bulk_data_dir', hb.BULK_DATA_DIR)
    kw['external_bulk_data_dir'] = kw.get('external_bulk_data_dir', hb.EXTERNAL_BULK_DATA_DIR)

    ### Generic project-specific dirs from kwargs.
    kw['input_dir'] = kw.get('input_dir', os.path.join(kw['project_dir'], 'input'))  # New inputs specific to this project.
    kw['project_base_data_dir'] = kw.get('project_base_data_dir', os.path.join(kw['project_dir'], 'base_data'))  # Data that must be redistributed with this project for it to work. Do not put actual base data here that might be used across many projects.
    kw['temporary_dir'] = kw.get('temporary_dir', hb.TEMPORARY_DIR)  # Generates new run_dirs here. Useful also to set the numdal temporary_dir to here for the run.
    kw['test_dir'] = kw.get('temporary_dir', hb.TEST_DATA_DIR)  # Generates new run_dirs here. Useful also to set the numdal temporary_dir to here for the run.
    # kw['intermediate_dir'] =  kw.get('input_dir', os.path.join(kw['project_dir'], kw['temporary_dir']))  # If generating lots of data, set this to temporary_dir so that you don't put huge data into the cloud.
    kw['intermediate_dir'] = kw.get('intermediate_dir', os.path.join(kw['project_dir'], 'intermediate'))  # If generating lots of data, set this to temporary_dir so that you don't put huge data into the cloud.
    kw['output_dir'] = kw.get('output_dir', os.path.join(kw['project_dir'], 'output'))  # the final working run is move form Intermediate to here and any hand-made docs are put here.
    kw['run_string'] = kw.get('run_string', nd.pretty_time())  # unique string with time-stamp. To be used on run_specific identifications.
    kw['run_dir'] = kw.get('run_dir', os.path.join(kw['temporary_dir'], '0_seals_' + kw['run_string']))  # ready to delete dir containing the results of one run.
    kw['basis_name'] = kw.get('basis_name', '')  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.
    kw['basis_dir'] = kw.get('basis_dir', os.path.join(kw['intermediate_dir'], kw['basis_name']))  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.

    ### Common base data references
    kw['base_data_country_names_uri'] = os.path.join(kw['base_data_dir'], 'misc', 'country_names.csv')
    kw['base_data_country_ids_raster_uri'] = os.path.join(kw['base_data_dir'], 'misc', 'country_ids.tif')
    kw['base_data_calories_per_cell_uri'] = os.path.join(kw['base_data_dir'], 'publications/ag_tradeoffs/land_econ', 'calories_per_cell.tif')
    kw['proportion_cropland_uri'] = os.path.join(kw['base_data_dir'], 'crops/earthstat', 'proportion_cropland.tif')
    kw['base_data_precip_uri'] = os.path.join(kw['base_data_dir'], 'climate/worldclim/baseline/5min', 'baseline_bio12_Annual_Precipitation.tif')
    kw['base_data_temperature_uri'] = os.path.join(kw['base_data_dir'], 'climate/worldclim/baseline/5min', 'baseline_bio1_Annual_Mean_Temperature.tif')
    kw['base_data_gdp_2000_uri'] = os.path.join(kw['input_dir'], 'gdp_2000.tif')
    kw['base_data_price_per_ha_masked_dir'] = os.path.join(kw['base_data_dir'], 'crops\\crop_prices_and_production_value_2000\\price_per_ha_masked')
    kw['base_data_crop_calories_dir'] = os.path.join(kw['base_data_dir'], 'crops\\crop_calories')
    kw['base_data_ag_value_2000_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'ag_value_2000.tif')
    kw['base_data_minutes_to_market_uri'] = os.path.join(kw['base_data_dir'], 'socioeconomic\\distance_to_market\\uchida_and_nelson_2009\\access_50k', 'minutes_to_market_5m.tif')
    kw['base_data_ag_value_2000_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'ag_value_2000.tif')
    kw['base_data_pop_30s_uri'] = os.path.join(kw['base_data_dir'], 'population\\ciesin', 'pop_30s.tif')
    kw['base_data_proportion_pasture_uri'] = os.path.join(hb.BASE_DATA_DIR, 'crops/earthstat', 'proportion_pasture.tif')
    kw['base_data_faostat_pasture_uri'] = os.path.join(hb.BASE_DATA_DIR, 'socioeconomic\\fao', 'faostat', 'Production_LivestockPrimary_E_All_Data_(Norm).csv')
    kw['base_data_ag_value_2005_spam_uri'] = os.path.join(hb.BASE_DATA_DIR, 'crops', 'ag_value_2005_spam.tif')

    # Common base data references GAEZ
    kw['base_data_workability_index_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "workability_index.tif")
    kw['base_data_toxicity_index_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "toxicity_index.tif")
    kw['base_data_rooting_conditions_index_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "rooting_conditions_index.tif")
    kw['base_data_rainfed_land_percent_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "rainfed_land_percent.tif") # REMOVE?
    kw['base_data_protected_areas_index_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "protected_areas_index.tif")
    kw['base_data_oxygen_availability_index_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "oxygen_availability_index.tif")
    kw['base_data_nutrient_retention_index_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "nutrient_retention_index.tif")
    kw['base_data_nutrient_availability_index_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "nutrient_availability_index.tif")
    kw['base_data_irrigated_land_percent_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "irrigated_land_percent.tif")
    kw['base_data_excess_salts_index_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "excess_salts_index.tif")
    kw['base_data_cultivated_land_percent_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "cultivated_land_percent.tif")
    kw['base_data_crop_suitability_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'gaez', "crop_suitability.tif")
    kw['base_data_precip_2070_uri'] = os.path.join(kw['base_data_dir'], 'climate/worldclim/ar5_projections/5min', "ensemble_mean_85bi7012_Annual_Precipitation.tif")
    kw['base_data_temperature_2070_uri'] = os.path.join(kw['base_data_dir'], 'climate/worldclim/ar5_projections/5min', "ensemble_mean_85bi701_Annual_Mean_Temperature.tif")
    kw['base_data_slope_uri'] = os.path.join(kw['base_data_dir'], 'elevation', "slope.tif")
    kw['base_data_altitude_uri'] = os.path.join(kw['base_data_dir'], 'elevation', "altitude.tif")

    kw['crop_names'] = [
        'barley',
        'cassava',
        'groundnut',
        'maize',
        'millet',
        'oilpalm',
        'potato',
        'rapeseed',
        'rice',
        'rye',
        'sorghum',
        'soybean',
        'sugarcane',
        'wheat',
    ]

    kw['crop_types'] = [
        'c3_annual',
        'c3_perennial',
        'c4_annual',
        'c4_perennial',
        'nitrogen_fixer',
    ]

    kw['data_registry'] = OrderedDict() # a name, uri air to indicate any map used int he regression.
    kw['data_registry']['workability'] = kw['base_data_workability_index_uri']
    kw['data_registry']['toxicity'] = kw['base_data_toxicity_index_uri']
    kw['data_registry']['rooting_conditions'] = kw['base_data_rooting_conditions_index_uri']
    kw['data_registry']['rainfed_land_p'] = kw['base_data_rainfed_land_percent_uri']
    kw['data_registry']['protected_areas'] = kw['base_data_protected_areas_index_uri']
    kw['data_registry']['oxygen_availability'] = kw['base_data_oxygen_availability_index_uri']
    kw['data_registry']['nutrient_retention'] = kw['base_data_nutrient_retention_index_uri']
    kw['data_registry']['nutrient_availability'] = kw['base_data_nutrient_availability_index_uri']
    kw['data_registry']['irrigated_land_percent'] = kw['base_data_irrigated_land_percent_uri']
    kw['data_registry']['excess_salts'] = kw['base_data_excess_salts_index_uri']
    kw['data_registry']['cultivated_land_percent'] = kw['base_data_cultivated_land_percent_uri']
    kw['data_registry']['crop_suitability'] = kw['base_data_crop_suitability_uri']
    kw['data_registry']['temperature'] = kw['base_data_temperature_2070_uri']
    kw['data_registry']['slope'] = kw['base_data_slope_uri']
    kw['data_registry']['altitude'] = kw['base_data_altitude_uri']
    # kw['data_registry']['base_data_country_names_uri'] = kw['base_data_country_names_uri']
    # kw['data_registry']['base_data_country_ids_raster_uri'] = kw['base_data_country_ids_raster_uri']
    # kw['data_registry']['base_data_calories_per_cell_uri'] = kw['base_data_calories_per_cell_uri']
    kw['data_registry']['proportion_cropland'] = kw['proportion_cropland_uri']
    kw['data_registry']['precip'] = kw['base_data_precip_uri']
    kw['data_registry']['temperature'] = kw['base_data_temperature_uri']
    kw['data_registry']['gdp_gecon'] = os.path.join(kw['base_data_dir'], 'socioeconomic\\nordhaus_gecon', 'gdp_per_capita_2000_5m.tif')
    kw['data_registry']['minutes_to_market'] = kw['base_data_minutes_to_market_uri']






    kw['sample_fraction'] = kw.get('sample_fraction', 0.2)

    return kw
kw = get_default_kw()

kw['aggregated_crop_data_dir'] = os.path.join(kw['basis_dir'], 'aggregated_crop_data')
kw['aggregated_crop_data_csv_uri'] = os.path.join(kw['aggregated_crop_data_dir'], 'aggregated_crop_data.csv')

# data_path = kw['aggregated_crop_data_csv_uri']
# df = pd.read_csv(data_path)
# # hb.describe_dataframe(df)
#
# # hb.enumerate_array_as_histogram(df['c3_annual_calories_per_ha'])
# print(len(np.where(df['c3_annual_calories_per_ha'] < 1000000000000)[0]))
# print(len(np.where(df['c3_annual_calories_per_ha'] > 1000000000000)[0]))





data_path = os.path.join(kw['basis_dir'], 'baseline_regression_data.csv')
df = pd.read_csv(data_path)
for colname in df.columns:
    print(colname, np.nanmean(df[colname]), np.max(df[colname]))


climate_path = os.path.join(kw['basis_dir'], 'climate_scenarios.csv')
df = pd.read_csv(climate_path)
for colname in df.columns:
    print(colname, np.nanmean(df[colname]), np.max(df[colname]))