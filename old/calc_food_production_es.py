# coding=utf-8

"""
o   Final regression results aggregated by the five crop functional types (already done?)
o   Change in calories produced for each combination of scenarios
o   Where do we fall outside of observed range when we apply rcp8.5 (color the places where we move outside range of where c3s are grown)
o   Map self-sufficiency index for food security: for calories, Vitamin A, Fe (although missing animal products is a big problem)


Note: final scenarios agreed on:
§  Keep LU constant (2015) and run RCP2.6, 6.0, 8.5
§  Keep climate constant (closest to 2015) and run SSP1, 3, 5
§  Climate & LU combinations: SSP1-RCP2.6, SSP3-RCP6.0, SSP5-RCP8.5


"""

import math, os, sys, time, random, shutil, logging, csv, json

import numpy as np
from osgeo import gdal, osr, ogr
import pandas as pd
import geopandas as gpd
from collections import OrderedDict
import logging
import scipy

import numdal as nd
import geoecon as ge
import hazelbean as hb
import multiprocessing


import hazelbean as hb
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


def execute(**kw):
    L.info('Executing script.')
    if not kw:
        kw = get_default_kw()

    hb.create_dirs(kw['run_dir'])

    kw = setup_dirs(**kw)

    if kw['copy_base_data']:
        # NOTE Asymmetry, because this is just a copy process, we have both cases go to basis_dir. This allows recopy based on existence.
        kw['base_data_copy_dir'] = os.path.join(kw['basis_dir'], 'base_data_copy')
        kw = copy_base_data(**kw)
    else:
        kw['base_data_copy_dir'] = os.path.join(kw['basis_dir'], 'base_data_copy')

    if kw['sum_earthstat']:
        kw['crop_production_tons_per_cell_uri'] = os.path.join(kw['run_dir'], 'crop_production_tons_per_cell.tif')
        kw['crop_yield_per_ha_uri'] = os.path.join(kw['run_dir'], 'crop_yield_per_ha.tif')
        kw['crop_harvested_area_ha_uri'] = os.path.join(kw['run_dir'], 'crop_harvested_area_ha.tif')
        kw['crop_harvested_area_fraction_uri'] = os.path.join(kw['run_dir'], 'crop_harvested_area_fraction.tif')

        kw = sum_earthstat(**kw)
    else:
        kw['crop_production_tons_per_cell_uri'] = os.path.join(kw['basis_dir'], 'crop_production_tons_per_cell.tif')
        kw['crop_yield_per_ha_uri'] = os.path.join(kw['basis_dir'], 'crop_yield_per_ha.tif')
        kw['crop_harvested_area_ha_uri'] = os.path.join(kw['basis_dir'], 'crop_harvested_area_ha.tif')
        kw['crop_harvested_area_fraction_uri'] = os.path.join(kw['basis_dir'], 'crop_harvested_area_fraction.tif')

    if kw['resample_from_30s']:
        kw['resampled_data_dir'] = os.path.join(kw['run_dir'], 'resampled')
        hb.create_dirs(kw['resampled_data_dir'])
        kw = resample_from_30s(**kw)
    else:
        kw['resampled_data_dir'] = os.path.join(kw['basis_dir'], 'resampled')

    if kw['process_gaez_inputs']:
        kw['gaez_data_dir'] = os.path.join(kw['run_dir'], 'gaez')
        hb.create_dirs(kw['gaez_data_dir'])
        kw = process_gaez_inputs(**kw)
    else:
        kw['gaez_data_dir'] = os.path.join(kw['basis_dir'], 'gaez')

    if kw['create_spatial_lags']:
        kw['spatial_lags_dir'] = os.path.join(kw['run_dir'], 'spatial_lags')
        kw['adjacent_neighbors_uri'] = os.path.join(kw['spatial_lags_dir'], 'adjacent_neighbors.tif')
        kw['distance_weighted_5x5_neighbors_uri'] = os.path.join(kw['spatial_lags_dir'], 'distance_weighted_5x5_neighbors.tif')

        hb.create_dirs(kw['spatial_lags_dir'])
        kw = create_spatial_lags(**kw)
    else:
        kw['spatial_lags_dir'] = os.path.join(kw['basis_dir'], 'spatial_lags')
        kw['adjacent_neighbors_uri'] = os.path.join(kw['spatial_lags_dir'], 'adjacent_neighbors.tif')
        kw['distance_weighted_5x5_neighbors_uri'] = os.path.join(kw['spatial_lags_dir'], 'distance_weighted_5x5_neighbors.tif')


    if kw['create_baseline_regression_data']:
        kw['baseline_regression_data_uri'] = os.path.join(kw['run_dir'], 'baseline_regression_data.csv')
        kw['nan_mask_uri'] = os.path.join(kw['run_dir'], 'nan_mask.csv')
        kw = create_baseline_regression_data(**kw)
    else:
        kw['baseline_regression_data_uri'] = os.path.join(kw['basis_dir'], 'baseline_regression_data.csv')
        kw['nan_mask_uri'] = os.path.join(kw['basis_dir'], 'nan_mask.csv')

    if kw['clean_baseline_regression_data']:
        kw = clean_baseline_regression_data(**kw)
    else:
        pass

    if kw['create_nan_mask']:
        kw['nan_mask_uri'] = os.path.join(kw['run_dir'], 'nan_mask.csv')
        kw = create_nan_mask(**kw)
    else:
        kw['nan_mask_uri'] = os.path.join(kw['basis_dir'], 'nan_mask.csv')

    if kw['aggregate_crops_by_type']:
        kw['aggregated_crop_data_dir'] = os.path.join(kw['run_dir'], 'aggregated_crop_data')
        kw['aggregated_crop_data_csv_uri'] = os.path.join(kw['aggregated_crop_data_dir'], 'aggregated_crop_data.csv')
        hb.create_dirs(kw['aggregated_crop_data_dir'])
        kw = aggregate_crops_by_type(**kw)
    else:
        kw['aggregated_crop_data_dir'] = os.path.join(kw['basis_dir'], 'aggregated_crop_data')
        kw['aggregated_crop_data_csv_uri'] = os.path.join(kw['aggregated_crop_data_dir'], 'aggregated_crop_data.csv')

    if kw['convert_aggregated_crop_type_dfs_to_geotiffs']:
        kw['aggregated_crop_data_dir_2'] = os.path.join(kw['run_dir'], 'aggregated_crop_data')
        kw['data_registry']['c3_annual_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_PotassiumApplication_Rate.tif')
        kw['data_registry']['c3_annual_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_PhosphorusApplication_Rate.tif')
        kw['data_registry']['c3_annual_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_NitrogenApplication_Rate.tif')
        kw['data_registry']['c3_perennial_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_PotassiumApplication_Rate.tif')
        kw['data_registry']['c3_perennial_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_PhosphorusApplication_Rate.tif')
        kw['data_registry']['c3_perennial_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_NitrogenApplication_Rate.tif')
        kw['data_registry']['c4_annual_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_PotassiumApplication_Rate.tif')
        kw['data_registry']['c4_annual_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_PhosphorusApplication_Rate.tif')
        kw['data_registry']['c4_annual_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_NitrogenApplication_Rate.tif')
        kw['data_registry']['c4_perennial_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_PotassiumApplication_Rate.tif')
        kw['data_registry']['c4_perennial_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_PhosphorusApplication_Rate.tif')
        kw['data_registry']['c4_perennial_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_NitrogenApplication_Rate.tif')
        kw['data_registry']['nitrogen_fixer_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_PotassiumApplication_Rate.tif')
        kw['data_registry']['nitrogen_fixer_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_PhosphorusApplication_Rate.tif')
        kw['data_registry']['nitrogen_fixer_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_NitrogenApplication_Rate.tif')

        hb.create_dirs(kw['aggregated_crop_data_dir_2'])
        kw = convert_aggregated_crop_type_dfs_to_geotiffs(**kw)
    else:
        kw['aggregated_crop_data_dir_2'] = os.path.join(kw['basis_dir'], 'aggregated_crop_data')
        kw['data_registry']['c3_annual_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_PotassiumApplication_Rate.tif')
        kw['data_registry']['c3_annual_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_PhosphorusApplication_Rate.tif')
        kw['data_registry']['c3_annual_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_NitrogenApplication_Rate.tif')
        kw['data_registry']['c3_perennial_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_PotassiumApplication_Rate.tif')
        kw['data_registry']['c3_perennial_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_PhosphorusApplication_Rate.tif')
        kw['data_registry']['c3_perennial_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_NitrogenApplication_Rate.tif')
        kw['data_registry']['c4_annual_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_PotassiumApplication_Rate.tif')
        kw['data_registry']['c4_annual_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_PhosphorusApplication_Rate.tif')
        kw['data_registry']['c4_annual_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_NitrogenApplication_Rate.tif')
        kw['data_registry']['c4_perennial_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_PotassiumApplication_Rate.tif')
        kw['data_registry']['c4_perennial_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_PhosphorusApplication_Rate.tif')
        kw['data_registry']['c4_perennial_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_NitrogenApplication_Rate.tif')
        kw['data_registry']['nitrogen_fixer_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_PotassiumApplication_Rate.tif')
        kw['data_registry']['nitrogen_fixer_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_PhosphorusApplication_Rate.tif')
        kw['data_registry']['nitrogen_fixer_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_NitrogenApplication_Rate.tif')


    if kw['calc_optimal_regression_equations_among_linear_cubed']:
        kw['optimal_regression_equations_among_linear_cubed_dir'] = os.path.join(kw['run_dir'], 'optimal_regression_equations_among_linear_cubed')
        kw['optimal_regression_equations_among_linear_cubed_results_uri'] = os.path.join(kw['optimal_regression_equations_among_linear_cubed_dir'], 'combined_regression_results.json')
        hb.create_dirs(kw['optimal_regression_equations_among_linear_cubed_dir'])
        kw = calc_optimal_regression_equations_among_linear_cubed(**kw)
    else:
        kw['optimal_regression_equations_among_linear_cubed_dir'] = os.path.join(kw['basis_dir'], 'optimal_regression_equations_among_linear_cubed')
        kw['optimal_regression_equations_among_linear_cubed_results_uri'] = os.path.join(kw['optimal_regression_equations_among_linear_cubed_dir'], 'combined_regression_results.json')

    if kw['calc_simple_regression']:
        kw['simple_regression_dir'] = os.path.join(kw['run_dir'], 'simple_regression')
        kw['simple_regression_results_uri'] = os.path.join(kw['simple_regression_dir'], 'combined_regression_results.json')

        hb.create_dirs(kw['simple_regression_dir'])
        kw = calc_simple_regression(**kw)
    else:
        kw['simple_regression_dir'] = os.path.join(kw['basis_dir'], 'simple_regression')
        kw['simple_regression_results_uri'] = os.path.join(kw['simple_regression_dir'], 'combined_regression_results.json')


    if kw['calc_crop_types_regression']:
        kw['crop_types_regression_dir'] = os.path.join(kw['run_dir'], 'crop_types_regression')
        kw['crop_types_regression_results_uri'] = os.path.join(kw['crop_types_regression_dir'], 'combined_regression_results.json')
        hb.create_dirs(kw['crop_types_regression_dir'])
        kw = calc_crop_types_regression(**kw)
    else:
        kw['crop_types_regression_dir'] = os.path.join(kw['basis_dir'], 'crop_types_regression')
        kw['crop_types_regression_results_uri'] = os.path.join(kw['crop_types_regression_dir'], 'combined_regression_results.json')


    if kw['combine_crop_types_regressions_into_single_file']:
        kw['crop_types_regression_dir_2'] = os.path.join(kw['run_dir'], 'crop_types_regression')
        kw['crop_types_regression_results_uri'] = os.path.join(kw['crop_types_regression_dir_2'], 'crop_types_regression_results.json')

        hb.create_dirs(kw['crop_types_regression_dir_2'])
        kw = combine_crop_types_regressions_into_single_file(**kw)
    else:
        kw['crop_types_regression_dir_2'] = os.path.join(kw['basis_dir'], 'crop_types_regression')
        kw['crop_types_regression_results_uri'] = os.path.join(kw['crop_types_regression_dir_2'], 'crop_types_regression_results.json')


    if kw['create_climate_scenarios_df']:
        kw['climate_scenarios_csv_uri'] = os.path.join(kw['run_dir'], 'climate_scenarios.csv')
        kw['climate_scenarios_csv_with_nan_uri'] = os.path.join(kw['run_dir'], 'climate_scenarios_with_nan.csv')
        kw = create_climate_scenarios_df(**kw)
    else:
        kw['climate_scenarios_csv_uri'] = os.path.join(kw['basis_dir'], 'climate_scenarios.csv')
        kw['climate_scenarios_csv_with_nan_uri'] = os.path.join(kw['basis_dir'], 'climate_scenarios_with_nan.csv')

    if kw['project_crop_specific_calories_per_cell_based_on_climate']:
        kw['crop_specific_projection_csvs_dir'] = os.path.join(kw['run_dir'], 'crop_specific_projections')
        hb.create_dirs(kw['crop_specific_projection_csvs_dir'])
        kw = project_crop_specific_calories_per_cell_based_on_climate(**kw)
    else:
        kw['crop_specific_projection_csvs_dir'] = os.path.join(kw['basis_dir'], 'crop_specific_projections')

    if kw['project_crop_types_calories_per_cell_based_on_climate']:
        kw['crop_types_projection_csvs_dir'] = os.path.join(kw['run_dir'], 'crop_types_projections')
        hb.create_dirs(kw['crop_types_projection_csvs_dir'])
        kw = project_crop_types_calories_per_cell_based_on_climate(**kw)
    else:
        kw['crop_types_projection_csvs_dir'] = os.path.join(kw['basis_dir'], 'crop_types_projections')

    if kw['write_crop_specific_projections_from_reg_results']:
        kw['crop_specific_projections_geotiffs_dir'] = os.path.join(kw['run_dir'], 'crop_specific_projections_geotiffs')
        hb.create_dirs(kw['crop_specific_projections_geotiffs_dir'])
        kw = write_crop_specific_projections_from_reg_results(**kw)
    else:
        kw['crop_specific_projections_geotiffs_dir'] = os.path.join(kw['basis_dir'], 'crop_specific_projections_geotiffs')


    if kw['write_crop_types_projections_from_reg_results']:
        kw['crop_types_projections_geotiffs_dir'] = os.path.join(kw['run_dir'], 'crop_types_projections_geotiffs')
        hb.create_dirs(kw['crop_types_projections_geotiffs_dir'])
        kw = write_crop_types_projections_from_reg_results(**kw)
    else:
        kw['crop_types_projections_geotiffs_dir'] = os.path.join(kw['basis_dir'], 'crop_types_projections_geotiffs')

    if kw['combine_regressions_into_single_table']:
        kw = combine_regressions_into_single_table(**kw)
    else:
        pass

    if kw['create_results_for_each_rcp_ssp_pair']:
        kw['results_for_each_rcp_ssp_pair_dir'] = os.path.join(kw['run_dir'], 'results_for_each_rcp_ssp_pair')
        hb.create_dirs(kw['results_for_each_rcp_ssp_pair_dir'])
        kw = create_results_for_each_rcp_ssp_pair(**kw)
    else:
        kw['results_for_each_rcp_ssp_pair_dir'] = os.path.join(kw['basis_dir'], 'results_for_each_rcp_ssp_pair')


    if kw['create_maps_for_each_rcp_ssp_pair']:
        kw['maps_for_each_rcp_ssp_pair_dir'] = os.path.join(kw['run_dir'], 'maps_for_each_rcp_ssp_pair')
        hb.create_dirs(kw['maps_for_each_rcp_ssp_pair_dir'])
        kw = create_maps_for_each_rcp_ssp_pair(**kw)
    else:
        kw['maps_for_each_rcp_ssp_pair_dir'] = os.path.join(kw['basis_dir'], 'maps_for_each_rcp_ssp_pair')


    return kw


def setup_dirs(**kw):
    L.debug('Making default dirs.')

    dirs = [kw['project_dir'], kw['input_dir'], kw['intermediate_dir'], kw['run_dir'], kw['output_dir']]
    hb.create_dirs(dirs)

    return kw

def copy_base_data(**kw):
    hb.create_dirs(kw['base_data_copy_dir'])

    to_copy = [
        kw['base_data_country_names_uri'],
        kw['base_data_country_ids_raster_uri'],
        kw['base_data_calories_per_cell_uri'],
        kw['base_data_precip_uri'],
        kw['base_data_temperature_uri'],
        kw['base_data_gdp_2000_uri'],
        kw['base_data_ag_value_2000_uri'],
        kw['base_data_minutes_to_market_uri'],
        kw['base_data_ag_value_2000_uri'],
        kw['base_data_pop_30s_uri'],
        kw['base_data_proportion_pasture_uri'],
        kw['base_data_faostat_pasture_uri'],
        kw['base_data_ag_value_2005_spam_uri'],
        kw['base_data_workability_index_uri'],
        kw['base_data_toxicity_index_uri'],
        kw['base_data_rooting_conditions_index_uri'],
        kw['base_data_rainfed_land_percent_uri'],
        kw['base_data_protected_areas_index_uri'],
        kw['base_data_oxygen_availability_index_uri'],
        kw['base_data_nutrient_retention_index_uri'],
        kw['base_data_nutrient_availability_index_uri'],
        kw['base_data_irrigated_land_percent_uri'],
        kw['base_data_excess_salts_index_uri'],
        kw['base_data_cultivated_land_percent_uri'],
        kw['base_data_crop_suitability_uri'],
        kw['base_data_precip_2070_uri'],
        kw['base_data_temperature_2070_uri'],
        kw['base_data_slope_uri'],
        kw['base_data_altitude_uri'],
    ]

    local_keys = [
        'country_names_uri',
        'country_ids_raster_uri',
        'calories_per_cell_uri',
        'precip_uri',
        'temperature_uri',
        'gdp_2000_uri',
        'ag_value_2000_uri',
        'minutes_to_market_uri',
        'ag_value_2000_uri',
        'pop_30s_uri',
        'proportion_pasture_uri',
        'faostat_pasture_uri',
        'ag_value_2005_spam_uri',
        'workability_index_uri',
        'toxicity_index_uri',
        'rooting_conditions_index_uri',
        'rainfed_land_percent_uri',
        'protected_areas_index_uri',
        'oxygen_availability_index_uri',
        'nutrient_retention_index_uri',
        'nutrient_availability_index_uri',
        'irrigated_land_percent_uri',
        'excess_salts_index_uri',
        'cultivated_land_percent_uri',
        'crop_suitability_uri',
        'precip_2070_uri',
        'temperature_uri',
        'slope_uri',
        'altitude_uri',
    ]

    for i, base_data_uri in enumerate(to_copy):
        local_filename = nd.explode_uri(base_data_uri)['filename'].replace('base_data_', '', 1)
        local_uri = os.path.join(kw['base_data_copy_dir'], local_filename)
        kw[local_keys[i]] = local_uri
        if not os.path.exists(local_uri):
            L.info(local_uri + ' does not exist, so copying it from base data at ' + base_data_uri)

            shutil.copy(base_data_uri, local_uri)
    return kw



def resample_from_30s(**kw):
    calories_per_cell_af = nd.ArrayFrame(kw['calories_per_cell_uri'])


    base_data_uris = [
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'bulk_density_1m_30s.tif'),
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'CEC_1m_30s.tif'),
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'ph_1m_30s.tif'),
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'soil_organic_content_1m_30s.tif'),
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'sand_percent_1m_30s.tif'),
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'silt_percent_1m_30s.tif'),
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'clay_percent_1m_30s.tif'),
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'erodibility_30s.tif'),
                      os.path.join(kw['base_data_dir'], 'soil/soilgrids', 'root_depth_30s.tif'),
                      # os.path.join(kw['base_data_dir'], 'climate/worldclim/baseline/30s', 'alt.tif'),
                      os.path.join(kw['input_dir'], 'he85bi501.tif'),
                      os.path.join(kw['input_dir'], 'he85bi5012.tif'),
                      os.path.join(kw['input_dir'], 'he85bi701.tif'),
                      os.path.join(kw['input_dir'], 'he85bi7012.tif'),
                      ]

    output_uris = [
                   os.path.join(kw['resampled_data_dir'], 'bulk_density.tif'),
                   os.path.join(kw['resampled_data_dir'], 'cation_exchange_capacity.tif'),
                   os.path.join(kw['resampled_data_dir'], 'ph.tif'),
                   os.path.join(kw['resampled_data_dir'], 'soil_organic_content.tif'),
                   os.path.join(kw['resampled_data_dir'], 'sand_percent.tif'),
                   os.path.join(kw['resampled_data_dir'], 'silt_percent.tif'),
                   os.path.join(kw['resampled_data_dir'], 'clay_percent.tif'),
                   os.path.join(kw['resampled_data_dir'], 'erodibility.tif'),
                   os.path.join(kw['resampled_data_dir'], 'root_depth.tif'),
                   # os.path.join(kw['resampled_data_dir'], 'altitude.tif'),
                   os.path.join(kw['resampled_data_dir'], 'he85bi501_temperature.tif'),
                   os.path.join(kw['resampled_data_dir'], 'he85bi5012_precip.tif'),
                   os.path.join(kw['resampled_data_dir'], 'he85bi701_temperature.tif'),
                   os.path.join(kw['resampled_data_dir'], 'he85bi7012_precip.tif'),
                   ]

    for i in range(len(base_data_uris)):
        L.info('Resampling for ' + str(output_uris[i]))
        base_data_af = nd.ArrayFrame(base_data_uris[i])
        output_af = base_data_af.resample(calories_per_cell_af, resample_method='average', output_uri=output_uris[i])

    return kw


# NOTE, Multiprocessing only works on module-top-level functions for some reason...
# NOTE, cannot use local scope to pass variables (must be args).
def f(base_data_dir, match_af_uri, output_uri, match_string):
    file_uris = nd.get_list_of_file_uris_recursively(os.path.join(base_data_dir,
                                                                  'crops/earthstat\\crop_production'), filter_extensions='.tif', filter_strings=match_string)
    match_af = nd.ArrayFrame(match_af_uri)
    sum_array = np.zeros(match_af.shape)
    for file_uri in file_uris:
        L.info('Summing ' + file_uri)
        ds = gdal.Open(file_uri)
        sum_array += ds.GetRasterBand(1).ReadAsArray()
        ds = None
    output_af = nd.ArrayFrame(sum_array, match_af, output_uri=output_uri)
    return sum_array


def sum_earthstat(**kw):
    match_af = nd.ArrayFrame(kw['calories_per_cell_uri'])

    output_uris = [
        kw['crop_production_tons_per_cell_uri'],
        kw['crop_yield_per_ha_uri'],
        kw['crop_harvested_area_ha_uri'],
        kw['crop_harvested_area_fraction_uri'],
    ]

    match_strings = ['_Production.tif', '_YieldPerHectare.tif', '_HarvestedAreaHectares.tif', '_HarvestedAreaFraction.tif']

    jobs = []
    for i, output_uri in enumerate(output_uris):
        match_string = match_strings[i]
        p = multiprocessing.Process(target=f, args=(kw['base_data_dir'], kw['calories_per_cell_uri'], output_uri, match_string))
        jobs.append(p)
        p.start()

    # Wait for all to finish
    for j in jobs:
        j.join()
    # Result 24 seconds!

    return kw

def process_gaez_inputs(**kw):
    names = ['workability_index',
             'toxicity_index',
             'rooting_conditions_index',
             'protected_areas_index',
             'oxygen_availability_index',
             'nutrient_retention_index',
             'nutrient_availability_index',
             'excess_salts_index',
             ]

    # Save a local, modified version to project where the categories are converted to a continuous 0-1.
    for name in names:
        base_data_uri = os.path.join(kw['base_data_dir'], 'crops', 'gaez', name + '.tif')
        output_uri = os.path.join(kw['gaez_data_dir'], name + '.tif').replace('index', 'continuous')

        af = nd.ArrayFrame(base_data_uri)
        rules = {0: 0.0,
                 1: 1.0,
                 2: 0.75,
                 3: 0.5,
                 4: 0.25,
                 5: 0.0,
                 6: 0.0,
                 7: 0.0,}

        if 'protected' in base_data_uri:
            rules = {0: 1.0,
                     1: 0.5,
                     2: 0.0,
                     3: 0.0,
                     4: 0.0,
                     5: 0.0,
                     6: 0.0,
                     7: 0.0, }
        af.reclassify_array_by_dict(rules, output_uri=output_uri)

    return kw

def create_spatial_lags(**kw):
    calories_per_cell_af = nd.ArrayFrame(kw['calories_per_cell_uri'])
    adjacent_neighbors = np.array([[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]])
    adjacent_neighbors = adjacent_neighbors / np.sum(adjacent_neighbors)

    input_array = np.where(calories_per_cell_af.data == calories_per_cell_af.no_data_value, 0.0, calories_per_cell_af.data)

    adjacent_neighbors_convolved = scipy.ndimage.filters.convolve(input_array, adjacent_neighbors, mode='constant', cval=0.0)
    nd.ArrayFrame(adjacent_neighbors_convolved, calories_per_cell_af, output_uri=kw['adjacent_neighbors_uri'])

    distance_weighted_5x5_neighbors = np.array([[0.35355339059, 0.414213562, .5, 0.414213562, 0.35355339059],  # .70710678 = 1/2^.5
                        [0.414213562, .70710678, 1, .70710678, 0.414213562],
                        [.5, 1, 0, 1, .5],
                        [0.414213562, .70710678, 1, .70710678, 0.414213562],
                        [0.35355339059, 0.414213562, .5, 0.414213562, 0.35355339059]])
    distance_weighted_5x5_neighbors = distance_weighted_5x5_neighbors / np.sum(distance_weighted_5x5_neighbors)

    distance_weighted_5x5_neighbors_convolved = scipy.ndimage.filters.convolve(input_array, distance_weighted_5x5_neighbors, mode='constant', cval=0.0)
    nd.ArrayFrame(distance_weighted_5x5_neighbors_convolved, calories_per_cell_af, output_uri=kw['distance_weighted_5x5_neighbors_uri'])

    return kw


def create_baseline_regression_data(**kw):

    # For specific files, you can specify both a name and a uri, which can be different from the dict key (name)
    input_uris = OrderedDict()
    input_uris['calories_per_cell'] = kw['calories_per_cell_uri']
    input_uris['precip'] = kw['precip_uri']
    input_uris['temperature'] = kw['temperature_uri']
    input_uris['gdp_2000'] = kw['gdp_2000_uri']
    input_uris['ag_value_2000'] = kw['ag_value_2000_uri']
    input_uris['minutes_to_market'] = kw['minutes_to_market_uri']
    input_uris['adjacent_neighbors'] = kw['adjacent_neighbors_uri']
    input_uris['distance_weighted_5x5_neighbors'] = kw['distance_weighted_5x5_neighbors_uri']
    input_uris['ag_value_2005_spam'] = kw['ag_value_2005_spam_uri']
    input_uris['proportion_cropland'] = kw['proportion_cropland_uri']
    input_uris['workability'] = os.path.join(kw['gaez_data_dir'], 'workability_continuous.tif')
    input_uris['toxicity'] = os.path.join(kw['gaez_data_dir'], 'toxicity_continuous.tif')
    input_uris['rooting_conditions'] = os.path.join(kw['gaez_data_dir'], 'rooting_conditions_continuous.tif')
    input_uris['protected_areas'] = os.path.join(kw['gaez_data_dir'], 'protected_areas_continuous.tif')
    input_uris['oxygen_availability'] = os.path.join(kw['gaez_data_dir'], 'oxygen_availability_continuous.tif')
    input_uris['nutrient_retention'] = os.path.join(kw['gaez_data_dir'], 'nutrient_retention_continuous.tif')
    input_uris['nutrient_availability'] = os.path.join(kw['gaez_data_dir'], 'nutrient_availability_continuous.tif')
    input_uris['excess_salts'] = os.path.join(kw['gaez_data_dir'], 'excess_salts_continuous.tif')
    input_uris['irrigated_land_percent'] = kw['base_data_irrigated_land_percent_uri']
    input_uris['rainfed_land_percent'] = kw['base_data_rainfed_land_percent_uri']
    input_uris['cultivated_land_percent'] = kw['base_data_cultivated_land_percent_uri']
    input_uris['crop_suitability'] = kw['base_data_crop_suitability_uri']
    input_uris['gdp_gecon'] = os.path.join(kw['base_data_dir'], 'socioeconomic\\nordhaus_gecon', 'gdp_per_capita_2000_5m.tif')
    input_uris['slope'] = kw['slope_uri']
    input_uris['altitude'] = kw['altitude_uri']

    for crop_name in kw['crop_names']:
        input_uris[crop_name + ''] = os.path.join(kw['base_data_price_per_ha_masked_dir'], crop_name + '_production_value_per_ha_gt01_national_price.tif')
        input_uris[crop_name + '_calories_per_ha'] = os.path.join(kw['base_data_crop_calories_dir'], crop_name + '_calories_per_ha_masked.tif')
        input_uris[crop_name + '_proportion_cultivated'] = os.path.join(kw['base_data_dir'], 'crops/earthstat/crop_production', crop_name + '_HarvAreaYield_Geotiff', crop_name + '_HarvestedAreaFraction.tif')

    for crop_name in kw['crop_names']:
        for nutrient in ['Potassium', 'Phosphorus', 'Nitrogen']:
            name = crop_name + '_' + nutrient + 'Application_Rate'
            input_uris[name] = os.path.join(kw['base_data_dir'], 'crops/earthstat/crop_fertilizer/Fertilizer_' + crop_name, crop_name + '_' + nutrient + 'Application_Rate.tif')

    # Specify Dirs where all tifs will be added.
    input_dirs = []
    input_dirs.append(kw['resampled_data_dir'])

    # Iterate through input_uris adding them.  Currently also fixes fertilizer nan issues.
    af_names_list = []
    dfs_list = []
    for name, uri in input_uris.items():
        # TODOO Some 933e18 values are slipping under calories. Prevo9us attempts ddnt eliminate. not sure why.
        # TODOO, Bad kludge. Decide how to deal with input-specific modifications like nan-to-zero
        if 'Fertilizer' in uri or '_calories_per_ha' in uri or '_HarvestedAreaFraction' in uri or 'altitude' in uri or 'slope' in uri:
            af = nd.ArrayFrame(uri)
            # NOTE, originaly i had this as af = af.where() which failed to modify the af before going in.
            modified_array = np.where((af.data < 0) | (af.data > 9999999999999999), 0, af.data)
            modified_af = nd.ArrayFrame(modified_array, af)
            af_names_list.append(name)
            df = ge.convert_af_to_1d_df(modified_af)
            dfs_list.append(df)
        else:
            af = nd.ArrayFrame(uri)
            af_names_list.append(name)
            df = ge.convert_af_to_1d_df(af)
            dfs_list.append(df)

    for dir in input_dirs:
        uris_list = nd.get_list_of_file_uris_recursively(dir, filter_extensions='.tif')
        for uri in uris_list:
            if 'Fertilizer' in uri or '_calories_per_ha_masked' in uri or '_HarvestedAreaFraction' in uri or 'altitude' in uri or 'slope' in uri:
                name = nd.explode_uri(uri)['file_root']
                af = nd.ArrayFrame(uri)
                # NOTE, originaly i had this as af = af.where() which failed to modify the af before going in.
                modified_array = np.where(af.data < 0, 0, af.data)
                modified_af = nd.ArrayFrame(modified_array, af)
                af_names_list.append(name)
                df = ge.convert_af_to_1d_df(modified_af)
                dfs_list.append(df)
            else:
                name = nd.explode_uri(uri)['file_root']
                af = nd.ArrayFrame(uri)
                af_names_list.append(name)
                df = ge.convert_af_to_1d_df(af)
                dfs_list.append(df)

    L.info('Concatenating all dataframes.')
    # CAREFUL, here all my data are indeed positive but this could change.
    # REMEMBER, we are just determining what gets written to disk here, not what is regressed.
    # LEARNING POINT: stack_dfs was only for time series space-time-frames and just happend to work because i didn't structure my data in the wrong way..
    # df = ge.stack_dfs(dfs_list, af_names_list)
    df = ge.concatenate_dfs_horizontally(dfs_list, af_names_list)
    df[df < 0] = 0.0


    # Rather than getting rid of all cells without crops, just get rid of those not on land.
    df[df['excess_salts'] == 255.0] = np.nan

    # kw['nan_mask_uri'] = 'nan_mask.csv'
    df_nan = df['excess_salts']
    df_nan.to_csv(kw['nan_mask_uri'])

    df = df.dropna()

    df.to_csv(kw['baseline_regression_data_uri'])

    return kw

def clean_baseline_regression_data(**kw):
    df = pd.read_csv(kw['baseline_regression_data_uri'], index_col=['Unnamed: 0'])  # Could speed up here with usecols='excess_salts

    for col_name in df.columns:
        print('col_name', col_name)
        df[col_name][df[col_name] > 1e+17] = np.nan

    df.to_csv(kw['baseline_regression_data_uri'])

    return kw


def create_nan_mask(**kw):
    # LEARNING POINT , ommitting index_col=['Unnamed: 0'] was a massive time-sink mistake. Deal with index columns more carefully.
    df = pd.read_csv(kw['baseline_regression_data_uri'], index_col=['Unnamed: 0']) # Could speed up here with usecols='excess_salts
    df[df['excess_salts'] == 255.0] = np.nan
    df_nan = df['excess_salts']
    df_nan.columns = ['not_nan']
    df_nan = df_nan.dropna()
    df_nan.to_csv(kw['nan_mask_uri'])

    return kw


def aggregate_crops_by_type(**kw):
    """CMIP6 and the land-use harmonization project have centered on 5 crop types: c3 annual, c3 perennial, c4 annual, c4 perennial, nitrogen fixer
    Aggregate the 15 crops to those four categories by modifying the baseline_regression_data."""

    baseline_regression_data_df = pd.read_csv(kw['baseline_regression_data_uri'])
    baseline_regression_data_df.set_index('Unnamed: 0', inplace=True)
    # baseline_regression_data_df = ge.read_csv_sample(kw['baseline_regression_data_uri'], .001)

    vars_names_to_aggregate = [
        'production_value_per_ha',
        'calories_per_ha',
        'proportion_cultivated',
        'PotassiumApplication_Rate',
        'PhosphorusApplication_Rate',
        'NitrogenApplication_Rate',
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

    # Create a DF of zeros, ready to hold the summed results for each crop type. Indix given will  be from baseline_regression_data_df so that spatial indices match.
    crop_types_df = pd.DataFrame(np.zeros(len(baseline_regression_data_df.index)), index=baseline_regression_data_df.index)

    # Iterate through crop_types
    for crop_type, crops in crop_membership.items():
        L.info('Aggregating ' + str(crop_type) + ' ' + str(crops))
        for var_name_to_aggregate in vars_names_to_aggregate:
            output_col_name = crop_type + '_' + var_name_to_aggregate
            crop_types_df[output_col_name] = np.zeros(len(baseline_regression_data_df.index))
            for crop in crops:
                input_col_name = crop + '_' + var_name_to_aggregate
                if input_col_name in baseline_regression_data_df:
                    crop_types_df[output_col_name] += baseline_regression_data_df[input_col_name]

    crop_types_df.to_csv(kw['aggregated_crop_data_csv_uri'])

    return kw

def convert_aggregated_crop_type_dfs_to_geotiffs(**kw):
    match_af = nd.ArrayFrame(kw['calories_per_cell_uri'])

    df = pd.read_csv(kw['aggregated_crop_data_csv_uri'])
    df.set_index('Unnamed: 0', inplace=True)

    cols_to_plot = ['c3_annual_production_value_per_ha',
    'c3_annual_calories_per_ha',
    'c3_annual_proportion_cultivated',
    'c3_annual_PotassiumApplication_Rate',
    'c3_annual_PhosphorusApplication_Rate',
    'c3_annual_NitrogenApplication_Rate',
    'c3_perennial_production_value_per_ha',
    'c3_perennial_calories_per_ha',
    'c3_perennial_proportion_cultivated',
    'c3_perennial_PotassiumApplication_Rate',
    'c3_perennial_PhosphorusApplication_Rate',
    'c3_perennial_NitrogenApplication_Rate',
    'c4_annual_production_value_per_ha',
    'c4_annual_calories_per_ha',
    'c4_annual_proportion_cultivated',
    'c4_annual_PotassiumApplication_Rate',
    'c4_annual_PhosphorusApplication_Rate',
    'c4_annual_NitrogenApplication_Rate',
    'c4_perennial_production_value_per_ha',
    'c4_perennial_calories_per_ha',
    'c4_perennial_proportion_cultivated',
    'c4_perennial_PotassiumApplication_Rate',
    'c4_perennial_PhosphorusApplication_Rate',
    'c4_perennial_NitrogenApplication_Rate',
    'nitrogen_fixer_production_value_per_ha',
    'nitrogen_fixer_calories_per_ha',
    'nitrogen_fixer_proportion_cultivated',
    'nitrogen_fixer_PotassiumApplication_Rate',
    'nitrogen_fixer_PhosphorusApplication_Rate',
    'nitrogen_fixer_NitrogenApplication_Rate',]

    for col_name in cols_to_plot:
        print('plotting', col_name)
        column = col_name
        output_uri = os.path.join(kw['aggregated_crop_data_dir_2'], col_name + '.tif')
        af = ge.convert_df_to_af_via_index(df, column, match_af, nd.temp('.tif'))

        af = nd.where(af>10000000000000000000, af.no_data_value, af, output_uri=output_uri)

        # print('unique values in ' + col_name + str(nd.get_value_count_odict_from_array(af.data)))
        # print('unique values in ' + col_name)
        # print(np.histogram(af.data))


        # try:
        #     af.show(output_uri=output_uri.replace('.tif', '.png'))
        # except:
        #     pass
    return kw

def calc_optimal_regression_equations_among_linear_cubed(**kw):
    def gen_r_code_for_crop(crop_name):
        r_string = """library(tidyverse)
            library(MASS)
            library(stats4)
            
            d <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/baseline_regression_data.csv")
            
            d$precip_2 <- d$precip ^ 2
            d$precip_3 <- d$precip ^ 3
            d$temperature_2 <- d$temperature ^ 2
            d$temperature_3 <- d$temperature ^ 3
            d$minutes_to_market_2 <- d$minutes_to_market ^ 2
            d$minutes_to_market_3 <- d$minutes_to_market ^ 3
            d$proportion_cropland_2 <- d$proportion_cropland ^ 2
            d$proportion_cropland_3 <- d$proportion_cropland ^ 3
            d$gdp_gecon_2 <- d$gdp_gecon ^ 2
            d$gdp_gecon_3 <- d$gdp_gecon ^ 3
            d$altitude_2 <- d$altitude ^ 2
            d$altitude_3 <- d$altitude ^ 3
            d$slope_2 <- d$slope ^ 2
            d$slope_3 <- d$slope ^ 3
            d$irrigated_land_percent_2 <- d$irrigated_land_percent ^ 2
            d$irrigated_land_percent_3 <- d$irrigated_land_percent ^ 3
            d$crop_suitability_2 <- d$crop_suitability ^ 2
            d$crop_suitability_3 <- d$crop_suitability ^ 3
            d$""" + crop_name + """_PotassiumApplication_Rate_2 <- d$""" + crop_name + """_PotassiumApplication_Rate ^ 2
            d$""" + crop_name + """_PotassiumApplication_Rate_3 <- d$""" + crop_name + """_PotassiumApplication_Rate ^ 3
            d$""" + crop_name + """_PhosphorusApplication_Rate_2 <- d$""" + crop_name + """_PhosphorusApplication_Rate ^ 2
            d$""" + crop_name + """_PhosphorusApplication_Rate_3 <- d$""" + crop_name + """_PhosphorusApplication_Rate ^ 3
            d$""" + crop_name + """_NitrogenApplication_Rate_2 <- d$""" + crop_name + """_NitrogenApplication_Rate ^ 2
            d$""" + crop_name + """_NitrogenApplication_Rate_3 <- d$""" + crop_name + """_NitrogenApplication_Rate ^ 3
            
            # # Make a copy of  the data, set the depvar variables that are 0 to be NA, then drop them.
            # d_""" + crop_name + """ = d
            # d_""" + crop_name + """$""" + crop_name + """_calories_per_ha[d_""" + crop_name + """$""" + crop_name + """_calories_per_ha==0] <- NA
            # d_""" + crop_name + """ = d_""" + crop_name + """[!is.na(d_""" + crop_name + """$""" + crop_name + """_calories_per_ha),]
                        
            """ + crop_name + """_linear_formula_string = """ + crop_name + """_calories_per_ha ~ precip + temperature + minutes_to_market + proportion_cropland + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + crop_suitability + gdp_gecon + altitude + slope + """ + crop_name + """_PotassiumApplication_Rate + """ + crop_name + """_PhosphorusApplication_Rate + """ + crop_name + """_NitrogenApplication_Rate
            """ + crop_name + """_linear_fit <- lm(""" + crop_name + """_linear_formula_string, data=d_""" + crop_name + """)
            summary(""" + crop_name + """_linear_fit)
            # "<^>""" + crop_name + """_linear_fit<^>"
            # step(""" + crop_name + """_linear_fit)
            # "<^>"
            """ + crop_name + """_full_formula_string = """ + crop_name + """_calories_per_ha ~ precip + precip_2  + precip_3 + temperature + temperature_2 + temperature_3 + minutes_to_market + minutes_to_market_2 + minutes_to_market_3 + proportion_cropland + proportion_cropland_2 + proportion_cropland_3 + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + irrigated_land_percent_2 + irrigated_land_percent_3 + crop_suitability + crop_suitability_2 + crop_suitability_3 + gdp_gecon + gdp_gecon_2 + gdp_gecon_3 + altitude + altitude_2 + altitude_3 + slope + slope_2 + slope_3 + """ + crop_name + """_PotassiumApplication_Rate + """ + crop_name + """_PotassiumApplication_Rate_2 + """ + crop_name + """_PotassiumApplication_Rate_3 + """ + crop_name + """_PhosphorusApplication_Rate + """ + crop_name + """_PhosphorusApplication_Rate_2 + """ + crop_name + """_PhosphorusApplication_Rate_3 + """ + crop_name + """_NitrogenApplication_Rate + """ + crop_name + """_NitrogenApplication_Rate_2 + """ + crop_name + """_NitrogenApplication_Rate_3
            """ + crop_name + """_full_fit <- lm(""" + crop_name + """_full_formula_string, data=d_""" + crop_name + """)
            summary(""" + crop_name + """_full_fit)
            "<^>""" + crop_name + """_full_fit<^>"
            step(""" + crop_name + """_full_fit)
            "<^>"
            
            """ + crop_name + """_value_full_string = """ + crop_name + """_production_value_per_ha ~ precip + precip_2 + precip_3 + temperature + temperature_2 + temperature_3 + minutes_to_market + minutes_to_market_2 + minutes_to_market_3 + proportion_cropland + proportion_cropland_2 + proportion_cropland_3 + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + irrigated_land_percent_2 + irrigated_land_percent_3 + crop_suitability + crop_suitability_2 + crop_suitability_3 + gdp_gecon + gdp_gecon_2 + gdp_gecon_3 + altitude + altitude_2 + altitude_3 + slope + slope_2 + slope_3 + """ + crop_name + """_PotassiumApplication_Rate + """ + crop_name + """_PotassiumApplication_Rate_2 + """ + crop_name + """_PotassiumApplication_Rate_3 + """ + crop_name + """_PhosphorusApplication_Rate + """ + crop_name + """_PhosphorusApplication_Rate_2 + """ + crop_name + """_PhosphorusApplication_Rate_3 + """ + crop_name + """_NitrogenApplication_Rate + """ + crop_name + """_NitrogenApplication_Rate_2 + """ + crop_name + """_NitrogenApplication_Rate_3
            """ + crop_name + """_value_fit <- lm(""" + crop_name + """_value_full_string, data=d_""" + crop_name + """)
            summary(""" + crop_name + """_value_fit)
            "<^>""" + crop_name + """_value_fit<^>"
            step(""" + crop_name + """_value_fit)
            "<^>"
    
        """

        return r_string

    also_make_sample = False
    if also_make_sample:
        df = ge.read_csv_sample(kw['baseline_regression_data_uri'], 0.01, index_col=0)
        df.to_csv(kw['baseline_regression_data_uri'].replace('.csv', '_sample.csv'))

    # I limited to 4 due to memory issues
    jobs = []
    current_thread = 0
    num_concurrent = 4
    for crop_name in kw['crop_names']:
        current_thread += 1


        r_string = gen_r_code_for_crop(crop_name)
        output_uri=os.path.join(kw['optimal_regression_equations_among_linear_cubed_dir'], crop_name + '_regression_results.txt')
        print('Starting regression process for ' + output_uri)
        script_save_uri=os.path.join(kw['optimal_regression_equations_among_linear_cubed_dir'], crop_name + '_regression_code.R')
        p = multiprocessing.Process(target=ge.execute_r_string, args=(r_string, output_uri, script_save_uri, True))
        jobs.append(p)
        p.start()

        # LEARNING POINT, I accidentally skipped every 6th one because it did the join INSTEAD of launching the thread.
        # LEARNING POINT, i had to add the len() check because the script would move on before finishing the last regression.
        if current_thread > num_concurrent or current_thread >= len(kw['crop_names']):
            # Wait for all to finish
            for j in jobs:
                j.join()
            jobs = []
            current_thread = 0


    L.info('Combining regressions into single file.')
    kw['optimal_regression_equations_among_linear_cubed_results'] = OrderedDict()
    for crop_name in kw['crop_names']:
        uri = os.path.join(kw['optimal_regression_equations_among_linear_cubed_dir'], crop_name + '_regression_results.txt')

        if os.path.exists(uri):
            with open(uri) as f:
                content = [i.replace('\n', '') for i in f.readlines()]
                content = '\n'.join(content)
                returned_odict = hb.parse_cat_ears_in_string(content)
        else:
            print(uri + ' does not exist.')

        L.info('Loaded regression results\n' + nd.pp(returned_odict, return_as_string=True))

        crop_output = OrderedDict()

        for regression_name, v in returned_odict.items():
            coefficients_raw_r_string = hb.get_strings_between_values(v, 'Coefficients:', '[1]')
            pair_lists = []
            for i in coefficients_raw_r_string:
                lines = i.split('\n')
                for remove_spaces in lines:
                    r = remove_spaces.split(' ')
                    pair_lists.append([i for i in r if len(i) > 0])

            pair_lists = [i for i in pair_lists if len(i) > 0]

            keys = []
            values = []
            for c, i in enumerate(pair_lists):
                if c % 2 == 0:
                    keys.extend(i)
                else:
                    values.extend(i)

            crop_output[regression_name] = OrderedDict(zip(keys, values))

        kw['optimal_regression_equations_among_linear_cubed_results'][crop_name] = crop_output



    with open(kw['optimal_regression_equations_among_linear_cubed_results_uri_uri'], 'w') as f:
        json.dump(kw['optimal_regression_equations_among_linear_cubed_results'], f)

    json_string = json.dumps(kw['optimal_regression_equations_among_linear_cubed_results'])

    return kw


def calc_simple_regression(**kw):
    def gen_r_code_for_crop(crop_name):
        r_string = """library(tidyverse)
            library(MASS)
            library(stats4)
            
            ## Enables reading subset of csv
            #library(sqldf)
            #query_string <- "select * from file order by random() limit 20000"
            #d <- read.csv.sql(file = "C:/OneDrive/Projects/ipbes/intermediate/baseline_regression_data.csv", sql = query_string)

            # NON subset way
            d <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/baseline_regression_data.csv")

            # d$precip_2 <- d$precip ^ 2
            # d$precip_3 <- d$precip ^ 3
            # d$temperature_2 <- d$temperature ^ 2
            # d$temperature_3 <- d$temperature ^ 3
            # d$minutes_to_market_2 <- d$minutes_to_market ^ 2
            # d$minutes_to_market_3 <- d$minutes_to_market ^ 3
            # d$proportion_cropland_2 <- d$proportion_cropland ^ 2
            # d$proportion_cropland_3 <- d$proportion_cropland ^ 3
            # d$gdp_gecon_2 <- d$gdp_gecon ^ 2
            # d$gdp_gecon_3 <- d$gdp_gecon ^ 3
            # d$altitude_2 <- d$altitude ^ 2
            # d$altitude_3 <- d$altitude ^ 3
            # d$slope_2 <- d$slope ^ 2
            # d$slope_3 <- d$slope ^ 3
            # d$irrigated_land_percent_2 <- d$irrigated_land_percent ^ 2
            # d$irrigated_land_percent_3 <- d$irrigated_land_percent ^ 3
            # d$crop_suitability_2 <- d$crop_suitability ^ 2
            # d$crop_suitability_3 <- d$crop_suitability ^ 3
            # d$""" + crop_name + """_PotassiumApplication_Rate_2 <- d$""" + crop_name + """_PotassiumApplication_Rate ^ 2
            # d$""" + crop_name + """_PotassiumApplication_Rate_3 <- d$""" + crop_name + """_PotassiumApplication_Rate ^ 3
            # d$""" + crop_name + """_PhosphorusApplication_Rate_2 <- d$""" + crop_name + """_PhosphorusApplication_Rate ^ 2
            # d$""" + crop_name + """_PhosphorusApplication_Rate_3 <- d$""" + crop_name + """_PhosphorusApplication_Rate ^ 3
            # d$""" + crop_name + """_NitrogenApplication_Rate_2 <- d$""" + crop_name + """_NitrogenApplication_Rate ^ 2
            # d$""" + crop_name + """_NitrogenApplication_Rate_3 <- d$""" + crop_name + """_NitrogenApplication_Rate ^ 3

            # Make a copy of  the data, set the depvar variables that are 0 to be NA, then drop them.
            d_""" + crop_name + """ = d
            d_""" + crop_name + """$""" + crop_name + """_calories_per_ha[d_""" + crop_name + """$""" + crop_name + """_calories_per_ha==0] <- NA
            d_""" + crop_name + """ = d_""" + crop_name + """[!is.na(d_""" + crop_name + """$""" + crop_name + """_calories_per_ha),]

            # Define the regression string and run  it
            """ + crop_name + """_linear_formula_string = """ + crop_name + """_calories_per_ha ~ precip + temperature + minutes_to_market + proportion_cropland + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + crop_suitability + gdp_gecon + altitude + slope + """ + crop_name + """_PotassiumApplication_Rate + """ + crop_name + """_PhosphorusApplication_Rate + """ + crop_name + """_NitrogenApplication_Rate
            """ + crop_name + """_linear_fit <- lm(""" + crop_name + """_linear_formula_string, data=d_""" + crop_name + """)
            
            # Now print a cat-ears version to be extracted by python
            "<^>""" + crop_name + """<^>"
            summary(""" + crop_name + """_linear_fit)
            "<^>"


        """

        return r_string

def calc_crop_types_regression_old(**kw):
    def gen_r_code_for_crop(crop_name):
        r_string = """library(tidyverse)
            library(MASS)
            library(stats4)
            
            options("width"=4000) # Because string formatting is used to extract results, dont want a split line.
                       
            d_baseline <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/baseline_regression_data.csv")
            d_crop_groups <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/aggregated_crop_data/aggregated_crop_data.csv")



            # # Make a copy of  the data, set the depvar variables that are 0 to be NA, then drop them.
            # d_baseline_copy = d_baseline
            # d_baseline_copy$maize_calories_per_ha[d_baseline_copy$maize_calories_per_ha==0] <- NA # Note manual use of  MAIZE here because the base data doesn't have the c3 stuff
            # d_baseline_copy = d_baseline_copy[!is.na(d_baseline_copy$maize_calories_per_ha),]

            full_data = cbind(d_baseline, d_crop_groups)

            full_data$""" + crop_name + """_calories_per_ha[full_data$""" + crop_name + """_calories_per_ha==0] <- NA
            full_data = full_data[!is.na(full_data$""" + crop_name + """_calories_per_ha),]


            # Define the regression string and run  it
            """ + crop_name + """_linear_formula_string = """ + crop_name + """_calories_per_ha ~ precip + temperature + minutes_to_market  + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + gdp_gecon + altitude + slope + """ + crop_name + """_PotassiumApplication_Rate + """ + crop_name + """_PhosphorusApplication_Rate + """ + crop_name + """_NitrogenApplication_Rate
            """ + crop_name + """_linear_fit <- lm(""" + crop_name + """_linear_formula_string, data=full_data)
            
            # Now print a cat-ears version to be extracted by python
            "<^>""" + crop_name + """<^>"
            summary(""" + crop_name + """_linear_fit)
            "<^>"


        """

        # r_string = """library(tidyverse)
        #     library(MASS)
        #     library(stats4)
        #
        #     options("width"=4000) # Because string formatting is used to extract results, dont want a split line.
        #
        #     d_baseline <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/baseline_regression_data.csv")
        #     d_crop_groups <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/aggregated_crop_data/aggregated_crop_data.csv")
        #
        #
        #
        #     # # Make a copy of  the data, set the depvar variables that are 0 to be NA, then drop them.
        #     # d_baseline_copy = d_baseline
        #     # d_baseline_copy$maize_calories_per_ha[d_baseline_copy$maize_calories_per_ha==0] <- NA # Note manual use of  MAIZE here because the base data doesn't have the c3 stuff
        #     # d_baseline_copy = d_baseline_copy[!is.na(d_baseline_copy$maize_calories_per_ha),]
        #
        #     full_data = cbind(d_baseline, d_crop_groups)
        #
        #     full_data$""" + crop_name + """_calories_per_ha[full_data$""" + crop_name + """_calories_per_ha==0] <- NA
        #     full_data = full_data[!is.na(full_data$""" + crop_name + """_calories_per_ha),]
        #
        #
        #     # Define the regression string and run  it
        #     """ + crop_name + """_linear_formula_string = """ + crop_name + """_calories_per_ha ~ precip + temperature + minutes_to_market + proportion_cropland + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + crop_suitability + gdp_gecon + altitude + slope + """ + crop_name + """_PotassiumApplication_Rate + """ + crop_name + """_PhosphorusApplication_Rate + """ + crop_name + """_NitrogenApplication_Rate
        #     """ + crop_name + """_linear_fit <- lm(""" + crop_name + """_linear_formula_string, data=full_data)
        #
        #     # Now print a cat-ears version to be extracted by python
        #     "<^>""" + crop_name + """<^>"
        #     summary(""" + crop_name + """_linear_fit)
        #     "<^>"
        #
        #
        # """
        #
        return r_string

    # I limited to 4 due to memory issues
    jobs = []
    current_thread = 0
    num_concurrent = 1
    for name in kw['crop_types']:
        current_thread += 1

        r_string = gen_r_code_for_crop(name)
        output_uri = os.path.join(kw['crop_types_regression_dir'], name + '_regression_results.txt')
        L.info('Starting regression process for ' + output_uri)

        script_save_uri = os.path.join(kw['crop_types_regression_dir'], name + '_regression_code.R')
        p = multiprocessing.Process(target=ge.execute_r_string, args=(r_string, output_uri, script_save_uri, True))
        jobs.append(p)
        p.start()

        # LEARNING POINT, I accidentally skipped every 6th one because it did the join INSTEAD of launching the thread.
        # LEARNING POINT, this had to go AFTER the p.start() else jobs wasn't the right lenegth
        if current_thread > num_concurrent or current_thread >= len(kw['crop_types']):
            # Wait for all to finish
            L.info('Waiting for threads to finish.')
            for j in jobs:
                j.join()
            jobs = []
            current_thread = 0




    return kw


def calc_crop_types_regression(**kw):
    def gen_r_code_for_crop(crop_name):
        r_string = """library(tidyverse)
            library(MASS)
            library(stats4)
            options("width"=4000) # Because string formatting is used to extract results, dont want a split line.


            ### Create and combine the data
            d_baseline <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/baseline_regression_data.csv")            
            d_crop_groups <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/aggregated_crop_data/aggregated_crop_data.csv")
            d = cbind(d_baseline, d_crop_groups)                       

            d$precip_2 <- d$precip ^ 2
            d$precip_3 <- d$precip ^ 3
            d$temperature_2 <- d$temperature ^ 2
            d$temperature_3 <- d$temperature ^ 3
            d$minutes_to_market_2 <- d$minutes_to_market ^ 2
            d$minutes_to_market_3 <- d$minutes_to_market ^ 3
            # d$proportion_cropland_2 <- d$proportion_cropland ^ 2
            # d$proportion_cropland_3 <- d$proportion_cropland ^ 3
            d$gdp_gecon_2 <- d$gdp_gecon ^ 2
            d$gdp_gecon_3 <- d$gdp_gecon ^ 3
            d$altitude_2 <- d$altitude ^ 2
            d$altitude_3 <- d$altitude ^ 3
            d$slope_2 <- d$slope ^ 2
            d$slope_3 <- d$slope ^ 3
            d$irrigated_land_percent_2 <- d$irrigated_land_percent ^ 2
            d$irrigated_land_percent_3 <- d$irrigated_land_percent ^ 3
            d$crop_suitability_2 <- d$crop_suitability ^ 2
            d$crop_suitability_3 <- d$crop_suitability ^ 3
            d$""" + crop_name + """_PotassiumApplication_Rate_2 <- d$""" + crop_name + """_PotassiumApplication_Rate ^ 2
            d$""" + crop_name + """_PotassiumApplication_Rate_3 <- d$""" + crop_name + """_PotassiumApplication_Rate ^ 3
            d$""" + crop_name + """_PhosphorusApplication_Rate_2 <- d$""" + crop_name + """_PhosphorusApplication_Rate ^ 2
            d$""" + crop_name + """_PhosphorusApplication_Rate_3 <- d$""" + crop_name + """_PhosphorusApplication_Rate ^ 3
            d$""" + crop_name + """_NitrogenApplication_Rate_2 <- d$""" + crop_name + """_NitrogenApplication_Rate ^ 2
            d$""" + crop_name + """_NitrogenApplication_Rate_3 <- d$""" + crop_name + """_NitrogenApplication_Rate ^ 3

            ### Make a copy of  the data, set the depvar variables that are 0 to be NA, then drop them.
            d_""" + crop_name + """ = d
            d_""" + crop_name + """$""" + crop_name + """_calories_per_ha[d_""" + crop_name + """$""" + crop_name + """_calories_per_ha==0] <- NA
            d_""" + crop_name + """ = d_""" + crop_name + """[!is.na(d_""" + crop_name + """$""" + crop_name + """_calories_per_ha),]

            ## Linear regression, all vars
            """ + crop_name + """_linear_formula_string = """ + crop_name + """_calories_per_ha ~ precip + temperature + minutes_to_market + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + crop_suitability + gdp_gecon + altitude + slope + """ + crop_name + """_PotassiumApplication_Rate + """ + crop_name + """_PhosphorusApplication_Rate + """ + crop_name + """_NitrogenApplication_Rate
            """ + crop_name + """_linear_fit <- lm(""" + crop_name + """_linear_formula_string, data=d_""" + crop_name + """)
            "<^>""" + crop_name + """_linear_fit<^>"
            summary(""" + crop_name + """_linear_fit)
            "<^>"
            
            ### Cubic regression, all vars
            """ + crop_name + """_full_formula_string = """ + crop_name + """_calories_per_ha ~ precip + precip_2  + precip_3 + temperature + temperature_2 + temperature_3 + minutes_to_market + minutes_to_market_2 + minutes_to_market_3 + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + irrigated_land_percent_2 + irrigated_land_percent_3 + crop_suitability + crop_suitability_2 + crop_suitability_3 + gdp_gecon + gdp_gecon_2 + gdp_gecon_3 + altitude + altitude_2 + altitude_3 + slope + slope_2 + slope_3 + """ + crop_name + """_PotassiumApplication_Rate + """ + crop_name + """_PotassiumApplication_Rate_2 + """ + crop_name + """_PotassiumApplication_Rate_3 + """ + crop_name + """_PhosphorusApplication_Rate + """ + crop_name + """_PhosphorusApplication_Rate_2 + """ + crop_name + """_PhosphorusApplication_Rate_3 + """ + crop_name + """_NitrogenApplication_Rate + """ + crop_name + """_NitrogenApplication_Rate_2 + """ + crop_name + """_NitrogenApplication_Rate_3
            """ + crop_name + """_full_fit <- lm(""" + crop_name + """_full_formula_string, data=d_""" + crop_name + """)
            "<^>""" + crop_name + """_full_fit<^>"
            summary(""" + crop_name + """_full_fit)
            "<^>"
            
            ### Linear regression no_endogenous
            """ + crop_name + """_linear_formula_string = """ + crop_name + """_calories_per_ha ~ precip + temperature + minutes_to_market + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + crop_suitability + gdp_gecon + altitude + slope 
            """ + crop_name + """_linear_fit <- lm(""" + crop_name + """_linear_formula_string, data=d_""" + crop_name + """)
            "<^>""" + crop_name + """_linear_fit_no_endogenous<^>"
            summary(""" + crop_name + """_linear_fit)
            "<^>"
            
            ### Cubic regression no_endogenous
            """ + crop_name + """_full_formula_string = """ + crop_name + """_calories_per_ha ~ precip + precip_2  + precip_3 + temperature + temperature_2 + temperature_3 + minutes_to_market + minutes_to_market_2 + minutes_to_market_3 + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts  + crop_suitability + crop_suitability_2 + crop_suitability_3 + gdp_gecon + gdp_gecon_2 + gdp_gecon_3 + altitude + altitude_2 + altitude_3 + slope + slope_2 + slope_3 
            """ + crop_name + """_full_fit <- lm(""" + crop_name + """_full_formula_string, data=d_""" + crop_name + """)
            "<^>""" + crop_name + """_full_fit_no_endogenous<^>"
            summary(""" + crop_name + """_full_fit)
            "<^>"



        """

        return r_string

    also_make_sample = False
    if also_make_sample:
        df = ge.read_csv_sample(kw['baseline_regression_data_uri'], 0.01, index_col=0)
        df.to_csv(kw['baseline_regression_data_uri'].replace('.csv', '_sample.csv'))




    # I limited to 4 due to memory issues
    jobs = []
    current_thread = 0
    num_concurrent = 4
    for name in kw['crop_types'][0:1]:
        current_thread += 1

        r_string = gen_r_code_for_crop(name)
        output_uri = os.path.join(kw['crop_types_regression_dir'], name + '_regression_results.txt')
        L.info('Starting regression process for ' + output_uri)

        script_save_uri = os.path.join(kw['crop_types_regression_dir'], name + '_regression_code.R')
        p = multiprocessing.Process(target=ge.execute_r_string, args=(r_string, output_uri, script_save_uri, True))
        jobs.append(p)
        p.start()

        # LEARNING POINT, I accidentally skipped every 6th one because it did the join INSTEAD of launching the thread.
        # LEARNING POINT, this had to go AFTER the p.start() else jobs wasn't the right lenegth
        if current_thread > num_concurrent or current_thread >= len(kw['crop_types']):
            # Wait for all to finish
            L.info('Waiting for threads to finish.')
            for j in jobs:
                j.join()
            jobs = []
            current_thread = 0



    return kw



def combine_crop_types_regressions_into_single_file(**kw):
    L.info('Combining regressions into single file.')
    kw['crop_types_regression_results'] = OrderedDict()
    for name in kw['crop_types']:
        print('crop type', name)
        uri = os.path.join(kw['crop_types_regression_dir'], name + '_regression_results.txt')
        print('uri', uri)
        if os.path.exists(uri):
            with open(uri) as f:
                content = [i.replace('\n', '') for i in f.readlines()]
                content = '\n'.join(content)
                returned_odict = hb.parse_cat_ears_in_string(content)
        else:
            print(uri + ' does not exist.')

        L.info('Loaded regression results\n' + nd.pp(returned_odict, return_as_string=True))

        crop_output = OrderedDict()

        for regression_name, v in returned_odict.items():

            # coefficients_raw_r_string = hb.get_strings_between_values(v, 'Coefficients:', '[1]')
            coefficients_raw_r_string = hb.get_strings_between_values(v, 'Coefficients:', '---')
            # coefficients_raw_r_string = hb.get_strings_between_values(v, 'Pr(>|t|)\n', '[1]')

            pair_lists = []
            for i in coefficients_raw_r_string:
                lines = i.split('\n')
                for remove_spaces in lines:
                    r = remove_spaces.split(' ')
                    pair_lists.append([i for i in r if len(i) > 0])

            keys = []
            values = []
            for c, i in enumerate(pair_lists):
                if len(i) > 0 and i[0] != 'Estimate':
                    keys.append(i[0])
                    values.append(i[1])

            crop_output[regression_name] = OrderedDict(zip(keys, values))

        kw['crop_types_regression_results'][name] = crop_output

    with open(kw['crop_types_regression_results_uri'], 'w') as f:
        json.dump(kw['crop_types_regression_results'], f)

    json_string = json.dumps(kw['crop_types_regression_results'])

    return kw

def create_climate_scenarios_df(**kw):
    # TODOO, Because I already have the regression equations, there is no reason to load the tifs as DFs. Just iterate through the regression vars andd multiply the geotiffs. This means defining apriori the var names.
    reg_dir = kw['crop_types_regression_results_uri']
    nan_mask_df = pd.read_csv(kw['nan_mask_uri'])

    input_uris = OrderedDict()
    # input_uris['he85bi501'] = os.path.join(kw['resampled_data_dir'], 'he85bi501_temperature.tif')
    # input_uris['he85bi5012'] = os.path.join(kw['resampled_data_dir'], 'he85bi5012_precip.tif')
    # input_uris['he85bi701'] = os.path.join(kw['resampled_data_dir'], 'he85bi701_temperature.tif')
    # input_uris['he85bi7012'] = os.path.join(kw['resampled_data_dir'], 'he85bi7012_precip.tif')

    input_uris['cur01'] = os.path.join(kw['input_dir'], 'bio1.bil')
    input_uris['cur012'] = os.path.join(kw['input_dir'], 'bio12.bil')
    input_uris['hd26bi701'] = os.path.join(kw['input_dir'], 'hd26bi701.tif')
    input_uris['hd26bi7012'] = os.path.join(kw['input_dir'], 'hd26bi7012.tif')
    input_uris['hd60bi701'] = os.path.join(kw['input_dir'], 'hd60bi701.tif')
    input_uris['hd60bi7012'] = os.path.join(kw['input_dir'], 'hd60bi7012.tif')
    input_uris['hd85bi701'] = os.path.join(kw['input_dir'], 'hd85bi701.tif')
    input_uris['hd85bi7012'] = os.path.join(kw['input_dir'], 'hd85bi7012.tif')
    # "C:\OneDrive\Projects\ipbes\input\bio1.bil"
    af_names_list = []
    dfs_list = []

    match_uri = kw['calories_per_cell_uri']
    match_af = nd.ArrayFrame(kw['calories_per_cell_uri'])

    aligned_uris = OrderedDict()
    for name, uri in input_uris.items():
        dst_uri = os.path.join(kw['run_dir'], os.path.split(uri)[1])
        hb.align_dataset_to_match(uri, match_uri, dst_uri)

        aligned_uris[name] = dst_uri

    # BIO VARS are 4320 by 1800 you fool!

    for name, uri in aligned_uris.items():
        print('name, uri', name, uri)
        af = nd.ArrayFrame(uri)
        af_names_list.append(name)
        df = ge.convert_af_to_1d_df(af)
        dfs_list.append(df)
    df = ge.concatenate_dfs_horizontally(dfs_list, af_names_list)
    df[df < 0] = 0.0

    # Rather than getting rid of all cells without crops, just get rid of those not on land.
    # df[nan_mask_df.ix[:, 0] == np.nan] = np.nan
    print('df1', df)
    df[nan_mask_df == np.nan] = np.nan
    df.to_csv(kw['climate_scenarios_csv_with_nan_uri'])

    df = df.dropna()
    print('df2', df)

    df.to_csv(kw['climate_scenarios_csv_uri'])
    df = None

    return kw



def project_crop_specific_calories_per_cell_based_on_climate(**kw):
    L.info('Loading baseline_regression_data_uri')
    data_df = pd.read_csv(kw['baseline_regression_data_uri'])
    L.info('Loading nan_mask_uri')
    nan_mask_df = pd.read_csv(kw['nan_mask_uri'])
    L.info('Loading climate_scenarios_csv_uri')
    modified_df = pd.read_csv(kw['climate_scenarios_csv_uri'])
    modified_df.rename(columns={'he85bi701': 'temperature', 'he85bi7012': 'precip'}, inplace=True)

    regression_results_odict = nd.file_to_python_object(kw['optimal_regression_equations_among_linear_cubed_results_uri'])

    # crop_names = list(regression_results_odict.keys())
    crop_names = kw['crop_names']
    # regression_names = ['_full_fit', '_value_fit']
    regression_names = ['_full_fit']

    for crop in crop_names:
        for regression_type in regression_names:

            change_df = pd.DataFrame(np.zeros(len(data_df.index)), index=data_df.index)

            regression_key = crop + regression_type
            r = regression_results_odict[crop][regression_key]
            vars_in_regression = list(r.keys())

            for var in vars_in_regression:
                if var in modified_df.columns or var.replace('_2', '') in modified_df.columns or var.replace('_3', '') in modified_df.columns:
                    var_value = r[var]
                    if var.endswith('_2'):
                        var = var.replace('_2', '')
                        change_df[0] += (modified_df[var] ** 2 - data_df[var] ** 2) * float(var_value)
                    elif var.endswith('_3'):
                        var = var.replace('_3', '')
                        change_df[0] += (modified_df[var] ** 3 - data_df[var] ** 3) * float(var_value)
                    else:
                        change_df[0] += (modified_df[var] - data_df[var]) * float(var_value)
            # print(crop, regression_type, sum(change_df[0]))

        change_df.to_csv(os.path.join(kw['crop_specific_projection_csvs_dir'], crop + '_projections.csv'))

    return kw

def project_crop_types_calories_per_cell_based_on_climate(**kw):
    L.info('Loading baseline_regression_data_uri')
    data_df = pd.read_csv(kw['baseline_regression_data_uri'], index_col=0)
    L.info('Loading nan_mask_uri')
    nan_mask_df = pd.read_csv(kw['nan_mask_uri'])
    L.info('Loading climate_scenarios_csv_uri')
    # modified_df = pd.read_csv(kw['climate_scenarios_csv_uri'])
    modified_df = pd.read_csv(kw['climate_scenarios_csv_with_nan_uri'])

    print('data_df', data_df)
    print('modified_df', modified_df)
    print('nan_mask_df', nan_mask_df)

    # print('modified_df', modified_df)

    # TODO I think he problem is here, that i zome nonzero values for climate in modified, butthey have zero in base-data, thus big-ass changes.
    modified_df.rename(columns={'he85bi701': 'temperature', 'he85bi7012': 'precip'}, inplace=True)

    regression_results_odict = nd.file_to_python_object(kw['crop_types_regression_results_uri'])

    # crop_names = list(regression_results_odict.keys())
    crop_types = kw['crop_types']
    # regression_names = ['_full_fit', '_value_fit']
    # regression_names = ['_crop_types']
    regression_names = ['_linear_fit', '_full_fit']

    for crop_type in crop_types:
        print('crop_type', crop_type)
        for regression_type in regression_names:

            change_df = pd.DataFrame(np.zeros(len(data_df.index)), index=data_df.index)
            regression_key = crop_type + regression_type

            r = regression_results_odict[crop_type][regression_key]
            vars_in_regression = list(r.keys())

            print(r)
            print(vars_in_regression)

            for var in vars_in_regression:
                if var in modified_df.columns or var.replace('_2', '') in modified_df.columns or var.replace('_3', '') in modified_df.columns:
                    var_value = r[var]

                    print(33, var, np.count_nonzero(modified_df[var]))
                    print(33, var, np.count_nonzero(data_df[var]))

                    if var.endswith('_2'):
                        var = var.replace('_2', '')
                        change_df[0] += (modified_df[var] ** 2 - data_df[var] ** 2) * float(var_value)
                    elif var.endswith('_3'):
                        var = var.replace('_3', '')
                        change_df[0] += (modified_df[var] ** 3 - data_df[var] ** 3) * float(var_value)
                    else:
                        change_df[0] += (modified_df[var] - data_df[var]) * float(var_value)
            # print(crop_type, regression_type, sum(change_df[0]))

        change_df.to_csv(os.path.join(kw['crop_types_projection_csvs_dir'], crop_type + '_projections.csv'))
        print('change_df', np.sum(change_df))
    return kw

#
# def project_single_crop_type_calories_per_cell_based_on_climate(**kw):
#     L.info('Loading baseline_regression_data_uri')
#     data_df = pd.read_csv(kw['baseline_regression_data_uri'], index_col=0)
#     L.info('Loading nan_mask_uri')
#     nan_mask_df = pd.read_csv(kw['nan_mask_uri'])
#     L.info('Loading climate_scenarios_csv_uri')
#     modified_df = pd.read_csv(kw['climate_scenarios_csv_uri'])
#     print('modified_df', modified_df)
#     modified_df.rename(columns={'he85bi701': 'temperature', 'he85bi7012': 'precip'}, inplace=True)
#
#     regression_results_odict = nd.file_to_python_object(kw['crop_types_regression_results_uri'])
#
#     # crop_names = list(regression_results_odict.keys())
#     crop_types = kw['crop_types']
#     # regression_names = ['_full_fit', '_value_fit']
#     regression_names = ['_crop_types']
#
#     for crop_type in crop_types:
#         for regression_type in regression_names:
#
#             change_df = pd.DataFrame(np.zeros(len(data_df.index)), index=data_df.index)
#
#             regression_key = crop_type + regression_type
#
#             # NOTE asymmetery on difficult line. because i only have 1 reg type, it happens to have the same type but this might change.
#             regression_key = crop_type
#             r = regression_results_odict[crop_type][regression_key]
#             vars_in_regression = list(r.keys())
#
#             for var in vars_in_regression:
#                 if var in modified_df.columns or var.replace('_2', '') in modified_df.columns or var.replace('_3', '') in modified_df.columns:
#                     var_value = r[var]
#                     if var.endswith('_2'):
#                         var = var.replace('_2', '')
#                         change_df[0] += (modified_df[var] ** 2 - data_df[var] ** 2) * float(var_value)
#                     elif var.endswith('_3'):
#                         var = var.replace('_3', '')
#                         change_df[0] += (modified_df[var] ** 3 - data_df[var] ** 3) * float(var_value)
#                     else:
#                         change_df[0] += (modified_df[var] - data_df[var]) * float(var_value)
#             print(crop_type, regression_type, sum(change_df[0]))
#
#         change_df.to_csv(os.path.join(kw['crop_types_projection_csvs_dir'], crop_type + '_projections.csv'))
#
#     return kw


def write_crop_specific_projections_from_reg_results(**kw):
    calories_per_cell_af = nd.ArrayFrame(kw['calories_per_cell_uri'])
    calories_per_cell_df = ge.convert_af_to_1d_df(calories_per_cell_af)
    calories_per_cell_af.show(keep_output=True)

    for crop_name in kw['crop_names']:
        L.info('Writing projections for ' + crop_name + ' to tif.')
        # reg_name = crop_name + '_regression'
        reg_name = crop_name

        crop_name_current_uri = os.path.join(kw['base_data_crop_calories_dir'], crop_name + '_calories_per_ha_masked.tif')
        crop_name_current = nd.ArrayFrame(crop_name_current_uri)
        crop_name_current.show()

        projected_change_full_df = pd.DataFrame(np.zeros(len(calories_per_cell_df.index)), index=calories_per_cell_df.index)
        projected_change_subset_df = pd.read_csv(os.path.join(kw['crop_specific_projection_csvs_dir'], reg_name + '_projections.csv'), index_col=0)

        projected_change_full_df[0][projected_change_subset_df.index] = projected_change_subset_df
        projected_change_array = projected_change_full_df.values.reshape(calories_per_cell_af.shape)

        projected_change_uri = os.path.join(kw['crop_specific_projections_geotiffs_dir'], crop_name + '_delta_calories_per_ha.tif')
        projected_change = nd.ArrayFrame(projected_change_array, crop_name_current, output_uri=projected_change_uri)
        # projected_change.show(keep_output=True)

        projected_change_current_mask_array = np.where(crop_name_current.data > 0, projected_change_array, np.nan)
        projected_change_current_mask_uri = os.path.join(kw['crop_specific_projections_geotiffs_dir'], crop_name + '_delta_calories_per_ha_current_mask.tif')
        projected_change_current_mask = nd.ArrayFrame(projected_change_current_mask_array, crop_name_current, output_uri=projected_change_current_mask_uri)
        projected_change_current_mask.show(keep_output=True)

        projected_calories_array = crop_name_current.data + projected_change_array
        output_uri = os.path.join(kw['crop_specific_projections_geotiffs_dir'], crop_name + '_projected_calories_per_ha.tif')
        projected_calories = nd.ArrayFrame(projected_calories_array, calories_per_cell_af, output_uri=output_uri)
        projected_calories.show(keep_output=True)

        projected_calories_current_mask_array = np.where(crop_name_current.data > 0, projected_calories_array, np.nan)
        projected_calories_current_mask_uri = os.path.join(kw['crop_specific_projections_geotiffs_dir'], crop_name + '_projected_calories_per_ha_current_mask.tif')
        projected_calories_current_mask = nd.ArrayFrame(projected_calories_current_mask_array, crop_name_current, output_uri=projected_calories_current_mask_uri)
        projected_calories_current_mask.show(keep_output=True)

    return kw


def write_crop_types_projections_from_reg_results(**kw):
    calories_per_cell_af = nd.ArrayFrame(kw['calories_per_cell_uri'])
    calories_per_cell_df = ge.convert_af_to_1d_df(calories_per_cell_af)
    calories_per_cell_af.show(keep_output=True)

    for crop_type in kw['crop_types']:
        L.info('Writing projections for ' + crop_type + ' to tif.')
        # reg_name = crop_type + '_regression'
        reg_name = crop_type

        crop_type_current_uri = os.path.join(kw['aggregated_crop_data_dir_2'], crop_type + '_calories_per_ha.tif')
        crop_type_current = nd.ArrayFrame(crop_type_current_uri)
        # crop_type_current.show()

        # NOTE!!!!! all the cells go to the upper left instead of skipping to legit cells.

        projected_change_full_df = pd.DataFrame(np.zeros(len(calories_per_cell_df.index)), index=calories_per_cell_df.index)

        projected_change_subset_df = pd.read_csv(os.path.join(kw['crop_types_projection_csvs_dir'], reg_name + '_projections.csv'), index_col=0)

        # print('projected_change_subset_df', projected_change_subset_df)


        # projected_change_full_df[0][projected_change_subset_df.index] = projected_change_subset_df
        projected_change_full_df[0][projected_change_subset_df.index] = projected_change_subset_df.ix[:,0]
        projected_change_array = projected_change_full_df.values.reshape(calories_per_cell_af.shape)

        projected_change_uri = os.path.join(kw['crop_types_projections_geotiffs_dir'], crop_type + '_delta_calories_per_ha.tif')
        projected_change = nd.ArrayFrame(projected_change_array, crop_type_current, output_uri=projected_change_uri)
        # projected_change.show(keep_output=True)

        projected_change_current_mask_array = np.where(crop_type_current.data > 0, projected_change_array, np.nan)
        projected_change_current_mask_uri = os.path.join(kw['crop_types_projections_geotiffs_dir'], crop_type + '_delta_calories_per_ha_current_mask.tif')
        projected_change_current_mask = nd.ArrayFrame(projected_change_current_mask_array, crop_type_current, output_uri=projected_change_current_mask_uri)
        # projected_change_current_mask.show(keep_output=True)

        projected_calories_array = crop_type_current.data + projected_change_array
        output_uri = os.path.join(kw['crop_types_projections_geotiffs_dir'], crop_type + '_projected_calories_per_ha.tif')
        projected_calories = nd.ArrayFrame(projected_calories_array, calories_per_cell_af, output_uri=output_uri)
        # projected_calories.show(keep_output=True)

        projected_calories_current_mask_array = np.where(crop_type_current.data > 0, projected_calories_array, np.nan)
        projected_calories_current_mask_uri = os.path.join(kw['crop_types_projections_geotiffs_dir'], crop_type + '_projected_calories_per_ha_current_mask.tif')
        projected_calories_current_mask = nd.ArrayFrame(projected_calories_current_mask_array, crop_type_current, output_uri=projected_calories_current_mask_uri)
        # projected_calories_current_mask.show(keep_output=True)

    return kw


def combine_regressions_into_single_table(**kw):
    # reg_dir = kw['crop_types_regression_dir_2']
    reg_dir = kw['crop_types_projection_csvs_dir']

    # kw['simple_regression_results']
    # optimal_regression_equations_among_linear_cubed_results_uri
    combined_regression_results = OrderedDict()
    for reg_name in kw['crop_types']:
        # print('reg_name', reg_name)
        coefficients_csv_uri = os.path.join(reg_dir, reg_name + '_projections.csv')
        coefficients_odict, metadata = hb.file_to_python_object(coefficients_csv_uri, declare_type='DD', return_all_parts=True)

        # Write just first row on first pass.
        if len(combined_regression_results) == 0:
            for row_name in metadata['row_headers']:
                combined_regression_results[row_name] = OrderedDict()

        # Write coefficient values
        for row_name in metadata['row_headers']:
            value_to_write = coefficients_odict[row_name]['Estimate'] + ge.calc_significance_stars_from_p_value(coefficients_odict[row_name]['Pr(>|t|)'])
            combined_regression_results[row_name][reg_name] = value_to_write

    optimal_regression_equations_among_linear_cubed_results_uri = os.path.join(kw['crop_types_regression_results_uri'], 'combined_regression_results.csv')
    hb.python_object_to_csv(combined_regression_results, optimal_regression_equations_among_linear_cubed_results_uri)
    return kw


def create_results_for_each_rcp_ssp_pair(**kw):

    scenarios = ['sspcur', 'ssp1', 'ssp3', 'ssp5']
    rcps = ['rcpcur', 'rcp26', 'rcp60', 'rcp85'] # NOTE, alll Hadgem2-AO

    scenario_permutations = ['sspcur_rcp26', 'sspcur_rcp60', 'sspcur_rcp85', 'ssp1_rcpcur', 'ssp3_rcpcur', 'ssp5_rcpcur', 'ssp1_rcp26', 'ssp3_rcp60', 'ssp5_rcp85',]

    crop_type_extent_prefix = [
        'c4ann',
        'c4per',
        'c3ann',
        'c3nfx',
        'c3per',
    ]

    crop_type_extent_filename = [
        'C4 annual crops',
        'C4 perennial crops',
        'C3 annual crops',
        'C3 nitrogen-fixing crops',
        'C3 perennial crops',
    ]

    climate_col_names = 'bio1,bio12,hd26bi701,hd26bi7012,hd60bi701,hd60bi7012,hd85bi701,hd85bi7012'.split(',')
    temperature_names_by_scenario = {
                                'sspcur_rcp26': 'hd26bi701.tif',
                                'sspcur_rcp60': 'hd60bi701.tif',
                                'sspcur_rcp85': 'hd85bi701.tif',
                                'ssp1_rcpcur': 'cur01.bil', # NOTE awful name mangling here. Caused because on last day of deadline and i didnt create a good enough regression framework.
                                'ssp3_rcpcur': 'cur01.bil',
                                'ssp5_rcpcur': 'cur01.bil',
                                'ssp1_rcp26': 'hd26bi701.tif',
                                'ssp3_rcp60': 'hd60bi701.tif',
                                'ssp5_rcp85': 'hd85bi701.tif',
                                }

    precip_names_by_scenario = {
                                'sspcur_rcp26': 'hd26bi7012.tif',
                                'sspcur_rcp60': 'hd60bi7012.tif',
                                'sspcur_rcp85': 'hd85bi7012.tif',
                                'ssp1_rcpcur': 'cur012.bil',
                                'ssp3_rcpcur': 'cur012.bil',
                                'ssp5_rcpcur': 'cur012.bil',
                                'ssp1_rcp26': 'hd26bi7012.tif',
                                'ssp3_rcp60': 'hd60bi7012.tif',
                                'ssp5_rcp85': 'hd85bi7012.tif',
                                }

    def get_ssp_scenario_string_from_label(scenario_label):
        if 'sspcur' in scenario_label:
            return 'IMAGE-ssp126/2015'
        elif 'ssp1' in scenario_label:
            return 'IMAGE-ssp126/2070'
        elif 'ssp3' in scenario_label:
            return 'AIM-ssp370/2070'
        elif 'ssp5' in scenario_label:
            return 'MAGPIE-ssp585/2070'


    # Load all necessary  data
    regression_results_odict = nd.file_to_python_object(kw['crop_types_regression_results_uri'])

    L.info('Loading baseline_regression_data_uri')
    data_df = pd.read_csv(kw['baseline_regression_data_uri'], index_col=0)


    # Create a df with anything that has changed
    L.info('Loading climate_scenarios_csv_uri')
    # modified_df = pd.read_csv(kw['climate_scenarios_csv_uri'], index_col=0)


    # joined = pd.merge(data_df, modified_df, left_index=True, right_index=True)
    # print('joined', joined)
    # modified_df = joined

    # TODO I've gotten myself confused here. The join was necessary to make sure meged_df was the same length, as data_df,
    # But then when I regressed, the vars now had the wrong names and everything came up as no change.


    modified_df = pd.read_csv(kw['climate_scenarios_csv_with_nan_uri'], index_col=0)
    # print('modified_df2', modified_df)

    # Rename anythin in the DF so it matches the name int he regression
    # TODO MISTAKE HERE? doesnt have all types
    # modified_df.rename(columns={'he85bi701': 'temperature', 'he85bi7012': 'precip'}, inplace=True)

    # modified_df[modified_df['precip']==0] = np.nan
    # modified_df = modified_df.dropna()

    for scenario in scenarios[1:2]:
        for rcp in rcps[0:2]:
            scenario_label = scenario + '_' + rcp
            print ('scenario_label', scenario_label)
            if scenario_label in scenario_permutations:
                ssp_scenario_string = get_ssp_scenario_string_from_label(scenario_label)
                for i, crop_type in enumerate(kw['crop_types'][0:1]):

                    regression_name = crop_type + '_linear_fit_no_endogenous'
                    print('     crop_type', crop_type)

                    # STEP - Calculate yield map given climate params
                    change_df = pd.DataFrame(np.zeros(len(data_df.index)), index=data_df.index)

                    current_regression_coefficients = regression_results_odict[crop_type][regression_name]
                    current_regression_coefficient_names = list(current_regression_coefficients.keys())
                    current_regression_coefficient_values = list(current_regression_coefficients.values())


                    for modified_var_name in list(modified_df.columns.values):
                        in_regression = False
                        # Fix the wrongly named precip-temp vars
                        if modified_var_name == os.path.splitext(temperature_names_by_scenario[scenario_label])[0] or modified_var_name == os.path.splitext(precip_names_by_scenario[scenario_label])[0]:
                            in_regression = True
                            if modified_var_name.endswith('1'):
                                data_df_name = 'temperature'
                            elif modified_var_name.endswith('12'):
                                data_df_name = 'precip'
                        else:
                            data_df_name = modified_var_name

                        if modified_var_name in current_regression_coefficient_names or in_regression:
                            if modified_var_name.endswith('_2'):
                                modified_var_name = modified_var_name.replace('_2', '')
                                change_df[0] += (modified_df[modified_var_name] ** 2 - data_df[data_df_name] ** 2) * float(current_regression_coefficients[data_df_name])
                            elif modified_var_name.endswith('_3'):
                                modified_var_name = modified_var_name.replace('_3', '')
                                change_df[0] += (modified_df[modified_var_name] ** 3 - data_df[data_df_name] ** 3) * float(current_regression_coefficients[data_df_name])

                            else:
                                to_add = (modified_df[modified_var_name] - data_df[data_df_name]) * float(current_regression_coefficients[data_df_name])
                                print('to_add', to_add)
                                for i in range(len(modified_df[modified_var_name])):
                                    print(modified_df[modified_var_name])
                                    print(np.ndarray(modified_df[modified_var_name])[i])
                                    print(np.ndarray(data_df[data_df_name])[i])
                                change_df[0] += to_add
                               # change_df[0] += (modified_df[modified_var_name] - data_df[data_df_name]) * float(current_regression_coefficients[data_df_name])

                            print('nanmean', np.nanmean(change_df[0]))
                            print('sum' ,np.sum(change_df[0]))


                    # change_df[change_df > 1e+18] = 0.0
                    # change_df[change_df < -1e+18] = 0.0
                    projection_df_path = os.path.join(kw['results_for_each_rcp_ssp_pair_dir'], scenario_label + '_' + crop_type + '.csv')
                    change_df.to_csv(projection_df_path)

    return kw


def create_maps_for_each_rcp_ssp_pair(**kw):

    calories_per_cell_af = nd.ArrayFrame(kw['calories_per_cell_uri'])
    calories_per_cell_df = ge.convert_af_to_1d_df(calories_per_cell_af)
    # calories_per_cell_af.show(keep_output=True)

    L.info('Loading nan_mask_uri')
    nan_mask_df = pd.read_csv(kw['nan_mask_uri'], index_col=0)

    for crop_type in kw['crop_types']:
        L.info('Writing projections for ' + crop_type + ' to tif.')

        crop_type_current_uri = os.path.join(kw['aggregated_crop_data_dir_2'], crop_type + '_calories_per_ha.tif')
        crop_type_current_af = nd.ArrayFrame(crop_type_current_uri)

        df = ge.convert_af_to_df(crop_type_current_af)

        # # ERROR CHECK that converting by reshape works for the input data
        # new_df = pd.DataFrame(np.zeros(len(df.index)), index=df.index)
        # new_df[0][df_data_only.index] = df_data_only.ix[:,0]
        # new_array = new_df.values.reshape(crop_type_current_af.shape)
        # ge.show_array(new_array)


        for filename in os.listdir(kw['results_for_each_rcp_ssp_pair_dir']):
            if crop_type in filename:
                print('Processing crop_type', crop_type, filename)

                crop_type_current_uri = os.path.join(kw['aggregated_crop_data_dir_2'], crop_type + '_calories_per_ha.tif')
                crop_type_current = nd.ArrayFrame(crop_type_current_uri)
                file_path = os.path.join(kw['results_for_each_rcp_ssp_pair_dir'], filename)
                projected_change_full_df = pd.DataFrame(np.zeros(len(calories_per_cell_df.index)), index=calories_per_cell_df.index)
                projected_change_subset_df = pd.read_csv(file_path, index_col=0)

                projected_change_full_df[0][projected_change_subset_df.index] = projected_change_subset_df.ix[:, 0]
                projected_change_array = projected_change_full_df.values.reshape(calories_per_cell_af.shape)

                projected_change_path = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], filename.replace('.csv', '.tif'))
                hb.save_array_as_geotiff(projected_change_array, projected_change_path, kw['calories_per_cell_uri'], data_type_override=7)

                ge.show_array(projected_change_array, output_uri=projected_change_path.replace('.tif', '.png'))

                #
                # projected_change_current_mask_array = np.where(crop_type_current.data > 0, projected_change_array, np.nan)
                # projected_change_current_mask_uri = os.path.join(kw['crop_types_projections_geotiffs_dir'], crop_type + '_delta_calories_per_ha_current_mask.tif')
                # projected_change_current_mask = nd.ArrayFrame(projected_change_current_mask_array, crop_type_current, output_uri=projected_change_current_mask_uri)
                # # projected_change_current_mask.show(keep_output=True)
                #
                # projected_calories_array = crop_type_current.data + projected_change_array
                # output_uri = os.path.join(kw['crop_types_projections_geotiffs_dir'], crop_type + '_projected_calories_per_ha.tif')
                # projected_calories = nd.ArrayFrame(projected_calories_array, calories_per_cell_af, output_uri=output_uri)
                # # projected_calories.show(keep_output=True)
                #
                # projected_calories_current_mask_array = np.where(crop_type_current.data > 0, projected_calories_array, np.nan)
                # projected_calories_current_mask_uri = os.path.join(kw['crop_types_projections_geotiffs_dir'], crop_type + '_projected_calories_per_ha_current_mask.tif')
                # projected_calories_current_mask = nd.ArrayFrame(projected_calories_current_mask_array, crop_type_current, output_uri=projected_calories_current_mask_uri)
                # projected_calories_current_mask.show(keep_output=True)

                # file_path = os.path.join(kw['results_for_each_rcp_ssp_pair_dir'], filename)
                # projected_change_subset_df = pd.read_csv(file_path, index_col=0)
                #
                #
                #
                # projected_change_uri = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], crop_type + '_delta_calories_per_ha.tif')
                # af = ge.convert_df_to_af_via_index(projected_change_subset_df, '0', calories_per_cell_af, projected_change_uri)
                # ge.show_array(af.data)





                # projected_change_full_df = pd.DataFrame(np.zeros(len(calories_per_cell_df.index)), index=calories_per_cell_df.index)
                # projected_change_full_df[0][projected_change_subset_df.index] = projected_change_subset_df.ix[:,0]
                # print(33, len(projected_change_full_df.values))
                # projected_change_array = projected_change_full_df.values.reshape(calories_per_cell_af.shape)

                # print('projected_change_array', projected_change_array)
                #
                #
                # projected_change = nd.ArrayFrame(projected_change_array, crop_type_current_af, output_uri=projected_change_uri)
                # projected_change.show(keep_output=True)
                # #



                # print(filename)
                # file_path = os.path.join(kw['results_for_each_rcp_ssp_pair_dir'], filename)
                #
                #
                #
                # print('len(calories_per_cell_df.index)', len(calories_per_cell_df.index))
                # print(33, calories_per_cell_df, calories_per_cell_df.index)
                # print(44, nan_mask_df, nan_mask_df.index)
                # nan_mask_df = nan_mask_df.dropna()
                # print(55, nan_mask_df, nan_mask_df.index)
                #
                # projected_change_full_df = pd.DataFrame(np.zeros(len(calories_per_cell_df.index)), index=calories_per_cell_df.index)
                #
                # projected_change_subset_df = pd.read_csv(file_path, index_col=0)
                #
                # projected_change_uri = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], crop_type + '_delta_calories_per_ha.tif')
                #
                # print('projected_change_subset_df', projected_change_subset_df)
                # af = ge.convert_df_to_af_via_index(projected_change_subset_df, '0', calories_per_cell_af, projected_change_uri)
                #
                # projected_change_array = af.data
                #
                # print(1, 'projected_change_array', projected_change_array.shape)
                #
                # # projected_change_full_df[0][projected_change_subset_df.index] = projected_change_subset_df
                # projected_change_full_df[0][projected_change_subset_df.index] = projected_change_subset_df.ix[:,0]
                # projected_change_array = projected_change_full_df.values.reshape(calories_per_cell_af.shape)
                #
                # print(2, 'projected_change_array', projected_change_array.shape)
                #
                # # projected_change = nd.ArrayFrame(projected_change_array, crop_type_current, output_uri=projected_change_uri)
                # # projected_change.show(keep_output=True)
                #
                #
                # projected_change_current_mask_array = np.where(crop_type_current.data > 0, projected_change_array, np.nan)
                #
                # print(3, 'projected_change_current_mask_array', projected_change_current_mask_array.shape)
                #
                #
                # projected_change_current_mask_uri = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], crop_type + '_delta_calories_per_ha_current_mask.tif')
                # projected_change_current_mask = nd.ArrayFrame(projected_change_current_mask_array, crop_type_current, output_uri=projected_change_current_mask_uri)
                # projected_change_current_mask.show(keep_output=True)
                #
                # projected_calories_array = crop_type_current.data + projected_change_array
                #
                # print(4, 'projected_calories_array', projected_calories_array.shape)
                # output_uri = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], crop_type + '_projected_calories_per_ha.tif')
                # projected_calories = nd.ArrayFrame(projected_calories_array, calories_per_cell_af, output_uri=output_uri)
                # projected_calories.show(keep_output=True)
                #
                # projected_calories_current_mask_array = np.where(crop_type_current.data > 0, projected_calories_array, np.nan)
                # projected_calories_current_mask_uri = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], crop_type + '_projected_calories_per_ha_current_mask.tif')
                # projected_calories_current_mask = nd.ArrayFrame(projected_calories_current_mask_array, crop_type_current, output_uri=projected_calories_current_mask_uri)
                # projected_calories_current_mask.show(keep_output=True)
                # print(5, 'projected_calories_current_mask_array', projected_calories_current_mask_array.shape)

    return kw






main = 'here'
if __name__ == '__main__':
    kw = get_default_kw()

    ## Set runtime_conditionals
    kw['copy_base_data'] = 1 # This one ALWAYS needs to be run because it sets some of the default uris but it doesn't slow down because it checks for existence first
    kw['sum_earthstat'] = 0
    kw['resample_from_30s'] = 0 # Resample fails because calories.tif not projected. Used manual fix.
    kw['process_gaez_inputs'] = 0
    kw['create_spatial_lags'] = 0 # Skipping for now.
    kw['create_baseline_regression_data'] = 0
    kw['clean_baseline_regression_data'] = 0
    kw['create_nan_mask'] = 0
    kw['aggregate_crops_by_type'] = 0
    kw['convert_aggregated_crop_type_dfs_to_geotiffs'] = 0
    kw['calc_optimal_regression_equations_among_linear_cubed'] = 0
    kw['calc_simple_regression'] = 0
    kw['calc_crop_types_regression'] = 0
    kw['combine_crop_types_regressions_into_single_file'] = 0
    kw['create_climate_scenarios_df'] = 0
    kw['project_crop_specific_calories_per_cell_based_on_climate'] = 0
    kw['project_crop_types_calories_per_cell_based_on_climate'] = 0
    kw['write_crop_specific_projections_from_reg_results'] = 0
    kw['write_crop_types_projections_from_reg_results'] = 0
    kw['combine_regressions_into_single_table'] = 0
    kw['create_results_for_each_rcp_ssp_pair'] = 1
    kw['create_maps_for_each_rcp_ssp_pair'] = 1

    kw['sample_fraction'] = 1.0

    # Select subsample of crops if desired
    # kw['crop_names'] = ['wheat']

    # Hacky way to set match af
    kw['calories_per_cell_uri'] = os.path.join(kw['basis_dir'], 'base_data_copy', 'calories_per_cell.tif')

    kw = execute(**kw)

    L.info('Script complete.')
