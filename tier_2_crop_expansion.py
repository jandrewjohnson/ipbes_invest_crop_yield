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

import geoecon as ge
import numdal as nd



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

    ### Generic project-specific dirs from kwargs.
    kw['input_dir'] = kw.get('input_dir', os.path.join(kw['project_dir'], 'input'))  # New inputs specific to this project.
    kw['project_base_data_dir'] = kw.get('project_base_data_dir', os.path.join(kw['project_dir'], 'base_data'))  # Data that must be redistributed with this project for it to work. Do not put actual base data here that might be used across many projects.

    # kw['intermediate_dir'] =  kw.get('input_dir', os.path.join(kw['project_dir'], kw['temporary_dir']))  # If generating lots of data, set this to temporary_dir so that you don't put huge data into the cloud.
    kw['intermediate_dir'] = kw.get('intermediate_dir', os.path.join(kw['project_dir'], 'intermediate'))  # If generating lots of data, set this to temporary_dir so that you don't put huge data into the cloud.
    kw['output_dir'] = kw.get('output_dir', os.path.join(kw['project_dir'], 'output'))  # the final working run is move form Intermediate to here and any hand-made docs are put here.
    kw['run_string'] = kw.get('run_string', hb.pretty_time())  # unique string with time-stamp. To be used on run_specific identifications.
    kw['run_dir'] = kw.get('run_dir', os.path.join(kw['intermediate_dir'], '0_ipbes_' + kw['run_string']))  # ready to delete dir containing the results of one run.
    kw['basis_name'] = kw.get('basis_name', '')  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.
    kw['basis_dir'] = kw.get('basis_dir', os.path.join(kw['intermediate_dir'], kw['basis_name']))  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.

    kw['gaez_data_dir'] = kw['project_base_data_dir']

    ### Common base data references
    # kw['proportion_cropland_uri'] = os.path.join(kw['project_base_data_dir'], 'crops/earthstat', 'proportion_cropland.tif')
    kw['country_names_uri'] = os.path.join(kw['project_base_data_dir'], 'country_names.csv')
    kw['country_ids_raster_uri'] = os.path.join(kw['project_base_data_dir'], 'country_ids.tif')
    kw['calories_per_cell_uri'] = os.path.join(kw['project_base_data_dir'], 'calories_per_cell.tif')
    kw['precip_uri'] = os.path.join(kw['project_base_data_dir'], 'bio12.bil')
    kw['temperature_uri'] = os.path.join(kw['project_base_data_dir'], 'bio1.bil')
    kw['gdp_2000_uri'] = os.path.join(kw['project_base_data_dir'], 'gdp_2000.tif')
    # kw['price_per_ha_masked_dir'] = os.path.join(kw['project_base_data_dir'], 'crops\\crop_prices_and_production_value_2000\\price_per_ha_masked')
    # kw['crop_calories_dir'] = os.path.join(kw['project_base_data_dir'], 'crops\\crop_calories')
    # kw['ag_value_2000_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'ag_value_2000.tif')
    kw['minutes_to_market_uri'] = os.path.join(kw['project_base_data_dir'], 'minutes_to_market_5m.tif')
    # kw['ag_value_2000_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'ag_value_2000.tif')
    kw['pop_30s_uri'] = os.path.join(kw['project_base_data_dir'], 'population\\ciesin', 'pop_30s.tif')
    # kw['proportion_pasture_uri'] = os.path.join(kw['project_base_data_dir'], 'crops/earthstat', 'proportion_pasture.tif')
    # kw['faostat_pasture_uri'] = os.path.join(kw['project_base_data_dir'], 'socioeconomic\\fao', 'faostat', 'Production_LivestockPrimary_E_All_Data_(Norm).csv')
    # kw['ag_value_2005_spam_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'ag_value_2005_spam.tif')

    kw['ha_per_cell_5m_path'] = os.path.join(kw['project_base_data_dir'], 'misc', 'ha_per_cell_5m.tif')

    # Common base data references GAEZ
    kw['workability_index_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "workability_index.tif")
    kw['toxicity_index_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "toxicity_index.tif")
    kw['rooting_conditions_index_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "rooting_conditions_index.tif")
    kw['rainfed_land_percent_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "rainfed_land_percent.tif") # REMOVE?
    kw['protected_areas_index_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "protected_areas_index.tif")
    kw['oxygen_availability_index_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "oxygen_availability_index.tif")
    kw['nutrient_retention_index_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "nutrient_retention_index.tif")
    kw['nutrient_availability_index_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "nutrient_availability_index.tif")
    kw['irrigated_land_percent_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "irrigated_land_percent.tif")
    kw['excess_salts_index_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "excess_salts_index.tif")
    kw['cultivated_land_percent_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "cultivated_land_percent.tif")
    kw['crop_suitability_uri'] = os.path.join(kw['project_base_data_dir'], 'crops', 'gaez', "crop_suitability.tif")
    # kw['precip_2070_uri'] = os.path.join(kw['project_base_data_dir'], 'climate/worldclim/ar5_projections/5min', "ensemble_mean_85bi7012_Annual_Precipitation.tif")
    # kw['temperature_2070_uri'] = os.path.join(kw['project_base_data_dir'], 'climate/worldclim/ar5_projections/5min', "ensemble_mean_85bi701_Annual_Mean_Temperature.tif")
    kw['slope_uri'] = os.path.join(kw['project_base_data_dir'], "slope.tif")
    kw['altitude_uri'] = os.path.join(kw['project_base_data_dir'], "altitude.tif")

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

    # kw['data_registry'] = OrderedDict() # a name, uri air to indicate any map used int he regression.
    # kw['data_registry']['workability'] = kw['base_data_workability_index_uri']
    # kw['data_registry']['toxicity'] = kw['base_data_toxicity_index_uri']
    # kw['data_registry']['rooting_conditions'] = kw['base_data_rooting_conditions_index_uri']
    # kw['data_registry']['rainfed_land_p'] = kw['base_data_rainfed_land_percent_uri']
    # kw['data_registry']['protected_areas'] = kw['base_data_protected_areas_index_uri']
    # kw['data_registry']['oxygen_availability'] = kw['base_data_oxygen_availability_index_uri']
    # kw['data_registry']['nutrient_retention'] = kw['base_data_nutrient_retention_index_uri']
    # kw['data_registry']['nutrient_availability'] = kw['base_data_nutrient_availability_index_uri']
    # kw['data_registry']['irrigated_land_percent'] = kw['base_data_irrigated_land_percent_uri']
    # kw['data_registry']['excess_salts'] = kw['base_data_excess_salts_index_uri']
    # kw['data_registry']['cultivated_land_percent'] = kw['base_data_cultivated_land_percent_uri']
    # kw['data_registry']['crop_suitability'] = kw['base_data_crop_suitability_uri']
    # kw['data_registry']['temperature'] = kw['base_data_temperature_2070_uri']
    # kw['data_registry']['slope'] = kw['base_data_slope_uri']
    # kw['data_registry']['altitude'] = kw['base_data_altitude_uri']
    # # kw['data_registry']['base_data_country_names_uri'] = kw['base_data_country_names_uri']
    # # kw['data_registry']['base_data_country_ids_raster_uri'] = kw['base_data_country_ids_raster_uri']
    # # kw['data_registry']['base_data_calories_per_cell_uri'] = kw['base_data_calories_per_cell_uri']
    # # kw['data_registry']['proportion_cropland'] = kw['proportion_cropland_uri']
    # # kw['data_registry']['precip'] = kw['base_data_precip_uri']
    # # kw['data_registry']['temperature'] = kw['base_data_temperature_uri']
    # kw['data_registry']['gdp_gecon'] = os.path.join(kw['base_data_dir'], 'socioeconomic\\nordhaus_gecon', 'gdp_per_capita_2000_5m.tif')
    # kw['data_registry']['minutes_to_market'] = kw['base_data_minutes_to_market_uri']

    # NOTE: Here i made a unique lists of vars used in eachregression
    kw['linear_fit_no_endogenous_var_names'] = ['precip', 'temperature', 'minutes_to_market', 'workability', 'toxicity', 'rooting_conditions', 'protected_areas', 'oxygen_availability', 'nutrient_retention', 'nutrient_availability', 'excess_salts', 'crop_suitability', 'gdp_gecon', 'altitude', 'slope']
    kw['full_fit_no_endogenous_var_names'] = ['precip', 'precip_2 ', 'precip_3', 'temperature', 'temperature_2', 'temperature_3', 'minutes_to_market', 'minutes_to_market_2', 'minutes_to_market_3', 'workability', 'toxicity', 'rooting_conditions', 'protected_areas', 'oxygen_availability', 'nutrient_retention', 'nutrient_availability', 'excess_salts ', 'crop_suitability', 'crop_suitability_2', 'crop_suitability_3', 'gdp_gecon', 'gdp_gecon_2', 'gdp_gecon_3', 'altitude', 'altitude_2', 'altitude_3', 'slope', 'slope_2', 'slope_3']

    kw['sample_fraction'] = kw.get('sample_fraction', 0.2)

    return kw



def execute(**kw):
    L.info('Executing script.')
    if not kw:
        kw = get_default_kw()

    hb.create_dirs(kw['run_dir'])

    kw = setup_dirs(**kw)

    if kw['create_baseline_regression_data']:
        kw['baseline_regression_data_uri'] = os.path.join(kw['run_dir'], 'baseline_regression_data.csv')
        kw['nan_mask_uri'] = os.path.join(kw['run_dir'], 'nan_mask.csv')
        kw = create_baseline_regression_data(**kw)
    else:
        kw['baseline_regression_data_uri'] = os.path.join(kw['basis_dir'], 'baseline_regression_data.csv')
        kw['nan_mask_uri'] = os.path.join(kw['basis_dir'], 'nan_mask.csv')

    if kw['create_crop_types_regression_data']:
        kw['crop_types_regression_data_uri'] = os.path.join(kw['run_dir'], 'crop_types_regression_data.csv')
        kw = create_crop_types_regression_data(**kw)
    else:
        kw['crop_types_regression_data_uri'] = os.path.join(kw['basis_dir'], 'crop_types_regression_data.csv')

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

    if kw['create_crop_types_depvars']:
        kw['crop_type_depvars_uri'] = os.path.join(kw['run_dir'], 'crop_type_depvars.csv')
        kw = create_crop_types_depvars(**kw)
    else:
        kw['crop_type_depvars_uri'] = os.path.join(kw['basis_dir'], 'crop_type_depvars.csv')

    # if kw['convert_aggregated_crop_type_dfs_to_geotiffs']:
    #     kw['aggregated_crop_data_dir_2'] = os.path.join(kw['run_dir'], 'aggregated_crop_data')
    #     kw['data_registry']['c3_annual_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['c3_annual_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['c3_annual_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_NitrogenApplication_Rate.tif')
    #     kw['data_registry']['c3_perennial_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['c3_perennial_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['c3_perennial_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_NitrogenApplication_Rate.tif')
    #     kw['data_registry']['c4_annual_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['c4_annual_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['c4_annual_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_NitrogenApplication_Rate.tif')
    #     kw['data_registry']['c4_perennial_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['c4_perennial_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['c4_perennial_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_NitrogenApplication_Rate.tif')
    #     kw['data_registry']['nitrogen_fixer_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['nitrogen_fixer_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['nitrogen_fixer_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_NitrogenApplication_Rate.tif')
    #
    #     hb.create_dirs(kw['aggregated_crop_data_dir_2'])
    #     kw = convert_aggregated_crop_type_dfs_to_geotiffs(**kw)
    # else:
    #     kw['aggregated_crop_data_dir_2'] = os.path.join(kw['basis_dir'], 'aggregated_crop_data')
    #     kw['data_registry']['c3_annual_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['c3_annual_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['c3_annual_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_annual_NitrogenApplication_Rate.tif')
    #     kw['data_registry']['c3_perennial_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['c3_perennial_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['c3_perennial_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c3_perennial_NitrogenApplication_Rate.tif')
    #     kw['data_registry']['c4_annual_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['c4_annual_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['c4_annual_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_annual_NitrogenApplication_Rate.tif')
    #     kw['data_registry']['c4_perennial_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['c4_perennial_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['c4_perennial_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'c4_perennial_NitrogenApplication_Rate.tif')
    #     kw['data_registry']['nitrogen_fixer_PotassiumApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_PotassiumApplication_Rate.tif')
    #     kw['data_registry']['nitrogen_fixer_PhosphorusApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_PhosphorusApplication_Rate.tif')
    #     kw['data_registry']['nitrogen_fixer_NitrogenApplication_Rate'] = os.path.join(kw['aggregated_crop_data_dir_2'], 'nitrogen_fixer_NitrogenApplication_Rate.tif')


    if kw['calc_optimal_regression_equations_among_linear_cubed']:
        kw['optimal_regression_equations_among_linear_cubed_dir'] = os.path.join(kw['run_dir'], 'optimal_regression_equations_among_linear_cubed')
        kw['optimal_regression_equations_among_linear_cubed_results_uri'] = os.path.join(kw['optimal_regression_equations_among_linear_cubed_dir'], 'combined_regression_results.json')
        hb.create_dirs(kw['optimal_regression_equations_among_linear_cubed_dir'])
        kw = calc_optimal_regression_equations_among_linear_cubed(**kw)
    else:
        kw['optimal_regression_equations_among_linear_cubed_dir'] = os.path.join(kw['basis_dir'], 'optimal_regression_equations_among_linear_cubed')
        kw['optimal_regression_equations_among_linear_cubed_results_uri'] = os.path.join(kw['optimal_regression_equations_among_linear_cubed_dir'], 'combined_regression_results.json')


    if kw['do_crop_types_regression']:
        kw['crop_types_regression_dir'] = os.path.join(kw['run_dir'], 'crop_types_regression')
        kw['crop_types_regression_results_uri'] = os.path.join(kw['crop_types_regression_dir'], 'crop_types_regression_results.json')
        hb.create_dirs(kw['crop_types_regression_dir'])
        kw = do_crop_types_regression(**kw)
    else:
        kw['crop_types_regression_dir'] = os.path.join(kw['basis_dir'], 'crop_types_regression')
        kw['crop_types_regression_results_uri'] = os.path.join(kw['crop_types_regression_dir'], 'crop_types_regression_results.json')

    if kw['create_climate_scenarios_df']:
        kw['climate_scenarios_csv_uri'] = os.path.join(kw['run_dir'], 'climate_scenarios.csv')
        kw['climate_scenarios_csv_with_nan_uri'] = os.path.join(kw['run_dir'], 'climate_scenarios_with_nan.csv')
        kw = create_climate_scenarios_df(**kw)
    else:
        kw['climate_scenarios_csv_uri'] = os.path.join(kw['basis_dir'], 'climate_scenarios.csv')
        kw['climate_scenarios_csv_with_nan_uri'] = os.path.join(kw['basis_dir'], 'climate_scenarios_with_nan.csv')


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

    if kw['create_aggregated_results']:
        kw['aggregated_results_dir'] = os.path.join(kw['run_dir'], 'aggregated_results')
        hb.create_dirs(kw['aggregated_results_dir'])
        kw = create_aggregated_results(**kw)
    else:
        kw['aggregated_results_dir'] = os.path.join(kw['basis_dir'], 'aggregated_results')


    if kw['create_percent_changes']:
        kw['percent_changes_dir'] = os.path.join(kw['run_dir'], 'percent_changes')
        hb.create_dirs(kw['percent_changes_dir'])
        kw = create_percent_changes(**kw)
    else:
        kw['percent_changes_dir'] = os.path.join(kw['basis_dir'], 'percent_changes')


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
        local_filename = hb.explode_uri(base_data_uri)['filename'].replace('base_data_', '', 1)
        local_uri = os.path.join(kw['base_data_copy_dir'], local_filename)
        kw[local_keys[i]] = local_uri
        if not os.path.exists(local_uri):
            L.info(local_uri + ' does not exist, so copying it from base data at ' + base_data_uri)

            shutil.copy(base_data_uri, local_uri)
    return kw


def create_baseline_regression_data(**kw):

    kw['precip_uri'] = os.path.join(kw['project_base_data_dir'], 'bio12.bil')

    # For specific files, you can specify both a name and a uri, which can be different from the dict key (name)
    input_uris = OrderedDict()
    input_uris['calories_per_cell'] = kw['calories_per_cell_uri']
    input_uris['precip'] = kw['precip_uri']
    input_uris['temperature'] = kw['temperature_uri']
    input_uris['gdp_2000'] = kw['gdp_2000_uri']
    # input_uris['ag_value_2000'] = kw['ag_value_2000_uri']
    input_uris['minutes_to_market'] = kw['minutes_to_market_uri']
    # input_uris['adjacent_neighbors'] = kw['adjacent_neighbors_uri']
    # input_uris['distance_weighted_5x5_neighbors'] = kw['distance_weighted_5x5_neighbors_uri']
    # input_uris['ag_value_2005_spam'] = kw['ag_value_2005_spam_uri']
    # input_uris['proportion_cropland'] = kw['proportion_cropland_uri']
    input_uris['workability'] = os.path.join(kw['gaez_data_dir'], 'workability_continuous.tif')
    input_uris['toxicity'] = os.path.join(kw['gaez_data_dir'], 'toxicity_continuous.tif')
    input_uris['rooting_conditions'] = os.path.join(kw['gaez_data_dir'], 'rooting_conditions_continuous.tif')
    input_uris['protected_areas'] = os.path.join(kw['gaez_data_dir'], 'protected_areas_continuous.tif')
    input_uris['oxygen_availability'] = os.path.join(kw['gaez_data_dir'], 'oxygen_availability_continuous.tif')
    input_uris['nutrient_retention'] = os.path.join(kw['gaez_data_dir'], 'nutrient_retention_continuous.tif')
    input_uris['nutrient_availability'] = os.path.join(kw['gaez_data_dir'], 'nutrient_availability_continuous.tif')
    input_uris['excess_salts'] = os.path.join(kw['gaez_data_dir'], 'excess_salts_continuous.tif')
    # input_uris['irrigated_land_percent'] = kw['base_data_irrigated_land_percent_uri']
    # input_uris['rainfed_land_percent'] = kw['base_data_rainfed_land_percent_uri']
    # input_uris['cultivated_land_percent'] = kw['base_data_cultivated_land_percent_uri']
    # input_uris['crop_suitability'] = kw['base_data_crop_suitability_uri']
    input_uris['gdp_gecon'] = os.path.join(kw['project_base_data_dir'], 'gdp_per_capita_2000_5m.tif')
    input_uris['slope'] = kw['slope_uri']
    input_uris['altitude'] = kw['altitude_uri']

    # START HERE and re-add these while clarifying when we are using the top 16 crops vs ALL the crops.
    # for crop_name in kw['crop_names']:
    #     input_uris[crop_name + ''] = os.path.join(kw['base_data_price_per_ha_masked_dir'], crop_name + '_production_value_per_ha_gt01_national_price.tif')
    #     input_uris[crop_name + '_calories_per_ha'] = os.path.join(kw['base_data_crop_calories_dir'], crop_name + '_calories_per_ha_masked.tif')
    #     input_uris[crop_name + '_proportion_cultivated'] = os.path.join(kw['project_base_data_dir'], 'crops/earthstat/crop_production', crop_name + '_HarvAreaYield_Geotiff', crop_name + '_HarvestedAreaFraction.tif')

    # for crop_name in kw['crop_names']:
    #     for nutrient in ['Potassium', 'Phosphorus', 'Nitrogen']:
    #         name = crop_name + '_' + nutrient + 'Application_Rate'
    #         input_uris[name] = os.path.join(kw['project_base_data_dir'], 'crops/earthstat/crop_fertilizer/Fertilizer_' + crop_name, crop_name + '_' + nutrient + 'Application_Rate.tif')
    #
    # # Specify Dirs where all tifs will be added.
    # input_dirs = []
    # input_dirs.append(kw['resampled_data_dir'])

    # Iterate through input_uris adding them.  Currently also fixes fertilizer nan issues.
    af_names_list = []
    dfs_list = []
    for name, uri in input_uris.items():
        if 'Fertilizer' in uri or '_calories_per_ha' in uri or '_HarvestedAreaFraction' in uri or 'altitude' in uri or 'slope' in uri:
            af = hb.ArrayFrame(uri)
            # NOTE, originaly i had this as af = af.where() which failed to modify the af before going in.
            modified_array = np.where((af.data < 0) | (af.data > 9999999999999999), 0, af.data)
            temp1_path = hb.temp('.tif', remove_at_exit=True, folder=kw['run_dir'])
            hb.save_array_as_geotiff(modified_array, temp1_path, kw['calories_per_cell_uri'])
            modified_af = hb.ArrayFrame(temp1_path)
            af_names_list.append(name)
            df = hb.convert_af_to_1d_df(modified_af)
            dfs_list.append(df)
        else:
            af = hb.ArrayFrame(uri)
            af_names_list.append(name)
            df = hb.convert_af_to_1d_df(af)
            dfs_list.append(df)

    # for dir in input_dirs:
    #     uris_list = hb.get_list_of_file_uris_recursively(dir, filter_extensions='.tif')
    #     for uri in uris_list:
    #         if 'Fertilizer' in uri or '_calories_per_ha_masked' in uri or '_HarvestedAreaFraction' in uri or 'altitude' in uri or 'slope' in uri:
    #             name = hb.explode_uri(uri)['file_root']
    #             af = hb.ArrayFrame(uri)
    #             # NOTE, originaly i had this as af = af.where() which failed to modify the af before going in.
    #             modified_array = np.where(af.data < 0, 0, af.data)
    #             modified_af = hb.ArrayFrame(modified_array, af)
    #             af_names_list.append(name)
    #             df = hb.convert_af_to_1d_df(modified_af)
    #             dfs_list.append(df)
    #         else:
    #             name = hb.explode_uri(uri)['file_root']
    #             af = hb.ArrayFrame(uri)
    #             af_names_list.append(name)
    #             df = hb.convert_af_to_1d_df(af)
    #             dfs_list.append(df)

    L.info('Concatenating all dataframes.')
    # CAREFUL, here all my data are indeed positive but this could change.
    # REMEMBER, we are just determining what gets written to disk here, not what is regressed.
    # LEARNING POINT: stack_dfs was only for time series space-time-frames and just happend to work because i didn't structure my data in the wrong way..
    # df = ge.stack_dfs(dfs_list, af_names_list)
    df = hb.concatenate_dfs_horizontally(dfs_list, af_names_list)
    df[df < 0] = 0.0


    # Rather than getting rid of all cells without crops, just get rid of those not on land.
    df[df['excess_salts'] == 255.0] = np.nan

    # kw['nan_mask_uri'] = 'nan_mask.csv'
    df_nan = df['excess_salts']
    df_nan.to_csv(kw['nan_mask_uri'])

    df = df.dropna()

    df.to_csv(kw['baseline_regression_data_uri'])

    return kw

def create_crop_types_regression_data(**kw):
    L.info('Running create_crop_types_regression_data. ')
    L.debug('   Open the beasline regression_data which is HUGE and also has data on crop-specific stuff, then write it to a file specific to the IPBES tasks. In the event that I go back to crop-specific analysis, this may need to be switched back to crop specific.')
    df = pd.read_csv(kw['baseline_regression_data_uri'], index_col=['Unnamed: 0'])  # Could speed up here with usecols='excess_salts

    output_df = pd.DataFrame(index=df.index)

    print(output_df)
    for col_name in df.columns:
        if col_name in kw['linear_fit_no_endogenous_var_names']:
            output_df[col_name] = df[col_name]
    output_df.to_csv(kw['crop_types_regression_data_uri'])

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

    # # Create a DF of zeros, ready to hold the summed results for each crop type. Indix given will  be from baseline_regression_data_df so that spatial indices match.
    # crop_types_df = pd.DataFrame(np.zeros(len(baseline_regression_data_df.index)), index=baseline_regression_data_df.index)

    # # Iterate through crop_types
    # for crop_type, crops in crop_membership.items():
    #     L.info('Aggregating ' + str(crop_type) + ' ' + str(crops))
    #     for var_name_to_aggregate in vars_names_to_aggregate:
    #         L.info('  var_name_to_aggregate ' + var_name_to_aggregate)
    #         output_col_name = crop_type + '_' + var_name_to_aggregate
    #         crop_types_df[output_col_name] = np.zeros(len(baseline_regression_data_df.index))
    #         for crop in crops:
    #             input_col_name = crop + '_' + var_name_to_aggregate
    #             if input_col_name in baseline_regression_data_df:
    #                 crop_types_df[output_col_name] += baseline_regression_data_df[input_col_name]
    #
    #         crop_types_df[output_col_name][crop_types_df[output_col_name] > 1e+12] = 0.0

    kw['c3_annual_calories_path'] = os.path.join(kw['project_base_data_dir'], "c3_annual_calories.tif")
    kw['c3_perennial_calories_path'] = os.path.join(kw['project_base_data_dir'], "c3_perennial_calories.tif")
    kw['c4_annual_calories_path'] = os.path.join(kw['project_base_data_dir'], "c4_annual_calories.tif")
    kw['c4_perennial_calories_path'] = os.path.join(kw['project_base_data_dir'], "c4_perennial_calories.tif")
    kw['nitrogen_fixer_calories_path'] = os.path.join(kw['project_base_data_dir'], "nitrogen_fixer_calories.tif")

    crop_types_df = pd.DataFrame(np.zeros(len(baseline_regression_data_df.index)),
                                 index=baseline_regression_data_df.index)
    # for crop in crops:
    var_name_to_aggregate = 'calories'
    for crop_type in kw['crop_types']:
        input_col_name = crop + '_' + var_name_to_aggregate
        output_col_name = crop_type + '_' + var_name_to_aggregate
        crop_types_df[output_col_name] = np.zeros(len(baseline_regression_data_df.index))

        crop_types_df[output_col_name] += baseline_regression_data_df[input_col_name]


        crop_types_df[output_col_name][crop_types_df[output_col_name] > 1e+12] = 0.0

    crop_types_df.to_csv(kw['aggregated_crop_data_csv_uri'])

    return kw


def create_crop_types_depvars(**kw):
    L.info('Running create_crop_types_depvars. ')
    L.debug('   Open aggregated_crop_data.csv and pull out ipbes specific stuff.')
    df = pd.read_csv(kw['aggregated_crop_data_csv_uri'], index_col=['Unnamed: 0'])  # Could speed up here with usecols='excess_salts

    output_df = pd.DataFrame(index=df.index)

    print(output_df)
    for col_name in df.columns:
        if '_calories_per_ha' in col_name:
            output_df[col_name] = df[col_name]
            hb.describe_dataframe(output_df)
    output_df.to_csv(kw['crop_type_depvars_uri'])

    return kw




def convert_aggregated_crop_type_dfs_to_geotiffs(**kw):
    match_af = hb.ArrayFrame(kw['calories_per_cell_uri'])

    df = pd.read_csv(kw['aggregated_crop_data_csv_uri'])
    df.set_index('Unnamed: 0', inplace=True)

    cols_to_plot = [
        'c3_annual_production_value_per_ha',
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
        af = hb.convert_df_to_af_via_index(df, column, match_af, hb.temp('.tif'))
        af = hb.where(af>10000000000000000000, af.no_data_value, af, output_uri=output_uri)

    return kw


def do_crop_types_regression(**kw):
    def gen_r_code_for_crop(crop_name):

        r_string = """library(tidyverse)
            library(MASS)
            library(stats4)
            options("width"=4000) # Because string formatting is used to extract results, dont want a split line.


            ### Create and combine the data
            d_1 <- read_csv(\"""" + kw['crop_types_regression_data_uri'] + """\")            
            d_2 <- read_csv(\"""" + kw['crop_type_depvars_uri'] + """\")
            d = cbind(d_1, d_2)                       

            d$precip_2 <- d$precip ^ 2
            d$precip_3 <- d$precip ^ 3
            d$temperature_2 <- d$temperature ^ 2
            d$temperature_3 <- d$temperature ^ 3
            d$minutes_to_market_2 <- d$minutes_to_market ^ 2
            d$minutes_to_market_3 <- d$minutes_to_market ^ 3
            d$gdp_gecon_2 <- d$gdp_gecon ^ 2
            d$gdp_gecon_3 <- d$gdp_gecon ^ 3
            d$altitude_2 <- d$altitude ^ 2
            d$altitude_3 <- d$altitude ^ 3
            d$slope_2 <- d$slope ^ 2
            d$slope_3 <- d$slope ^ 3
            d$crop_suitability_2 <- d$crop_suitability ^ 2
            d$crop_suitability_3 <- d$crop_suitability ^ 3
   
            ### Make a copy of  the data, set the depvar variables that are 0 to be NA, then drop them.
            d_""" + crop_name + """ = d
            d_""" + crop_name + """$""" + crop_name + """_calories_per_ha[d_""" + crop_name + """$""" + crop_name + """_calories_per_ha==0] <- NA
            d_""" + crop_name + """ = d_""" + crop_name + """[!is.na(d_""" + crop_name + """$""" + crop_name + """_calories_per_ha),]

            ### Linear regression no_endogenous
            """ + crop_name + """_formula_string = """ + crop_name + """_calories_per_ha ~ """ + ' + '.join(kw['linear_fit_no_endogenous_var_names']) + """
            """ + crop_name + """_fit <- lm(""" + crop_name + """_formula_string, data=d_""" + crop_name + """)
            "<^>""" + crop_name + """_linear_fit_no_endogenous<^>"
            summary(""" + crop_name + """_fit)
            "<^>"
            
            
            ### Full regression no_endogenous
            """ + crop_name + """_formula_string = """ + crop_name + """_calories_per_ha ~ """ + ' + '.join(kw['full_fit_no_endogenous_var_names']) + """
            """ + crop_name + """_fit <- lm(""" + crop_name + """_formula_string, data=d_""" + crop_name + """)
            "<^>""" + crop_name + """_full_fit_no_endogenous<^>"
            summary(""" + crop_name + """_fit)
            "<^>"
           
        """

        return r_string

    jobs = []
    current_thread = 0
    num_concurrent = 6
    for name in kw['crop_types']:
        current_thread += 1

        r_string = gen_r_code_for_crop(name)
        output_uri = os.path.join(kw['crop_types_regression_dir'], name + '_regression_results.txt')
        L.info('Starting regression process for ' + output_uri)

        script_save_uri = os.path.join(kw['crop_types_regression_dir'], name + '_regression_code.R')
        p = multiprocessing.Process(target=hb.execute_r_string, args=(r_string, output_uri, script_save_uri, True))
        jobs.append(p)
        p.start()

        if current_thread > num_concurrent or current_thread >= len(kw['crop_types']):
            L.info('Waiting for threads to finish.')
            for j in jobs:
                j.join()
            jobs = []
            current_thread = 0

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

            L.info('Loaded regression results\n' + hb.pp(returned_odict, return_as_string=True))

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

    kw['crop_types_regression_results_uri'] = os.path.join(kw['crop_types_regression_dir'], 'crop_types_regression_results.json')
    kw = combine_crop_types_regressions_into_single_file(**kw)

    return kw



def create_climate_scenarios_df(**kw):
    # TODOO, Because I already have the regression equations, there is no reason to load the tifs as DFs. Just iterate through the regression vars andd multiply the geotiffs. This means defining apriori the var names.
    reg_dir = kw['crop_types_regression_results_uri']
    nan_mask_df = pd.read_csv(kw['nan_mask_uri'])

    input_uris = OrderedDict()
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
    match_af = hb.ArrayFrame(kw['calories_per_cell_uri'])

    aligned_uris = OrderedDict()
    for name, uri in input_uris.items():
        dst_uri = os.path.join(kw['run_dir'], os.path.split(uri)[1])
        hb.align_dataset_to_match(uri, match_uri, dst_uri)

        aligned_uris[name] = dst_uri

    for name, uri in aligned_uris.items():
        af = hb.ArrayFrame(uri)
        af_names_list.append(name)
        df = hb.convert_af_to_1d_df(af)
        dfs_list.append(df)
    df = hb.concatenate_dfs_horizontally(dfs_list, af_names_list)
    df[df < 0] = 0.0

    # Rather than getting rid of all cells without crops, just get rid of those not on land.
    # df[nan_mask_df.ix[:, 0] == np.nan] = np.nan
    df[nan_mask_df == np.nan] = np.nan
    df.to_csv(kw['climate_scenarios_csv_with_nan_uri'])

    df = df.dropna()
    df.to_csv(kw['climate_scenarios_csv_uri'])
    df = None

    return kw


def create_results_for_each_rcp_ssp_pair(**kw):

    scenarios = ['sspcur', 'ssp1', 'ssp3', 'ssp5']
    rcps = ['rcpcur', 'rcp26', 'rcp60', 'rcp85'] # NOTE, alll Hadgem2-AO

    # scenario_permutations = ['sspcur_rcpcur']
    scenario_permutations = ['sspcur_rcpcur', 'sspcur_rcp26', 'sspcur_rcp60', 'sspcur_rcp85', 'ssp1_rcpcur', 'ssp3_rcpcur', 'ssp5_rcpcur', 'ssp1_rcp26', 'ssp3_rcp60', 'ssp5_rcp85',]
    # scenario_permutations = ['sspcur_rcpcur', 'sspcur_rcp26']

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
                                'sspcur_rcpcur': 'cur01.bil',
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
                                'sspcur_rcpcur': 'cur012.bil',
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

    # def get_ssp_scenario_string_from_label(scenario_label):
    #     if 'sspcur' in scenario_label:
    #         return 'IMAGE-ssp126/2015'
    #     elif 'ssp1' in scenario_label:
    #         return 'IMAGE-ssp126/2070'
    #     elif 'ssp3' in scenario_label:
    #         return 'AIM-ssp370/2070'
    #     elif 'ssp5' in scenario_label:
    #         return 'MAGPIE-ssp585/2070'


    # Load all necessary  data
    regression_results_odict = hb.file_to_python_object(kw['crop_types_regression_results_uri'])

    # kw['crop_types_regression_data_uri']
    # kw['crop_type_depvars_uri']

    L.info('Loading crop_types_regression_data_uri')
    data_df = pd.read_csv(kw['crop_types_regression_data_uri'], index_col=0)

    # NOTE, baseline regression data reported degrees wihle rcps report .1 degrees.
    data_df['temperature'] = data_df['temperature'] * 10.0
    colname = 'temperature'
    print(colname, np.nanmean(data_df[colname]), np.max(data_df[colname]))

    # Create a df with anything that has changed
    L.info('Loading climate_scenarios_csv_uri')
    modified_df = pd.read_csv(kw['climate_scenarios_csv_with_nan_uri'], index_col=0)
    # hb.describe_dataframe(modified_df)

    # modified_df[modified_df['precip']==0] = np.nan
    # modified_df = modified_df.dropna()

    for scenario in scenarios:
        for rcp in rcps:
            scenario_label = scenario + '_' + rcp
            print ('scenario_label', scenario_label)
            if scenario_label in scenario_permutations:
                # ssp_scenario_string = get_ssp_scenario_string_from_label(scenario_label)
                for i, crop_type in enumerate(kw['crop_types']):
                # for i, crop_type in enumerate(kw['crop_types'][0:1]):

                    regression_name = crop_type + '_full_fit_no_endogenous'
                    regression_name = crop_type + '_linear_fit_no_endogenous'
                    print('     crop_type', crop_type)

                    # STEP - Calculate yield map given climate params
                    change_df = pd.DataFrame(np.zeros(len(data_df.index)), index=data_df.index)

                    current_regression_coefficients = regression_results_odict[crop_type][regression_name]
                    current_regression_coefficient_names = list(current_regression_coefficients.keys())
                    current_regression_coefficient_values = list(current_regression_coefficients.values())


                    for modified_var_name in list(modified_df.columns.values):
                        in_regression = False
                        if modified_var_name == os.path.splitext(temperature_names_by_scenario[scenario_label])[0] or modified_var_name == os.path.splitext(precip_names_by_scenario[scenario_label])[0]:
                            in_regression = True
                            if modified_var_name.endswith('1'):
                                coefficients_name = 'temperature'
                                data_df_name = 'cur01'
                            elif modified_var_name.endswith('12'):
                                coefficients_name = 'precip'
                                data_df_name = 'cur012'
                        else:
                            data_df_name = modified_var_name



                        if modified_var_name in current_regression_coefficient_names or in_regression:
                            if modified_var_name.endswith('_2'):
                                modified_var_name = modified_var_name.replace('_2', '')
                                change_df[0] += (modified_df[modified_var_name] ** 2 - modified_df[data_df_name] ** 2) * float(current_regression_coefficients[coefficients_name])
                            elif modified_var_name.endswith('_3'):
                                modified_var_name = modified_var_name.replace('_3', '')
                                change_df[0] += (modified_df[modified_var_name] ** 3 - modified_df[data_df_name] ** 3) * float(current_regression_coefficients[coefficients_name])

                            else:
                                to_add = (modified_df[modified_var_name] - modified_df[data_df_name]) * float(current_regression_coefficients[coefficients_name])
                                change_df[0] += to_add

                    projection_df_path = os.path.join(kw['results_for_each_rcp_ssp_pair_dir'], scenario_label + '_' + crop_type + '.csv')
                    change_df.to_csv(projection_df_path)

    return kw


def create_maps_for_each_rcp_ssp_pair(**kw):
    # year_2050_area_fraction = OrderedDict()
    # year_2050_area_fraction['c4_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2050\c4per ^ area_fraction ^ C4 perennial crops.tif"
    # year_2050_area_fraction['c3_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2050\c3ann ^ area_fraction ^ C3 annual crops.tif"
    # year_2050_area_fraction['nitrogen_fixer'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2050\c3nfx ^ area_fraction ^ C3 nitrogen-fixing crops.tif"
    # year_2050_area_fraction['c3_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2050\c3per ^ area_fraction ^ C3 perennial crops.tif"
    # year_2050_area_fraction['c4_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2050\c4ann ^ area_fraction ^ C4 annual crops.tif"

    year_2050_area_fraction = OrderedDict()
    year_2050_area_fraction['ssp1'] = OrderedDict()
    year_2050_area_fraction['ssp1']['c4_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\IMAGE-ssp126\2050\c4per ^ area_fraction ^ C4 perennial crops.tif"
    year_2050_area_fraction['ssp1']['c3_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\IMAGE-ssp126\2050\c3ann ^ area_fraction ^ C3 annual crops.tif"
    year_2050_area_fraction['ssp1']['nitrogen_fixer'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\IMAGE-ssp126\2050\c3nfx ^ area_fraction ^ C3 nitrogen-fixing crops.tif"
    year_2050_area_fraction['ssp1']['c3_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\IMAGE-ssp126\2050\c3per ^ area_fraction ^ C3 perennial crops.tif"
    year_2050_area_fraction['ssp1']['c4_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\IMAGE-ssp126\2050\c4ann ^ area_fraction ^ C4 annual crops.tif"

    year_2050_area_fraction['ssp3'] = OrderedDict()
    year_2050_area_fraction['ssp3']['c4_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\AIM-ssp370\2050\c4per ^ area_fraction ^ C4 perennial crops.tif"
    year_2050_area_fraction['ssp3']['c3_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\AIM-ssp370\2050\c3ann ^ area_fraction ^ C3 annual crops.tif"
    year_2050_area_fraction['ssp3']['nitrogen_fixer'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\AIM-ssp370\2050\c3nfx ^ area_fraction ^ C3 nitrogen-fixing crops.tif"
    year_2050_area_fraction['ssp3']['c3_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\AIM-ssp370\2050\c3per ^ area_fraction ^ C3 perennial crops.tif"
    year_2050_area_fraction['ssp3']['c4_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\AIM-ssp370\2050\c4ann ^ area_fraction ^ C4 annual crops.tif"

    year_2050_area_fraction['ssp5'] = OrderedDict()
    year_2050_area_fraction['ssp5']['c4_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\MAGPIE-ssp585\2050\c4per ^ area_fraction ^ C4 perennial crops.tif"
    year_2050_area_fraction['ssp5']['c3_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\MAGPIE-ssp585\2050\c3ann ^ area_fraction ^ C3 annual crops.tif"
    year_2050_area_fraction['ssp5']['nitrogen_fixer'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\MAGPIE-ssp585\2050\c3nfx ^ area_fraction ^ C3 nitrogen-fixing crops.tif"
    year_2050_area_fraction['ssp5']['c3_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\MAGPIE-ssp585\2050\c3per ^ area_fraction ^ C3 perennial crops.tif"
    year_2050_area_fraction['ssp5']['c4_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\extract_lulc\MAGPIE-ssp585\2050\c4ann ^ area_fraction ^ C4 annual crops.tif"



    current_area_fraction = OrderedDict()
    current_area_fraction['c4_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2015\c4per ^ area_fraction ^ C4 perennial crops.tif"
    current_area_fraction['c3_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2015\c3ann ^ area_fraction ^ C3 annual crops.tif"
    current_area_fraction['nitrogen_fixer'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2015\c3nfx ^ area_fraction ^ C3 nitrogen-fixing crops.tif"
    current_area_fraction['c3_perennial'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2015\c3per ^ area_fraction ^ C3 perennial crops.tif"
    current_area_fraction['c4_annual'] = r"C:\OneDrive\Projects\ipbes\intermediate\states_2015\c4ann ^ area_fraction ^ C4 annual crops.tif"

    calories_per_cell_af = hb.ArrayFrame(kw['calories_per_cell_uri'])
    calories_per_cell_df = hb.convert_af_to_1d_df(calories_per_cell_af)
    # calories_per_cell_af.show(keep_output=True)

    L.info('Loading nan_mask_uri')
    nan_mask_df = pd.read_csv(kw['nan_mask_uri'], index_col=0)

    for crop_type in kw['crop_types']:
        L.info('Writing projections for ' + crop_type + ' to tif.')

        earthstat_crop_type_current_uri = os.path.join(kw['aggregated_crop_data_dir_2'], crop_type + '_calories_per_ha.tif')
        earthstat_crop_type_current_af = hb.ArrayFrame(earthstat_crop_type_current_uri)

        current_area_fraction_input_path =  current_area_fraction[crop_type]
        current_area_fraction_resampled_path = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], crop_type + '_current_area_fraction_resampled.tif')

        tmp1 = hb.temp('.tif', remove_at_exit=True)

        print('current_area_fraction_input_path', current_area_fraction_input_path)

        hb.align_dataset_to_match(current_area_fraction_input_path, kw['calories_per_cell_uri'], tmp1, resample_method='near')


        a = hb.as_array(tmp1).astype(np.float64)
        b = np.where(np.isfinite(a), a, 0.0)
        c = np.where((b > 0.0) & (b <= 1.0), b, 0)
        hb.save_array_as_geotiff(c, current_area_fraction_resampled_path, kw['calories_per_cell_uri'])

        current_area_fraction_resampled = hb.as_array(current_area_fraction_resampled_path)
        current_area_fraction_resampled = np.where((current_area_fraction_resampled > 0.001) & (current_area_fraction_resampled < 1.01), current_area_fraction_resampled, 0)
        current_crop_type_projected_extent = np.where((current_area_fraction_resampled > 0.001) & (current_area_fraction_resampled < 1.01), 1, 0)

        ha_per_cell_5m_path = kw['ha_per_cell_5m_path']
        ha_per_cell_5m = hb.as_array(ha_per_cell_5m_path)

        for filename in os.listdir(kw['results_for_each_rcp_ssp_pair_dir']):
            if crop_type in filename:

                ssp_string = filename[0:4]
                print('Processing crop_type', crop_type, filename)

                earthstat_crop_type_current_uri = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], 'current_' + crop_type + '_calories_per_ha.tif')
                earthstat_crop_type_current = hb.ArrayFrame(earthstat_crop_type_current_uri)
                file_path = os.path.join(kw['results_for_each_rcp_ssp_pair_dir'], filename)
                projected_change_full_df = pd.DataFrame(np.zeros(len(calories_per_cell_df.index)), index=calories_per_cell_df.index)
                projected_change_subset_df = pd.read_csv(file_path, index_col=0)

                projected_change_full_df[0][projected_change_subset_df.index] = projected_change_subset_df.ix[:, 0]
                projected_change_array = projected_change_full_df.values.reshape(calories_per_cell_af.shape)

                projected_change_path = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], filename.replace('.csv', '.tif'))
                hb.save_array_as_geotiff(projected_change_array, projected_change_path, kw['calories_per_cell_uri'], data_type_override=7)

                projected_change_2015_extent_array = np.where(current_area_fraction_resampled > 0, projected_change_array, np.nan)
                projected_change_2015_extent_uri =  projected_change_path.replace('.tif', '_2015_extent.tif')
                projected_change_2015_extent = hb.ArrayFrame(projected_change_2015_extent_array, earthstat_crop_type_current, output_uri=projected_change_2015_extent_uri)

                if 'sspcur' in filename:
                    area_fraction = current_area_fraction_resampled
                else:

                    year_2050_area_fraction_input_path = year_2050_area_fraction[ssp_string][crop_type]
                    print('year_2050_area_fraction_input_path', year_2050_area_fraction_input_path)
                    year_2050_area_fraction_resampled_path = os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], ssp_string + '_' + crop_type + '_year_2050_area_fraction_resampled.tif')
                    tmp2 = hb.temp('.tif', remove_at_exit=True)

                    hb.align_dataset_to_match(year_2050_area_fraction_input_path, kw['calories_per_cell_uri'], tmp2, resample_method='near', force_to_match=True)

                    a = hb.as_array(tmp2).astype(np.float64)
                    b = np.where(np.isfinite(a), a, 0.0)
                    c = np.where((b > 0.0) & (b <= 1.0), b, 0)
                    hb.save_array_as_geotiff(c, year_2050_area_fraction_resampled_path, kw['calories_per_cell_uri'])

                    year_2050_area_fraction_resampled = hb.as_array(year_2050_area_fraction_resampled_path)
                    year_2050_area_fraction_resampled = np.where((year_2050_area_fraction_resampled > 0.001) & (year_2050_area_fraction_resampled < 1.01), year_2050_area_fraction_resampled, 0)
                    year_2050_crop_type_projected_extent = np.where((year_2050_area_fraction_resampled > 0.001) & (year_2050_area_fraction_resampled < 1.01), 1, 0)

                    area_fraction = year_2050_area_fraction_resampled

                    projected_change_2050_extent_array = np.where(year_2050_crop_type_projected_extent > 0, projected_change_array, np.nan)
                    projected_change_2050_extent_uri =  projected_change_path.replace('.tif', '_2050_extent.tif')
                    projected_change_2050_extent = hb.ArrayFrame(projected_change_2050_extent_array, earthstat_crop_type_current, output_uri=projected_change_2050_extent_uri)

                calories_per_cell = area_fraction * ha_per_cell_5m * projected_change_array
                calories_per_cell_uri =  os.path.join(kw['maps_for_each_rcp_ssp_pair_dir'], filename.replace('.csv', '_change_per_cell.tif'))
                calories_per_cell_af = hb.ArrayFrame(calories_per_cell, earthstat_crop_type_current, output_uri=calories_per_cell_uri)
    return kw

def create_aggregated_results(**kw):
    # Though, as desired, this would have zero change in climate, and thus zero

    # scenario_permutations = ['sspcur_rcpcur']
    scenario_permutations = ['sspcur_rcpcur', 'sspcur_rcp26', 'sspcur_rcp60', 'sspcur_rcp85', 'ssp1_rcpcur', 'ssp3_rcpcur', 'ssp5_rcpcur', 'ssp1_rcp26', 'ssp3_rcp60', 'ssp5_rcp85']
    # scenario_permutations = ['sspcur_rcpcur', 'sspcur_rcp26']

    match_path = kw['ha_per_cell_5m_path']
    match_af = hb.ArrayFrame(match_path)
    map_filenames = hb.list_filtered_paths_recursively(kw['maps_for_each_rcp_ssp_pair_dir'], include_extensions='.tif')


    for scenario in scenario_permutations:
        output_path = os.path.join(kw['aggregated_results_dir'], scenario + '_caloric_production.tif')
        output_array = np.zeros(match_af.shape).astype(np.float64)
        print('scenario', scenario)
        for crop_type in kw['crop_types']:
            filter = scenario + '_' + crop_type + '_change_per_cell'
            filenames = hb.list_filtered_paths_recursively(kw['maps_for_each_rcp_ssp_pair_dir'], include_extensions='.tif', include_strings=filter)

            if len(filenames) > 1:
                raise NameError

            a = hb.as_array(filenames[0])
            output_array += a
        print(np.sum(output_array))
        hb.save_array_as_geotiff(output_array, output_path, match_path)

        overlay_shp_uri = os.path.join(kw['project_base_data_dir'], 'misc', 'countries')
    return kw




def create_percent_changes(**kw):

    baseline_calories_path = kw['calories_per_cell_uri']
    baseline_calories = hb.as_array(baseline_calories_path).astype(np.float64)

    print('sum baseline_calories', np.sum(baseline_calories))

    output_baseline_calories_path = os.path.join(kw['percent_changes_dir'], 'baseline_calories.png')

    overlay_shp_uri = os.path.join(kw['project_base_data_dir'], 'misc', 'countries')

    # Because we will be dividing the new by the old, we put a 1 for all cells with zero production to avoid div by zero errors. Note that this means the percent change will have limited relevance past some %
    baseline_calories_zerofix = np.where(baseline_calories==0, 1, baseline_calories).astype(np.float64)

    for path in hb.list_filtered_paths_recursively(kw['aggregated_results_dir'], include_extensions='.tif'):
        output_path = os.path.join(kw['percent_changes_dir'], hb.explode_path(path)['file_root'] + '.png').replace('_caloric_production', '_total_caloric_production')
        calorie_change = hb.as_array(path).astype(np.float64)
        # TODOO START HERE, Switching to the linear regression results helped, but now I need to consider scaling to have them all meet future demand so as to match IAMs.
        # This doesn't make sense and isntead I should somehow have thenegative changes BOTTOM OUT.
        # total = (calorie_change/1000) + baseline_calories
        total = calorie_change + baseline_calories
        total = np.where(total<0, 0, total)
        print('sum', path, np.sum(total))

        percent_change = total / baseline_calories_zerofix
        output_path = os.path.join(kw['percent_changes_dir'], hb.explode_path(path)['file_root'] + '.png').replace('_caloric_production', '_percent_change')
    return kw




main = 'here'
if __name__ == '__main__':
    kw = get_default_kw()

    ## Set runtime_conditionals
    kw['create_baseline_regression_data'] = 0
    kw['create_crop_types_regression_data'] = 0
    kw['create_crop_types_depvars'] = 0
    kw['create_nan_mask'] = 0
    kw['aggregate_crops_by_type'] = 1
    kw['convert_aggregated_crop_type_dfs_to_geotiffs'] = 0
    kw['calc_optimal_regression_equations_among_linear_cubed'] = 0
    kw['do_crop_types_regression'] = 0
    kw['combine_crop_types_regressions_into_single_file'] = 0
    kw['create_climate_scenarios_df'] = 0
    kw['combine_regressions_into_single_table'] = 0
    kw['create_results_for_each_rcp_ssp_pair'] = 1
    kw['create_maps_for_each_rcp_ssp_pair'] = 0
    kw['create_aggregated_results'] = 0
    kw['create_percent_changes'] = 0



    kw['sample_fraction'] = 1.0

    # Select subsample of crops if desired
    # kw['crop_names'] = ['wheat']

    # Hacky way to set match af
    kw['calories_per_cell_uri'] = os.path.join(kw['basis_dir'], 'base_data_copy', 'calories_per_cell.tif')

    kw = execute(**kw)

    L.info('Script complete.')
