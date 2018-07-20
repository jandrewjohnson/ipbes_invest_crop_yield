import geoecon as ge
import numdal as nd
import hazelbean as hb



# coding=utf-8

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

CONFIG = nd.config
L = nd.get_logger()
L.setLevel(logging.INFO)


# This required function is called outside of this code's scope to create a kwargs dicitonary. It will be passed to each
# step in the projects logic, potentially modified, then passed to the next.


kw = OrderedDict()

### These should be the only lines that need editing for a new project.
kw['project_name'] = kw.get('project_name', 'ipbes')  # Name of the project being run. A project is a specific implementation of the repository's code to some input data relative to the workspace_dir.
kw['project_dir'] = kw.get('project_dir', os.path.join('c:/onedrive/projects', 'ipbes'))  # This is the ONLY absolute path and it is specific to the researcher and the researcher's current project.
kw['repository_dir'] = 'ipbes_0.1'  # This is the only dir that will be under Version Control. Don't put code anywhere else.

### Generic non-project-specific dir links.
kw['base_data_dir'] = kw.get('base_data_dir', CONFIG.BASE_DATA_DIR)
kw['bulk_data_dir'] = kw.get('bulk_data_dir', CONFIG.BULK_DATA_DIR)
kw['external_bulk_data_dir'] = kw.get('external_bulk_data_dir', CONFIG.EXTERNAL_BULK_DATA_DIR)

### Generic project-specific dirs from kwargs.
kw['input_dir'] = kw.get('input_dir', os.path.join(kw['project_dir'], 'input'))  # New inputs specific to this project.
kw['project_base_data_dir'] = kw.get('project_base_data_dir', os.path.join(kw['project_dir'], 'base_data'))  # Data that must be redistributed with this project for it to work. Do not put actual base data here that might be used across many projects.
kw['temporary_dir'] = kw.get('temporary_dir', CONFIG.TEMPORARY_DIR)  # Generates new run_dirs here. Useful also to set the numdal temporary_dir to here for the run.
kw['test_dir'] = kw.get('temporary_dir', CONFIG.TEST_DATA_DIR)  # Generates new run_dirs here. Useful also to set the numdal temporary_dir to here for the run.
# kw['intermediate_dir'] =  kw.get('input_dir', os.path.join(kw['project_dir'], kw['temporary_dir']))  # If generating lots of data, set this to temporary_dir so that you don't put huge data into the cloud.
kw['intermediate_dir'] = kw.get('intermediate_dir', os.path.join(kw['project_dir'], 'intermediate'))  # If generating lots of data, set this to temporary_dir so that you don't put huge data into the cloud.
kw['output_dir'] = kw.get('output_dir', os.path.join(kw['project_dir'], 'output'))  # the final working run is move form Intermediate to here and any hand-made docs are put here.
kw['run_string'] = kw.get('run_string', nd.pretty_time())  # unique string with time-stamp. To be used on run_specific identifications.
kw['run_dir'] = kw.get('run_dir', os.path.join(kw['temporary_dir'], '0_seals_' + kw['run_string']))  # ready to delete dir containing the results of one run.
kw['basis_name'] = kw.get('basis_name', '')  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.
kw['basis_dir'] = kw.get('basis_dir', os.path.join(kw['intermediate_dir'], kw['basis_name']))  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.



kw['country_names_uri'] = os.path.join(kw['base_data_dir'], 'misc', 'country_names.csv')
kw['country_ids_raster_uri'] = os.path.join(kw['base_data_dir'], 'misc', 'country_ids.tif')
kw['calories_per_cell_uri'] = os.path.join(kw['base_data_dir'], 'publications/ag_tradeoffs/land_econ', 'calories_per_cell.tif')
kw['precip_uri'] = os.path.join(kw['base_data_dir'], 'worldclim/baseline/5min', 'baseline_bio12_Annual_Precipitation.tif')
kw['temperature_uri'] = os.path.join(kw['base_data_dir'], 'worldclim/baseline/5min', 'baseline_bio1_Annual_Mean_Temperature.tif')
kw['gdp_2000_uri'] = os.path.join(kw['input_dir'], 'gdp_2000.tif')
kw['ag_value_2000_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'ag_value_2000.tif')
kw['minutes_to_market_uri'] = os.path.join(kw['base_data_dir'], 'distance_to_market\\uchida_and_nelson_2009\\access_50k', 'minutes_to_market_5m.tif')
kw['ag_value_2000_uri'] = os.path.join(kw['base_data_dir'], 'crops', 'ag_value_2000.tif')
kw['pop_30s_uri'] = os.path.join(kw['base_data_dir'], 'ciesin', 'pop_30s.tif')
kw['proportion_pasture_uri'] = os.path.join(CONFIG.BASE_DATA_DIR, 'earthstat', 'proportion_pasture.tif')
kw['faostat_pasture_uri'] = os.path.join(CONFIG.BASE_DATA_DIR, 'fao', 'faostat', 'Production_LivestockPrimary_E_All_Data_(Norm).csv')
kw['ag_value_2005_spam_uri'] = os.path.join(CONFIG.BASE_DATA_DIR, 'crops', 'ag_value_2005_spam.tif')

# for i in [kw['calories_per_cell_uri'],
#           kw['precip_uri'],
#           kw['temperature_uri'],
#           kw['gdp_2000_uri'],
#           kw['ag_value_2000_uri'],
#           kw['minutes_to_market_uri'],
#           kw['ag_value_2000_uri'],
#           kw['proportion_pasture_uri'],
#           kw['ag_value_2005_spam_uri'] ]:
#
#     print(i)
#     nd.show(i, output_uri=nd.temp('.png', nd.explode_uri(i)['file_root']), cbar_percentiles=[10, 50, 90])
#
#


kw['plot_intermediate_dir'] = 0
if kw['plot_intermediate_dir']:
    for i in nd.get_list_of_file_uris_recursively(kw['intermediate_dir'], filter_extensions='.tif'):

        output_uri = i.replace('.tif', '.png')
        if not os.path.exists(output_uri) and 'pop_30s.tif' not in i:
            # if 'pop_30s.tif' not in i:
            print(i)
            if 'precipitation' in i or 'temperature' in i:
                use_basemap = False
            else:
                use_basemap = True

            if 'gaez' in i:
                title = nd.explode_uri(i)['file_root'].replace('_continuous', '').replace('_', ' ').title()
            else:
                title = i

            nd.show(i, output_uri=output_uri, title=title, resolution='c',cbar_percentiles=[2, 50, 98], use_basemap=use_basemap)

kw['plot_fertilizer_inputs_dir'] = 0
if kw['plot_fertilizer_inputs_dir']:
    dir = os.path.join(CONFIG.BASE_DATA_DIR, 'crops', 'earthstat', 'crop_fertilizer', 'fertilizer_wheat')
    print(dir)
    for i in nd.get_list_of_file_uris_recursively(dir, filter_extensions='.tif'):


        output_uri = i.replace('.tif', '.png')
        if not os.path.exists(output_uri) and 'pop_30s.tif' not in i:
            # if 'pop_30s.tif' not in i:
            print(i)
            if 'precipitation' in i or 'temperature' in i:
                use_basemap = False
            else:
                use_basemap = True
            nd.show(i, output_uri=output_uri, resolution='c',cbar_percentiles=[2, 50, 98], use_basemap=use_basemap)


kw['plot_gdp_dir'] = 0
if kw['plot_gdp_dir']:
    dir = os.path.join(kw['base_data_dir'], 'socioeconomic\\nordhaus_gecon')
    print(dir)
    for i in nd.get_list_of_file_uris_recursively(dir, filter_extensions='.tif'):
        print(i)

        output_uri = i.replace('.tif', '.png')
        if not os.path.exists(output_uri) and 'pop_30s.tif' not in i:
            # if 'pop_30s.tif' not in i:
            print(i)
            if 'precipitation' in i or 'temperature' in i:
                use_basemap = False
            else:
                use_basemap = True
            nd.show(i, output_uri=output_uri, resolution='c',cbar_percentiles=[2, 50, 98], use_basemap=use_basemap)



kw['plot_maps_dir'] = 1
if kw['plot_maps_dir']:
    kw['create_graphics_from_tifs'] = 1
    maps_dir = os.path.join(kw['intermediate_dir'], 'maps_for_each_rcp_ssp_pair')

    luc_and_cc_paths = [
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcp26_nitrogen_fixer_2050_extent.tif',

        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcp60_nitrogen_fixer_2050_extent.tif',

        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcp85_nitrogen_fixer_2050_extent.tif',

    ]


    luc_no_cc_paths = [
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp1_rcpcur_nitrogen_fixer_2050_extent.tif',

        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp3_rcpcur_nitrogen_fixer_2050_extent.tif',

        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\ssp5_rcpcur_nitrogen_fixer_2050_extent.tif',

    ]

    no_luc_with_cc_paths = [
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp26_nitrogen_fixer_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp60_nitrogen_fixer_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c3_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c3_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c3_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c3_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c3_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c3_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c4_annual.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c4_annual_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c4_annual_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c4_perennial.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c4_perennial_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_c4_perennial_2050_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_nitrogen_fixer.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_nitrogen_fixer_2015_extent.tif',
        r'c:/onedrive/projects\ipbes\intermediate\maps_for_each_rcp_ssp_pair\sspcur_rcp85_nitrogen_fixer_2050_extent.tif',

    ]

    # for file_path in hb.list_filtered_paths_recursively(maps_dir, include_extensions='.tif'):
    #     print('file_path', file_path)

    for i in luc_and_cc_paths:
        a = hb.as_array(i)
        output_uri = os.path.join(kw['output_dir'], os.path.split(i)[1].replace('.tif', '.png'))
        ge.show_array(a, output_uri=output_uri, use_basemap=True, resolution='i', cbar_label='Change in kcal per ha')

    for i in luc_no_cc_paths:
        a = hb.as_array(i)
        output_uri = os.path.join(kw['output_dir'], os.path.split(i)[1].replace('.tif', '.png'))
        ge.show_array(a, output_uri=output_uri, use_basemap=True, resolution='i', cbar_label='Change in kcal per ha')

    for i in no_luc_with_cc_paths:
        a = hb.as_array(i)
        output_uri = os.path.join(kw['output_dir'], os.path.split(i)[1].replace('.tif', '.png'))
        ge.show_array(a, output_uri=output_uri, use_basemap=True, resolution='i', cbar_label='Change in kcal per ha')



















