# coding=utf-8

STATUS =  'ON HOLD, until I hear back from herrero, who may have already done this.'
import math, os, sys, time, random, shutil, logging, csv, json

import numpy as np
from osgeo import gdal, osr, ogr
import pandas as pd
import geopandas as gpd
from collections import OrderedDict
import logging

import numdal as nd
import geoecon as ge
import hazelbean as hb

CONFIG = nd.config
L = CONFIG.LOGGER
L.setLevel(logging.INFO)

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
    kw['repository_dir'] = 'ipbes_0.1' # This is the only dir that will be under Version Control. Don't put code anywhere else.

    ### Generic non-project-specific dir links.
    kw['base_data_dir'] = kw.get('base_data_dir', CONFIG.BASE_DATA_DIR)
    kw['bulk_data_dir'] = kw.get('bulk_data_dir', CONFIG.BULK_DATA_DIR)
    kw['external_bulk_data_dir'] = kw.get('external_bulk_data_dir', CONFIG.EXTERNAL_BULK_DATA_DIR)

    ### Generic project-specific dirs from kwargs.
    kw['input_dir'] = kw.get('input_dir', os.path.join(kw['project_dir'], 'input'))  # New inputs specific to this project.
    kw['project_base_data_dir'] = kw.get('project_base_data_dir', os.path.join(kw['project_dir'], 'base_data'))  # Data that must be redistributed with this project for it to work. Do not put actual base data here that might be used across many projects.
    kw['temporary_dir'] = kw.get('temporary_dir', CONFIG.TEMPORARY_DIR)  # Generates new run_dirs here. Useful also to set the numdal temporary_dir to here for the run.
    kw['test_dir'] = kw.get('temporary_dir', CONFIG.TEST_DATA_DIR)  # Generates new run_dirs here. Useful also to set the numdal temporary_dir to here for the run.
    #kw['intermediate_dir'] =  kw.get('input_dir', os.path.join(kw['project_dir'], kw['temporary_dir']))  # If generating lots of data, set this to temporary_dir so that you don't put huge data into the cloud.
    kw['intermediate_dir'] =  kw.get('intermediate_dir', os.path.join(kw['project_dir'], 'intermediate')) # If generating lots of data, set this to temporary_dir so that you don't put huge data into the cloud.
    kw['output_dir'] = kw.get('output_dir', os.path.join(kw['project_dir'], 'output'))  # the final working run is move form Intermediate to here and any hand-made docs are put here.
    kw['run_string'] = kw.get('run_string', nd.pretty_time())  # unique string with time-stamp. To be used on run_specific identifications.
    kw['run_dir'] = kw.get('run_dir', os.path.join(kw['temporary_dir'], '0_seals_' + kw['run_string']))  # ready to delete dir containing the results of one run.
    kw['basis_name'] = kw.get('basis_name', '')  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.
    kw['basis_dir'] = kw.get('basis_dir', os.path.join(kw['intermediate_dir'], kw['basis_name']))  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.

    ### Common base data references
    kw['country_names_uri'] = os.path.join(kw['base_data_dir'], 'misc', 'country_names.csv')
    kw['country_ids_raster_uri'] = os.path.join(kw['base_data_dir'], 'misc', 'country_ids.tif')

    ### Base-data links
    kw['proportion_pasture_uri'] = os.path.join(CONFIG.BASE_DATA_DIR, 'earthstat', 'proportion_pasture.tif')
    kw['faostat_pasture_uri'] = os.path.join(CONFIG.BASE_DATA_DIR, 'fao', 'faostat', 'Production_LivestockPrimary_E_All_Data_(Norm).csv')

    ### Project-specific data inputs
    # kw['input_csv_uri'] = kw.get('input_csv_uri', os.path.join(kw['input_dir'], 'LinkFile.csv'))


    # Runtime conditionals.
    kw['extract_pasture_data_from_faostat'] = kw.get('extract_pasture_data_from_faostat', True)
    kw['write_pasture_production_to_raster'] = kw.get('write_pasture_production_to_raster', True)
    kw['plot_global_production'] = kw.get('plot_global_production', True)

    return kw


def execute(**kw):
    L.info('Executing script.')
    if not kw:
        kw = get_default_kw()

    kw = setup_dirs(**kw)

    if kw['extract_pasture_data_from_faostat']:
        kw['pasture_csv_uri'] = os.path.join(kw['run_dir'], 'pasture_by_country.csv')
        kw = extract_pasture_data_from_faostat(**kw)
    else:
        kw['pasture_csv_uri'] = os.path.join(kw['basis_dir'], 'pasture_by_country.csv')

    if kw['write_pasture_production_to_raster']:
        kw['production_by_country_uri'] = os.path.join(kw['run_dir'], 'production_by_country.tif')
        kw['production_per_cell_uri'] = os.path.join(kw['run_dir'], 'production_per_cell.tif')
        kw = write_pasture_production_to_raster(**kw)
    else:
        kw['production_by_country_uri'] = os.path.join(kw['basis_dir'], 'production_by_country.tif')
        kw['production_per_cell_uri'] = os.path.join(kw['basis_dir'], 'production_per_cell.tif')

    if kw['plot_global_production']:
        kw = plot_global_production(**kw)
    else:
        pass

    return kw

def setup_dirs(**kw):
    L.debug('Making default dirs.')

    dirs = [kw['project_dir'], kw['input_dir'], kw['intermediate_dir'], kw['run_dir'], kw['output_dir']]
    hb.create_dirs(dirs)

    return kw

def extract_pasture_data_from_faostat(**kw):
    L.info('Running extract_pasture_data_from_faostat')

    country_names_df = pd.read_csv(kw['country_names_uri'], encoding='latin-1', index_col=False)
    pasture_fao_df = pd.read_csv(kw['faostat_pasture_uri'], encoding='latin-1', index_col=False, converters={'Year': lambda x: str(x)})

    df = pd.merge(country_names_df[['FAOSTAT_augmented', 'id']], pasture_fao_df,'inner', left_on='FAOSTAT_augmented', right_on='Country Code', )
    df.drop(['FAOSTAT_augmented', 'Item Code', 'Element Code', 'Flag', 'Year Code', 'Country Code'], axis=1, inplace=True)
    df.set_index(['id', 'Item', 'Year', 'Element', 'Unit'], inplace=True)


    df = df.xs('2000', level='Year')
    df = df.xs('Production', level='Element')
    df = df.xs('tonnes', level='Unit')

    df = ge.explode_df(df)

    df = df.reset_index()
    df.set_index(['id'], inplace=True) # Had to remove previous indices because it fell out in the XS being only = 2000.


    df = df[df['Item'].str.contains("Meat|meat", na=False)]

    df = df.drop('Item', axis=1)


    # START HERE, get this to properly add in country names.
    df_grouped = df.groupby([df.index.get_level_values('id')]).sum()
    # print(df_grouped)
    # print(df)
    # df = df.reset_index()
    # print(df)
    df = pd.merge(df_grouped, df, left_index=True, right_index=True, how='inner')
    # df.set_index(['id'], inplace=True) # Had to remove previous indices because it fell out in the XS being only = 2000.

    df.to_csv(kw['pasture_csv_uri'])

    return kw

def write_pasture_production_to_raster(**kw):
    L.info('Running write_pasture_production_to_raster')

    ids = nd.ArrayFrame(kw['country_ids_raster_uri'])
    ids = ids.set_data_type(7)
    ids = ids.set_no_data_value(-9999.0)
    ids_present = nd.get_value_count_odict_from_array(ids.data)


    df = pd.read_csv(kw['pasture_csv_uri'])
    rules = dict(zip(df['id'], df['Value']))

    # For countries that are not in the database, we don't want to write the id value and instead want zero.
    for k, v in ids_present.items():
        if k not in rules:
            rules[k] = 0

    # Write the values in the rules (Production tons) to the locations of the countrys.
    production_by_country_array = nd.reclassify_int_array_by_dict_to_floats(ids.data.astype(np.int), rules).astype(np.float64)
    production_by_country = nd.ArrayFrame(production_by_country_array, ids, data_type=7, output_uri=kw['production_by_country_uri'])

    proportion_pasture = nd.ArrayFrame(kw['proportion_pasture_uri'])

    global_production = np.zeros(proportion_pasture.shape)
    for country_id, production_total in rules.items():
        L.info(str(country_id) + ', ' + str(production_total))

        if production_total > 0:

            proportion_in_country = np.where(ids.data == country_id, proportion_pasture.data, 0)

            total_proportion_in_country = np.sum(proportion_in_country)

            if total_proportion_in_country > 0:
                L.info('production_total ' + str(production_total))
                L.info('total_proportion_in_country ' + str(total_proportion_in_country))
                production_per_proportion = production_total / total_proportion_in_country
                L.info('production_per_proportion ' + str(production_per_proportion))


                # production = production_per_proportion * proportion_pasture.data
                production = np.where(ids.data == country_id, production_per_proportion * proportion_pasture.data, 0)



                L.info('production ' + str(np.sum(production)))

                global_production += production
                L.info('global_production ' + str(np.sum(global_production)))

    global_production_af = nd.ArrayFrame(global_production, proportion_pasture, output_uri=kw['production_per_cell_uri'])

    return kw

def plot_global_production(**kw):
    af = nd.ArrayFrame(kw['production_per_cell_uri'])
    af.show(vmin=0, vmax=1500, use_basemap=True, title='Meat Production', cbar_label='Tons per cell')


main = 'Here'
if __name__ == '__main__':
    kw = get_default_kw()

    kw['extract_pasture_data_from_faostat'] = 1
    kw['write_pasture_production_to_raster'] = 0
    kw['plot_global_production'] = 0

    kw = execute(**kw)

    L.info('Script complete.')
