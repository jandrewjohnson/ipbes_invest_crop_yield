import math, os, sys, time, random, shutil, logging, csv, json, types

import numpy as np
from osgeo import gdal, osr, ogr
import pandas as pd
import geopandas as gpd
from collections import OrderedDict
import logging
import scipy
import geoecon as ge
import hazelbean as hb
import multiprocessing



L = hb.get_logger()

def extract_lulc(**kw):
    if kw['runtime_conditionals']['extract_lulc']:
        for scenario_name in kw['scenario_names']:
            scenario_dir = os.path.join(dirs['extract_lulc'], scenario_name)
            os.mkdir(scenario_dir)

            filename = 'multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-' + scenario_name + '-2-1-f_gn_2015-2100.nc'
            states_path = os.path.join(kw['scenarios_data_dir'], filename)

            for year in kw['years']:
                year_dir = os.path.join(scenario_dir, str(year))
                os.mkdir(year_dir)
                L.info('Extracting from ' + states_path)

                ge.extract_geotiff_from_netcdf(states_path, year_dir, year - 2015) #0 = 2015, last year is 85=2100

    return kw

def resample_lulc(**kw):
    if kw['runtime_conditionals']['resample_lulc']:
        for scenario in kw['scenario_names']:
            for year  in kw['years']:
                read_dir = os.path.join(dirs['extract_lulc'], scenario, str(year))
                write_dir = os.path.join(dirs['resample_lulc'], scenario, str(year))
                print('read_dir', read_dir)
                hb.create_dirs(write_dir)
                for filename in hb.list_filtered_paths_recursively(read_dir, include_extensions=['.tif']):
                    input_path = os.path.join(read_dir, filename)
                    output_path = os.path.join(write_dir, os.path.basename(filename))
                    L.info('Aligning ' + input_path)
                    hb.align_dataset_to_match(input_path, kw['5min_floats_match_path'], output_path)


    return kw
def aggregate_crops_by_type(**kw):
    """CMIP6 and the land-use harmonization project have centered on 5 crop types: c3 annual, c3 perennial, c4 annual, c4 perennial, nitrogen fixer
    Aggregate the 15 crops to those four categories by modifying the baseline_regression_data."""

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

    match_path = kw['5min_floats_match_path']
    match_array = hb.as_array(match_path)
    # Iterate through crop_types

    if kw['runtime_conditionals']['aggregate_crops_by_type']:

        df = pd.DataFrame(index=range(1, 100), columns=crop_membership.keys())

        for crop_type, crops in crop_membership.items():
            L.info('Aggregating ' + str(crop_type) + ' ' + str(crops))



            crop_type_calories_output_path = os.path.join(dirs['aggregate_crops_by_type'], crop_type + '_calories.tif')
            crop_type_calories_array = np.zeros(match_array.shape)

            current_crop_calories_array = None
            for crop in crops:
                crop_calories_path = os.path.join(hb.BASE_DATA_DIR, 'crops/crop_calories', crop + '_calories_per_ha_masked.tif')
                current_crop_calories_array = hb.as_array(crop_calories_path)
                current_crop_calories_array[np.isnan(current_crop_calories_array)] = 0.0
                current_crop_calories_array[current_crop_calories_array > 1e+14] = 0.0
                current_crop_calories_array[current_crop_calories_array < 0] = 0.0

                current_crop_climate_bins_path = os.path.join(hb.BASE_DATA_DIR, r'crops\invest\extended_climate_bin_maps\extendedclimatebins' + crop +  '.tif')
                current_crop_climate_bins = hb.as_array(current_crop_climate_bins_path)

                for i in range(1, 101):
                    sum_ = np.sum(np.where(current_crop_climate_bins == i, current_crop_calories_array, 0))




                # print(np.sum(current_crop_climate_bins))

                crop_type_calories_array += current_crop_calories_array

            #     print('crop_calories_path', crop_calories_path, np.sum(current_crop_calories_array), np.sum(crop_type_calories_array))
            #
            # print(crop_type, np.sum(crop_type_calories_array))
            hb.save_array_as_geotiff(crop_type_calories_array, crop_type_calories_output_path, match_path)

    return kw


def caloric_production_change(**kw):


    if kw['runtime_conditionals']['caloric_production_change']:
        base_year = 2015
        for scenario in kw['scenario_names']:

            for year  in kw['years']:
                if year != base_year:
                    for c, crop_type in enumerate(kw['crop_types_short']):
                        base_dir = os.path.join(dirs['resample_lulc'], scenario, str(base_year))
                        base_year_path = hb.list_filtered_paths_recursively(base_dir, include_strings=crop_type, include_extensions='.tif')[0]
                        base_year_array = hb.as_array(base_year_path)

                        base_year_array[np.isnan(base_year_array)] = 0.0
                        base_year_array[base_year_array > 1e+14] = 0.0
                        base_year_array[base_year_array < 0] = 0.0



                        input_dir = os.path.join(dirs['resample_lulc'], scenario, str(year))
                        input_path = hb.list_filtered_paths_recursively(input_dir, include_strings=crop_type, include_extensions='.tif')[0]
                        input_array = hb.as_array(input_path)

                        input_array[np.isnan(input_array)] = 0.0
                        input_array[input_array > 1e+14] = 0.0
                        input_array[input_array < 0] = 0.0

                        calories_per_ha_array = hb.as_array(os.path.join(dirs['aggregate_crops_by_type'], kw['crop_types'][c] + '_calories.tif'))
                        calories_per_ha_array[np.isnan(calories_per_ha_array)] = 0.0
                        calories_per_ha_array[calories_per_ha_array > 1e+14] = 0.0
                        calories_per_ha_array[calories_per_ha_array < 0] = 0.0


                        ha_per_cell_array = hb.as_array(os.path.join(hb.BASE_DATA_DIR, 'misc', 'ha_per_cell_5m.tif'))

                        extent_difference_array = base_year_array - input_array
                        baseline_calorie_provision = calories_per_ha_array * ha_per_cell_array * base_year_array
                        calorie_provision_per_cell = calories_per_ha_array * ha_per_cell_array * input_array
                        caloric_change_per_cell = calories_per_ha_array * ha_per_cell_array * extent_difference_array
                        # calorie_provision_percent_change = (calorie_provision_per_cell / baseline_calorie_provision) * 100.0 - 100.0

                        calorie_provision_percent_change = np.divide(calorie_provision_per_cell, baseline_calorie_provision, out=np.zeros_like(calorie_provision_per_cell), where=baseline_calorie_provision != 0)
                        calorie_provision_percent_change = np.multiply(calorie_provision_percent_change, 100.0, out=np.zeros_like(calorie_provision_per_cell), where=baseline_calorie_provision != 0)
                        calorie_provision_percent_change = np.subtract(calorie_provision_percent_change, 100.0, out=np.zeros_like(calorie_provision_per_cell), where=baseline_calorie_provision != 0)

                        hb.create_dirs(os.path.join(dirs['caloric_production_change'], scenario, str(year)))

                        extent_difference_path = os.path.join(dirs['caloric_production_change'], scenario, str(year), crop_type + '_extent_difference.tif')
                        hb.save_array_as_geotiff(extent_difference_array, extent_difference_path, kw['5min_floats_match_path'], no_data_value_override=-9999.0)

                        caloric_change_per_cell_path = os.path.join(dirs['caloric_production_change'], scenario, str(year), crop_type + '_caloric_change_per_cell.tif')
                        hb.save_array_as_geotiff(caloric_change_per_cell, caloric_change_per_cell_path, kw['5min_floats_match_path'], no_data_value_override=-9999.0)

                        caloric_production_per_cell_path = os.path.join(dirs['caloric_production_change'], scenario, str(year), crop_type + '_calories_per_cell.tif')
                        hb.save_array_as_geotiff(calorie_provision_per_cell, caloric_production_per_cell_path, kw['5min_floats_match_path'], no_data_value_override=-9999.0)

                        calorie_provision_percent_change_path = os.path.join(dirs['caloric_production_change'], scenario, str(year), crop_type + '_calories_percent_change.tif')
                        hb.save_array_as_geotiff(calorie_provision_percent_change, calorie_provision_percent_change_path, kw['5min_floats_match_path'], no_data_value_override=-9999.0, data_type_override=6)

                        produce_final = True
                        if produce_final:


                            overlay_shp_uri = os.path.join(hb.BASE_DATA_DIR, 'misc', 'countries')

                            scenario_string = scenario.split('-')[1][0:4].upper() + 'xRCP' + scenario.split('-')[1][4] + '.' + scenario.split('-')[1][5] + '_' + scenario.split('-')[0] + '_global_' + str(year)
                            kw['output_dir'] = kw['output_dir'].replace('\\', '/')
                            output_path = os.path.join(kw['output_dir'], scenario_string + '_' + crop_type + '_kcal_production_per_cell.tif')
                            shutil.copy(caloric_production_per_cell_path, output_path)
                            ge.show_raster_uri(output_path, output_uri=output_path.replace('.tif', '.png'), title=hb.explode_uri(output_path)['file_root'].replace('_', ' ').title(),
                                               cbar_label='Kcal production per grid-cell given 2050 land-use', cbar_percentiles=[2,50,98],
                                               overlay_shp_uri=overlay_shp_uri, use_basemap=True, bounding_box='clip_poles') #cbar_percentiles=[1, 50, 99],

                            output_path = os.path.join(kw['output_dir'], scenario_string + '_' + crop_type + '_percent_change.tif')
                            shutil.copy(calorie_provision_percent_change_path, output_path)
                            ge.show_raster_uri(output_path, output_uri=output_path.replace('.tif', '.png'), title=hb.explode_uri(output_path)['file_root'].replace('_', ' ').title(),
                                               cbar_label='Percent change in kcal production from land-use change', vmin=-50, vmid=0, vmax=50,
                                               overlay_shp_uri=overlay_shp_uri, use_basemap=True, bounding_box='clip_poles') #cbar_percentiles=[1, 50, 99],

    return kw



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
    kw['project_dir'] = kw.get('project_dir', os.path.join(hb.PRIMARY_DRIVE, 'onedrive/projects', 'ipbes'))  # This is the ONLY absolute path and it is specific to the researcher and the researcher's current project.
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
    kw['run_string'] = kw.get('run_string', hb.pretty_time())  # unique string with time-stamp. To be used on run_specific identifications.
    kw['run_dir'] = kw.get('run_dir', os.path.join(kw['temporary_dir'], '_ipbes_' + kw['run_string']))  # ready to delete dir containing the results of one run.
    kw['basis_name'] = kw.get('basis_name', '')  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.
    kw['basis_dir'] = kw.get('basis_dir', os.path.join(kw['intermediate_dir'], kw['basis_name']))  # Specify a manually-created dir that contains a subset of results that you want to use. For any input that is not created fresh this run, it will instead take the equivilent file from here. Default is '' because you may not want any subsetting.

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

    kw['crop_types_short'] = [
        'c3ann',
        'c3per',
        'c4ann',
        'c4per',
        'c3nfx',
    ]

    kw['runtime_conditionals'] = OrderedDict()

    kw['scenarios_data_dir'] = os.path.join(kw['bulk_data_dir'], 'lulc\\luh2')
    kw['scenario_names'] = [
        'IMAGE-ssp126',
        'AIM-ssp360',
        'MAGPIE-ssp585',
    ]
    kw['years'] = [2015, 2050]

    kw['5min_floats_match_path'] = os.path.join(kw['input_dir'], "5min_floats_match.tif")

    return kw
def setup_dirs(**kw):
    global dirs
    L.debug('Making default dirs.')
    dirs_list = [kw['project_dir'], kw['input_dir'], kw['intermediate_dir'], kw['run_dir'], kw['output_dir']]
    print(dirs_list)
    dirs = OrderedDict()
    for conditional in kw['runtime_conditionals']:
        if kw['runtime_conditionals'][conditional]:
            new_dir = os.path.join(kw['run_dir'], conditional)
            dirs[conditional] = new_dir
            dirs_list.append(new_dir)
        else:
            dirs[conditional] = os.path.join(kw['intermediate_dir'], conditional)

    hb.create_dirs(dirs_list)

    return kw
def execute(**kw):
    L.info('Executing script.')

    if not kw:
        kw = get_default_kw()

    hb.create_dirs(kw['run_dir'])

    kw = setup_dirs(**kw)

    runtime_functions = OrderedDict()
    for k, runtime_function in dict(globals()).items():
        if isinstance(runtime_function, types.FunctionType):
            if runtime_function.__name__ in kw['runtime_conditionals']:
                runtime_functions[runtime_function.__name__] = runtime_function

    global conditional_name  # Makes is so that the function being called knows its own name

    for conditional_name, conditional_value in kw['runtime_conditionals'].items():
        if conditional_value:
            L.info('Running ' + conditional_name)
        else:
            L.info('Skipping calculation of ' + conditional_name)

        kw = runtime_functions[conditional_name](**kw)

    return kw
main = 'here'
if __name__ == '__main__':

    kw = get_default_kw()
    kw['runtime_conditionals']['extract_lulc'] = 0
    kw['runtime_conditionals']['resample_lulc'] = 0
    kw['runtime_conditionals']['aggregate_crops_by_type'] = 0
    kw['runtime_conditionals']['caloric_production_change'] = 1

    kw = execute(**kw)

    L.info('Script complete.')
