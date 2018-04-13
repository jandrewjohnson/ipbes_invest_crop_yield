import math, os, sys, time, random, shutil, logging, csv, json

import netCDF4
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

L = hb.get_logger('process_ssp_scenarios')

def extract_lulc(p):
    if p.tasks['extract_lulc']:

        for scenario_name in p.scenario_names:
            scenario_dir = os.path.join(p.task_dirs['extract_lulc'], scenario_name)
            hb.create_dirs(scenario_dir)

            filename = 'multiple-states_input4MIPs_landState_ScenarioMIP_UofMD-' + scenario_name + '-2-1-f_gn_2015-2100.nc'
            states_path = os.path.join(p.scenarios_data_dir, filename)

            for year in p.years:
                year_dir = os.path.join(scenario_dir, str(year))
                os.mkdir(year_dir)
                L.info('Extracting from ' + states_path)

                ge.extract_geotiff_from_netcdf(states_path, year_dir, year - 2015) #0 = 2015, last year is 85=2100

    else:
        pass

def resample_lulc(p):
    if p.tasks['resample_lulc']:
        match_af = hb.ArrayFrame(p.base_data_ha_per_cell_path)
        match_r_path = p.match_r_path
        hb.reproject_to_cylindrical(match_af.uri, match_r_path)
        # hb.reproject_to_epsg(match_af.uri, match_r_path, 54012)
        match_r_af = hb.ArrayFrame(match_r_path)

        for scenario in p.scenario_names:
            for year  in p.years:
                read_dir = os.path.join(p.task_dirs['extract_lulc'], scenario, str(year))
                write_dir = os.path.join(p.resample_lulc_dir, scenario, str(year))
                hb.create_dirs(write_dir)
                for filename in hb.list_files_in_dir_recursively(read_dir, filter_extensions=['.tif']):
                    input_path = os.path.join(read_dir, filename)
                    output_path = os.path.join(write_dir, os.path.basename(filename) + '.tif')
                    print('input output', input_path, output_path)

                    hb.align_dataset_to_match(input_path, match_r_path, output_path)
    else:
        pass

def extract_management(p):
    if p.tasks['extract_management']:
        for scenario_name in p.scenario_names:
            scenario_dir = os.path.join(p.scenarios_data_dir, scenario_name)
            hb.create_directories(scenario_dir)

            management_path = os.path.join(scenario_dir, 'management.nc')

            # hb.create_dirs([p.2015_management_dir, p.2030_management_dir, p.2050_management_dir, p.2070_management_dir, p.2100_management_dir])
            #
            # ge.extract_geotiff_from_netcdf(management_path, p.2015_management_dir, 0) #0 = 2015, last year is 85=2100
            # ge.extract_geotiff_from_netcdf(management_path, p.2030_management_dir, 15) #0 = 2015, last year is 85=2100
            # ge.extract_geotiff_from_netcdf(management_path, p.2050_management_dir, 35) #0 = 2015, last year is 85=2100
            # ge.extract_geotiff_from_netcdf(management_path, p.2070_management_dir, 55) #0 = 2015, last year is 85=2100
            # ge.extract_geotiff_from_netcdf(management_path, p.2100_management_dir, 85) #0 = 2015, last year is 85=2100
    else:
        pass
def convert_states_to_ag_extent(p):
    if p.tasks['convert_states_to_ag_extent']:

        def add_crop_layers_from_dir(input_dir):

            crop_layer_names = [
                "c4per ^ area_fraction ^ C4 perennial crops.tif",
                "c4ann ^ area_fraction ^ C4 annual crops.tif",
                "c3per ^ area_fraction ^ C3 perennial crops.tif",
                "c3nfx ^ area_fraction ^ C3 nitrogen-fixing crops.tif",
                "c3ann ^ area_fraction ^ C3 annual crops.tif",
            ]
            uris_to_combine = [os.path.join(input_dir, i) for i in crop_layer_names]
            print('uris_to_combine', uris_to_combine)
            match_af = hb.ArrayFrame(uris_to_combine[0])
            proportion_cultivated = np.zeros(match_af.shape)
            mask = np.where((match_af.data >= 0.0) & (match_af.data <= 1.0))
            for uri in uris_to_combine:
                proportion_cultivated[mask] += hb.ArrayFrame(uri).data[mask]

            return proportion_cultivated

        match_path = os.path.join(p.task_dirs['extract_lulc'], p.scenario_names[0], str(p.years[0]), "c4per ^ area_fraction ^ C4 perennial crops.tif")


        for scenario_name in p.scenario_names:
            print('task_dirs', p.task_dirs['extract_lulc'])
            scenario_dir = os.path.join(p.task_dirs['extract_lulc'], scenario_name)
            for year in p.years:
                input_dir = os.path.join(p.task_dirs['extract_lulc'], scenario_name, str(year))
                print(input_dir)

                array = add_crop_layers_from_dir(input_dir)

                output_dir = os.path.join(p.task_dirs['convert_states_to_ag_extent'], scenario_name, str(year))
                hb.create_dirs(output_dir)
                output_path = os.path.join(output_dir, 'proportion_ag.tif')

                hb.save_array_as_geotiff(array, output_path, match_path)
        else:
            pass
if __name__ == '__main__':

    p = hb.ProjectFlow('process_ssp_scenarios')

    p.scenarios_data_dir = os.path.join(hb.EXTERNAL_BULK_DATA_DIR, 'lulc\\luh2')
    p.scenario_names = [
        'IMAGE-ssp126',
        'AIM-ssp360',
        'MAGPIE-ssp585',
    ]

    # p.years = [2015, 2050]
    p.years = [2015, 2050]


    p.tasks['extract_lulc'] = 1
    p.tasks['resample_lulc'] = 1
    p.tasks['extract_management'] = 1
    p.tasks['convert_states_to_ag_extent'] = 1

    args = OrderedDict()
    args['skip_extract_lulc'] = 1
    args['skip_resample_lulc'] = 1
    args['skip_extract_management'] = 1
    args['skip_convert_states_to_ag_extent'] = 0
    p.execute(args)

