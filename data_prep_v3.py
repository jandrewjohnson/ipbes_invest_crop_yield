import os, sys, shutil
from collections import OrderedDict
import numpy as np
import pandas as pd
import geopandas as gpd

import hazelbean as hb

L = hb.get_logger('data_prep_v3')

def setup_dirs(p):
    L.debug('Making default dirs.')

    p.input_dir = os.path.join(p.project_dir, 'input')
    p.intermediate_dir = os.path.join(p.project_dir, 'intermediate')
    p.run_dir = os.path.join(p.project_dir, 'intermediate')
    p.output_dir = os.path.join(p.project_dir, 'output')


    dirs = [p.project_dir, p.input_dir, p.intermediate_dir, p.run_dir, p.output_dir]
    hb.create_dirs(dirs)


def link_base_data(p):
    p.calories_per_cell_path = os.path.join(p.input_dir, 'calories_per_cell.tif')


    
    # Cartographic
    p.country_names_path = os.path.join(p.input_dir, 'cartographic/country_names.csv')
    p.country_ids_raster_path = os.path.join(p.input_dir, 'cartographic/country_ids.tif')    #
    p.ha_per_cell_5m_path = os.path.join(p.input_dir, 'cartographic/ha_per_cell_5m.tif')

    # Crop

    # Climate
    p.precip_path = os.path.join(p.input_dir, 'climate/worldclim/bio12.bil')
    p.temperature_path = os.path.join(p.input_dir, 'climate/worldclim/bio1.bil')

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
    p.pop_30s_path = os.path.join(p.input_dir, 'demographic/ciesin', 'pop_30s_REDO_FROM_WEB.tif')


def create_baseline_regression_data(p):
    p.baseline_regression_data_path = os.path.join(p.cur_dir, 'baseline_regression_data.csv')
    # Iterate through input_paths adding them.  Currently also fixes fertilizer nan issues.
    af_names_list = []
    dfs_list = []
    paths_to_add = [
        # p.country_names_path,
        p.country_ids_raster_path,
        p.ha_per_cell_5m_path,
        p.precip_path,
        p.temperature_path,
        p.slope_path,
        p.altitude_path,
        p.workability_index_path,
        p.toxicity_index_path,
        p.rooting_conditions_index_path,
        # p.rainfed_land_percent_path,
        # p.protected_areas_index_path,
        p.oxygen_availability_index_path,
        # p.nutrient_retention_index_path,
        p.nutrient_retention_index_path,
        p.nutrient_availability_index_path,
        # p.irrigated_land_percent_path,
        p.excess_salts_index_path,
        # p.cultivated_land_percent_path,
        # p.crop_suitability_path,
        p.gdp_2000_path,
        p.gdp_gecon,
        p.minutes_to_market_path,
        p.pop_30s_path,
    ]

    if p.run_this:
        match_af = hb.ArrayFrame(paths_to_add[0])
        for path in paths_to_add:
            print('path', path)
            print()
            # if 'altitude' in path or 'slope' in path and 0:
            #     name = hb.explode_path(path)['file_root']
            #     af = hb.ArrayFrame(path)
            #     modified_array = np.where(af.data < 0, 0, af.data)
            #     tmp1 = hb.temp(remove_at_exit=True)
            #     hb.save_array_as_geotiff(modified_array, tmp1, p.precip_path, projection_override=match_af.projection)
            #     modified_af = hb.ArrayFrame(tmp1)
            #     af_names_list.append(name)
            #     df = hb.convert_af_to_1d_df(modified_af)
            #     dfs_list.append(df)
            # else:

            name = hb.explode_path(path)['file_root']
            af = hb.ArrayFrame(path)
            af_names_list.append(name)
            df = hb.convert_af_to_1d_df(af)
            dfs_list.append(df)

        L.info('Concatenating all dataframes.')
        df = hb.concatenate_dfs_horizontally(dfs_list, af_names_list)
        df[df < 0] = 0.0

        # Rather than getting rid of all cells without crops, just get rid of those not on land.
        df[df['excess_salts'] == 255.0] = np.nan

        # p.nan_mask_path'] = 'nan_mask.csv'
        df_nan = df['excess_salts']
        df_nan.to_csv(p.nan_mask_path)

        df = df.dropna()

        df.to_csv(p.baseline_regression_data_path)


def aggregate_crops_by_type(p):
    """CMIP6 and the land-use harmonization project have centered on 5 crop types: c3 annual, c3 perennial, c4 annual, c4 perennial, nitrogen fixer
    Aggregate the 15 crops to those four categories by modifying the baseline_regression_data."""

    baseline_regression_data_df = pd.read_csv(p.baseline_regression_data_path)
    baseline_regression_data_df.set_index('Unnamed: 0', inplace=True)

    vars_names_to_aggregate = [
        # 'production_value_per_ha',
        # 'calories_per_ha',
        'yield_per_ha'
        'proportion_cultivated',
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

    # Create a DF of zeros, ready to hold the summed results for each crop type. Indix given will  be from baseline_regression_data_df so that spatial indices match.
    crop_types_df = pd.DataFrame(np.zeros(len(baseline_regression_data_df.index)), index=baseline_regression_data_df.index)

    # Iterate through crop_types
    for crop_type, crops in crop_membership.items():
        L.info('Aggregating ' + str(crop_type) + ' ' + str(crops))
        for var_name_to_aggregate in vars_names_to_aggregate:
            L.info('  var_name_to_aggregate ' + var_name_to_aggregate)
            output_col_name = crop_type + '_' + var_name_to_aggregate
            crop_types_df[output_col_name] = np.zeros(len(baseline_regression_data_df.index))
            for crop in crops:
                input_col_name = crop + '_' + var_name_to_aggregate
                if input_col_name in baseline_regression_data_df:
                    crop_types_df[output_col_name] += baseline_regression_data_df[input_col_name]

            crop_types_df[output_col_name][crop_types_df[output_col_name] > 1e+12] = 0.0

    p.c3_annual_calories_path = os.path.join(p.run_dir, "c3_annual_calories.tif")
    p.c3_perennial_calories_path = os.path.join(p.run_dir, "c3_perennial_calories.tif")
    p.c4_annual_calories_path = os.path.join(p.run_dir, "c4_annual_calories.tif")
    p.c4_perennial_calories_path = os.path.join(p.run_dir, "c4_perennial_calories.tif")
    p.nitrogen_fixer_calories_path = os.path.join(p.run_dir, "nitrogen_fixer_calories.tif")

    crop_types_df = pd.DataFrame(np.zeros(len(baseline_regression_data_df.index)),
                                 index=baseline_regression_data_df.index)
    # for crop in crops:
    var_name_to_aggregate = 'calories'
    for crop_type in p.crop_types:
        input_col_name = crop_type + '_' + var_name_to_aggregate
        output_col_name = crop_type + '_' + var_name_to_aggregate  ####Why duplicate since input_col_name == output_col_name?
        crop_types_df[output_col_name] = np.zeros(len(baseline_regression_data_df.index))

        crop_types_df[output_col_name] += baseline_regression_data_df[input_col_name]

        crop_types_df[output_col_name][crop_types_df[output_col_name] > 1e+12] = 0.0

    crop_types_df.to_csv(p.aggregated_crop_data_csv_path)

    p.crop_types = [
        'c3_annual',
        'c3_perennial',
        'c4_annual',
        'c4_perennial',
        'nitrogen_fixer',
    ]

main = 'here'
if __name__ =='__main__':
    p = hb.ProjectFlow('../ipbes_invest_crop_yield_project')

    setup_dirs_task = p.add_task(setup_dirs)
    link_base_data_task = p.add_task(link_base_data)
    create_baseline_regression_data_task = p.add_task(create_baseline_regression_data)

    setup_dirs_task.run = 0
    link_base_data_task.run = 1
    create_baseline_regression_data_task.run = 1

    setup_dirs_task.skip_existing = 0
    link_base_data_task.skip_existing = 0
    create_baseline_regression_data_task.skip_existing = 0

    p.execute()



