import os, sys, shutil
from collections import OrderedDict
import numpy as np
import pandas as pd
import geopandas as gpd

import hazelbean as hb

L = hb.get_logger('data_preprocessing')


def load_data(p):
    crop_types_df = pd.read_csv(p.aggregated_crop_data_csv_path)
    df_land = pd.read_csv(p.baseline_regression_data_path)
    print(df_land.shape,crop_types_df.shape)

main = 'here'
if __name__ =='__main__':
    p = hb.ProjectFlow('../ipbes_invest_crop_yield_project')

    load_data_task = p.add_task(load_data)

    load_data_task.run = 1

    load_data_task.skip_existing = 1

    p.execute()
