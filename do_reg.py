import os, sys, shutil
from collections import OrderedDict
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
import mpl_toolkits
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.basemap import Basemap

import

import hazelbean as hb

L = hb.get_logger('data_prep_v3')

p = hb.ProjectFlow('../ipbes_invest_crop_yield_project')

def setup_dirs(p):
    L.debug('Making default dirs.')

    p.input_dir = os.path.join(p.project_dir, 'input')
    p.intermediate_dir = os.path.join(p.project_dir, 'intermediate')
    p.run_dir = os.path.join(p.project_dir, 'intermediate')
    p.output_dir = os.path.join(p.project_dir, 'output')

    dirs = [p.project_dir, p.input_dir, p.intermediate_dir, p.run_dir, p.output_dir]
    hb.create_dirs(dirs)


def load_data(p):
    p.aggregate_crops_by_type_path = os.path.join(p.intermediate_dir, 'aggregate_crops_by_type', 'aggregated_crop_data.csv')
    p.baseline_regression_data_path = os.path.join(p.intermediate_dir, 'create_baseline_regression_data', 'baseline_regression_data.csv')

    if p.run_this:

        crops_df = pd.read_csv(p.aggregate_crops_by_type_path)
        baseline_df = pd.read_csv(p.baseline_regression_data_path)

        df = pd.merge(crops_df, baseline_df, left_on='pixel_id', right_on='pixel_id', how='inner')

        print(df.columns)


main = 'here'
if __name__ == '__main__':

    setup_dirs_task = p.add_task(setup_dirs)
    setup_dirs_task.run = 1
    setup_dirs_task.skip_existing = 0

    load_data_task = p.add_task(load_data)
    load_data_task.run = 1
    load_data_task.skip_existing = 0

    p.execute()
