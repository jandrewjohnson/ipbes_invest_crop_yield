import hazelbean as hb
import geoecon as ge

import os, sys

paths = [
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c3ann_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c3ann_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c3nfx_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c3nfx_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c3per_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c3per_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c4ann_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c4ann_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c4per_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP3xRCP6.0_AIM_global_2050_c4per_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c3ann_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c3ann_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c3nfx_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c3nfx_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c3per_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c3per_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c4ann_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c4ann_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c4per_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP5xRCP8.5_MAGPIE_global_2050_c4per_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c3ann_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c3ann_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c3nfx_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c3nfx_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c3per_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c3per_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c4ann_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c4ann_percent_change.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c4per_kcal_production_per_cell.tif",
    r"C:\OneDrive\Projects\ipbes\output\SSP1xRCP2.6_IMAGE_global_2050_c4per_percent_change.tif",
]

for path in paths:
    file_root = hb.explode_path(path)['file_root']
    scenario_name = file_root.split('_')[0].replace('x', ' ').upper()
    data_name = file_root.split('_', 3)[3:][0].replace('_', ' ')

    print('scenario_name', scenario_name)
    print('data_name', data_name)
    a = hb.as_array(path)
    output_uri = path.replace('.tif', '.png')
    if 'percent_change' in path:
        ge.show_array(a, output_uri=output_uri, use_basemap=True, resolution='i', title=scenario_name, cbar_label=data_name, vmin=0, vmax=0.2, move_ticks_in=True)
    else:
        ge.show_array(a, output_uri=output_uri, use_basemap=True, resolution='i', title=scenario_name, cbar_label=data_name, vmin=0, move_ticks_in=True)