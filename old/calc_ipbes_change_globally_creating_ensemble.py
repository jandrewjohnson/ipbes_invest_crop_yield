import os, sys
import numpy as np

import hazelbean as hb
import geoecon as ge

aoi_path = r"C:\OneDrive\Projects\lccmr\data\mn_aoi.shp"
lulc_path = r"C:\OneDrive\Projects\base_data\lulc\esacci\ESACCI-LC-L4-LCCS-Map-300m-P1Y-2013-v2.0.7.tif"
match_30km_path = r"C:\OneDrive\Projects\base_data\ipbes\convert_states_to_ag_extent\IMAGE-ssp126\2015\proportion_ag.tif"
match_300m_path = r"C:\OneDrive\Projects\lccmr\data\match_300m_floats.tif"

run_dir = hb.make_run_dir(run_name='aggregating_ssps')

# lulc_clipped_path = os.path.join(run_dir, 'lulc_esa_2013.tif')
# hb.clip_dataset_uri(lulc_path, aoi_path, lulc_clipped_path)


base_ag_path = r"C:\OneDrive\Projects\base_data\ipbes\convert_states_to_ag_extent\IMAGE-ssp126\2015\proportion_ag.tif"
base_urban_path = r"C:\OneDrive\Projects\base_data\ipbes\extract_lulc\IMAGE-ssp126\2015\urban ^ area_fraction ^ urban land.tif"
base_pasture_path = r"C:\OneDrive\Projects\base_data\ipbes\extract_lulc\IMAGE-ssp126\2015\pastr ^ area_fraction ^ managed pasture.tif"
ssp1_ag_path = r"C:\OneDrive\Projects\base_data\ipbes\convert_states_to_ag_extent\IMAGE-ssp126\2050\proportion_ag.tif"
ssp3_ag_path = r"C:\OneDrive\Projects\base_data\ipbes\convert_states_to_ag_extent\AIM-ssp360\2050\proportion_ag.tif"
ssp5_ag_path = r"C:\OneDrive\Projects\base_data\ipbes\convert_states_to_ag_extent\MAGPIE-ssp585\2050\proportion_ag.tif"
ssp1_urban_path = r"C:\OneDrive\Projects\base_data\ipbes\extract_lulc\IMAGE-ssp126\2050\urban ^ area_fraction ^ urban land.tif"
ssp3_urban_path = r"C:\OneDrive\Projects\base_data\ipbes\extract_lulc\AIM-ssp360\2050\urban ^ area_fraction ^ urban land.tif"
ssp5_urban_path = r"C:\OneDrive\Projects\base_data\ipbes\extract_lulc\MAGPIE-ssp585\2050\urban ^ area_fraction ^ urban land.tif"
ssp1_pasture_path = r"C:\OneDrive\Projects\base_data\ipbes\extract_lulc\IMAGE-ssp126\2050\pastr ^ area_fraction ^ managed pasture.tif"
ssp3_pasture_path = r"C:\OneDrive\Projects\base_data\ipbes\extract_lulc\AIM-ssp360\2050\pastr ^ area_fraction ^ managed pasture.tif"
ssp5_pasture_path = r"C:\OneDrive\Projects\base_data\ipbes\extract_lulc\MAGPIE-ssp585\2050\pastr ^ area_fraction ^ managed pasture.tif"

# base_ag_clipped_path = os.path.join(run_dir, r"IMAGE-ssp126\2015\proportion_ag_clipped.tif")
# base_urban_clipped_path = os.path.join(run_dir, r"IMAGE-ssp126\2015\urban ^ area_fraction ^ urban land_clipped.tif")
# base_pasture_clipped_path = os.path.join(run_dir, r"IMAGE-ssp126\2015\pastr ^ area_fraction ^ managed pasture_clipped.tif")
# ssp1_ag_clipped_path = os.path.join(run_dir, r"IMAGE-ssp126\2050\proportion_ag_clipped.tif")
# ssp3_ag_clipped_path = os.path.join(run_dir, r"AIM-ssp360\2050\proportion_ag_clipped.tif")
# ssp5_ag_clipped_path = os.path.join(run_dir, r"MAGPIE-ssp585\2050\proportion_ag_clipped.tif")
# ssp1_urban_clipped_path = os.path.join(run_dir, r"IMAGE-ssp126\2050\urban ^ area_fraction ^ urban land_clipped.tif")
# ssp3_urban_clipped_path = os.path.join(run_dir, r"AIM-ssp360\2050\urban ^ area_fraction ^ urban land_clipped.tif")
# ssp5_urban_clipped_path = os.path.join(run_dir, r"MAGPIE-ssp585\2050\urban ^ area_fraction ^ urban land_clipped.tif")
# ssp1_pasture_clipped_path = os.path.join(run_dir, r"IMAGE-ssp126\2050\pastr ^ area_fraction ^ managed pasture_clipped.tif")
# ssp3_pasture_clipped_path = os.path.join(run_dir, r"AIM-ssp360\2050\pastr ^ area_fraction ^ managed pasture_clipped.tif")
# ssp5_pasture_clipped_path = os.path.join(run_dir, r"MAGPIE-ssp585\2050\pastr ^ area_fraction ^ managed pasture_clipped.tif")

# hb.clip_dataset_uri(base_ag_path, aoi_path, base_ag_clipped_path)
# hb.clip_dataset_uri(base_urban_path, aoi_path, base_urban_clipped_path)
# hb.clip_dataset_uri(base_pasture_path, aoi_path, base_pasture_clipped_path)
# hb.clip_dataset_uri(ssp1_ag_path, aoi_path, ssp1_ag_clipped_path)
# hb.clip_dataset_uri(ssp3_ag_path, aoi_path, ssp3_ag_clipped_path)
# hb.clip_dataset_uri(ssp5_ag_path, aoi_path, ssp5_ag_clipped_path)
# hb.clip_dataset_uri(ssp1_urban_path, aoi_path, ssp1_urban_clipped_path)
# hb.clip_dataset_uri(ssp3_urban_path, aoi_path, ssp3_urban_clipped_path)
# hb.clip_dataset_uri(ssp5_urban_path, aoi_path, ssp5_urban_clipped_path)
# hb.clip_dataset_uri(ssp1_pasture_path, aoi_path, ssp1_pasture_clipped_path)
# hb.clip_dataset_uri(ssp3_pasture_path, aoi_path, ssp3_pasture_clipped_path)
# hb.clip_dataset_uri(ssp5_pasture_path, aoi_path, ssp5_pasture_clipped_path)


base_ag_array = hb.as_array(base_ag_path)
base_urban_array = hb.as_array(base_urban_path)
base_pasture_array = hb.as_array(base_pasture_path)
ssp1_ag_array = hb.as_array(ssp1_ag_path)
ssp3_ag_array = hb.as_array(ssp3_ag_path)
ssp5_ag_array = hb.as_array(ssp5_ag_path)
ssp1_urban_array = hb.as_array(ssp1_urban_path)
ssp3_urban_array = hb.as_array(ssp3_urban_path)
ssp5_urban_array = hb.as_array(ssp5_urban_path)
ssp1_pasture_array = hb.as_array(ssp1_pasture_path)
ssp3_pasture_array = hb.as_array(ssp3_pasture_path)
ssp5_pasture_array = hb.as_array(ssp5_pasture_path)

ssp1_ag_change_array = ssp1_ag_array - base_ag_array
ssp3_ag_change_array = ssp3_ag_array - base_ag_array
ssp5_ag_change_array = ssp5_ag_array - base_ag_array
ssp1_urban_change_array = ssp1_urban_array - base_urban_array
ssp3_urban_change_array = ssp3_urban_array - base_urban_array
ssp5_urban_change_array = ssp5_urban_array - base_urban_array
ssp1_pasture_change_array = ssp1_pasture_array - base_pasture_array
ssp3_pasture_change_array = ssp3_pasture_array - base_pasture_array
ssp5_pasture_change_array = ssp5_pasture_array - base_pasture_array

ssp1_ag_change_path = os.path.join(run_dir, 'ssp1_ag_change.tif')
ssp3_ag_change_path = os.path.join(run_dir, 'ssp3_ag_change.tif')
ssp5_ag_change_path = os.path.join(run_dir, 'ssp5_ag_change.tif')
ssp1_urban_change_path = os.path.join(run_dir, 'ssp1_urban_change.tif')
ssp3_urban_change_path = os.path.join(run_dir, 'ssp3_urban_change.tif')
ssp5_urban_change_path = os.path.join(run_dir, 'ssp5_urban_change.tif')
ssp1_pasture_change_path = os.path.join(run_dir, 'ssp1_pasture_change.tif')
ssp3_pasture_change_path = os.path.join(run_dir, 'ssp3_pasture_change.tif')
ssp5_pasture_change_path = os.path.join(run_dir, 'ssp5_pasture_change.tif')

hb.save_array_as_geotiff(ssp1_ag_change_array, ssp1_ag_change_path, match_30km_path)
hb.save_array_as_geotiff(ssp3_ag_change_array, ssp3_ag_change_path, match_30km_path)
hb.save_array_as_geotiff(ssp5_ag_change_array, ssp5_ag_change_path, match_30km_path)
hb.save_array_as_geotiff(ssp1_urban_change_array, ssp1_urban_change_path, match_30km_path)
hb.save_array_as_geotiff(ssp3_urban_change_array, ssp3_urban_change_path, match_30km_path)
hb.save_array_as_geotiff(ssp5_urban_change_array, ssp5_urban_change_path, match_30km_path)
hb.save_array_as_geotiff(ssp1_pasture_change_array, ssp1_pasture_change_path, match_30km_path)
hb.save_array_as_geotiff(ssp3_pasture_change_array, ssp3_pasture_change_path, match_30km_path)
hb.save_array_as_geotiff(ssp5_pasture_change_array, ssp5_pasture_change_path, match_30km_path)

ag_change = (ssp1_ag_change_array + ssp3_ag_change_array + ssp1_ag_change_array) / 3.0
pasture_change = (ssp1_pasture_change_array + ssp3_pasture_change_array + ssp1_pasture_change_array) / 3.0
urban_change = (ssp1_urban_change_array + ssp3_urban_change_array + ssp1_urban_change_array) / 3.0

ag_change_path = os.path.join(run_dir,'ag_change.tif')
pasture_change_path = os.path.join(run_dir,'pasture_change.tif')
urban_change_path = os.path.join(run_dir,'urban_change.tif')

hb.save_array_as_geotiff(ag_change, ag_change_path, match_30km_path)
hb.save_array_as_geotiff(pasture_change, pasture_change_path, match_30km_path)
hb.save_array_as_geotiff(urban_change, urban_change_path, match_30km_path)

# ag_change_300m_path = os.path.join(run_dir,'ag_change_300m.tif')
# pasture_change_300m_path = os.path.join(run_dir,'pasture_change_300m.tif')
# urban_change_300m_path = os.path.join(run_dir,'urban_change_300m.tif')

# hb.align_dataset_to_match(ag_change_path, match_30km_path, ag_change_300m_path, mode='dataset', resample_method='lanczos', align_to_match=True, output_data_type=6)
# hb.align_dataset_to_match(pasture_change_path, match_300m_path, pasture_change_300m_path, mode='dataset', resample_method='lanczos', align_to_match=True, output_data_type=6)
# hb.align_dataset_to_match(urban_change_path, match_300m_path, urban_change_300m_path, mode='dataset', resample_method='lanczos', align_to_match=True, output_data_type=6)


# ag_change_300m = hb.as_array(ag_change_300m_path)
# ag_change_300m[ag_change_300m<0] = 0.0
# urban_change_300m = hb.as_array(urban_change_300m_path)
# urban_change_300m[urban_change_300m<0] = 0.0
# crop_adjacency_effect = hb.as_array(crop_adjacency_effect_path)
# crop_adjacency_effect[crop_adjacency_effect<0] = 0.0
# urban_adjacency_effect = hb.as_array(urban_adjacency_effect_path)
# print('urban_adjacency_effect', urban_adjacency_effect)
# urban_adjacency_effect[urban_adjacency_effect<0] = 0.0

# ag_expansion_likelihood = ag_change_300m * crop_adjacency_effect
# urban_expansion_likelihood = urban_change_300m * urban_adjacency_effect
#
# m = 878.805 / 22224.3
# # overall_expansion_likelihood = ag_expansion_likelihood + urban_expansion_likelihood
# overall_expansion_likelihood = m * crop_adjacency_effect + urban_adjacency_effect
# # normalized_expansion_likelihood = hb.normalize_array(overall_expansion_likelihood, 0, 1, False)
# normalized_expansion_likelihood = overall_expansion_likelihood
# normalized_expansion_likelihood_path = os.path.join(run_dir, 'normalized_expansion_likelihood.tif')
#
# hb.save_array_as_geotiff(normalized_expansion_likelihood, normalized_expansion_likelihood_path, match_300m_path)






