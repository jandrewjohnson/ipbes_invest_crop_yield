### For reference:
[Paper draft](https://docs.google.com/document/d/1MK59KWyGKPNDi4ZZ8vHMsSxYdsalcxMv1AamdkMARy0/edit?ts=5b51ed41#)
[gsheet lit review etc](https://docs.google.com/spreadsheets/d/1f-in1bvPsU_y19rAu3XNtgUrr6Tr52wR0mWT5KZ-yQw/edit#gid=361776245)

# Workplan

### Now
-	Justin: fix the latitude column 

### Aug 1-10
-	Input data: Climate data augmentation – climate add other worldclim features (bioclim ones like diurnal variation…)
-	Input data: Soil features – Charlie reads about GAEZ and choose metric(s) + how to encode them.
-	Input data: continuous metric for spatial adjacency (read on spatial regression and ask Chris/Stace/David): abs(lat), sin(lat), sq(lat2+lon2) ..?
-	Justin tries to win his bet about categorical ordered variable encoding, which would be the most “true” way to encode the soil variables
-	Justin fails at winning his bet because it’s impossible (thus Charlie wins)
-	Charlie: Run baseline_data_version1

With df_v1 : 
-	Charlie settles on an (or a few) evaluation metric(s) (likely R2) – stats help!
-	Charlie tries different modeling approaches (linear, poly2, poly3, poly2+interact°, poly3+interact°, Ridge (≠ params), Lasso (≠ params), Tree-based approaches)
-	… and builds big R2 table to compare results

We find datasets problems and best encoding approaches and build baseline_df_v2:
-	Re-run this big R2 table with v2. Based on this, dig deeper into the best models and tune parameters… 
-	Select the few (3-6) best models for the paper
-	Confirm with Justin about these models

-	(Now and only now, not before /!\DataLeakageIsAThing) run validation set to get the R2 for unseen data.

-	Calc. coefficient of importance/std for each feature of each model: Table 1
-	Figure to compare pro/cons of each model
-	Figure to discuss importance of features (relative importance of climate, soil, demographics, spatial adjacency) ..
-	Other cool figures or maybe not
-	Write the paper (outline drafted here)

## (Also at some point):
-	A little more lit review on GLI + Lobell + recent global crop modeling? (Search Lobell cites Monfreda)

## Stretch goals
-	Add censored regression to modeling approaches explored
-	Compare our models performance when using SPAM data 


