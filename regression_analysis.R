# install.packages("tidyverse")
# install.packages("MASS")

library(tidyverse)
library(MASS)
library(stats4)

d <- read_csv("C:/OneDrive/Projects/ipbes/intermediate/regression_input.csv")

d$precip_2 <- d$precip ^ 2
d$precip_3 <- d$precip ^ 3
d$temperature_2 <- d$temperature ^ 2
d$temperature_3 <- d$temperature ^ 3
d$minutes_to_market_2 <- d$minutes_to_market ^ 2
d$minutes_to_market_3 <- d$minutes_to_market ^ 3
d$proportion_cropland_2 <- d$proportion_cropland ^ 2
d$proportion_cropland_3 <- d$proportion_cropland ^ 3
d$gdp_gecon_2 <- d$gdp_gecon ^ 2
d$gdp_gecon_3 <- d$gdp_gecon ^ 3
d$irrigated_land_percent_2 <- d$irrigated_land_percent ^ 2
d$irrigated_land_percent_3 <- d$irrigated_land_percent ^ 3
d$crop_suitability_2 <- d$crop_suitability ^ 2
d$crop_suitability_3 <- d$crop_suitability ^ 3
d$maize_PotassiumApplication_Rate_2 <- d$maize_PotassiumApplication_Rate ^ 2
d$maize_PotassiumApplication_Rate_3 <- d$maize_PotassiumApplication_Rate ^ 3
d$maize_PhosphorusApplication_Rate_2 <- d$maize_PhosphorusApplication_Rate ^ 2
d$maize_PhosphorusApplication_Rate_3 <- d$maize_PhosphorusApplication_Rate ^ 3
d$maize_NitrogenApplication_Rate_2 <- d$maize_NitrogenApplication_Rate ^ 2
d$maize_NitrogenApplication_Rate_3 <- d$maize_NitrogenApplication_Rate ^ 3
d$wheat_PotassiumApplication_Rate_2 <- d$wheat_PotassiumApplication_Rate ^ 2
d$wheat_PotassiumApplication_Rate_3 <- d$wheat_PotassiumApplication_Rate ^ 3
d$wheat_PhosphorusApplication_Rate_2 <- d$wheat_PhosphorusApplication_Rate ^ 2
d$wheat_PhosphorusApplication_Rate_3 <- d$wheat_PhosphorusApplication_Rate ^ 3
d$wheat_NitrogenApplication_Rate_2 <- d$wheat_NitrogenApplication_Rate ^ 2
d$wheat_NitrogenApplication_Rate_3 <- d$wheat_NitrogenApplication_Rate ^ 3

# Make a copy of  the data, set the depvar variables that are 0 to be NA, then drop them.
d_wheat = d
d_wheat$wheat_calories_per_ha[d_wheat$wheat_calories_per_ha==0] <- NA
d_wheat = d_wheat[!is.na(d_wheat$wheat_calories_per_ha),]

## Manual add and drop example. Switching to step()
# wheat_linear_formula_string = wheat_calories_per_ha ~ precip + temperature
# wheat_linear_fit <- lm(wheat_linear_formula_string, data=d_wheat)
# summary(wheat_linear_fit)
# drop1(wheat_linear_fit, tests="AIC")
# add1(wheat_linear_fit, scope=~.+minutes_to_market)
# add1(wheat_linear_fit, scope=~.+minutes_to_market + proportion_cropland + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + crop_suitability + gdp_gecon + wheat_PotassiumApplication_Rate + wheat_PhosphorusApplication_Rate + wheat_NitrogenApplication_Rate)

wheat_linear_formula_string = wheat_calories_per_ha ~ precip + temperature + minutes_to_market + proportion_cropland + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + crop_suitability + gdp_gecon + wheat_PotassiumApplication_Rate + wheat_PhosphorusApplication_Rate + wheat_NitrogenApplication_Rate
wheat_linear_fit <- lm(wheat_linear_formula_string, data=d_wheat)
summary(wheat_linear_fit)
"<^ wheat>"
step(wheat_linear_fit)
"^>"
wheat_full_formula_string = wheat_calories_per_ha ~ precip + precip_2  + precip_3 + temperature + temperature_2 + temperature_3 + minutes_to_market + minutes_to_market_2 + minutes_to_market_3 + proportion_cropland + proportion_cropland_2 + proportion_cropland_3 + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + irrigated_land_percent_2 + irrigated_land_percent_3 + crop_suitability + crop_suitability_2 + crop_suitability_3 + gdp_gecon + gdp_gecon_2 + gdp_gecon_3 + wheat_PotassiumApplication_Rate + wheat_PotassiumApplication_Rate_2 + wheat_PotassiumApplication_Rate_3 + wheat_PhosphorusApplication_Rate + wheat_PhosphorusApplication_Rate_2 + wheat_PhosphorusApplication_Rate_3 + wheat_NitrogenApplication_Rate + wheat_NitrogenApplication_Rate_2 + wheat_NitrogenApplication_Rate_3
wheat_full_fit <- lm(wheat_full_formula_string, data=d_wheat)
summary(wheat_full_fit)
step(wheat_full_fit)

wheat_value_full_string = wheat_production_value_per_ha ~ precip + precip_2 + precip_3 + temperature + temperature_2 + temperature_3 + minutes_to_market + minutes_to_market_2 + minutes_to_market_3 + proportion_cropland + proportion_cropland_2 + proportion_cropland_3 + workability + toxicity + rooting_conditions + protected_areas + oxygen_availability + nutrient_retention + nutrient_availability + excess_salts + irrigated_land_percent + irrigated_land_percent_2 + irrigated_land_percent_3 + crop_suitability + crop_suitability_2 + crop_suitability_3 + gdp_gecon + gdp_gecon_2 + gdp_gecon_3 + wheat_PotassiumApplication_Rate + wheat_PotassiumApplication_Rate_2 + wheat_PotassiumApplication_Rate_3 + wheat_PhosphorusApplication_Rate + wheat_PhosphorusApplication_Rate_2 + wheat_PhosphorusApplication_Rate_3 + wheat_NitrogenApplication_Rate + wheat_NitrogenApplication_Rate_2 + wheat_NitrogenApplication_Rate_3
wheat_value_fit <- lm(wheat_value_full_string, data=d_wheat)
summary(wheat_value_fit)
step(wheat_value_fit)


