library(tidyverse)
library(readxl)
library(writexl)
library(here)
library(ggplot2)
library(dplyr)
library(patchwork)
library(broom)


# Nomenclature
#  ds: Dataset
#  ds_sel: Dataset of features selected from a larger dataset


# Selected features from each dataset (NOTE: Outdated)
# nyc_population_ds_sel:
#  1:21: Count in each 5 year age bracket
#  27: Male count
#  28:45: Male in each 5yr bracket
#  51: Female count
#  52:69 Female in each 5yr bracket
#  75:77, median age (both, male, female)
#  115:128: Relation to household
#  135:148: Married/unmarried/living alone etc
#  149:153, 160:162 Housing unit status

# nyc_income_ds_sel:
#  5:23 (odds only): % of households with each income level
#  25: Median income

# nyc_vehicles_ds_sel
#  

# nyc_education_ds_sel
#  12:15: 25+ education levels
#  The rest: Education levels for other age groups


############################## Parameters ##############################

population_threshold = 300 # Remove tracts below this population level



########################################################################

# 1. Load datasets from files (files are downloaded raw from data.census.gov)
nyc_population_ds <- read.csv(here("Datasets", "NYC census data", "NYC_DECENNIALDP2020.DP1-Data.csv"), skip = 1)
nyc_income_ds <- read.csv(here("Datasets", "NYC census data", "NYC_ACSST5Y2020.S1901-Data.csv"), skip = 1)
nyc_vehicle_ds <- read.csv(here("Datasets", "NYC census data", "NYC_ACSDT5Y2020.B08201-Data.csv"), skip = 1)
nyc_education_ds <- read.csv(here("Datasets", "NYC census data", "NYC_ACSST5Y2020.S1501-Data.csv"), skip = 1) 
nyc_geoinfo_ds <- read.csv(here("Datasets", "NYC census data", "NYC_GEOINFO2023.GEOINFO-Data.csv"), skip = 1)


# Select only the relevant features from each dataset
nyc_population_ds_sel <- nyc_population_ds %>% select(1:21, 27:45, 51:69, 75:77, 115:128, 131, 135:148, 149:153, 160:162)
nyc_income_ds_sel <- nyc_income_ds %>% select(1, 5,7,9,11,13,15,17,19,21,23, 25, 27)
nyc_vehicle_ds_sel <- nyc_vehicle_ds %>% select(-starts_with("Margin")) %>% select(-(2:3))
nyc_education_ds_sel <- nyc_education_ds %>% select(1,2,starts_with("Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population."))

nyc_density <- nyc_population_ds_sel %>%
  select(Geography = Geography, Geography_Name = Geographic.Area.Name, total_population = Count..SEX.AND.AGE..Total.population) %>%  # Just tract name and total population
  left_join(nyc_geoinfo_ds %>% select(Geography = Geographic.Identifier, land_area_sqmi = Area..Land..in.square.miles.), by = "Geography") %>%
  mutate(population_density = total_population / land_area_sqmi) %>%
  mutate(population_density = ifelse(is.finite(population_density), population_density, NA))

glimpse(nyc_density)


# 5. Combine all datasets, joining by census tract ID
nyc_combined_data_large <- nyc_population_ds_sel %>%
  left_join(nyc_density, by = "Geography") %>%
  left_join(nyc_income_ds_sel, by = "Geography") %>%
  left_join(nyc_vehicle_ds_sel, by = "Geography") %>%
  left_join(nyc_education_ds_sel, by = "Geography") %>%
  filter(total_population >= population_threshold) %>%
  filter(Geography != "1400000US36005000100")
  
  
nyc_combined_data_concise <- nyc_combined_data_large %>%
    transmute(
      Geography = Geography,
      Geography_name = Geographic.Area.Name.x,
      total_population = total_population,
      population_density = population_density,
      
      male_percent = Count..SEX.AND.AGE..Male.population / Count..SEX.AND.AGE..Total.population,
      
      percent_inhouseholds = Count..RELATIONSHIP..Total.population..In.households / Count..SEX.AND.AGE..Total.population,
      percent_marriedhouseholds = Count..HOUSEHOLDS.BY.TYPE..Total.households..Married.couple.household /  Count..SEX.AND.AGE..Total.population,
      percent_cohabitingcouple = Count..HOUSEHOLDS.BY.TYPE..Total.households..Cohabiting.couple.household /  Count..SEX.AND.AGE..Total.population,
      percent_solomale = Count..HOUSEHOLDS.BY.TYPE..Total.households..Male.householder..no.spouse.or.partner.present...Living.alone /  Count..SEX.AND.AGE..Total.population,
      percent_solofemale = Count..HOUSEHOLDS.BY.TYPE..Total.households..Female.householder..no.spouse.or.partner.present...Living.alone /  Count..SEX.AND.AGE..Total.population,
      
      percent_occupiedhousingunits = Count..HOUSING.OCCUPANCY..Total.housing.units..Occupied.housing.units / Count..HOUSING.OCCUPANCY..Total.housing.units,
      percent_owneroccupied = Count..HOUSING.TENURE..Occupied.housing.units..Owner.occupied.housing.units / Count..HOUSING.OCCUPANCY..Total.housing.units,
      percent_renteroccupied = Count..HOUSING.TENURE..Occupied.housing.units..Renter.occupied.housing.units / Count..HOUSING.OCCUPANCY..Total.housing.units,
      
      # Age distribution (both sexes)
      median_age = Count..MEDIAN.AGE.BY.SEX..Both.sexes,
      percent_under5 = Count..SEX.AND.AGE..Total.population..Under.5.years / Count..SEX.AND.AGE..Total.population,
      percent_5to9 =   Count..SEX.AND.AGE..Total.population..5.to.9.years / Count..SEX.AND.AGE..Total.population,
      percent_10to14 = Count..SEX.AND.AGE..Total.population..10.to.14.years / Count..SEX.AND.AGE..Total.population,
      percent_15to19 = Count..SEX.AND.AGE..Total.population..15.to.19.years / Count..SEX.AND.AGE..Total.population,
      percent_20to24 = Count..SEX.AND.AGE..Total.population..20.to.24.years / Count..SEX.AND.AGE..Total.population,
      percent_25to29 = Count..SEX.AND.AGE..Total.population..25.to.29.years / Count..SEX.AND.AGE..Total.population,
      percent_30to34 = Count..SEX.AND.AGE..Total.population..30.to.34.years / Count..SEX.AND.AGE..Total.population,
      percent_35to39 = Count..SEX.AND.AGE..Total.population..35.to.39.years / Count..SEX.AND.AGE..Total.population,
      percent_40to44 = Count..SEX.AND.AGE..Total.population..40.to.44.years / Count..SEX.AND.AGE..Total.population,
      percent_45to49 = Count..SEX.AND.AGE..Total.population..45.to.49.years / Count..SEX.AND.AGE..Total.population,
      percent_50to54 = Count..SEX.AND.AGE..Total.population..50.to.54.years / Count..SEX.AND.AGE..Total.population,
      percent_55to59 = Count..SEX.AND.AGE..Total.population..55.to.59.years / Count..SEX.AND.AGE..Total.population,
      percent_60to64 = Count..SEX.AND.AGE..Total.population..60.to.64.years / Count..SEX.AND.AGE..Total.population,
      percent_65to69 = Count..SEX.AND.AGE..Total.population..65.to.69.years / Count..SEX.AND.AGE..Total.population,
      percent_70to74 = Count..SEX.AND.AGE..Total.population..70.to.74.years / Count..SEX.AND.AGE..Total.population,
      percent_75to79 = Count..SEX.AND.AGE..Total.population..75.to.79.years / Count..SEX.AND.AGE..Total.population,
      percent_80to84 = Count..SEX.AND.AGE..Total.population..80.to.84.years / Count..SEX.AND.AGE..Total.population,
      percent_over85 = Count..SEX.AND.AGE..Total.population..85.years.and.over / Count..SEX.AND.AGE..Total.population,
      
      # Age distribution (male)
      male_median_age = Count..MEDIAN.AGE.BY.SEX..Male,
      male_percent_under5 = Count..SEX.AND.AGE..Male.population..Under.5.years / Count..SEX.AND.AGE..Male.population,
      male_percent_5to9 = Count..SEX.AND.AGE..Male.population..5.to.9.years / Count..SEX.AND.AGE..Male.population,
      male_percent_10to14 = Count..SEX.AND.AGE..Male.population..10.to.14.years / Count..SEX.AND.AGE..Male.population,
      male_percent_15to19 = Count..SEX.AND.AGE..Male.population..15.to.19.years / Count..SEX.AND.AGE..Male.population,
      male_percent_20to24 = Count..SEX.AND.AGE..Male.population..20.to.24.years / Count..SEX.AND.AGE..Male.population,
      male_percent_25to29 = Count..SEX.AND.AGE..Male.population..25.to.29.years / Count..SEX.AND.AGE..Male.population,
      male_percent_30to34 = Count..SEX.AND.AGE..Male.population..30.to.34.years / Count..SEX.AND.AGE..Male.population,
      male_percent_35to39 = Count..SEX.AND.AGE..Male.population..35.to.39.years / Count..SEX.AND.AGE..Male.population,
      male_percent_40to44 = Count..SEX.AND.AGE..Male.population..40.to.44.years / Count..SEX.AND.AGE..Male.population,
      male_percent_45to49 = Count..SEX.AND.AGE..Male.population..45.to.49.years / Count..SEX.AND.AGE..Male.population,
      male_percent_50to54 = Count..SEX.AND.AGE..Male.population..50.to.54.years / Count..SEX.AND.AGE..Male.population,
      male_percent_55to59 = Count..SEX.AND.AGE..Male.population..55.to.59.years / Count..SEX.AND.AGE..Male.population,
      male_percent_60to64 = Count..SEX.AND.AGE..Male.population..60.to.64.years / Count..SEX.AND.AGE..Male.population,
      male_percent_65to69 = Count..SEX.AND.AGE..Male.population..65.to.69.years / Count..SEX.AND.AGE..Male.population,
      male_percent_70to74 = Count..SEX.AND.AGE..Male.population..70.to.74.years / Count..SEX.AND.AGE..Male.population,
      male_percent_75to79 = Count..SEX.AND.AGE..Male.population..75.to.79.years / Count..SEX.AND.AGE..Male.population,
      male_percent_80to84 = Count..SEX.AND.AGE..Male.population..80.to.84.years / Count..SEX.AND.AGE..Male.population,
      male_percent_over85 = Count..SEX.AND.AGE..Male.population..85.years.and.over / Count..SEX.AND.AGE..Male.population,
      
      # Age distribution (female)
      Female_median_age = Count..MEDIAN.AGE.BY.SEX..Female,
      female_percent_under5 = Count..SEX.AND.AGE..Female.population..Under.5.years / Count..SEX.AND.AGE..Female.population,
      female_percent_5to9 = Count..SEX.AND.AGE..Female.population..5.to.9.years / Count..SEX.AND.AGE..Female.population,
      female_percent_10to14 = Count..SEX.AND.AGE..Female.population..10.to.14.years / Count..SEX.AND.AGE..Female.population,
      female_percent_15to19 = Count..SEX.AND.AGE..Female.population..15.to.19.years / Count..SEX.AND.AGE..Female.population,
      female_percent_20to24 = Count..SEX.AND.AGE..Female.population..20.to.24.years / Count..SEX.AND.AGE..Female.population,
      female_percent_25to29 = Count..SEX.AND.AGE..Female.population..25.to.29.years / Count..SEX.AND.AGE..Female.population,
      female_percent_30to34 = Count..SEX.AND.AGE..Female.population..30.to.34.years / Count..SEX.AND.AGE..Female.population,
      female_percent_35to39 = Count..SEX.AND.AGE..Female.population..35.to.39.years / Count..SEX.AND.AGE..Female.population,
      female_percent_40to44 = Count..SEX.AND.AGE..Female.population..40.to.44.years / Count..SEX.AND.AGE..Female.population,
      female_percent_45to49 = Count..SEX.AND.AGE..Female.population..45.to.49.years / Count..SEX.AND.AGE..Female.population,
      female_percent_50to54 = Count..SEX.AND.AGE..Female.population..50.to.54.years / Count..SEX.AND.AGE..Female.population,
      female_percent_55to59 = Count..SEX.AND.AGE..Female.population..55.to.59.years / Count..SEX.AND.AGE..Female.population,
      female_percent_60to64 = Count..SEX.AND.AGE..Female.population..60.to.64.years / Count..SEX.AND.AGE..Female.population,
      female_percent_65to69 = Count..SEX.AND.AGE..Female.population..65.to.69.years / Count..SEX.AND.AGE..Female.population,
      female_percent_70to74 = Count..SEX.AND.AGE..Female.population..70.to.74.years / Count..SEX.AND.AGE..Female.population,
      female_percent_75to79 = Count..SEX.AND.AGE..Female.population..75.to.79.years / Count..SEX.AND.AGE..Female.population,
      female_percent_80to84 = Count..SEX.AND.AGE..Female.population..80.to.84.years / Count..SEX.AND.AGE..Female.population,
      female_percent_over85 = Count..SEX.AND.AGE..Female.population..85.years.and.over / Count..SEX.AND.AGE..Female.population,
      
      # Income level
      median_household_income = as.numeric(Estimate..Households..Median.income..dollars.),
      #mean_household_income = Estimate..Households..Mean.income..dollars.,
      percent_income_under10k = as.numeric(Estimate..Households..Total..Less.than..10.000)/100,
      percent_income_10to15k = as.numeric(Estimate..Households..Total...10.000.to..14.999)/100,
      percent_income_15to25k = as.numeric(Estimate..Households..Total...15.000.to..24.999)/100,
      percent_income_25to35k = as.numeric(Estimate..Households..Total...25.000.to..34.999)/100,
      percent_income_35to50k = as.numeric(Estimate..Households..Total...35.000.to..49.999)/100,
      percent_income_50to75k = as.numeric(Estimate..Households..Total...50.000.to..74.999)/100,
      percent_income_75to100k = as.numeric(Estimate..Households..Total...75.000.to..99.999)/100,
      percent_income_100to150k = as.numeric(Estimate..Households..Total...100.000.to..149.999)/100,
      percent_income_150to200k = as.numeric(Estimate..Households..Total...150.000.to..199.999)/100,
      percent_income_over200k = as.numeric(Estimate..Households..Total...200.000.or.more)/100,
      
      # Vehicles available
      # TODO are the numbers households or people in a household? Need to scale? E.g. 1 person + 2*2person + 3*3person?
      percent_novehicles = Estimate..Total...No.vehicle.available / Count..SEX.AND.AGE..Total.population,
      percent_1vehicle_perperson = (Estimate..Total...1.person.household...1.vehicle.available +
                                     Estimate..Total...2.person.household...2.vehicles.available + 
                                     Estimate..Total...3.person.household...3.vehicles.available + 
                                     Estimate..Total...4.or.more.person.household...4.or.more.vehicles.available)/Count..SEX.AND.AGE..Total.population,
      
      percent_lessthan1vehicle_perperson = (Estimate..Total...1.person.household...No.vehicle.available +
                                               Estimate..Total...2.person.household...No.vehicle.available + 
                                               Estimate..Total...2.person.household...1.vehicle.available + 
                                               Estimate..Total...3.person.household...No.vehicle.available + 
                                               Estimate..Total...3.person.household...1.vehicle.available + 
                                               Estimate..Total...3.person.household...2.vehicles.available + 
                                               
                                               Estimate..Total...4.or.more.person.household...No.vehicle.available + 
                                               Estimate..Total...4.or.more.person.household...1.vehicle.available + 
                                               Estimate..Total...4.or.more.person.household...2.vehicles.available + 
                                               Estimate..Total...4.or.more.person.household...3.vehicles.available)/Count..SEX.AND.AGE..Total.population,
      
      # Education level
      # TODO the under 25/over25 perecntages are not scaled by population size
      
      #under25y_percent_lessthanhighschool = Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.18.to.24.years..Less.than.high.school.graduate,
      under25y_percent_highschool = as.numeric(Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.18.to.24.years..High.school.graduate..includes.equivalency.)/100,
      under25y_percent_somecollege = as.numeric(Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.18.to.24.years..Some.college.or.associate.s.degree)/100,
      under25y_percent_bachelors = as.numeric(Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.18.to.24.years..Bachelor.s.degree.or.higher)/100,
      
      #over25y_percent_lessthanhighschool = Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..Less.than.9th.grade + 
       # Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..9th.to.12th.grade..no.diploma,
      over25y_percent_highschool = as.numeric(Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..High.school.graduate..includes.equivalency.)/100,
      #over25y_percent_somecollege = Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..Some.college..no.degree + Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..Associate.s.degree,
      over25y_percent_somecollege = as.numeric(Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..Some.college..no.degree)/100 + 
        as.numeric(Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..Associate.s.degree)/100,
      over25y_percent_bachelors = as.numeric(Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..Bachelor.s.degree)/100,
      over25y_percent_graduatedegree = as.numeric(Estimate..Percent..AGE.BY.EDUCATIONAL.ATTAINMENT..Population.25.years.and.over..Graduate.or.professional.degree)/100
    ) %>%
    mutate(across(-c(Geography, Geography_name), ~as.numeric(as.character(.))))
    
  glimpse(nyc_combined_data_concise)
  
  
nyc_data_numeric <- nyc_combined_data_concise %>%
  mutate(across(-c(Geography, Geography_name), ~as.numeric(as.character(.)))) %>%
  drop_na()


# Normalize all features
nyc_data_normalized <- nyc_data_numeric %>%
  mutate(across(where(is.numeric), ~scale(.))) %>%
  mutate(across(where(is.numeric), ~as.numeric(.)))

glimpse(nyc_data_normalized)
    

# Save combined dataset to Excel file
write.csv(nyc_data_numeric, "nyc_data_numeric.csv")
write.csv(nyc_data_normalized, "nyc_data_normalized.csv")



################################ PCA ################################ 
# 
# # Load necessary libraries
# 
# 
# # Example with built-in iris dataset
# # Step 1: Prepare your data
# pca_data <- nyc_data_numeric %>%
#   select(where(is.numeric)) # Select only numeric columns for PCA
# 
# # Step 2: Perform PCA
# pca_result <- pca_data %>%
#   scale() %>% # Standardize variables (important for PCA)
#   prcomp()
# 
# # Step 3: Examine the results
# # Summary of PCA
# summary(pca_result)
# 
# # Extract component loadings
# loadings <- pca_result %>%
#   tidy(matrix = "rotation") %>%
#   pivot_wider(names_from = "PC", values_from = "value", names_prefix = "PC")
# 
# # Extract the variance explained by each component
# variance_explained <- pca_result %>%
#   tidy(matrix = "eigenvalues")
# 
# print(pca_result$rotation)
# 
# # Extract the PC scores from the PCA result
# pca_scores <- as.data.frame(pca_result$x)
# 
# # Add back the identifying columns from the original dataset
# pca_dataset <- bind_cols(
#   nyc_data_numeric %>% select(Geography, Geography_name),
#   pca_scores
# )
# 
# # Save the PCA dataset to a CSV file
# write.csv(pca_dataset, "nyc_data_pca.csv", row.names = FALSE)
# 
# # If you want to save only the first few PCs that explain most of the variance
# # For example, to keep only the first 10 PCs:
# pca_dataset_reduced <- bind_cols(
#   nyc_data_numeric %>% select(Geography, Geography_name),
#   pca_scores %>% select(1:10)  # Adjust the number based on your variance analysis
# )
# 
# # Save the reduced PCA dataset to a CSV file
# write.csv(pca_dataset_reduced, "nyc_data_pca_reduced.csv", row.names = FALSE)


################################ Plots ################################ 

p1 <- ggplot(nyc_data_numeric, aes(x = "", y = total_population)) +
  geom_violin(fill = "skyblue", alpha = 0.7) +
  geom_boxplot(width = 0.1, fill = "white", alpha = 0.5) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "NYC - Distribution of Total Population",
       x = NULL,
       y = "Total Population") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

p2 <- ggplot(nyc_data_numeric, aes(x = "", y = population_density)) +
  geom_violin(fill = "lightgreen", alpha = 0.7) +
  geom_boxplot(width = 0.1, fill = "white", alpha = 0.5) +
  scale_y_continuous(labels = scales::comma) +
  labs(title = "NYC - Distribution of Population Density",
       x = NULL,
       y = "Population Density (per sq. mile)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

combined_plot <- p1 + p2 + plot_layout(ncol = 2)
combined_plot


################################ TESTING ################################ 

# na_locations <- nyc_data_numeric %>%
#   rowwise() %>%
#   mutate(na_count = sum(is.na(c_across(-c(Geography, Geography_name))))) %>%
#   filter(na_count > 0) %>%
#   select(Geography_name, na_count, total_population)
# print(na_locations, n = Inf)  # n = Inf shows all rows