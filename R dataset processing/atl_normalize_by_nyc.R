library(tidyverse)
library(readr)

# 1. Load the NYC and Atlanta datasets
nyc_data_numeric <- read.csv("nyc_data_numeric.csv")
atl_data_numeric <- read.csv("atl_data_numeric.csv")

# 2. Extract non-numeric columns
nyc_non_numeric <- nyc_data_numeric %>%
  select_if(function(x) !is.numeric(x))

atl_non_numeric <- atl_data_numeric %>%
  select_if(function(x) !is.numeric(x))

# 3. Extract numeric columns
nyc_numeric <- nyc_data_numeric %>%
  select_if(is.numeric)

atl_numeric <- atl_data_numeric %>%
  select_if(is.numeric)

# 4. Ensure columns match between datasets
common_columns <- intersect(names(nyc_numeric), names(atl_numeric))
nyc_numeric <- nyc_numeric %>% select(all_of(common_columns))
atl_numeric <- atl_numeric %>% select(all_of(common_columns))

# 5. Calculate means and standard deviations from NYC dataset
nyc_means <- colMeans(nyc_numeric, na.rm = TRUE)
nyc_sds <- apply(nyc_numeric, 2, sd, na.rm = TRUE)

# 6. Normalize Atlanta data using NYC parameters in a vectorized way
# Replace zero standard deviations with 1 to avoid division by zero
nyc_sds[nyc_sds == 0] <- 1

# Create a scale_external function
scale_external <- function(data, center, scale) {
  result <- data
  for (col in names(data)) {
    if (col %in% names(center) && col %in% names(scale)) {
      result[[col]] <- (data[[col]] - center[[col]]) / scale[[col]]
    }
  }
  return(result)
}

# Apply normalization using NYC parameters
atl_normalized_with_nyc <- scale_external(
  atl_numeric,
  nyc_means,
  nyc_sds
)

# 7. Combine with non-numeric columns
atl_data_normalized_with_nyc <- bind_cols(
  atl_non_numeric,
  atl_normalized_with_nyc
)

# 8. Save the normalized dataset
write.csv(atl_data_normalized_with_nyc, "atl_data_normalized_with_nyc.csv", row.names = FALSE)

# 9. Generate visualization to compare distributions (optional)
# Create a function to plot comparison of distributions
plot_comparison <- function(nyc_data, atl_data, variable_name) {
  # Create a combined dataset for plotting
  nyc_plot_data <- data.frame(
    city = "NYC",
    value = nyc_data[[variable_name]]
  )
  
  atl_plot_data <- data.frame(
    city = "ATL (NYC normalized)",
    value = atl_data[[variable_name]]
  )
  
  combined_data <- rbind(nyc_plot_data, atl_plot_data)
  
  # Plot comparison
  ggplot(combined_data, aes(x = value, fill = city)) +
    geom_density(alpha = 0.5) +
    labs(
      title = paste("Comparison of", variable_name),
      x = "Normalized Value",
      y = "Density"
    ) +
    theme_minimal()
}

# Print summary statistics to verify normalization
cat("\n--- NYC Original Data Summary (Sample) ---\n")
summary(nyc_numeric[, 1:4])

cat("\n--- ATL Normalized with NYC Parameters (Sample) ---\n")
summary(atl_normalized_with_nyc[, 1:4])