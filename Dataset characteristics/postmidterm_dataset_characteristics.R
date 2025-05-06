# NYC Grid Data Analysis for Neural Network Training
# This script analyzes the characteristics of the NYC grid dataset
# to inform neural network training with bike_volume as the target variable

# Load necessary libraries
library(tidyverse)
library(corrplot)
library(moments) # For skewness and kurtosis
library(caret)   # For preprocessing

# Read the data
nyc_data <- read.csv("merged_nyc_grid_data.csv")

# 1. Basic dataset characteristics
cat("DATASET DIMENSIONS\n")
cat("Number of datapoints (rows):", nrow(nyc_data), "\n")
cat("Number of features (columns):", ncol(nyc_data) - 1, "\n") # Excluding bike_volume
cat("\n")

# 2. Missing values analysis
cat("MISSING VALUES ANALYSIS\n")
missing_values <- colSums(is.na(nyc_data))
cat("Total missing values:", sum(missing_values), "\n")
if(sum(missing_values) > 0) {
  cat("Features with missing values:\n")
  print(missing_values[missing_values > 0])
}
cat("\n")

# 3. Ground truth (bike_volume) analysis
cat("GROUND TRUTH (bike_volume) ANALYSIS\n")
summary(nyc_data$bike_volume)
cat("Variance:", var(nyc_data$bike_volume), "\n")
cat("Standard deviation:", sd(nyc_data$bike_volume), "\n")
cat("Skewness:", skewness(nyc_data$bike_volume), "\n")
cat("Excess kurtosis:", kurtosis(nyc_data$bike_volume) - 3, "\n") # Excess kurtosis (normal = 0)

# Visualize bike_volume distribution
png("bike_volume_distribution.png", width = 800, height = 600)
par(mfrow = c(2, 1))
hist(nyc_data$bike_volume, breaks = 30, main = "Histogram of bike_volume", 
     xlab = "Bike Volume", col = "lightblue")
boxplot(nyc_data$bike_volume, main = "Boxplot of bike_volume", horizontal = TRUE)
dev.off()
cat("\n")

# 4. Feature analysis
cat("FEATURE ANALYSIS\n")

# Calculate basic statistics for all features
feature_stats <- data.frame(
  Feature = names(nyc_data)[1:(ncol(nyc_data)-1)],
  Min = sapply(nyc_data[, 1:(ncol(nyc_data)-1)], min, na.rm = TRUE),
  Max = sapply(nyc_data[, 1:(ncol(nyc_data)-1)], max, na.rm = TRUE),
  Mean = sapply(nyc_data[, 1:(ncol(nyc_data)-1)], mean, na.rm = TRUE),
  SD = sapply(nyc_data[, 1:(ncol(nyc_data)-1)], sd, na.rm = TRUE),
  Variance = sapply(nyc_data[, 1:(ncol(nyc_data)-1)], var, na.rm = TRUE),
  Skewness = sapply(nyc_data[, 1:(ncol(nyc_data)-1)], skewness, na.rm = TRUE),
  Kurtosis = sapply(nyc_data[, 1:(ncol(nyc_data)-1)], function(x) kurtosis(x, na.rm = TRUE) - 3)
)

# Identify features with high variance
high_var_features <- feature_stats %>%
  arrange(desc(Variance)) %>%
  head(10)
cat("Top 10 features by variance:\n")
print(high_var_features[, c("Feature", "Variance", "Mean", "SD")])
cat("\n")

# Identify features with extreme values (potential outliers)
cat("Features with extreme values (>5 SD from mean):\n")
for (col in names(nyc_data)[1:(ncol(nyc_data)-1)]) {
  feature_data <- nyc_data[[col]]
  feature_mean <- mean(feature_data, na.rm = TRUE)
  feature_sd <- sd(feature_data, na.rm = TRUE)
  
  # Count values more than 5 standard deviations from the mean
  extreme_count <- sum(abs(feature_data - feature_mean) > 5 * feature_sd, na.rm = TRUE)
  
  if (extreme_count > 0) {
    cat(col, ": ", extreme_count, " extreme values\n", sep = "")
  }
}
cat("\n")

# 5. Correlation with target variable
cat("CORRELATION WITH TARGET VARIABLE\n")
target_correlations <- sapply(nyc_data[, 1:(ncol(nyc_data)-1)], 
                              function(x) cor(x, nyc_data$bike_volume, 
                                              use = "pairwise.complete.obs"))
target_correlations_df <- data.frame(
  Feature = names(target_correlations),
  Correlation = target_correlations
) %>%
  arrange(desc(abs(Correlation)))

cat("Top 10 features by absolute correlation with bike_volume:\n")
print(head(target_correlations_df, 10))

# Visualize top correlations
top_corr_features <- head(target_correlations_df, 10)$Feature
png("top_correlations.png", width = 1000, height = 800)
par(mfrow = c(3, 4))
for (feature in top_corr_features) {
  plot(nyc_data[[feature]], nyc_data$bike_volume, 
       main = paste("bike_volume vs", feature),
       xlab = feature, ylab = "bike_volume")
  abline(lm(bike_volume ~ nyc_data[[feature]], data = nyc_data), col = "red")
}
dev.off()
cat("\n")

# 6. Feature multicollinearity
cat("FEATURE MULTICOLLINEARITY\n")
# Compute correlation matrix
correlation_matrix <- cor(nyc_data[, 1:(ncol(nyc_data)-1)], use = "pairwise.complete.obs")

# Identify highly correlated features (|r| > 0.9)
high_correlations <- which(abs(correlation_matrix) > 0.9 & abs(correlation_matrix) < 1, arr.ind = TRUE)
if (nrow(high_correlations) > 0) {
  correlated_pairs <- data.frame(
    Feature1 = rownames(correlation_matrix)[high_correlations[, 1]],
    Feature2 = colnames(correlation_matrix)[high_correlations[, 2]],
    Correlation = correlation_matrix[high_correlations]
  ) %>%
    arrange(desc(abs(Correlation)))
  
  # Remove duplicates (e.g., A-B and B-A)
  correlated_pairs <- correlated_pairs[!duplicated(t(apply(correlated_pairs[, 1:2], 1, sort))), ]
  
  cat("Highly correlated feature pairs (|r| > 0.9):\n")
  print(correlated_pairs)
  
  # Visualize correlation matrix 
  png("correlation_matrix.png", width = 1200, height = 1200)
  corrplot(correlation_matrix, method = "circle", type = "upper", 
           tl.cex = 0.6, tl.col = "black")
  dev.off()
} else {
  cat("No feature pairs with correlation > 0.9 found.\n")
}
cat("\n")

# 7. Feature variance analysis
cat("FEATURE VARIANCE ANALYSIS\n")
# Check for near-zero variance predictors
nzv_stats <- nearZeroVar(nyc_data[, 1:(ncol(nyc_data)-1)], saveMetrics = TRUE)
if (any(nzv_stats$nzv)) {
  cat("Near-zero variance features that may need to be removed:\n")
  print(rownames(nzv_stats)[nzv_stats$nzv])
} else {
  cat("No near-zero variance features found.\n")
}
cat("\n")

# 8. Recommendations for preprocessing
cat("RECOMMENDATIONS FOR NEURAL NETWORK PREPROCESSING\n")

# Check if target variable needs transformation
if (abs(skewness(nyc_data$bike_volume)) > 1) {
  cat("1. Consider log or sqrt transformation of bike_volume (skewed distribution)\n")
} else {
  cat("1. bike_volume distribution is relatively normal, transformation may not be needed\n")
}

# Feature scaling recommendation
cat("2. Scale all features (min-max or standardization) before training\n")

# High correlation handling
if (nrow(high_correlations) > 0) {
  cat("3. Address multicollinearity by either:\n")
  cat("   - Using dimensionality reduction techniques (PCA)\n")
  cat("   - Removing some highly correlated features\n")
}

# Outlier recommendation
if (any(feature_stats$Skewness > 3 | feature_stats$Skewness < -3)) {
  cat("4. Consider robust scaling or outlier removal for highly skewed features\n")
  cat("   Highly skewed features (|skewness| > 3):\n")
  print(feature_stats$Feature[feature_stats$Skewness > 3 | feature_stats$Skewness < -3])
}

# Data splitting recommendation
cat("5. Split data into training (70%), validation (15%), and test (15%) sets\n")

# Class imbalance check
if (sd(nyc_data$bike_volume) > 2 * mean(nyc_data$bike_volume)) {
  cat("6. The bike_volume has high variance; consider stratified sampling for train/test split\n")
}

cat("7. Consider using regularization techniques (L1, L2) to prevent overfitting\n")

# 9. Export preprocessed data for model training (example)
cat("\nEXAMPLE PREPROCESSING PIPELINE\n")
cat("# Example code for preprocessing:\n")
cat("
# 1. Remove near-zero variance features (if any)
if (any(nzv_stats$nzv)) {
  nyc_data_filtered <- nyc_data[, !c(names(nyc_data) %in% rownames(nzv_stats)[nzv_stats$nzv])]
} else {
  nyc_data_filtered <- nyc_data
}

# 2. Transform target variable if needed (example with log transformation)
if (min(nyc_data_filtered$bike_volume) > 0) {
  nyc_data_filtered$bike_volume_log <- log(nyc_data_filtered$bike_volume)
} else {
  nyc_data_filtered$bike_volume_log <- log(nyc_data_filtered$bike_volume + 1)
}

# 3. Apply normalization to features
preprocessParams <- preProcess(nyc_data_filtered[, 1:(ncol(nyc_data_filtered)-2)], 
                              method = c('center', 'scale'))
nyc_data_normalized <- predict(preprocessParams, nyc_data_filtered)

# 4. Split data into training, validation and test sets
set.seed(123)
trainIndex <- createDataPartition(nyc_data_normalized$bike_volume, p = 0.7, list = FALSE)
train_data <- nyc_data_normalized[trainIndex, ]
temp_data <- nyc_data_normalized[-trainIndex, ]

valIndex <- createDataPartition(temp_data$bike_volume, p = 0.5, list = FALSE)
val_data <- temp_data[valIndex, ]
test_data <- temp_data[-valIndex, ]

# Now train_data, val_data, and test_data are ready for neural network training
")

# Summary of findings
cat("\nSUMMARY OF FINDINGS\n")
cat("1. Dataset size: ", nrow(nyc_data), " observations with ", ncol(nyc_data) - 1, " features\n", sep = "")
cat("2. Target variable (bike_volume) characteristics:\n")
cat("   - Range: ", min(nyc_data$bike_volume), " to ", max(nyc_data$bike_volume), "\n", sep = "")
cat("   - Mean: ", mean(nyc_data$bike_volume), "\n", sep = "")
cat("   - Skewness: ", skewness(nyc_data$bike_volume), "\n", sep = "")

# Top features by correlation
cat("3. Top features by correlation with target:\n")
top_features <- head(target_correlations_df, 25)
for (i in 1:nrow(top_features)) {
  cat("   - ", top_features$Feature[i], ": ", round(top_features$Correlation[i], 3), "\n", sep = "")
}

# Check for major preprocessing needs
if (any(nzv_stats$nzv) || nrow(high_correlations) > 0 || 
    abs(skewness(nyc_data$bike_volume)) > 1) {
  cat("4. Key preprocessing needs identified: ")
  if (any(nzv_stats$nzv)) cat("remove near-zero variance features; ")
  if (nrow(high_correlations) > 0) cat("address multicollinearity; ")
  if (abs(skewness(nyc_data$bike_volume)) > 1) cat("transform target variable; ")
  cat("\n")
} else {
  cat("4. No major preprocessing issues identified beyond standard scaling\n")
}

# Dataset size adequacy for neural networks
if (nrow(nyc_data) < 1000) {
  cat("5. Dataset size may be small for complex neural networks; consider simpler architectures or regularization\n")
} else {
  cat("5. Dataset size appears adequate for neural network training\n")
}