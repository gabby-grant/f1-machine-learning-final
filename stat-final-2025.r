library(readr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(tidyr)
library(lubridate)
library(randomForest)
library(e1071)
library(caret)
library(ISLR)
library(tidyverse)      # for data manipulation
library(caret)          # for machine learning framework
library(randomForest)   # for random forest
library(rpart)          # for decision trees
library(rpart.plot)     # for tree visualization
library(glmnet)         # for regularized logistic regression
library(corrplot)       # for correlation plots
library(pROC)          # for ROC curves
circuits <- read_csv("circuits.csv")
constructor_results <- read_csv("constructor_results.csv")
constructor_standings <- read_csv("constructor_standings.csv")
constructors <- read_csv("constructors.csv")
driver_standings <- read_csv("driver_standings.csv")
drivers <- read_csv("drivers.csv")
lap_times <- read_csv("lap_times.csv")
pit_stops <- read_csv("pit_stops.csv")
qualifying <- read_csv("qualifying.csv")
races <- read_csv("races.csv")
results <- read_csv("results.csv")
seasons <- read_csv("seasons.csv")
sprint_results <- read_csv("sprint_results.csv")
status <- read_csv("status.csv")
# Helper function to convert time strings to seconds
time_to_seconds <- function(time_str) {
  if (is.na(time_str) || is.null(time_str)) return(NA)
  
  tryCatch({
    if (grepl(":", as.character(time_str))) {
      parts <- strsplit(as.character(time_str), ":")[[1]]
      if (length(parts) == 2) {
        minutes <- as.numeric(parts[1])
        seconds <- as.numeric(parts[2])
        return(minutes * 60 + seconds)
      }
    } else {
      return(as.numeric(time_str))
    }
    return(NA)
  }, error = function(e) {
    return(NA)
  })
}

# Helper function to classify circuit types
classify_circuit_type <- function(circuit_ref) {
  street_circuits <- c("monaco", "singapore", "baku", "miami", "las_vegas", "detroit", "phoenix")
  ifelse(tolower(circuit_ref) %in% street_circuits, "street", "permanent")
}

print("Helper functions defined!")
# ==== Function to create the master dataset =====
create_master_dataset <- function() {
  print("Starting to join tables...")
  
  master_data <- results %>%
    left_join(races, by = "raceId") %>%
    left_join(drivers, by = "driverId") %>%
    left_join(constructors, by = "constructorId") %>%
    left_join(circuits, by = "circuitId") %>%
    left_join(qualifying, by = c("raceId", "driverId")) %>%
    left_join(status, by = "statusId")
  
  print("Basic joins completed...")
  
  master_data$year <- year(as.Date(master_data$date))
  
  print("Adding derived features...")
  
  master_data <- master_data %>%
    mutate(
      points_scored = ifelse(is.na(points), 0, points),
      finished_race = ifelse(statusId == 1, 1, 0),
      podium_finish = ifelse(positionOrder <= 3 & finished_race == 1, 1, 0),
      points_finish = ifelse(positionOrder <= 10 & finished_race == 1, 1, 0),
      grid_position = ifelse(is.na(grid), 20, grid),
      grid_advantage = 21 - grid_position,
      circuit_type = classify_circuit_type(circuitRef),
      driver_age = ifelse(is.na(dob), 30, 
                          as.numeric(difftime(as.Date(date), as.Date(dob), units = "days")) / 365.25),
      position_change = ifelse(finished_race == 1, grid_position - positionOrder, NA)
    )
  
  print("Feature engineering completed!")
  return(master_data)
}

# Function to add historical features 
add_historical_features <- function(data) {
  
  print("Adding rolling performance metrics...")
  
  data <- data %>%
    arrange(driverId, date) %>%
    group_by(driverId) %>%
    mutate(
      # Rolling averages (last 5 races) - handle NA values
      avg_position_last5 = rollmeanr(lag(positionOrder), k = 5, fill = NA, na.rm = TRUE),
      avg_points_last5 = rollmeanr(lag(points_scored), k = 5, fill = NA, na.rm = TRUE),
      
      # DNF rate over last 10 races
      dnf_rate_last10 = rollapplyr(lag(finished_race), width = 10, 
                                   FUN = function(x) ifelse(length(x[!is.na(x)]) == 0, 0, 
                                                            1 - mean(x, na.rm = TRUE)), 
                                   fill = NA, partial = TRUE),
      
      # Experience metrics
      races_completed = row_number(),
      career_wins = cumsum(lag(ifelse(positionOrder == 1 & finished_race == 1, 1, 0), default = 0)),
      career_podiums = cumsum(lag(podium_finish, default = 0)),
      
      # Form indicators
      last_race_position = lag(positionOrder),
      last_race_points = lag(points_scored),
      
      # Season points (cumulative)
      season_points = ave(points_scored, year, FUN = cumsum)
    ) %>%
    ungroup()
  
  print("Adding constructor performance metrics...")
  
  # Add constructor performance
  data <- data %>%
    arrange(constructorId, date) %>%
    group_by(constructorId) %>%
    mutate(
      constructor_avg_position_last5 = rollmeanr(lag(positionOrder), k = 5, fill = NA, na.rm = TRUE),
      constructor_wins_season = ave(ifelse(positionOrder == 1 & finished_race == 1, 1, 0), 
                                    year, FUN = cumsum)
    ) %>%
    ungroup()
  
  print("Historical features completed!")
  return(data)
}

add_simple_historical_features <- function(data) {
  print("Adding historical features...")
  
  data <- data %>%
    arrange(driverId, date) %>%
    group_by(driverId) %>%
    mutate(
      last_race_position = lag(positionOrder),
      last_race_points = lag(points_scored),
      last_race_finished = lag(finished_race),
      races_completed = row_number(),
      career_wins = cumsum(lag(ifelse(positionOrder == 1 & finished_race == 1, 1, 0), default = 0)),
      career_podiums = cumsum(lag(podium_finish, default = 0)),
      season_points = ave(points_scored, year, FUN = cumsum)
    ) %>%
    ungroup() %>%
    arrange(constructorId, date) %>%
    group_by(constructorId) %>%
    mutate(
      constructor_last_position = lag(positionOrder)
    ) %>%
    ungroup()
  
  print("Historical features completed!")
  return(data)
}

# Function to create the master dataset
create_master_dataset <- function() {
  print("Starting to join tables...")
  
  master_data <- results %>%
    left_join(races, by = "raceId") %>%
    left_join(drivers, by = "driverId") %>%
    left_join(constructors, by = "constructorId") %>%
    left_join(circuits, by = "circuitId") %>%
    left_join(qualifying, by = c("raceId", "driverId")) %>%
    left_join(status, by = "statusId")
  
  print("Basic joins completed...")
  
  master_data$year <- year(as.Date(master_data$date))
  
  print("Adding derived features...")
  
  master_data <- master_data %>%
    mutate(
      points_scored = ifelse(is.na(points), 0, points),
      finished_race = ifelse(statusId == 1, 1, 0),
      podium_finish = ifelse(positionOrder <= 3 & finished_race == 1, 1, 0),
      points_finish = ifelse(positionOrder <= 10 & finished_race == 1, 1, 0),
      grid_position = ifelse(is.na(grid), 20, grid),
      grid_advantage = 21 - grid_position,
      circuit_type = classify_circuit_type(circuitRef),
      driver_age = ifelse(is.na(dob), 30, 
                          as.numeric(difftime(as.Date(date), as.Date(dob), units = "days")) / 365.25),
      position_change = ifelse(finished_race == 1, grid_position - positionOrder, NA)
    )
  
  print("Feature engineering completed!")
  return(master_data)
}

# Function to add historical features (simplified version)
add_simple_historical_features <- function(data) {
  print("Adding historical features...")
  
  data <- data %>%
    arrange(driverId, date) %>%
    group_by(driverId) %>%
    mutate(
      last_race_position = lag(positionOrder),
      last_race_points = lag(points_scored),
      last_race_finished = lag(finished_race),
      races_completed = row_number(),
      career_wins = cumsum(lag(ifelse(positionOrder == 1 & finished_race == 1, 1, 0), default = 0)),
      career_podiums = cumsum(lag(podium_finish, default = 0)),
      season_points = ave(points_scored, year, FUN = cumsum)
    ) %>%
    ungroup() %>%
    arrange(constructorId.x, date) %>%
    group_by(constructorId.x) %>%
    mutate(
      constructor_last_position = lag(positionOrder)
    ) %>%
    ungroup()
  
  print("Historical features completed!")
  return(data)
}

# NOW CREATE THE MASTER DATASET
print("Creating master dataset...")
master_data <- create_master_dataset()
print("Master dataset created successfully!")

# Add historical features
print("Adding historical features...")
master_data <- add_simple_historical_features(master_data)
print("All features added!")

# Check the result
print(paste("Final dataset contains", nrow(master_data), "rows and", ncol(master_data), "columns"))
head(master_data)
df <- (master_data)
# Set seed for reproducibility
set.seed(123)

# ================ data cleaning =====================

# Basic data exploration
cat("Dataset dimensions:", dim(df), "\n")
cat("Target variable (podium_finish) distribution:\n")
table(df$podium_finish)

# Check for missing values in key variables
key_vars <- c("podium_finish", "grid_position", "driver_age", "points_scored", 
              "career_wins", "career_podiums", "season_points", "races_completed")
missing_summary <- df[key_vars] %>% 
  summarise_all(~sum(is.na(.))) %>%
  gather(variable, missing_count)
#print(missing_summary)

# Verify target variable is binary (0 and 1)
cat("Unique values in podium_finish:", unique(df$podium_finish), "\n")
cat("Podium finish rate:", mean(df$podium_finish, na.rm = TRUE), "\n")

# Clean the dataset (keep ALL variables for exploration)
df_clean <- df %>%
  # Remove rows with missing target variable and ensure it's binary
  filter(!is.na(podium_finish), podium_finish %in% c(0, 1)) %>%
  # Clean and create additional features without removing variables
  mutate(
    # Clean existing variables
    grid_pos = as.numeric(grid_position),
    qual_pos = as.numeric(position.y),
    driver_age = as.numeric(driver_age),
    points = as.numeric(points_scored),
    wins = as.numeric(career_wins),
    podiums = as.numeric(career_podiums),
    season_pts = as.numeric(season_points),
    races = as.numeric(races_completed),
    team = as.factor(as.character(name.y)),
    # circuit features
    circuit_name = as.factor(as.character(circuitRef)),
    circuit_type = as.factor(as.character(circuit_type)),
    race_country = as.factor(as.character(country)),
    # Create additional engineered features
    win_rate = wins / pmax(races, 1),
    podium_rate = podiums / pmax(races, 1),
    avg_season_pts = season_pts / pmax(races, 1),
    # Add more useful features
    experience_level = case_when(
      races < 50 ~ "Rookie",
      races < 150 ~ "Experienced", 
      TRUE ~ "Veteran"
    ),
    grid_advantage = case_when(
      grid_pos <= 3 ~ "Front_Row",
      grid_pos <= 10 ~ "Top_10",
      TRUE ~ "Back_Field"
    )
  ) %>%
filter(
  !is.na(grid_pos),     # Remove missing grid positions
  grid_pos >= 1         # Remove invalid grid positions (fixes the tree!)
)
# Convert target to factor with meaningful labels for classification
df_clean$podium_finish <- factor(df_clean$podium_finish, 
                                 levels = c(0, 1), 
                                 labels = c("No_Podium", "Podium"))

cat("Cleaned dataset dimensions:", dim(df_clean), "\n")
cat("Original dataset had", ncol(df), "variables, cleaned dataset has", ncol(df_clean), "variables\n")
cat("Final target distribution:\n")
table(df_clean$podium_finish)
cat("Podium finish rate:", round(mean(df_clean$podium_finish == "Podium"), 3), "\n")
# Get numeric variables for correlation analysis
numeric_vars <- df_clean %>% 
  select_if(is.numeric) %>%
  names()

cat("Available numeric variables for analysis:", length(numeric_vars), "\n")
cat("First 10:", head(numeric_vars, 10), "\n")

# Correlation matrix of key predictors (subset for readability)
key_predictors <- c("grid_pos", "qual_pos", "driver_age", "points", "wins", "career_podiums", "last_race_position", "contructorId", "races", "win_rate", "podium_rate")

# Filter to existing variables
available_predictors <- key_predictors[key_predictors %in% names(df_clean)]
cor_matrix <- df_clean %>%
  select(all_of(available_predictors)) %>%
  cor(use = "complete.obs")

# ==== eda =====
# Plot correlation matrix
corrplot(cor_matrix, 
         method = "circle", 
         type = "upper", 
         order = "hclust", 
         tl.cex = 0.8, 
         tl.col = "black")
title("Correlation Matrix - Key Predictors", line = 2)

# Box plots for key variables
p1 <- ggplot(df_clean, aes(x = podium_finish, y = grid_pos, fill = podium_finish)) +
  geom_boxplot() + 
  labs(title = "Grid Position vs Podium Finish", y = "Grid Position") +
  theme_minimal()

p2 <- ggplot(df_clean, aes(x = podium_finish, y = win_rate, fill = podium_finish)) +
  geom_boxplot() + 
  labs(title = "Win Rate vs Podium Finish", y = "Career Win Rate") +
  theme_minimal()

# Additional exploratory plots
p3 <- ggplot(df_clean, aes(x = podium_finish, y = driver_age, fill = podium_finish)) +
  geom_boxplot() + 
  labs(title = "Driver Age vs Podium Finish", y = "Driver Age") +
  theme_minimal()

p4 <- ggplot(df_clean, aes(x = experience_level, fill = podium_finish)) +
  geom_bar(position = "fill") +
  labs(title = "Podium Rate by Experience Level", y = "Proportion") +
  theme_minimal()

p5 <- ggplot(df_clean, aes(x = podium_finish, y = career_wins, fill = podium_finish)) +
  geom_boxplot() + 
  labs(title = "Career Wins vs Podium Finish", y = "Career Wins") +
  theme_minimal()


print(p1)
print(p2)
print(p3)
print(p4)
print(p5)
# Summary average statistics by podium finish
summary_stats <- df_clean %>%
  group_by(podium_finish) %>%
  summarise(
    count = n(),
    avg_grid = mean(grid_pos, na.rm = TRUE),
    avg_age = mean(driver_age, na.rm = TRUE),
    avg_wins = mean(wins, na.rm = TRUE),
    avg_win_rate = mean(win_rate, na.rm = TRUE),
    avg_podium_rate = mean(podium_rate, na.rm = TRUE),
    .groups = 'drop'
  )
print("Summary Statistics by Podium Finish:")
print(summary_stats)
# ==================feature selection - correlation ===================
# 4. FEATURE SELECTION FOR MODELING

# Now select final features for modeling (after exploration)
# Remove highly correlated features from available predictors
high_cor <- findCorrelation(cor_matrix, cutoff = 0.8)
if(length(high_cor) > 0) {
  high_cor_names <- colnames(cor_matrix)[high_cor]
  cat("Highly correlated features to remove:", high_cor_names, "\n")
  
  # Remove these from our modeling variables
  original_modeling_vars <- c("podium_finish", "grid_pos", "driver_age", 
                              "races_completed", "win_rate","last_race_position", "constructorId",
                              "circuitId","career_podiums","qual_pos",
                              "last_race_points",
                              "circuit_type", "alt", "lat")
  
  # Keep only variables that are NOT highly correlated
  modeling_vars <- original_modeling_vars[!original_modeling_vars %in% high_cor_names]
  cat("Variables after removing highly correlated ones:", modeling_vars, "\n")
  
} else {
  cat("No highly correlated features found\n")
  modeling_vars <- c("podium_finish", "grid_pos", "driver_age", "constructorId", "circuitId",
                     "alt", "lat", "win_rate", "races_completed", "last_race_position",)
}
# Check which variables are available
available_modeling_vars <- modeling_vars[modeling_vars %in% names(df_clean)]
cat("Variables selected for modeling:", available_modeling_vars, "\n")
# Create modeling dataset
df_modeling <- df_clean %>% 
  select(all_of(available_modeling_vars)) %>%
  # Remove rows with missing values in key predictors for modeling
  filter(!is.na(grid_pos), !is.na(driver_age), !is.na(races_completed)) %>%
  drop_na()  # Remove any remaining missing values

cat("Full dataset for exploration:", nrow(df_clean), "rows,", ncol(df_clean), "columns\n")
cat("Modeling dataset:", nrow(df_modeling), "rows,", ncol(df_modeling), "columns\n")
cat("Variables kept for modeling:", names(df_modeling), "\n")

# ================ test/train split =====================

# Create train/test split (80/20) using the modeling dataset
train_index <- createDataPartition(df_modeling$podium_finish, p = 0.8, list = FALSE)
train_data <- df_modeling[train_index, ]
test_data <- df_modeling[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")

# Check class balance
print("Training Set:")
prop.table(table(train_data$podium_finish))
print("Test Set:")
prop.table(table(test_data$podium_finish))

# ==== feature selection - rf importance
rf_model <- randomForest(podium_finish ~ ., 
                         data = train_data, 
                         ntree = 500, 
                         mtry = 3,  # Choose one value instead of grid search
                         importance = TRUE)

cat("\n--- RANDOM FOREST MODEL SUMMARY ---\n")
print(rf_model)

# ==== feature importance analysis ====
cat("\n=== FEATURE IMPORTANCE ANALYSIS ===\n")

# Method 1: Using importance() function (works with direct randomForest objects)
# Extract importance matrix
importance_matrix <- importance(rf_model)
print("Importance matrix structure:")
print(head(importance_matrix))

# Create data frame - use MeanDecreaseGini for classification
# (For regression, you'd use %IncMSE)
importance_df <- data.frame(
  Feature = rownames(importance_matrix),
  Importance = importance_matrix[, "MeanDecreaseGini"]  # For classification trees
)

# Sort by importance (highest first)
importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]

# Print top 10 features
cat("\nTop 10 Most Important Features for F1 Podium Prediction:\n")
print(head(importance_df, 10))

# Create the feature importance plot (your original style)
ggplot(head(importance_df, 10), aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "forestgreen") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top 10 Feature Importance (Random Forest)", 
    subtitle = "Formula 1 Podium Finish Prediction",
    x = "Features",
    y = "Mean Decrease Gini"
  ) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 11)
  )
# ==== compare with f1 specific categories =====
cat("\n=== F1-SPECIFIC FEATURE ANALYSIS ===\n")

# Define F1 feature categories based on your dataset
f1_categories <- list(
  "Grid/Qualifying" = c("grid", "grid_pos", "grid_advantage", "qualifyId"),
  "Driver Experience" = c("driver_age", "career_wins", "career_podiums", "races_completed", "win_rate", "races"),
  "Recent Performance" = c("last_race_points", "last_race_position", "last_race_finished"),
  "Race Position" = c("position", "positionOrder", "position_change"),
  "Circuit" = c("circuit_type", "circuitId", "circuitRef"),
  "Constructor" = c("constructorId", "constructor_last_position", "constructorRef"),
  "Points/Results" = c("points", "points_scored", "points_finish")
)

# Categorize features
importance_df$Category <- "Other"
for (category in names(f1_categories)) {
  importance_df$Category[importance_df$Feature %in% f1_categories[[category]]] <- category
}

# Summary by category
category_summary <- importance_df %>%
  group_by(Category) %>%
  summarise(
    Avg_Importance = mean(Importance),
    Max_Importance = max(Importance),
    Count = n(),
    Top_Feature = Feature[which.max(Importance)],
    .groups = 'drop'
  ) %>%
  arrange(desc(Avg_Importance))

print("Feature Importance by F1 Category:")
print(category_summary)

# Plot by category
ggplot(category_summary, aes(x = reorder(Category, Avg_Importance), y = Avg_Importance)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Average Feature Importance by F1 Category",
    subtitle = "Which types of features matter most for podium finishes?",
    x = "Feature Category",
    y = "Average Mean Decrease Gini"
  ) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 12)
  )

# Show most important feature in each category
cat("\nMost important feature in each category:\n")
for (i in 1:nrow(category_summary)) {
  cat(sprintf("%s: %s (%.2f)\n", 
              category_summary$Category[i], 
              category_summary$Top_Feature[i], 
              category_summary$Max_Importance[i]))
}
# ==== model formula setup ====
# Define different model formulas to test
formula1 <- podium_finish ~ driver_age + grid_pos
formula2 <- podium_finish ~ driver_age + grid_pos + last_race_points
formula3 <- podium_finish ~ driver_age + grid_pos + last_race_position
formula5 <- podium_finish ~ races_completed + last_race_position
formula6 <- podium_finish ~ driver_age + races_completed
formula7 <- podium_finish ~ driver_age + last_race_position + races_completed
formula8 <- podium_finish ~ driver_age + last_race_position + circuit_type

formulas <- list(
  "Basic" = formula1,
  "With_Points" = formula2, 
  "With_Last Position" = formula3,
  "Recent_Performance" = formula5,
  "Driver Experience 1" = formula6,
  "Driver Experience 2" = formula7,
  "Ciruit" = formula8
)

# Initialize storage for results
model_results <- list()
# ==== logistic loop ====
logit_models <- list()
for(i in 1:length(formulas)) {
  formula_name <- names(formulas)[i]
  cat("\n--- Testing Formula:", formula_name, "---\n")
  print(formulas[[i]])
  
  # Train model
  logit_models[[formula_name]] <- glm(formulas[[i]], data = train_data, family = "binomial")
  
  # Make predictions
  pred_prob <- predict(logit_models[[formula_name]], test_data, type = "response")
  pred_class <- factor(ifelse(pred_prob > 0.5, "Podium", "No_Podium"), 
                      levels = c("No_Podium", "Podium"))
  
  # Calculate metrics
  accuracy <- mean(pred_class == test_data$podium_finish)
  auc_score <- auc(roc(as.numeric(test_data$podium_finish == "Podium"), pred_prob))
  
  # Store results
  model_results[[paste0("Logit_", formula_name)]] <- list(
    model = logit_models[[formula_name]],
    predictions = pred_class,
    probabilities = pred_prob,
    accuracy = accuracy,
    auc = auc_score,
    formula = formulas[[i]]
  )
  
  cat("Accuracy:", round(accuracy, 4), "AUC:", round(auc_score, 4), "\n")
}

# ==== rf loop =====
cat("\n=== TESTING RANDOM FOREST FORMULAS ===\n")
rf_models <- list()
for(i in 1:length(formulas)) {
  formula_name <- names(formulas)[i]
  cat("\n--- Testing Formula:", formula_name, "---\n")
  print(formulas[[i]])
  
  # Train model
  rf_models[[formula_name]] <- randomForest(formulas[[i]], data = train_data, 
                                            ntree = 500, mtry = 2, importance = TRUE)
  
  # Make predictions
  pred_class <- predict(rf_models[[formula_name]], test_data, type = "class")
  pred_prob <- predict(rf_models[[formula_name]], test_data, type = "prob")[,"Podium"]
  
  # Calculate metrics
  accuracy <- mean(pred_class == test_data$podium_finish)
  auc_score <- auc(roc(as.numeric(test_data$podium_finish == "Podium"), pred_prob))
  
  # Store results
  model_results[[paste0("RF_", formula_name)]] <- list(
    model = rf_models[[formula_name]],
    predictions = pred_class,
    probabilities = pred_prob,
    accuracy = accuracy,
    auc = auc_score,
    formula = formulas[[i]]
  )
  
  cat("Accuracy:", round(accuracy, 4), "AUC:", round(auc_score, 4), "\n")
}
# ==== decision tree loop ====
cat("\n=== TESTING DECISION TREE FORMULAS ===\n")

tree_models <- list()
for(i in 1:length(formulas)) {
  formula_name <- names(formulas)[i]
  cat("\n--- Testing Formula:", formula_name, "---\n")
  print(formulas[[i]])
  
  # Train model
  tree_models[[formula_name]] <- rpart(formulas[[i]], data = train_data, 
                                       method = "class", cp = 0.001,
                                       control = rpart.control(minsplit = 20, minbucket = 7))
  
  # Prune tree
  optimal_cp <- tree_models[[formula_name]]$cptable[which.min(tree_models[[formula_name]]$cptable[,"xerror"]),"CP"]
  tree_pruned <- prune(tree_models[[formula_name]], cp = optimal_cp)
  
  # Make predictions
  pred_class <- predict(tree_pruned, test_data, type = "class")
  pred_prob <- predict(tree_pruned, test_data, type = "prob")[,"Podium"]
  
  # Calculate metrics
  accuracy <- mean(pred_class == test_data$podium_finish)
  auc_score <- auc(roc(as.numeric(test_data$podium_finish == "Podium"), pred_prob))
  
  # Store results
  model_results[[paste0("Tree_", formula_name)]] <- list(
    model = tree_pruned,
    predictions = pred_class,
    probabilities = pred_prob,
    accuracy = accuracy,
    auc = auc_score,
    formula = formulas[[i]]
  )
  
  cat("Accuracy:", round(accuracy, 4), "AUC:", round(auc_score, 4), "\n")
}

# === compare moels and formulas =====
cat("\n=== FORMULA COMPARISON RESULTS ===\n")

# Create comparison table
comparison_df <- data.frame(
  Model_Formula = names(model_results),
  Accuracy = sapply(model_results, function(x) round(x$accuracy, 4)),
  AUC = sapply(model_results, function(x) round(x$auc, 4)),
  stringsAsFactors = FALSE
)

# Add model type and formula name
comparison_df$Model_Type <- sub("_.*", "", comparison_df$Model_Formula)
comparison_df$Formula_Name <- sub(".*_", "", comparison_df$Model_Formula)

# Print sorted by AUC
comparison_df_sorted <- comparison_df[order(-comparison_df$AUC), ]
print(comparison_df_sorted)

# Find best performing combination
best_combo <- comparison_df_sorted[1, ]
cat("\nBest performing combination:\n")
cat("Model:", best_combo$Model_Formula, "\n")
cat("Accuracy:", best_combo$Accuracy, "\n") 
cat("AUC:", best_combo$AUC, "\n")

# Print the formula of best model
best_model_result <- model_results[[best_combo$Model_Formula]]
cat("Best Formula:\n")
print(best_model_result$formula)

# Store the best models for later use
best_model <- best_model_result$model
best_predictions <- best_model_result$predictions
best_probabilities <- best_model_result$probabilities

# ================evaluation =====================


# Get the top 3 performing models for detailed evaluation
top_3_models <- head(comparison_df_sorted, 3)

cat("\n=== DETAILED EVALUATION OF TOP 3 MODELS ===\n")

for(i in 1:nrow(top_3_models)) {
  model_name <- top_3_models$Model_Formula[i]
  model_result <- model_results[[model_name]]
  
  cat("\n", rep("=", 50), "\n")
  cat("RANK", i, ":", model_name, "\n")
  cat("Accuracy:", model_result$accuracy, "| AUC:", model_result$auc, "\n")
  cat("Formula:", deparse(model_result$formula), "\n")
  cat(rep("=", 50), "\n")
  
  # Create detailed confusion matrix
  conf_matrix <- confusionMatrix(model_result$predictions, test_data$podium_finish, positive = "Podium")
  print(conf_matrix)
  
  cat("\n")
}

# Create comparison visualizations
cat("\n=== VISUAL COMPARISONS ===\n")

# ROC Curves for top models
plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1), 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = "ROC Curves - Formula Comparison")
colors <- c("red", "blue", "green", "purple", "orange")
legend_names <- c()

for(i in 1:min(5, length(model_results))) {
  model_name <- names(model_results)[i]
  test_labels_numeric <- as.numeric(test_data$podium_finish == "Podium")
  roc_obj <- roc(test_labels_numeric, model_results[[model_name]]$probabilities)
  lines(roc_obj, col = colors[i], lwd = 2)
  legend_names <- c(legend_names, paste0(model_name, " (AUC=", round(auc(roc_obj), 3), ")"))
}
legend("bottomright", legend = legend_names, col = colors[1:length(legend_names)], lty = 1, lwd = 2)

# Performance comparison bar plot
library(ggplot2)
performance_plot <- ggplot(comparison_df_sorted, aes(x = reorder(Model_Formula, AUC), y = AUC, fill = Model_Type)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Model Performance Comparison (by AUC)", 
       x = "Model + Formula Combination", y = "AUC Score") +
  theme_minimal() +
  geom_text(aes(label = round(AUC, 3)), hjust = -0.1, size = 3)

print(performance_plot)

# ==== logistic regression w lasso - points ====
cat("\n=== TRAINING LOGISTIC REGRESSION ===\n")

# Simple logistic regression
logit_model_points <- glm(podium_finish ~ driver_age + grid_pos + last_race_points, data = train_data, family = "binomial")
print(summary(logit_model_points))

# Regularized logistic regression (LASSO)
x_train <- model.matrix(podium_finish ~ driver_age + grid_pos + last_race_points, data = train_data)[,-1]
y_train <- train_data$podium_finish
x_test <- model.matrix(podium_finish ~ driver_age + grid_pos + last_race_points, data = test_data)[,-1]

# Cross-validation to find optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, nfolds = 5)
plot(cv_lasso)
title("CV Lasso Model - Points")

# Fit LASSO model with optimal lambda
lasso_model_points <- glmnet(x_train, y_train, family = "binomial", 
                      alpha = 1, lambda = cv_lasso$lambda.min)
cat("Optimal lambda:", cv_lasso$lambda.min, "\n")
cat("LASSO Coefficients:\n")
print(coef(lasso_model_points))


# ==== rf - position ====
rf_model_position <- randomForest(podium_finish ~ driver_age + grid_pos + last_race_position, 
                         data = train_data, 
                         ntree = 500, 
                         mtry = 3,  # Choose one value instead of grid search
                         importance = TRUE)

print(rf_model_position)
varImpPlot(rf_model_position, main = "Variable Importance - Position Formula")

# ==== Make predictions on test set ====
# Make predictions on test set
logit_model <- logit_model_points
pred_logit <- predict(logit_model, test_data, type = "response")
pred_logit_class <- factor(ifelse(pred_logit > 0.5, "Podium", "No_Podium"), 
                           levels = c("No_Podium", "Podium"))
lasso_model <- lasso_model_points
pred_lasso <- predict(lasso_model, x_test, type = "response", s = cv_lasso$lambda.min)
pred_lasso_class <- factor(ifelse(pred_lasso > 0.5, "Podium", "No_Podium"), 
                           levels = c("No_Podium", "Podium"))

rf_model <- rf_model_position
pred_rf <- predict(rf_model, test_data, type = "class")
pred_rf_prob <- predict(rf_model, test_data, type = "prob")[,"Podium"]

# Convert test labels to numeric for ROC calculation (1 for Podium, 0 for No_Podium)
test_labels_numeric <- as.numeric(test_data$podium_finish == "Podium")

# Calculate accuracy for each model
accuracy_logit <- mean(pred_logit_class == test_data$podium_finish)
accuracy_lasso <- mean(pred_lasso_class == test_data$podium_finish)
accuracy_rf <- mean(pred_rf == test_data$podium_finish)
# Create detailed confusion matrices with full statistics
cat("\n=== DETAILED MODEL EVALUATION ===\n")
cat("\n--- LOGISTIC REGRESSION RESULTS ---\n")
conf_logit <- confusionMatrix(pred_logit_class, test_data$podium_finish, positive = "Podium")
print(conf_logit)
cat("\n--- LASSO LOGISTIC REGRESSION RESULTS ---\n")
conf_lasso <- confusionMatrix(pred_lasso_class, test_data$podium_finish, positive = "Podium")
print(conf_lasso)
cat("\n--- RANDOM FOREST RESULTS ---\n")
cat("Random Forest Model Call:\n")
print(rf_model$call)
cat("\n")
conf_rf <- confusionMatrix(pred_rf, test_data$podium_finish, positive = "Podium")
print(conf_rf)
# Compile results
results_summary <- data.frame(
  Model = c("Logistic Regression", "LASSO Logistic", "Random Forest"),
  Accuracy = c(accuracy_logit, accuracy_lasso, accuracy_rf),
  Sensitivity = c(conf_logit$byClass["Sensitivity"], 
                  conf_lasso$byClass["Sensitivity"],
                  conf_rf$byClass["Sensitivity"]),
  Specificity = c(conf_logit$byClass["Specificity"], 
                  conf_lasso$byClass["Specificity"],
                  conf_rf$byClass["Specificity"])
)

print("=== MODEL COMPARISON RESULTS ===")
print(results_summary)

# ROC Curves
roc_logit <- roc(test_labels_numeric, pred_logit)
roc_lasso <- roc(test_labels_numeric, as.numeric(pred_lasso))
roc_rf <- roc(test_labels_numeric, pred_rf_prob)

# Plot ROC curves
plot(roc_logit, col = "red", main = "ROC Curves Comparison")
lines(roc_lasso, col = "blue")
lines(roc_rf, col = "purple")
legend("bottomright", legend = c("Logistic", "LASSO", "Random Forest"),
       col = c("red", "blue", "purple"), lty = 1)



# AUC values
auc_values <- data.frame(
  Model = c("Logistic Regression", "LASSO Logistic", "Random Forest"),
  AUC = c(auc(roc_logit), auc(roc_lasso), auc(roc_rf))
)

print("=== AUC VALUES ===")
print(auc_values)

# Select best model based on AUC
best_model_idx <- which.max(auc_values$AUC)
best_model_name <- auc_values$Model[best_model_idx]
best_auc <- auc_values$AUC[best_model_idx]

cat("\n=== FINAL MODEL SELECTION ===\n")
cat("Best performing model:", best_model_name, "\n")
cat("Best AUC score:", round(best_auc, 4), "\n")

# Feature importance from best model (if Random Forest)
if(best_model_name == "Random Forest") {
  cat("\nTop 3 most important features:\n")
  importance_df <- data.frame(
    Feature = rownames(importance(rf_model)),
    Importance = importance(rf_model)[,1]
  ) %>% arrange(desc(Importance))
  print(head(importance_df, 3))
}
