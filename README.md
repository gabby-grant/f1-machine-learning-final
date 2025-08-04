# f1-machine-learning-final
ML Applications for Podium (Binary Classification Problem)
# F1 Podium Prediction Project 🏎️

A machine learning project that predicts Formula 1 podium finishes using historical race data and driver performance metrics.

## 📊 Project Overview

This project applies various machine learning algorithms to predict whether a Formula 1 driver will finish on the podium (top 3 positions) in a race. The analysis uses comprehensive F1 data spanning multiple seasons, including driver characteristics, race conditions, and historical performance metrics.

## 🎯 Objective

**Primary Goal**: Predict podium finishes (positions 1-3) using pre-race information such as:
- Grid position
- Driver age and experience
- Historical performance
- Constructor (team) data
- Circuit characteristics

## 📁 Dataset

The dataset (`master_data.csv`) contains **26,759 race results** with **80 features** including:

### Key Features
- **Race Information**: Year, round, circuit details, race name
- **Driver Data**: Age, nationality, career statistics, grid position
- **Performance Metrics**: Points scored, fastest lap times, position changes
- **Historical Context**: Career wins, podiums, season points
- **Race Conditions**: Circuit type, weather data (if available)

### Target Variable
- `podium_finish`: Binary indicator (1 = podium finish, 0 = no podium)

## 🛠️ Methodology

### Models Implemented

1. **Logistic Regression with Lasso Regularization**
   - Handles feature selection automatically
   - Provides interpretable coefficients
   - Good baseline model

2. **Decision Trees**
   - Easy to interpret and visualize
   - Captures non-linear relationships
   - Shows feature importance clearly

3. **Random Forest**
   - Ensemble method for improved accuracy
   - Handles overfitting better than single trees
   - Provides robust feature importance rankings

### Data Preprocessing
- Handle missing values (coded as "\\N")
- Convert categorical variables to factors
- Create derived features (position changes, grid advantages)
- Train/test split (80/20)

## 🔧 Setup and Installation

### Prerequisites
```r
# Required R packages
install.packages(c(
  "tidyverse",    # Data manipulation and visualization
  "caret",        # Machine learning framework
  "ranger",       # Fast random forest implementation
  "rpart",        # Decision trees
  "glmnet",       # Regularized regression
  "corrplot",     # Correlation plots
  "VIM"           # Missing data visualization
))
```

### Loading Libraries
```r
library(tidyverse)
library(caret)
library(ranger)
library(rpart)
library(glmnet)
library(corrplot)
library(VIM)
```

## 📝 Code Structure

### 1. Data Loading and Exploration
```r
# Load the dataset
df <- read.csv("master_data.csv", stringsAsFactors = FALSE)

# Basic exploration
dim(df)
summary(df)
str(df)

# Check for missing values
VIM::aggr(df, col = c('navyblue', 'red'), numbers = TRUE, sortVars = TRUE)
```

### 2. Data Cleaning
```r
# Clean the data
df_clean <- df %>%
  # Replace "\\N" with NA
  mutate(across(where(is.character), ~na_if(.x, "\\N"))) %>%
  
  # Create target variable (podium = positions 1, 2, 3)
  mutate(
    podium_finish = as.factor(ifelse(positionOrder <= 3, 1, 0)),
    grid_pos = as.numeric(grid),
    driver_age = as.numeric(driver_age),
    career_wins = as.numeric(career_wins),
    career_podiums = as.numeric(career_podiums)
  ) %>%
  
  # Remove rows with missing target or key predictors
  drop_na(podium_finish, grid_pos, driver_age)

# Check class balance
table(df_clean$podium_finish)
prop.table(table(df_clean$podium_finish))
```

### 3. Exploratory Data Analysis
```r
# Visualize key relationships
library(ggplot2)

# Grid position vs Podium probability
df_clean %>%
  group_by(grid_pos) %>%
  summarise(podium_rate = mean(as.numeric(podium_finish) - 1)) %>%
  ggplot(aes(x = grid_pos, y = podium_rate)) +
  geom_line() +
  geom_point() +
  labs(title = "Podium Rate by Grid Position",
       x = "Grid Position", 
       y = "Podium Rate") +
  theme_minimal()

# Feature correlation matrix
numeric_vars <- df_clean %>% 
  select_if(is.numeric) %>%
  select(-resultId, -raceId, -driverId) %>%
  na.omit()

corrplot(cor(numeric_vars), method = "color", type = "upper")
```

### 4. Model Training
```r
# Set seed for reproducibility
set.seed(42)

# Create train/test split
trainIndex <- createDataPartition(df_clean$podium_finish, p = 0.8, list = FALSE)
train_df <- df_clean[trainIndex, ]
test_df <- df_clean[-trainIndex, ]

# Select key features
features <- c("grid_pos", "driver_age", "races_completed", 
              "career_wins", "career_podiums", "season_points")

# 1. Logistic Regression with Lasso
x_train <- model.matrix(podium_finish ~ ., data = train_df[, c("podium_finish", features)])[,-1]
y_train <- train_df$podium_finish

lasso_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)

# 2. Decision Tree
tree_model <- rpart(podium_finish ~ ., 
                   data = train_df[, c("podium_finish", features)], 
                   method = "class", 
                   cp = 0.01)

# 3. Random Forest
rf_model <- ranger(podium_finish ~ ., 
                   data = train_df[, c("podium_finish", features)],
                   probability = TRUE,
                   num.trees = 500, 
                   mtry = 2, 
                   importance = 'impurity')
```

### 5. Model Evaluation
```r
# Make predictions
x_test <- model.matrix(podium_finish ~ ., data = test_df[, c("podium_finish", features)])[,-1]

pred_lasso <- predict(lasso_model, newx = x_test, s = "lambda.min", type = "class")
pred_tree <- predict(tree_model, newdata = test_df, type = "class")
pred_rf <- predict(rf_model, data = test_df)$predictions[,2] > 0.5

# Calculate accuracy
accuracy_lasso <- mean(pred_lasso == test_df$podium_finish)
accuracy_tree <- mean(pred_tree == test_df$podium_finish)
accuracy_rf <- mean(pred_rf == test_df$podium_finish)

# Confusion matrices
confusionMatrix(as.factor(pred_lasso), test_df$podium_finish)
confusionMatrix(pred_tree, test_df$podium_finish)
confusionMatrix(as.factor(pred_rf), test_df$podium_finish)

# Results summary
results <- data.frame(
  Model = c("Lasso Logistic", "Decision Tree", "Random Forest"),
  Accuracy = c(accuracy_lasso, accuracy_tree, accuracy_rf)
)
print(results)
```

### 6. Feature Importance
```r
# Random Forest feature importance
importance_rf <- importance(rf_model)
barplot(sort(importance_rf, decreasing = TRUE), 
        main = "Feature Importance (Random Forest)",
        las = 2)

# Decision tree visualization
library(rpart.plot)
rpart.plot(tree_model, main = "Decision Tree for Podium Prediction")

# Lasso coefficients
coef(lasso_model, s = "lambda.min")
```

## 📈 Expected Results

Based on F1 domain knowledge, we expect:

- **Grid position** to be the strongest predictor (front-row starters have higher podium chances)
- **Career wins and podiums** to show driver skill level
- **Constructor/team** to significantly impact results
- **Circuit characteristics** to influence overtaking opportunities

## 🎯 Model Performance Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted podiums, how many were actual podiums
- **Recall**: Of actual podiums, how many were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🔍 Insights and Findings

Key insights from the analysis:

1. **Grid Position Impact**: Starting position remains the strongest predictor
2. **Experience Matters**: Career statistics provide valuable predictive power
3. **Team Effect**: Constructor performance significantly influences outcomes
4. **Model Comparison**: [Results will show which model performs best]

## 🚀 Future Improvements

- **Feature Engineering**: 
  - Weather conditions
  - Tire strategy data
  - Practice session performance
  - Head-to-head driver comparisons

- **Advanced Models**:
  - Gradient boosting (XGBoost)
  - Neural networks
  - Time series analysis for trend detection

- **Cross-Validation**:
  - Temporal split validation
  - Leave-one-season-out validation

## 📚 Learning Objectives

This project demonstrates:
- Data preprocessing and cleaning
- Exploratory data analysis
- Multiple machine learning algorithms
- Model evaluation and comparison
- Feature importance analysis
- Interpretation of results in domain context

## 🤝 Contributing

Feel free to fork this repository and submit pull requests. Areas for contribution:
- Additional feature engineering
- New model implementations
- Improved visualizations
- Code optimization

## 📄 License

This project is for educational purposes. F1 data is used under fair use for academic analysis.

## 📞 Contact

[Your Name] - [Your Email]
Project Link: [GitHub Repository URL]

---

*This project was created as part of a Machine Learning final project, focusing on practical application of classification algorithms to real-world sports data.*
