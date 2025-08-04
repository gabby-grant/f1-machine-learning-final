
# F1 Podium Prediction Stat Final Project 🏎️
ML Applications for Podium (Binary Classification Problem)
A machine learning project that predicts Formula 1 podium finishes using historical race data and driver performance metrics.

**Project Overview**

This project applies various machine learning algorithms to predict whether a Formula 1 driver will finish on the podium (top 3 positions) in a race. The analysis uses an F1 data spanning multiple seasons, including driver characteristics, race conditions, and historical performance metrics. Dataset can be accessed [here](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020). 

**Objective**

Primary Goal: Predict podium finishes (positions 1-3) using pre-race information such as:
- Grid position
- Driver age and experience
- Historical performance
- Constructor (team) data
- Circuit characteristics

**Dataset**

The dataset (`master_data.csv`) contains **26,759 race results** with **80 features** including:

Key Features:
- Race Information: Year, round, circuit details, race name
- Driver Data: Age, nationality, career statistics, grid position
- Performance Metrics: Points scored, fastest lap times, position changes
- Historical Context: Career wins, podiums, season points
- Race Conditions: Circuit type, weather data (if available)

### Target Variable
- `podium_finish`: Binary indicator (1 = podium finish, 0 = no podium)

## Methodology

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
- Handle missing values 
- Convert categorical variables to factors
- Create derived features (position changes, grid advantages)
- Train/test split (80/20)

## Code Structure

1. Data Loading and Exploration
2. Data Cleaning
3. Exploratory Data Analysis
4. Model Training
5. Model Evaluation
6. Feature Importance

## Expected Results

Based on F1 domain knowledge, we expect:

- **Grid position** to be the strongest predictor (front-row starters have higher podium chances)
- **Career wins and podiums** to show driver skill level
- **Constructor/team** to significantly impact results
- **Circuit characteristics** to influence overtaking opportunities

## Model Performance Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: Of predicted podiums, how many were actual podiums
- **Recall**: Of actual podiums, how many were correctly predicted
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

##  Insights and Findings

1. **Grid Position Impact**: Starting position remains the strongest predictor
2. **Experience Matters**: Career statistics provide valuable predictive power
3. **Model Comparison**: [Results will show which model performs best]

## Future Improvements

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

This project is for educational purposes. F1 data is used under fair use for academic analysis.

*This project was created as part of a Machine Learning final project, focusing on practical application of classification algorithms to real-world  data.*
