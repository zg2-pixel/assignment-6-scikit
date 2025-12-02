# Assignment 6: Concrete Strength Prediction with Scikit-Learn

## Overview
In this assignment, you will build and evaluate machine learning models to predict concrete compressive strength using the UCI Concrete Compressive Strength dataset. You will use pandas to process an Excel file and perform data exploration and analysis.

## Dataset
The dataset is provided as an Excel file (`data/concrete_data.xlsx`) containing 1030 concrete samples with 8 input features:
- Cement (kg/m3)
- Blast Furnace Slag (kg/m3)
- Fly Ash (kg/m3)
- Water (kg/m3)
- Superplasticizer (kg/m3)
- Coarse Aggregate (kg/m3)
- Fine Aggregate (kg/m3)
- Age (days)

Target: Concrete compressive strength (MPa)

## Tasks

### Task 1: Data Loading (10 points)
Complete the `load_data()` function to load the concrete dataset and return a pandas DataFrame.

### Task 2: Data Exploration with Pandas (15 points)
Complete the `explore_data()` function to:
- Display DataFrame shape and column names
- Show summary statistics using pandas describe()
- Find correlations between features and target
- Return the feature most correlated with strength

### Task 3: Data Preprocessing (20 points)
Complete the `preprocess_data()` function to:
- Split features (X) and target (y) using pandas operations
- Split into training and testing sets (80/20)
- Standardize features using StandardScaler

### Task 4: Model Training (25 points)
Complete the `train_model()` function to train a Random Forest Regressor.

### Task 5: Model Evaluation (20 points)
Complete the `evaluate_model()` function to calculate:
- Mean Squared Error (MSE)
- R-squared (R2) score

### Task 6: Feature Importance (10 points)
Complete the `get_feature_importance()` function to return the top 3 most important features as a pandas DataFrame.

## Running Your Code

```bash
# Run your implementation
python main.py

# Run tests
pytest test_main.py -v
```

## Files
- `main.py` - Your implementation (complete the TODO sections)
- `test_main.py` - Test cases for your implementation
- `solution.py` - Reference solution (do not look until finished!)
- `test_solution.py` - Tests for the solution

## Grading
Your code must pass all tests in `test_main.py` to receive full credit.
