import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    """
    Load the concrete dataset from Excel file using pandas.
    
    Returns:
        pd.DataFrame: DataFrame containing the concrete data
    """
    # TODO: Use pandas to read the Excel file 'data/concrete_data.xlsx'
    df = None
    
    return df


def explore_data(df):
    """
    Explore the dataset using pandas operations.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        str: Name of the feature most correlated with concrete strength
    """
    print("DATA EXPLORATION")
    
    # TODO: Print the shape of the DataFrame
    print(f"\nDataset shape: (rows, columns)")
    
    # TODO: Print the column names
    print(f"\nColumn names:")
    
    # TODO: Display summary statistics using df.describe()
    print("\nSummary Statistics:")
    
    # TODO: Calculate correlation matrix and find feature most correlated with target
    # Target column is 'Concrete Compressive Strength'
    # Use df.corr() to get correlation matrix
    # Extract correlations with target and find the maximum (excluding target itself)
    
    most_correlated_feature = None  # TODO: Find the feature name with highest correlation to target
    
    print(f"\nMost correlated feature with strength: {most_correlated_feature}")
    
    return most_correlated_feature


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the data using pandas operations.
    
    Args:
        df: DataFrame containing the data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """
    # TODO: Use pandas to separate features (X) and target (y)
    # Target column is 'Concrete Compressive Strength'
    # Hint: Use df.drop() or df.iloc[] or column selection
    X = None  # All columns except target
    y = None  # Target column only
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # TODO: Split into training and testing sets
    X_train, X_test, y_train, y_test = None, None, None, None
    
    # TODO: Create a StandardScaler and fit it on training data
    scaler = None
    
    # TODO: Transform both training and testing data
    X_train_scaled = None
    X_test_scaled = None
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def train_model(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest Regressor.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        random_state: Random seed
        
    Returns:
        RandomForestRegressor: Trained model
    """
    # TODO: Create a RandomForestRegressor with the given parameters
    model = None
    
    # TODO: Fit the model on training data
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics.
    
    Args:
        model: Trained model
        X_test: Testing features
        y_test: Testing target
        
    Returns:
        dict: Dictionary with 'mse' and 'r2' keys
    """
    # TODO: Make predictions on test data
    y_pred = None
    
    # TODO: Calculate Mean Squared Error
    mse = None
    
    # TODO: Calculate R-squared score
    r2 = None
    
    return {'mse': mse, 'r2': r2}


def get_feature_importance(model, feature_names, top_n=3):
    """
    Get the top N most important features as a pandas DataFrame.
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Feature', 'Importance'] sorted by importance
    """
    # TODO: Get feature importances from the model
    importances = None
    
    # TODO: Create a pandas DataFrame with feature names and importances
    # Columns should be: 'Feature' and 'Importance'
    importance_df = None
    
    # TODO: Sort by importance (descending) and return top_n rows
    # Use pandas sort_values() and head()
    top_features_df = None
    
    return top_features_df


def main():
    """Main function to run the entire pipeline."""
    print("Concrete Strength Prediction - Assignment 6")
    
    # Load data
    print("\n1. Loading data from Excel file...")
    df = load_data()
    print(f"   Loaded {len(df)} samples from Excel")
    
    # Explore data with pandas
    print("\n2. Exploring data with pandas...")
    most_correlated = explore_data(df)
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Train model
    print("\n4. Training Random Forest model...")
    model = train_model(X_train, y_train)
    print("   Model trained successfully")
    
    # Evaluate model
    print("\n5. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"   Mean Squared Error: {metrics['mse']:.2f}")
    print(f"   R-squared Score: {metrics['r2']:.4f}")
    
    # Feature importance as DataFrame
    print("\n6. Top 3 Most Important Features (pandas DataFrame):")
    top_features_df = get_feature_importance(model, feature_names)
    print(top_features_df.to_string(index=False))
    
if __name__ == "__main__":
    main()
