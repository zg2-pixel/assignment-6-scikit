"""
Test cases for main.py
Run with: pytest test_main.py -v
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import main


def test_load_data():
    """Test that data loads correctly from Excel."""
    df = main.load_data()
    
    assert df is not None, "DataFrame should not be None"
    assert isinstance(df, pd.DataFrame), "Should return a pandas DataFrame"
    assert len(df) == 1030, "Dataset should have 1030 samples"
    assert df.shape[1] == 9, "Dataset should have 9 columns (8 features + 1 target)"
    assert not df.isnull().any().any(), "Dataset should not contain missing values"
    assert 'Concrete Compressive Strength' in df.columns, "Should have target column"


def test_explore_data():
    """Test data exploration with pandas."""
    df = main.load_data()
    most_correlated = main.explore_data(df)
    
    assert most_correlated is not None, "Should return a feature name"
    assert isinstance(most_correlated, str), "Should return a string feature name"
    assert most_correlated in df.columns, "Feature should be in the DataFrame columns"
    assert most_correlated != 'Concrete Compressive Strength', "Should not return target column"


def test_preprocess_data():
    """Test data preprocessing using pandas."""
    df = main.load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = main.preprocess_data(df)
    
    # Check shapes
    assert X_train.shape[0] == 824, "Training set should have 824 samples (80%)"
    assert X_test.shape[0] == 206, "Test set should have 206 samples (20%)"
    assert X_train.shape[1] == 8, "Should have 8 features"
    assert len(y_train) == 824, "y_train should have 824 samples"
    assert len(y_test) == 206, "y_test should have 206 samples"
    
    # Check that scaler was fitted
    assert isinstance(scaler, StandardScaler), "Should return a StandardScaler"
    assert hasattr(scaler, 'mean_'), "Scaler should be fitted"
    
    # Check that data is scaled (mean close to 0, std close to 1)
    assert np.abs(X_train.mean()) < 0.5, "Scaled training data should have mean close to 0"
    assert 0.5 < np.std(X_train) < 1.5, "Scaled training data should have std close to 1"
    
    # Check feature names
    assert len(feature_names) == 8, "Should have 8 feature names"
    assert 'Cement' in feature_names, "Should include 'Cement' feature"


def test_train_model():
    """Test model training."""
    df = main.load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = main.preprocess_data(df)
    
    model = main.train_model(X_train, y_train)
    
    assert model is not None, "Model should not be None"
    assert isinstance(model, RandomForestRegressor), "Should return a RandomForestRegressor"
    assert hasattr(model, 'estimators_'), "Model should be fitted"
    assert model.n_estimators == 100, "Model should have 100 estimators"


def test_evaluate_model():
    """Test model evaluation."""
    df = main.load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = main.preprocess_data(df)
    model = main.train_model(X_train, y_train)
    
    metrics = main.evaluate_model(model, X_test, y_test)
    
    assert isinstance(metrics, dict), "Should return a dictionary"
    assert 'mse' in metrics, "Should contain 'mse' key"
    assert 'r2' in metrics, "Should contain 'r2' key"
    assert metrics['mse'] > 0, "MSE should be positive"
    assert 0 <= metrics['r2'] <= 1, "R2 should be between 0 and 1"
    assert metrics['r2'] > 0.5, "R2 should be reasonably high (>0.5)"


def test_get_feature_importance():
    """Test feature importance extraction as pandas DataFrame."""
    df = main.load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = main.preprocess_data(df)
    model = main.train_model(X_train, y_train)
    
    top_features_df = main.get_feature_importance(model, feature_names, top_n=3)
    
    assert isinstance(top_features_df, pd.DataFrame), "Should return a pandas DataFrame"
    assert len(top_features_df) == 3, "Should return top 3 features"
    assert 'Feature' in top_features_df.columns, "Should have 'Feature' column"
    assert 'Importance' in top_features_df.columns, "Should have 'Importance' column"
    
    # Check that values are correct types
    assert all(isinstance(f, str) for f in top_features_df['Feature']), "Features should be strings"
    assert all(isinstance(i, (int, float, np.number)) for i in top_features_df['Importance']), "Importances should be numeric"
    
    # Check that DataFrame is sorted by importance (descending)
    importances = top_features_df['Importance'].tolist()
    assert importances == sorted(importances, reverse=True), "Features should be sorted by importance"


def test_end_to_end():
    """Test the entire pipeline runs without errors."""
    df = main.load_data()
    most_correlated = main.explore_data(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = main.preprocess_data(df)
    model = main.train_model(X_train, y_train)
    metrics = main.evaluate_model(model, X_test, y_test)
    top_features_df = main.get_feature_importance(model, feature_names)
    
    # Verify all components work together
    assert df is not None
    assert most_correlated is not None
    assert model is not None
    assert metrics is not None
    assert top_features_df is not None
    assert isinstance(top_features_df, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
