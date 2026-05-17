import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
import joblib

def load_data():
    print("Fetching Ames Housing dataset from OpenML...")
    housing = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    X = housing.data
    y = housing.target
    return X, y

def build_pipeline(X):
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42))
    ])
    
    return pipeline

def train_and_save():
    X, y = load_data()
    
    # Drop 'Id' column from X to prevent it from being used as a feature
    if 'Id' in X.columns:
        X = X.drop('Id', axis=1)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = build_pipeline(X)
    
    # Hyperparameter tuning setup
    param_distributions = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1],
        'model__max_depth': [3, 4, 5],
        'model__subsample': [0.8, 1.0]
    }
    
    print("Training model with RandomizedSearchCV...")
    search = RandomizedSearchCV(pipeline, param_distributions, n_iter=5, cv=3, 
                                scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42, verbose=1)
    
    search.fit(X_train, y_train)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best RMSE: {-search.best_score_:.2f}")
    
    # Evaluate on test set
    best_model = search.best_estimator_
    test_score = best_model.score(X_test, y_test)
    print(f"Test R^2 Score: {test_score:.4f}")
    
    # Save the best model
    print("Saving model to model.pkl...")
    joblib.dump(best_model, 'model.pkl')
    
    # Save a sample dataframe and full data for EDA in Streamlit
    print("Saving data for Streamlit UI...")
    df = X.copy()
    df['SalePrice'] = y
    df.to_csv('ames_housing.csv', index=False)
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    train_and_save()
