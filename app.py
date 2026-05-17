import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="House Price Predictor", layout="wide", page_icon="🏠")

st.title("🏠 House Price Predictor")
st.markdown("End-to-end regression pipeline — from raw data to deployed model.")

# Load model and data
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('ames_housing.csv')

# Check if model exists
if not os.path.exists('model.pkl') or not os.path.exists('ames_housing.csv'):
    st.warning("Model or data not found. Please run `python pipeline.py` first.")
    st.stop()

model = load_model()
df = load_data()

tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "🔮 Price Prediction"])

with tab1:
    st.header("Exploratory Data Analysis")
    
    st.subheader("Distribution of Sale Price")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['SalePrice'], kde=True, color='blue', ax=ax)
    ax.set_title('Sale Price Distribution')
    st.pyplot(fig)
    
    st.subheader("Correlation Heatmap (Top 10 Numeric Features)")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    top_corr_features = corr.index[abs(corr["SalePrice"]) > 0.5]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)
    
    st.subheader("Outliers Analysis (Overall Quality vs Sale Price)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=df, ax=ax2)
    ax2.set_title('Overall Quality vs Sale Price')
    st.pyplot(fig2)

with tab2:
    st.header("Predict House Price")
    st.markdown("Adjust the key features to predict the house price. Other features will be kept at their median/mode values.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Features")
        gr_liv_area = st.slider("Above Ground Living Area (sqft)", min_value=300, max_value=6000, value=1500)
        total_bsmt_sf = st.slider("Total Basement Area (sqft)", min_value=0, max_value=6000, value=1000)
        first_flr_sf = st.slider("First Floor Area (sqft)", min_value=300, max_value=5000, value=1000)
        garage_cars = st.slider("Size of garage in car capacity", min_value=0, max_value=5, value=2)
        year_built = st.slider("Original Construction Date", min_value=1870, max_value=2010, value=1970)
        
    with col2:
        st.subheader("Categorical Features")
        overall_qual = st.selectbox("Overall Material and Finish Quality", sorted(df['OverallQual'].dropna().unique()))
        exter_qual = st.selectbox("Exterior Material Quality", df['ExterQual'].dropna().unique())
        kitchen_qual = st.selectbox("Kitchen Quality", df['KitchenQual'].dropna().unique())
        neighborhood = st.selectbox("Neighborhood", sorted(df['Neighborhood'].dropna().unique()))
    
    if st.button("Predict Price", type="primary"):
        # Create a base dataframe with median/mode values from the training data
        input_data = {}
        for col in df.columns:
            if col == 'SalePrice':
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                input_data[col] = [df[col].median()]
            else:
                input_data[col] = [df[col].mode()[0]]
                
        input_df = pd.DataFrame(input_data)
        
        # Override with user inputs
        input_df['GrLivArea'] = gr_liv_area
        input_df['TotalBsmtSF'] = total_bsmt_sf
        input_df['1stFlrSF'] = first_flr_sf
        input_df['GarageCars'] = garage_cars
        input_df['YearBuilt'] = year_built
        input_df['OverallQual'] = overall_qual
        input_df['ExterQual'] = exter_qual
        input_df['KitchenQual'] = kitchen_qual
        input_df['Neighborhood'] = neighborhood
        
        # Ensure Id is removed if present
        if 'Id' in input_df.columns:
            input_df = input_df.drop('Id', axis=1)
            
        prediction = model.predict(input_df)[0]
        
        st.success(f"### Predicted House Price: ${prediction:,.2f}")
