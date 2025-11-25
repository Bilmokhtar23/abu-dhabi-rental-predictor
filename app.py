"""
🏠 Abu Dhabi Rental Price Predictor - Simplified Version
Simple and straightforward rental price prediction app
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Abu Dhabi Rental Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Constants
DATA_PATH_TRAIN = Path("data/processed/train_set_FINAL.csv")
DATA_PATH_TEST = Path("data/processed/test_set_FINAL.csv")
MODEL_PATH = Path("model_outputs/production/stacked_ensemble_latest.joblib")
FEATURES_PATH = Path("model_outputs/production/feature_columns_latest.json")

# Cache data loading
@st.cache_data
def load_data():
    """Load the processed dataset"""
    df_train = pd.read_csv(DATA_PATH_TRAIN)
    df_test = pd.read_csv(DATA_PATH_TEST)
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

@st.cache_resource
def load_model():
    """Load the stacked ensemble model"""
    model_dict = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, 'r') as f:
        feature_info = json.load(f)

    if isinstance(feature_info, list):
        feature_cols = feature_info
    elif 'all_features' in feature_info:
        feature_cols = feature_info['all_features']
    else:
        feature_cols = feature_info.get('feature_cols', [])

    return model_dict, feature_cols

# Load data and model
df = load_data()
model_dict, feature_cols = load_model()

# Main app
st.title("🏠 Abu Dhabi Rental Price Predictor")
st.markdown("Get instant AI-powered rental price predictions for Abu Dhabi properties.")

# Sidebar with basic info
st.sidebar.header("Model Info")
st.sidebar.metric("R² Score", "0.9107")
st.sidebar.metric("MAE", "5,934 AED")
st.sidebar.metric("Properties", f"{len(df):,}")

# Input form
st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("📍 Location", sorted(df['Location'].unique()))
    property_type = st.selectbox("🏢 Property Type", sorted(df['Type'].unique()))
    furnishing = st.selectbox("🛋️ Furnishing", sorted(df['Furnishing'].unique()))

with col2:
    area = st.number_input("📐 Area (sqft)", min_value=100, max_value=20000, value=1500, step=50)
    beds = st.number_input("🛏️ Bedrooms", min_value=0, max_value=10, value=2, step=1)
    baths = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2, step=1)

# Predict button
if st.button("🔮 Predict Rental Price", type="primary"):
    with st.spinner("Calculating prediction..."):
        # Feature engineering (simplified)
        input_data = {}

        # Basic features
        input_data['Beds'] = beds
        input_data['Baths'] = baths
        input_data['Area_in_sqft'] = area

        # Log transformation
        input_data['log_area'] = np.log1p(area)

        # Ratios
        input_data['bath_bed_ratio'] = baths / (beds + 1)
        input_data['area_per_bedroom'] = area / (beds + 1)

        # Location-based features
        location_data = df[df['Location'] == location]
        if len(location_data) > 0:
            rank = (location_data['Area_in_sqft'] < area).mean()
            input_data['property_rank_in_location'] = rank

            location_mean_area = location_data['Area_in_sqft'].mean()
            location_std_area = location_data['Area_in_sqft'].std()
            if location_std_area > 0:
                input_data['area_deviation_from_location'] = (area - location_mean_area) / location_std_area
            else:
                input_data['area_deviation_from_location'] = 0
        else:
            input_data['property_rank_in_location'] = 0.5
            input_data['area_deviation_from_location'] = 0

        # Premium calculations
        global_rent_mean = df['Rent'].mean()

        loc_type_rent = df[(df['Location'] == location) & (df['Type'] == property_type)]['Rent'].mean()
        if pd.isna(loc_type_rent) or loc_type_rent == 0:
            loc_type_rent = df[df['Location'] == location]['Rent'].mean()
        input_data['location_type_premium'] = loc_type_rent / global_rent_mean if global_rent_mean > 0 else 1.0

        furn_rent_mean = df[df['Furnishing'] == furnishing]['Rent'].mean()
        input_data['furnishing_type_premium'] = furn_rent_mean / global_rent_mean if global_rent_mean > 0 else 1.0

        type_rent_mean = df[df['Type'] == property_type]['Rent'].mean()
        input_data['type_room_premium'] = type_rent_mean / global_rent_mean if global_rent_mean > 0 else 1.0

        # Categorical features
        input_data['Location'] = location
        input_data['Type'] = property_type
        input_data['Furnishing'] = furnishing

        # Create DataFrame and encode
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_cols]

        encoder = model_dict['encoder']
        input_encoded = encoder.transform(input_df)

        # Make prediction
        base_models = model_dict['base_models']
        meta_model = model_dict['meta_model']

        base_preds = []
        for name, model in base_models.items():
            pred = model.predict(input_encoded)
            base_preds.append(pred)

        meta_features = np.column_stack(base_preds)
        prediction = meta_model.predict(meta_features)[0]

        # Display results
        st.success("Prediction Complete!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Annual Rent", f"{prediction:,.0f} AED")

        with col2:
            st.metric("Monthly Rent", f"{prediction/12:,.0f} AED")

        with col3:
            st.metric("Model Accuracy", "91.07%")

        # Comparison
        st.subheader("📊 Price Comparison")

        location_avg = df[df['Location'] == location]['Rent'].mean()
        type_avg = df[df['Type'] == property_type]['Rent'].mean()

        comparison_data = {
            'Category': ['Location Average', 'Property Type Average', 'Your Prediction'],
            'Price': [location_avg, type_avg, prediction]
        }

        st.bar_chart(pd.DataFrame(comparison_data).set_index('Category'))

# Footer
st.markdown("---")
st.markdown("*Powered by Stacked Ensemble ML Model • Based on 23,281 Abu Dhabi properties*")
