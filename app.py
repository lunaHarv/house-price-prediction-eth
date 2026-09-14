import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Ethiopian Real Estate Dashboard", page_icon="🏡", layout="wide")

st.title("🏡 Real Estate Price Prediction Dashboard")
st.write("Multivariate Linear Regression Model for Property Valuation in Ethiopia (ETB)")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("full_house_price_model.pkl")

model = load_model()

# Sidebar for Key Metrics
st.sidebar.header("📊 Model Metrics")
st.sidebar.metric(label="Model R² Score", value="0.82")
st.sidebar.metric(label="Mean Absolute Error", value="~422,815 ETB")

# Input Form
st.subheader("Property Attribute Inputs")
with st.form("valuation_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Physical Structure**")
        rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=4)
        built_area = st.number_input("Built Area (sqm)", min_value=20, max_value=1000, value=200)
        site_area = st.number_input("Site Area (sqm)", min_value=50, max_value=2000, value=350)
        property_years = st.number_input("Property Age (Years)", min_value=0, max_value=50, value=10)

    with col2:
        st.markdown("**Building Type & Materials**")
        materials = st.selectbox("Construction Material", ["Concrete", "Mud&Wood"])
        typology = st.selectbox("Housing Typology", ["Detached", "Semi-detached", "Condominium"])
        land_grade = st.selectbox("Land Value Grading", ["High", "Medium", "Low"])
        road_type = st.selectbox("Type of Nearest Road", ["Asphalt", "Gravel"])

    with col3:
        st.markdown("**Location & Accessibility**")
        cbd_dist = st.number_input("Distance to CBD (km)", min_value=0.0, max_value=20.0, value=2.5)
        bus_dist = st.number_input("Distance to Bus Station (km)", min_value=0.0, max_value=10.0, value=1.2)
        school_dist = st.number_input("Distance to School (km)", min_value=0.0, max_value=10.0, value=1.5)

    submit = st.form_submit_button("Calculate Estimated Price")

if submit:
    # Format input row exactly matching dataset schema
    input_data = pd.DataFrame([{
        'Number_of_Rooms': rooms,
        'Site_Area_sqm': site_area,
        'Built_Area_sqm': built_area,
        'Property_Years': property_years,
        'Construction_Materials': materials,
        'Housing_Typology': typology,
        'Land_Value_Grading': land_grade,
        'Proximity_to_CBD_km': cbd_dist,
        'Proximity_to_Bus_Station_km': bus_dist,
        'Type_of_Nearest_Road': road_type,
        'Proximity_to_Schools_km': school_dist
    }])

    # Predict price
    predicted_price = model.predict(input_data)[0]
    
    st.divider()
    st.success(f"### Estimated Property Price: **{predicted_price:,.2f} ETB**")
