import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Ethiopian Real Estate Dashboard", 
    page_icon="🏡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement (Modern Cards & Clean Headers)
st.markdown("""
    <style>
    /* Main App Background Gradient */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Hero Header Banner */
    .hero-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    /* Input Card Containers */
    .input-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    /* Result Box Styling */
    .result-card {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        margin-top: 1.5rem;
    }
    .result-price {
        font-size: 2.8rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("full_house_price_model.pkl")

model = load_model()

# Sidebar: Model Performance & Information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/real-estate.png", width=80)
    st.title("Model Insights")
    
    st.metric(label="Model Accuracy (R²)", value="82.1%", delta="High Fit")
    st.metric(label="Mean Absolute Error", value="~422K ETB", delta="-Lower Variance", delta_color="normal")
    
    st.divider()
    st.markdown("### 📌 Features Included")
    st.markdown("""
    * **Size:** Built & Site Area
    * **Age:** Construction Years
    * **Location:** CBD & Transit Distances
    * **Quality:** Land Grading & Materials
    """)

# Hero Header Banner
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🏡 Ethiopian Real Estate Valuation Engine</div>
        <div class="hero-subtitle">Interactive Multivariate Price Estimation for Residential Properties</div>
    </div>
""", unsafe_allow_html=True)

# Input Sections arranged in 3 Visual Column Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("📐 Physical Structure")
    rooms = st.slider("Number of Rooms", min_value=1, max_value=8, value=4)
    built_area = st.number_input("Built Area (sqm)", min_value=20, max_value=600, value=200, step=10)
    site_area = st.number_input("Site Area (sqm)", min_value=50, max_value=1000, value=350, step=10)
    property_years = st.slider("Property Age (Years)", min_value=0, max_value=30, value=8)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("🏗️ Building Specs")
    materials = st.radio("Construction Material", ["Concrete", "Mud&Wood"], horizontal=True)
    typology = st.selectbox("Housing Typology", ["Detached", "Semi-detached", "Condominium"])
    land_grade = st.select_slider("Land Value Grading", options=["Low", "Medium", "High"], value="Medium")
    road_type = st.radio("Access Road Type", ["Asphalt", "Gravel"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.subheader("📍 Location & Access")
    cbd_dist = st.slider("Distance to CBD (km)", min_value=0.1, max_value=5.0, value=2.5, step=0.1)
    bus_dist = st.slider("Distance to Bus Station (km)", min_value=0.1, max_value=2.5, value=1.2, step=0.1)
    school_dist = st.slider("Distance to School (km)", min_value=0.2, max_value=3.0, value=1.5, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

# Predict Action Button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Calculate Market Valuation", use_container_width=True, type="primary"):
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

    predicted_price = model.predict(input_data)[0]
    
    # Styled Result Output
    st.markdown(f"""
        <div class="result-card">
            <div style="font-size: 1.2rem; text-transform: uppercase; letter-spacing: 1px;">Estimated Market Value</div>
            <div class="result-price">{predicted_price:,.2f} ETB</div>
            <div style="font-size: 0.95rem; opacity: 0.9; margin-top: 0.5rem;">
                Estimated valuation based on current multivariate regression parameters.
            </div>
        </div>
    """, unsafe_allow_html=True)
